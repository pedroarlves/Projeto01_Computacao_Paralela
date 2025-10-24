/*
================================================================================
main_hybrid.cc - Implementação K-Means Híbrida (MPI + OpenMP)
================================================================================

TEMPOS DE EXECUÇÃO (servidor parcode)
Dataset: MNIST train (60.000 amostras, 785 dimensões), K=15, max_iter=100

VERSÃO MPI+OpenMP (Híbrida):
  1 processo × 4 threads: 48.6451 s
  2 processos × 2 threads: 28.0142 s
  4 processos × 1 thread:  28.0967 s

VERSÃO OpenMP (para comparação):
  1 thread:  51.4609 s
  2 threads: 28.016 s
  4 threads: 17.3703 s
  8 threads: 12.275 s

================================================================================
MUDANÇAS REALIZADAS PARA PARALELIZAÇÃO
================================================================================

1. DISTRIBUIÇÃO DE DADOS COM MPI:
   - Rank 0 carrega dataset e faz broadcast de metadados (N, D)
   - Dataset dividido entre processos usando MPI_Scatterv
   - Cada processo recebe subconjunto de pontos para processar

2. INICIALIZAÇÃO E SINCRONIZAÇÃO DE CENTRÓIDES:
   - Rank 0 seleciona K centróides iniciais aleatoriamente
   - Centróides enviados para todos os processos via MPI_Bcast

3. PARALELIZAÇÃO HÍBRIDA NO LOOP DE ATRIBUIÇÃO:
   - Nível MPI: cada processo trabalha em seu subconjunto local
   - Nível OpenMP: threads paralelizam cálculo de distâncias no processo
   - Cada thread mantém buffers privativos (thread_sum_flat, thread_count)

4. REDUÇÃO EM DOIS NÍVEIS:
   - Nível OpenMP: buffers de threads somados aos buffers do processo
   - Nível MPI: MPI_Allreduce combina somas/contagens de todos processos

5. ATUALIZAÇÃO REPLICADA DE CENTRÓIDES:
   - Cada processo calcula novos centróides usando resultados globais
   - Centróides replicados e sincronizados em todos os processos

6. COLETA DE RESULTADOS:
   - Labels locais coletados no rank 0 via MPI_Gatherv
   - Rank 0 reconstrói vetor completo de labels

7. SINCRONIZAÇÃO E TIMING:
   - MPI_Barrier para medição precisa de tempo
   - MPI_Wtime para timing consistente entre processos

================================================================================
*/

#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>

using namespace std;
using vec = vector<double>;
using mat = vector<vec>;

static bool parse_csv_line(const string &line, vec &out) {
    out.clear();
    string cur;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (!cur.empty()) {
                out.push_back(stod(cur));
                cur.clear();
            } else {
                out.push_back(0.0);
            }
        } else if (!isspace((unsigned char)line[i])) cur.push_back(line[i]);
    }
    return !out.empty();
}

mat load_csv(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open file: " + path);
    mat data;
    string line;
    // skip header if present
    if (getline(in, line)) {
        // simple heuristic: if the first line contains non-digit letters, treat as header
        bool has_alpha = false;
        for (char c : line) if (isalpha((unsigned char)c)) { has_alpha = true; break; }
        if (!has_alpha) {
            // first line is numeric -> parse it
            vec row;
            if (parse_csv_line(line, row)) data.push_back(move(row));
        }
    }
    while (getline(in, line)) {
        if (line.empty()) continue;
        vec row;
        if (parse_csv_line(line, row)) data.push_back(move(row));
    }
    return data;
}

inline double dist2(const vec &a, const vec &b) {
    double s = 0.0;
    size_t n = min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

struct KMeansResult {
    mat centroids;
    vector<int> labels; // only filled on rank 0 after gather
    int iterations;
};

KMeansResult kmeans_mpi_omp(const mat &data_full_root, int K, int max_iter, mt19937 &rng) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root (rank 0) knows N_total and D; broadcast them to everyone.
    int N_total = 0;
    int D = 0;
    if (rank == 0) {
        N_total = (int)data_full_root.size();
        if (N_total > 0) D = (int)data_full_root[0].size();
    }
    MPI_Bcast(&N_total, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (N_total == 0 || D == 0) throw runtime_error("Empty dataset or unknown dimensionality.");

    // compute counts and displacements (in number of samples)
    vector<int> counts(size, 0), displs(size, 0);
    int base = N_total / size;
    int rem = N_total % size;
    for (int i = 0; i < size; ++i) {
        counts[i] = base + (i < rem ? 1 : 0);
        displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
    }
    int local_N = counts[rank];

    // Prepare flattened send buffer on root
    vector<double> sendbuf;
    if (rank == 0) {
        sendbuf.reserve((size_t)N_total * D);
        for (const auto &row : data_full_root) {
            sendbuf.insert(sendbuf.end(), row.begin(), row.end());
        }
    }

    // prepare recv buffer for local data (flattened)
    vector<double> recvbuf((size_t)local_N * D);
    vector<int> counts_d(size), displs_d(size);
    for (int i = 0; i < size; ++i) {
        counts_d[i] = counts[i] * D;
        displs_d[i] = displs[i] * D;
    }

    MPI_Scatterv(rank == 0 ? sendbuf.data() : nullptr, counts_d.data(), displs_d.data(), MPI_DOUBLE,
                 recvbuf.data(), local_N * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // build local_data matrix
    mat local_data(local_N, vec(D));
    for (int i = 0; i < local_N; ++i)
        for (int j = 0; j < D; ++j)
            local_data[i][j] = recvbuf[(size_t)i * D + j];

    // initialize centroids on root by random sampling unique indices
    mat centroids(K, vec(D, 0.0));
    if (rank == 0) {
        unordered_set<int> chosen;
        uniform_int_distribution<int> uid(0, N_total - 1);
        for (int k = 0; k < K; ++k) {
            int idx;
            do { idx = uid(rng); } while (!chosen.insert(idx).second);
            centroids[k] = data_full_root[idx];
        }
    }

    // Broadcast initial centroids to all ranks (flattened)
    vector<double> cent_flat(K * D);
    if (rank == 0) {
        for (int k = 0; k < K; ++k)
            for (int d = 0; d < D; ++d)
                cent_flat[k * D + d] = centroids[k][d];
    }
    MPI_Bcast(cent_flat.data(), K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        for (int k = 0; k < K; ++k) {
            for (int d = 0; d < D; ++d) centroids[k][d] = cent_flat[k * D + d];
        }
    }

    vector<int> local_labels(local_N, -1);
    int iter;
    const double tol = 1e-6; // loosened tolerance

    for (iter = 0; iter < max_iter; ++iter) {
        // local buffers for sums and counts
        vector<double> local_sum_flat((size_t)K * D, 0.0);
        vector<int> local_count(K, 0);

        int nthreads = omp_get_max_threads();
        vector<vector<double>> thread_sum_flat(nthreads, vector<double>((size_t)K * D, 0.0));
        vector<vector<int>> thread_count(nthreads, vector<int>(K, 0));

        // parallel assign step: each thread writes to its private buffer
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static)
            for (int i = 0; i < local_N; ++i) {
                double best = numeric_limits<double>::infinity();
                int bi = -1;
                for (int k = 0; k < K; ++k) {
                    // compute distance between local_data[i] and centroids[k]
                    double s = 0.0;
                    for (int d = 0; d < D; ++d) {
                        double diff = local_data[i][d] - centroids[k][d];
                        s += diff * diff;
                    }
                    if (s < best) { best = s; bi = k; }
                }
                local_labels[i] = bi;
                // accumulate to thread-local buffers
                for (int d = 0; d < D; ++d)
                    thread_sum_flat[tid][(size_t)bi * D + d] += local_data[i][d];
                thread_count[tid][bi]++;
            }
        }

        // reduce thread buffers into local_sum_flat and local_count
        for (int t = 0; t < nthreads; ++t) {
            for (int k = 0; k < K; ++k) {
                local_count[k] += thread_count[t][k];
                for (int d = 0; d < D; ++d)
                    local_sum_flat[(size_t)k * D + d] += thread_sum_flat[t][(size_t)k * D + d];
            }
        }

        // MPI Allreduce for sums (flattened) and counts
        vector<double> global_sum_flat((size_t)K * D, 0.0);
        vector<int> global_count(K, 0);
        MPI_Allreduce(local_sum_flat.data(), global_sum_flat.data(), K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_count.data(), global_count.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // update centroids and check convergence
        bool converged_local = true;
        for (int k = 0; k < K; ++k) {
            if (global_count[k] == 0) continue; // leave centroid unchanged
            vec updated(D);
            for (int d = 0; d < D; ++d)
                updated[d] = global_sum_flat[(size_t)k * D + d] / global_count[k];
            if (dist2(updated, centroids[k]) > tol) converged_local = false;
            centroids[k].swap(updated);
        }

        int local_conv_int = converged_local ? 1 : 0;
        int all_conv = 0;
        MPI_Allreduce(&local_conv_int, &all_conv, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (all_conv) break;
    }

    // gather labels to root
    vector<int> global_labels;
    if (rank == 0) global_labels.resize(N_total);
    MPI_Gatherv(local_labels.data(), local_N, MPI_INT,
                rank == 0 ? global_labels.data() : nullptr,
                counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    return {centroids, global_labels, iter + 1};
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 4) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " data.csv K max_iter [seed]\n";
        MPI_Finalize();
        return 1;
    }

    string path = argv[1];
    int K = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    unsigned seed = (argc >= 5) ? (unsigned)atoi(argv[4])                               : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();

    mat data; // only filled on rank 0
    if (rank == 0) {
        try {
            cerr << "Loading data from " << path << "...\n";
            data = load_csv(path);
            cerr << "Loaded " << data.size() << " samples. Dim=" << (data.empty() ? 0 : data[0].size()) << "\n";
        } catch (const exception &e) {
            cerr << "Error loading CSV: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // make RNG; different seed per rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    mt19937 rng(seed + world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    KMeansResult res = kmeans_mpi_omp(data, K, max_iter, rng);
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        cerr << "Converged in " << res.iterations << " iterations.\n";
        cerr << "Elapsed time: " << (t1 - t0) << " s\n";
        // Optionally print centroids or save labels
    }

    MPI_Finalize();
    return 0;
}
