/*
================================================================================
main.cc - Implementação K-Means com OpenMP
================================================================================

TEMPOS DE EXECUÇÃO (servidor parcode)
Dataset: MNIST train (60.000 amostras, 785 dimensões), K=15, max_iter=100

Versão sequencial (1 thread):  51.4609 s
Versão paralela:
  2 threads: 28.016 s   (speedup: 1.84x)
  4 threads: 17.3703 s  (speedup: 2.96x)
  8 threads: 12.275 s   (speedup: 4.19x)

================================================================================
MUDANÇAS REALIZADAS PARA PARALELIZAÇÃO
================================================================================

1. PARALELIZAÇÃO DO LOOP DE ATRIBUIÇÃO:
   - Adicionado #pragma omp parallel para criar região paralela
   - Adicionado #pragma omp for schedule(static) para dividir pontos entre threads
   - Cada thread processa subconjunto independente de pontos

2. ACUMULADORES LOCAIS POR THREAD:
   - Cada thread mantém local_sum e local_count privativos
   - Evita contenção e race conditions durante processamento paralelo

3. REDUÇÃO MANUAL COM SEÇÃO CRÍTICA:
   - Adicionado #pragma omp critical para somar acumuladores locais aos globais
   - Redução ocorre apenas uma vez por thread, minimizando overhead

4. ATUALIZAÇÃO SEQUENCIAL DOS CENTRÓIDES:
   - Cálculo de novos centróides feito fora da região paralela
   - Verificação de convergência permanece sequencial

================================================================================
*/

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

using vec = vector<double>;
using mat = vector<vec>;

// parse CSV line (floats separated by comma)
static bool parse_csv_line(const string &line, vec &out) {
    out.clear();
    string cur;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            if (!cur.empty()) {
                try {
                    out.push_back(stod(cur));
                } catch (...) { return false; }
                cur.clear();
            } else {
                // treat empty as 0
                out.push_back(0.0);
            }
        } else if (!isspace((unsigned char)line[i])) {
            cur.push_back(line[i]);
        }
    }
    return !out.empty();
}

// Load CSV into matrix (rows x cols)
mat load_csv(const string &path) {
    ifstream in(path);
    if (!in) throw runtime_error("Cannot open file: " + path);
    mat data;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        vec row;
        if (parse_csv_line(line, row)) data.push_back(std::move(row));
    }
    return data;
}

// Euclidean squared distance
inline double dist2(const vec &a, const vec &b) {
    double s = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// K-means sequential
struct KMeansResult {
    mat centroids;
    vector<int> labels;
    int iterations;
};

KMeansResult kmeans_openmp(const mat &data, int K, int max_iter, mt19937 &rng) {
    size_t N = data.size();
    if (N == 0) throw runtime_error("Empty dataset");
    size_t D = data[0].size();

    // inicialização dos centróides
    mat centroids;
    centroids.reserve(K);
    unordered_set<int> chosen;
    uniform_int_distribution<int> uid(0, (int)N - 1);
    while ((int)centroids.size() < K) {
        int idx = uid(rng);
        if (chosen.insert(idx).second) centroids.push_back(data[idx]);
    }

    vector<int> labels(N, -1);
    mat new_sum(K, vec(D, 0.0));
    vector<int> new_count(K, 0);

    int iter = 0;
    for (; iter < max_iter; ++iter) {
        // zera somas globais
        for (int k = 0; k < K; ++k) {
            fill(new_sum[k].begin(), new_sum[k].end(), 0.0);
            new_count[k] = 0;
        }

        // ========= PARALLELIZAÇÃO OpenMP =========
        int nthreads = 0;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();

            // cada thread mantém seus acumuladores locais
            mat local_sum(K, vec(D, 0.0));
            vector<int> local_count(K, 0);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < N; ++i) {
                double best = numeric_limits<double>::infinity();
                int bi = -1;
                for (int k = 0; k < K; ++k) {
                    double d = dist2(data[i], centroids[k]);
                    if (d < best) { best = d; bi = k; }
                }
                labels[i] = bi;
                for (size_t d = 0; d < D; ++d)
                    local_sum[bi][d] += data[i][d];
                local_count[bi]++;
            }

            // redução manual no final
            #pragma omp critical
            {
                for (int k = 0; k < K; ++k) {
                    new_count[k] += local_count[k];
                    for (size_t d = 0; d < D; ++d)
                        new_sum[k][d] += local_sum[k][d];
                }
            }
        }
        // ==========================================

        // atualização dos centróides
        bool converged = true;
        for (int k = 0; k < K; ++k) {
            if (new_count[k] == 0) continue;
            vec updated(D);
            for (size_t d = 0; d < D; ++d)
                updated[d] = new_sum[k][d] / new_count[k];
            if (dist2(updated, centroids[k]) > 1e-12)
                converged = false;
            centroids[k].swap(updated);
        }

        if (converged) break;
    }

    return {centroids, labels, iter + 1};
}

int main(int argc, char **argv) {

    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " data.csv K max_iter [seed]\n";
        return 1;
    }
    string path = argv[1];
    int K = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    unsigned seed = (argc >=5) ? (unsigned)atoi(argv[4]) : (unsigned)chrono::high_resolution_clock::now().time_since_epoch().count();

    cerr << "Loading data...\n";
    mat data = load_csv(path);
    cerr << "Loaded " << data.size() << " points with dimensionality " << data[0].size() << "\n";

    mt19937 rng(seed);

    auto t0 = chrono::high_resolution_clock::now();
    KMeansResult res = kmeans_openmp(data, K, max_iter, rng);
    auto t1 = chrono::high_resolution_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();

    cerr << "KMeans finished in " << res.iterations << " iterations.\n";
    cerr << "Elapsed time (seq): " << secs << " s\n";

    // // print centroids (first few elements) to stdout
    // cout.setf(std::ios::fixed); cout << setprecision(6);
    // for (int k = 0; k < K; ++k) {
    //     for (size_t d = 0; d < res.centroids[k].size(); ++d) {
    //         if (d) cout << ',';
    //         cout << res.centroids[k][d];
    //     }
    //     cout << '\n';
    // }

    return 0;
}
