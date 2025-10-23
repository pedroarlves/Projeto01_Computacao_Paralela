/*
================================================================================
main.cc - Implementação Paralela K-Means com OpenMP
================================================================================

DESCRIÇÃO:
  Implementação do algoritmo K-Means utilizando paralelização OpenMP.
  O algoritmo agrupa dados em K clusters minimizando a distância euclidiana
  entre os pontos e os centróides.

USO:
  ./main <dataset.csv> <K> <max_iter> [seed]
  
  Parâmetros:
    - dataset.csv: arquivo CSV onde cada linha é um ponto (valores separados por vírgula)
    - K: número de clusters desejado
    - max_iter: número máximo de iterações
    - seed: semente aleatória (opcional, para reprodutibilidade)

EXEMPLO:
  OMP_NUM_THREADS=4 ./main dataset/mnist_train.csv 15 100 42

================================================================================
TEMPOS DE EXECUÇÃO (medidos no servidor parcode)
================================================================================
Dataset: MNIST train (60.000 amostras, 785 dimensões)
Parâmetros: K=15 clusters, max_iter=100

Versão OpenMP:
  1 thread:  51.4609 s  (baseline sequencial)
  2 threads: 28.016 s   (speedup: 1.84x)
  4 threads: 17.3703 s  (speedup: 2.96x)
  8 threads: 12.275 s   (speedup: 4.19x)

================================================================================
MUDANÇAS REALIZADAS PARA PARALELIZAÇÃO (conforme exigido pelo enunciado)
================================================================================

1. PARALELIZAÇÃO DO LOOP PRINCIPAL (Assignment Step):
   - Usado #pragma omp parallel para criar região paralela
   - Cada thread processa um subconjunto dos pontos (scheduling estático)
   - Cada thread calcula distâncias e atribui pontos aos clusters
   - Cada thread mantém acumuladores locais (local_sum e local_count)

2. REDUÇÃO MANUAL DOS RESULTADOS:
   - Cada thread acumula suas somas e contagens localmente
   - Ao final do loop paralelo, usa #pragma omp critical para somar
     os acumuladores locais aos acumuladores globais
   - Evita race conditions sem usar redução automática

3. ATUALIZAÇÃO DOS CENTRÓIDES:
   - Feita sequencialmente após a região paralela
   - Calcula novos centróides: centroid[k] = sum[k] / count[k]
   - Verifica convergência comparando centróides antigos e novos

4. ESTRUTURA DE DADOS:
   - mat (matriz de doubles): representa o dataset e centróides
   - vec (vetor de doubles): representa um ponto ou centróide
   - Armazenamento contíguo facilita acesso paralelo

DIRETIVAS OpenMP UTILIZADAS:
  - #pragma omp parallel: cria região paralela
  - #pragma omp for schedule(static): divide iterações entre threads
  - #pragma omp critical: protege seção crítica (redução manual)
  - omp_get_thread_num(): identifica cada thread
  - omp_get_num_threads(): número total de threads

OBSERVAÇÕES:
  - Versão sequencial disponível em: implementação original (base deste código)
  - Dataset MNIST disponível em: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
  - Para compilar: cmake --build build
  - Para executar: OMP_NUM_THREADS=<n> ./build/main <dataset> <K> <max_iter>

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
