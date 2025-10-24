
# Projeto 01 - Computação Paralela
## Implementação Paralela de K-Means com OpenMP e MPI

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5+-green.svg)](https://www.openmp.org/)
[![MPI](https://img.shields.io/badge/MPI-3.0+-orange.svg)](https://www.mpi-forum.org/)

---

## 📋 Sobre o Projeto

Este projeto implementa o **algoritmo K-Means clustering** utilizando técnicas de paralelização:
- **OpenMP**: paralelismo de memória compartilhada
- **MPI + OpenMP**: paralelismo híbrido (memória distribuída + compartilhada)

### O que é K-Means?
K-Means é um algoritmo de aprendizado de máquina não supervisionado que agrupa dados em K clusters, minimizando a distância euclidiana entre os pontos e os centróides de seus respectivos clusters. É amplamente utilizado em análise de dados, reconhecimento de padrões e compressão de imagens.

### Dataset Utilizado
**MNIST** (Modified National Institute of Standards and Technology)
- 60.000 imagens de dígitos manuscritos (0-9)
- 784 features por imagem (28×28 pixels)
- Formato CSV para processamento eficiente
- Download: [Kaggle - MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

### Desenvolvimento
Código desenvolvido especificamente para este projeto educacional, implementando K-Means com extensões para paralelização OpenMP e MPI.


---

## 📁 Estrutura do Projeto

```
Projeto01_Computacao_Paralela/
├── main.cc                    # Implementação OpenMP
├── main_hybrid.cc             # Implementação híbrida MPI+OpenMP
├── CMakeLists.txt             # Configuração de build
├── README.md                  # Este arquivo
├── TESTES.txt                 # Resultados dos benchmarks
├── run_benchmarks.sh          # Script automatizado de testes
├── run_quick_test.sh          # Script de teste rápido
├── BENCHMARK_SCRIPTS.txt      # Documentação dos scripts
├── QUICK_START.txt            # Guia de início rápido
├── example_output.csv         # Exemplo de saída CSV
├── dataset/
│   ├── mnist_train.csv        # Dataset principal (60k amostras) - não versionado
│   └── mnist_test.csv         # Dataset de teste (10k amostras)
└── build/                     # Diretório de compilação (gerado)
    ├── main                   # Executável OpenMP
    └── main_hybrid            # Executável MPI+OpenMP
```

---

## 🔧 Requisitos

### Software Necessário
- **CMake** 3.15 ou superior
- **Compilador C++** com suporte a C++17 (g++ 7+, clang++ 5+)
- **OpenMP** 4.5 ou superior
- **MPI** (OpenMPI 3.0+ ou MPICH 3.2+)

### Dataset
- MNIST em formato CSV
- Download disponível em: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
- Extrair os arquivos na pasta `dataset/`

### Verificação dos Requisitos
```bash
# Verificar CMake
cmake --version

# Verificar compilador C++
g++ --version

# Verificar OpenMP
echo | cpp -fopenmp -dM | grep -i open

# Verificar MPI
mpirun --version
```

---


## 🔨 Compilação

### Método 1: CMake (Recomendado)

```bash
# Configurar o projeto
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build build

# Ou em uma linha
cmake -S . -B build && cmake --build build
```

**Executáveis gerados:**
- `build/main` - Versão OpenMP
- `build/main_hybrid` - Versão MPI+OpenMP

### Método 2: Compilação Manual

```bash
# Versão OpenMP
g++ -fopenmp -O3 -std=c++17 -o kmeans_openmp main.cc

# Versão Híbrida MPI+OpenMP
mpicxx -fopenmp -O3 -std=c++17 -o kmeans_hybrid main_hybrid.cc
```

---


## 🚀 Execução

### Sintaxe Geral

```bash
./executavel <dataset.csv> <K> <max_iter> [seed]
```

**Parâmetros:**
- `dataset.csv` - Arquivo CSV com os dados (cada linha é um ponto)
- `K` - Número de clusters desejado
- `max_iter` - Número máximo de iterações
- `seed` - Semente aleatória (opcional, para reprodutibilidade)

### Exemplos de Uso

#### 1️⃣ Versão OpenMP

```bash
# Com 4 threads
OMP_NUM_THREADS=4 ./build/main dataset/mnist_train.csv 10 100

# Teste com diferentes números de threads
OMP_NUM_THREADS=1 ./build/main dataset/mnist_train.csv 15 100
OMP_NUM_THREADS=2 ./build/main dataset/mnist_train.csv 15 100
OMP_NUM_THREADS=4 ./build/main dataset/mnist_train.csv 15 100
OMP_NUM_THREADS=8 ./build/main dataset/mnist_train.csv 15 100
```

#### 2️⃣ Versão Híbrida MPI+OpenMP

```bash
# 1 processo × 4 threads = 4 cores
OMP_NUM_THREADS=4 mpirun -np 1 ./build/main_hybrid dataset/mnist_train.csv 15 100

# 2 processos × 2 threads = 4 cores
OMP_NUM_THREADS=2 mpirun -np 2 ./build/main_hybrid dataset/mnist_train.csv 15 100

# 4 processos × 1 thread = 4 cores (MPI puro)
mpirun -np 4 ./build/main_hybrid dataset/mnist_train.csv 15 100
```

#### 3️⃣ Usando Scripts Automatizados

```bash
# Executar todos os benchmarks (K=15, 100 iterações)
./run_benchmarks.sh

# Teste rápido (K=5, 50 iterações)
./run_benchmarks.sh -k 5 -m 50

# Benchmark customizado com saída específica
./run_benchmarks.sh -k 10 -m 100 -o meus_resultados.csv

# Teste rápido para desenvolvimento
./run_quick_test.sh
```

**Opções do `run_benchmarks.sh`:**
```
-k, --clusters NUM     Número de clusters (padrão: 15)
-m, --max-iter NUM     Máximo de iterações (padrão: 100)
-s, --seed NUM         Semente aleatória (padrão: 42)
-o, --output FILE      Arquivo CSV de saída (padrão: benchmark_results.csv)
-d, --dataset PATH     Caminho do dataset (padrão: dataset/mnist_train.csv)
-h, --help             Mostrar ajuda
```

---


## 📊 Resultados de Desempenho

### Ambiente de Teste
- **Servidor**: parcode
- **Dataset**: MNIST train (60.000 amostras, 785 dimensões)
- **Configuração**: K=15 clusters, max_iter=100
- **Medição**: Tempo total de execução até convergência

### Resultados - OpenMP

| Threads | Tempo (s) | Speedup | Eficiência |
|---------|-----------|---------|------------|
| 1       | 51.46     | 1.00x   | 100%       |
| 2       | 28.02     | 1.84x   | 92%        |
| 4       | 17.37     | 2.96x   | 74%        |
| 8       | 12.28     | 4.19x   | 52%        |

### Resultados - MPI+OpenMP Híbrido (4 cores totais)

| Configuração      | Processos | Threads | Tempo (s) | Speedup |
|-------------------|-----------|---------|-----------|---------|
| 1 proc × 4 thr    | 1         | 4       | 48.65     | 1.06x   |
| 2 proc × 2 thr    | 2         | 2       | 28.01     | 1.84x   |
| 4 proc × 1 thr    | 4         | 1       | 28.10     | 1.83x   |

### Análise dos Resultados

✅ **OpenMP mostrou excelente escalabilidade:**
- Speedup quase linear até 4 threads (2.96x)
- Speedup de 4.19x com 8 threads demonstra boa utilização de recursos
- Eficiência de 74% com 4 threads e 52% com 8 threads

⚠️ **MPI+OpenMP híbrido:**
- Overhead de comunicação MPI impacta o desempenho
- Melhor resultado com 2 processos × 2 threads (1.84x)
- Para este problema, OpenMP puro é mais eficiente

📈 **Conclusão**: Para datasets que cabem em memória compartilhada, OpenMP oferece melhor desempenho. MPI+OpenMP é vantajoso para datasets maiores que não cabem em um único nó.

> **Nota**: Resultados completos com K=5, K=10 e K=15 disponíveis em `TESTES.txt`

---


## 🧪 Scripts de Teste Automatizados

### 🎯 `run_benchmarks.sh` - Benchmark Completo

Script automatizado que executa todos os testes requeridos e gera relatório CSV.

**Funcionalidades:**
- ✅ Compila automaticamente ambas as versões
- ✅ Executa testes OpenMP (1, 2, 4, 8 threads)
- ✅ Executa testes MPI+OpenMP (1×4, 2×2, 4×1)
- ✅ Salva resultados em CSV estruturado
- ✅ Calcula e exibe análise de speedup
- ✅ Validação de erros e feedback colorido

**Uso básico:**
```bash
./run_benchmarks.sh                    # Usa configuração padrão (K=15, iter=100)
./run_benchmarks.sh -k 10 -m 100       # K=10, 100 iterações
./run_benchmarks.sh -k 5 -m 50 -o test.csv  # Teste rápido com saída customizada
./run_benchmarks.sh --help             # Ver todas as opções
```

**Tempo estimado:** ~15-20 minutos para K=15 com dataset completo

### ⚡ `run_quick_test.sh` - Teste Rápido

Script para validação rápida durante desenvolvimento.

**Configuração:**
- K=5 clusters
- 50 iterações máximas
- 4 threads OpenMP
- Apenas versão OpenMP

**Tempo estimado:** ~30 segundos

```bash
./run_quick_test.sh
```

> 📖 **Documentação completa:** Veja `BENCHMARK_SCRIPTS.txt` para detalhes sobre os scripts

---


## 🔍 Detalhes da Implementação

### Algoritmo K-Means

1. **Inicialização**: Seleciona K centróides aleatórios
2. **Assignment**: Atribui cada ponto ao centróide mais próximo
3. **Update**: Recalcula centróides como média dos pontos atribuídos
4. **Convergência**: Repete passos 2-3 até centróides estabilizarem

### Paralelização OpenMP (`main.cc`)

**Estratégia:**
- Loop principal paralelizado com `#pragma omp parallel for`
- Cada thread processa subconjunto de pontos
- Acumuladores locais por thread evitam contenção
- Redução manual com seção crítica

**Diretivas utilizadas:**
```cpp
#pragma omp parallel              // Região paralela
#pragma omp for schedule(static)  // Divisão estática do trabalho
#pragma omp critical              // Proteção da redução
```

**Otimizações:**
- Scheduling estático para balanceamento de carga
- Buffers locais minimizam sincronização
- Cálculo de distância euclidiana otimizado

### Paralelização MPI+OpenMP (`main_hybrid.cc`)

**Arquitetura híbrida de dois níveis:**

**Nível 1 - MPI (entre nós):**
- `MPI_Scatterv`: Distribui dados entre processos
- `MPI_Bcast`: Sincroniza centróides
- `MPI_Allreduce`: Redução global de somas/contagens
- `MPI_Gatherv`: Coleta resultados finais

**Nível 2 - OpenMP (dentro do nó):**
- Threads paralelizam cálculo de distâncias
- Buffers por thread evitam race conditions
- Redução local antes da redução MPI

**Fluxo de execução:**
```
1. Rank 0 carrega dataset → MPI_Bcast metadados → MPI_Scatterv distribui dados
2. Loop de iterações:
   a. OpenMP paralelo: cada thread calcula distâncias localmente
   b. Redução OpenMP: combina resultados das threads
   c. MPI_Allreduce: combina resultados entre processos
   d. Atualização de centróides (replicado em todos os processos)
   e. Verificação de convergência global
3. MPI_Gatherv: Rank 0 coleta labels finais
```

---

## 💡 Observações Importantes

### Performance
- ⏱️ Tempo de execução varia com K (mais clusters = mais tempo)
- 🎯 Convergência ocorre quando centróides estabilizam (diferença < 1e-6)
- 🔄 Para reprodutibilidade, use a mesma seed em todas as execuções

### Limitações
- 💾 Dataset completo deve caber na memória (versão OpenMP)
- 🌐 Overhead de comunicação MPI pode superar benefícios em datasets pequenos
- 📊 Melhor escalabilidade observada com K ≥ 10

### Dataset
- ⚠️ `mnist_train.csv` (105MB) não está versionado no Git (excede limite GitHub)
- ✅ `mnist_test.csv` (18MB) incluído para testes rápidos
- 📥 Download do dataset completo: [Kaggle - MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

---

## 📚 Documentação Adicional

- **`main.cc`**: Comentários detalhados sobre paralelização OpenMP
- **`main_hybrid.cc`**: Documentação completa da implementação híbrida
- **`TESTES.txt`**: Resultados completos dos benchmarks
- **`BENCHMARK_SCRIPTS.txt`**: Guia completo dos scripts de teste
- **`QUICK_START.txt`**: Guia rápido para começar

---

