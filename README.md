
PROJETO 01 - COMPUTAÇÃO PARALELA
Implementação Paralela de K-Means com OpenMP e MPI


SOBRE A APLICAÇÃO:
------------------
Este projeto implementa o algoritmo K-Means clustering usando paralelização
com OpenMP e MPI+OpenMP (híbrido). K-Means é um algoritmo de aprendizado de
máquina não supervisionado que agrupa dados em K clusters, minimizando a
distância entre os pontos e os centróides de seus respectivos clusters.

O dataset utilizado é o MNIST (Modified National Institute of Standards and 
Technology), que contém 60.000 imagens de dígitos manuscritos (0-9), cada
uma com 784 features (28x28 pixels).


CÓDIGO BASE:
-----------
Este projeto foi desenvolvido do zero para fins educacionais, implementando
o algoritmo K-Means clássico com extensões para paralelização.


ARQUIVOS DO PROJETO:
-------------------
- main.cc                : Versão com OpenMP apenas
- main_hybrid.cc         : Versão híbrida (MPI + OpenMP)
- CMakeLists.txt         : Arquivo de configuração do CMake
- TESTES.txt             : Resultados dos testes de desempenho
- run_benchmarks.sh      : Script automatizado para executar todos os testes
- run_quick_test.sh      : Script para teste rápido durante desenvolvimento
- BENCHMARK_SCRIPTS.txt  : Documentação completa dos scripts
- README.txt             : Este arquivo


REQUISITOS:
----------
- CMake 3.15 ou superior
- Compilador C++ com suporte a C++17
- OpenMP
- MPI (OpenMPI ou MPICH)
- Dataset MNIST em formato CSV


COMPILAÇÃO:
----------

1) Usando CMake (recomendado):
   
   cmake -S . -B build
   cmake --build build

   Isso irá gerar dois executáveis:
   - build/main         (versão OpenMP)
   - build/main_hybrid  (versão MPI+OpenMP)


2) Compilação manual alternativa:

   Para versão OpenMP:
   g++ -fopenmp -O3 -std=c++17 -o kmeans_openmp main.cc

   Para versão híbrida:
   mpicxx -fopenmp -O3 -std=c++17 -o kmeans_hybrid main_hybrid.cc


EXECUÇÃO:
--------

SINTAXE:
./executavel <dataset.csv> <K> <max_iter> [seed]

Parâmetros:
  - dataset.csv : Arquivo CSV com os dados (cada linha é um ponto)
  - K           : Número de clusters desejado
  - max_iter    : Número máximo de iterações
  - seed        : Seed para geração aleatória (opcional)


EXEMPLOS DE EXECUÇÃO:

1) Versão OpenMP com 4 threads:
   
   export OMP_NUM_THREADS=4
   ./build/main dataset/mnist_train.csv 10 100

   ou

   OMP_NUM_THREADS=4 ./build/main dataset/mnist_train.csv 10 100


2) Versão OpenMP com diferentes números de threads:
   
   OMP_NUM_THREADS=1 ./build/main dataset/mnist_train.csv 15 100
   OMP_NUM_THREADS=2 ./build/main dataset/mnist_train.csv 15 100
   OMP_NUM_THREADS=4 ./build/main dataset/mnist_train.csv 15 100
   OMP_NUM_THREADS=8 ./build/main dataset/mnist_train.csv 15 100


3) Versão Híbrida MPI+OpenMP:

   a) 1 processo com 4 threads OpenMP:
      OMP_NUM_THREADS=4 mpirun -np 1 ./build/main_hybrid dataset/mnist_train.csv 15 100

   b) 2 processos com 2 threads OpenMP cada:
      OMP_NUM_THREADS=2 mpirun -np 2 ./build/main_hybrid dataset/mnist_train.csv 15 100

   c) 4 processos com 1 thread OpenMP cada (MPI puro):
      OMP_NUM_THREADS=1 mpirun -np 4 ./build/main_hybrid dataset/mnist_train.csv 15 100
      ou
      mpirun -np 4 ./build/main_hybrid dataset/mnist_train.csv 15 100


RESULTADOS DE DESEMPENHO:
------------------------
Testes realizados no servidor parcode com dataset MNIST (60000 pontos, 784 dimensões)
e 15 clusters:

OpenMP (main):
  1 thread:  51.4609 s  (baseline)
  2 threads: 28.016 s   (speedup: 1.84x)
  4 threads: 17.3703 s  (speedup: 2.96x)
  8 threads: 12.275 s   (speedup: 4.19x)

Híbrido MPI+OpenMP (main_hybrid):
  1 proc × 4 threads: 48.6451 s  (speedup: 1.06x vs 1 thread)
  2 proc × 2 threads: 28.0142 s  (speedup: 1.84x vs 1 thread)
  4 proc × 1 thread:  28.0967 s  (speedup: 1.83x vs 1 thread)

Ver arquivo TESTES.txt para resultados completos com 5, 10 e 15 clusters.


EXECUÇÃO AUTOMATIZADA COM SCRIPTS:
----------------------------------
Para facilitar os testes, há scripts bash disponíveis:

1) Script completo de benchmark (RECOMENDADO):
   ./run_benchmarks.sh
   
   Este script automaticamente:
   - Compila ambas as versões
   - Executa todos os testes requeridos (1,2,4,8 threads e configurações MPI)
   - Salva resultados em CSV estruturado
   - Exibe sumário com análise de speedup
   
   Uso com opções:
   ./run_benchmarks.sh -k 10 -m 100 -o results.csv
   
   Para ver todas as opções:
   ./run_benchmarks.sh --help

2) Script de teste rápido:
   ./run_quick_test.sh
   
   Para testes rápidos durante desenvolvimento (K=5, 50 iterações)

Veja BENCHMARK_SCRIPTS.txt para documentação completa dos scripts.


DETALHES DA PARALELIZAÇÃO:
-------------------------

OPENMP (main.cc):
- Paralelização do loop principal que calcula distâncias e atribui pontos aos clusters
- Cada thread mantém acumuladores locais (local_sum, local_count)
- Redução manual com seção crítica ao final de cada iteração
- Diretivas utilizadas: #pragma omp parallel, #pragma omp for, #pragma omp critical

MPI+OPENMP (main_hybrid.cc):
- Distribuição dos dados entre processos usando MPI_Scatterv
- Cada processo MPI trabalha em um subconjunto dos dados
- Dentro de cada processo, OpenMP paraleliza o loop de atribuição
- Redução global com MPI_Allreduce para somas e contagens
- Sincronização de centróides entre processos com MPI_Bcast e MPI_Allreduce
- Coleta final dos labels com MPI_Gatherv


OBSERVAÇÕES:
-----------
- O tempo de execução varia com o número de clusters (K)
- Convergência detectada quando centróides não mudam significativamente
- Para melhor reprodutibilidade, use a mesma seed em todas as execuções
- Os comentários no código fonte detalham as mudanças feitas para paralelização


CONTATO:
-------
Para dúvidas sobre este projeto, consulte os comentários detalhados nos
arquivos main.cc e main_hybrid.cc.

===============================================================================
