# sparse-bench
Benchmarks for sparse operations on GPU.

# Compilation
nvcc -w main.cu -lcusparse

# Execution
./a.out \<M\> \<K\> \<N\> \<sparsity in A\>

Note: Range of sparsity is [0,1]
