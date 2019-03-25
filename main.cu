#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <iostream>
#include <iomanip>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct  dense_matrix
{
	int rows;
	int cols;
	float* values;
	bool is_rowmajor;
}dense_matrix;

typedef struct  csr_matrix
{
	int rows; // Number of rows
	int cols; // Number of columns
	int nnz; // Number of nonzero elements.
	float* values; // stores elements
	int* indices;  // [i] stores the column index of element 'i'
	int* rowPtr; // [i] contains index of first element in row 'i'
}csr_matrix;

// Conversions between sparse matrix formats
dense_matrix convert_csr_to_dense(csr_matrix csr_mat, bool in_rowmajor){
	// Initialize and allocate memory for dense matrix.
	float* values = new float[csr_mat.rows*csr_mat.cols]();
	
	// Convert csr matrix to dense matrix
	for (int row = 0; row < csr_mat.rows; ++row)
	{
		for (int ind = csr_mat.rowPtr[row]; 
				ind < csr_mat.rowPtr[row+1]; ++ind)
		{
			int col = csr_mat.indices[ind];
			if(in_rowmajor)
				values[(row * csr_mat.cols) + col] = csr_mat.values[ind];
			else
				values[(col * csr_mat.rows) + row] = csr_mat.values[ind];
		}
	}

	// Packaging as dense matrix
	dense_matrix dense_mat;
	dense_mat.rows = csr_mat.rows;
	dense_mat.cols = csr_mat.cols;
	dense_mat.values = values;
	dense_mat.is_rowmajor = in_rowmajor;

	return dense_mat;
}

// print functionalities.
void print_matrix(float* data, int rows, int cols, bool is_rowmajor){
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if(is_rowmajor){
				std::cout << std::setw(3) << data[i*cols + j] << " ";
			}else{
				std::cout << std::setw(3) << data[j*rows + i] << " ";
			}
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void print_dense_matrix(dense_matrix dense_mat){
	print_matrix(dense_mat.values, dense_mat.rows, dense_mat.cols, dense_mat.is_rowmajor);
}

void print_csr_matrix(csr_matrix csr_mat){

	dense_matrix dense_mat = convert_csr_to_dense(csr_mat, false);
	print_dense_matrix(dense_mat);

	delete[] dense_mat.values;
}

// Utilites
void swap_memory_layout(float* mat, int rows, int cols, bool is_rowmajor){
	float* buff_mat = new float[rows*cols]();

	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			if(is_rowmajor)
				buff_mat[j*rows + i] = mat[i*cols + j];
			else
				buff_mat[i*cols + j] = mat[j*rows + i];
		}
	}

	for(int i=0; i < rows*cols; i++)
		mat[i] = buff_mat[i];

	delete[] buff_mat;
}

// Generate sparse matrices
csr_matrix generate_sparse_matrix(int rows, int cols, float sparsity, bool is_uniform){

	// Number of non zeros
	int nnz_per_row = int((1.0-sparsity) * cols);
	int nnz = nnz_per_row * rows;

	bool* flags = new bool[rows*cols]();

	// Picking non zero blocks randomly.
	if (is_uniform)
	{
		for (int row = 0; row < rows; ++row)
		{
			int cur_nnz_per_row = 0;
			while(cur_nnz_per_row < nnz_per_row)
			{
				int flat_id = row*cols + rand()%(cols);
				if(flags[flat_id] == false){
					flags[flat_id] = true;
					cur_nnz_per_row += 1;
				}
			}
		}
	}else{
		int cur_nnz_per_row = 0;
		while(cur_nnz_per_row < nnz)
		{
			int flat_id = rand()%(rows*cols);
			if(flags[flat_id] == false){
				flags[flat_id] = true;
				cur_nnz_per_row += 1;
			}
		}
	}
	

	// Initializing and allocating memory
	float* values = new float[nnz]();
	int* indices  = new int[nnz]();
	int* rowPtr = new int[rows+1]();

	// Populating indices and rowPtr
	int nz_id = 0;
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			int flat_id = row*cols + col;
			if(flags[flat_id] == true){
				indices[nz_id] = col;
				rowPtr[row] += 1;
				nz_id += 1;
			}
		}
	}

	// Generate rowPtr
	int sum = 0;
	for (int row = 0; row <= rows; ++row)
	{
		int temp = rowPtr[row];
		rowPtr[row] = sum;
		sum += temp;
	}


	// ALGO 0 : Random initialization
	for(int i=0; i< nnz; i++){
		values[i] = rand()%5 + 1;		
	}	

	csr_matrix csr_mat;

	csr_mat.rows = rows;
	csr_mat.cols = cols;
	csr_mat.nnz  = nnz;
	csr_mat.values = values;
	csr_mat.indices = indices;
	csr_mat.rowPtr = rowPtr;

	return csr_mat;
}

dense_matrix generate_dense_matrix(int rows, int cols, bool is_rowmajor){

	float* values = new float[rows * cols]();

	for(int i=0; i< rows*cols; i++){
		values[i] = rand()%5 + 1;
	}

	// Packaging dense matrix into structure.
	dense_matrix dense_mat;

	dense_mat.rows = rows;
	dense_mat.cols = cols;
	dense_mat.values = values;
	dense_mat.is_rowmajor = is_rowmajor;

	return dense_mat;
}

// GPU handles for different types of matrices
dense_matrix get_dense_matrix_gpu_handle(dense_matrix h_mat){
	dense_matrix d_mat;

	d_mat.rows = h_mat.rows;
	d_mat.cols = h_mat.cols;
	d_mat.is_rowmajor = h_mat.is_rowmajor;

	gpuErrchk(cudaMalloc((void**)&d_mat.values, sizeof(float) * h_mat.rows * h_mat.cols));
	gpuErrchk(cudaMemcpy(d_mat.values, h_mat.values, sizeof(float) * h_mat.rows * h_mat.cols, cudaMemcpyHostToDevice));

	return d_mat;	
}
csr_matrix get_csr_matrix_gpu_handle(csr_matrix h_mat){
	csr_matrix d_mat;

	d_mat.rows = h_mat.rows;
	d_mat.cols = h_mat.cols;
	d_mat.nnz = h_mat.nnz;
	
	gpuErrchk(cudaMalloc((void**)&d_mat.values, sizeof(float) * h_mat.nnz ));
	gpuErrchk(cudaMalloc((void**)&d_mat.indices, sizeof(int) * h_mat.nnz ));
	gpuErrchk(cudaMalloc((void**)&d_mat.rowPtr, sizeof(int) * (h_mat.rows+1) ));

	gpuErrchk(cudaMemcpy(d_mat.values, h_mat.values, sizeof(float) * h_mat.nnz ,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mat.indices, h_mat.indices, sizeof(int) * h_mat.nnz ,cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_mat.rowPtr, h_mat.rowPtr, sizeof(int) * (h_mat.rows+1) ,cudaMemcpyHostToDevice));

	return d_mat;	
}

// cusparse benchmark for spxdense mutliplication
void cusparse_benchmark(csr_matrix csr_mat,
					float* B, bool is_B_rowmajor, 
					int M, int K, int N, 
					float* C, bool is_C_rowmajor,
					bool use_csrmm2, 
					float &runtime, int num_iters){

	cusparseStatus_t status;
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;

	float alpha = 1.0f;
	float beta = 0.0f;

    status= cusparseCreate(&handle);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
    	fprintf(stderr, "!!!! CUSPARSE initialization error\n");
		exit(EXIT_FAILURE);
    }

     /* create and setup matrix descriptor */ 
    status= cusparseCreateMatDescr(&descr); 
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "Matrix descriptor initialization failed\n");
        exit(EXIT_FAILURE);
    }       
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    float *h_B;
    float *d_B, *d_C;

    h_B = B;
    if(is_B_rowmajor && !use_csrmm2){
    	h_B = new float[K * N]();
    	memcpy(h_B, B, (K * N * sizeof(float)) );
    	swap_memory_layout(h_B, K, N, true);
    }

    gpuErrchk(cudaMalloc((void**)&d_B, (K * N * sizeof(float))));
    gpuErrchk(cudaMalloc((void**)&d_C, (M * N * sizeof(float))));

    // Loading to device memory
    csr_matrix d_csr_mat = get_csr_matrix_gpu_handle(csr_mat);
    gpuErrchk(cudaMemcpy(d_B, h_B, (K * N * sizeof(float)), cudaMemcpyHostToDevice));


    float cum_runtime = 0;
    for (int iter_id = 0; iter_id < num_iters; ++iter_id)
    {
    	float elapased_time = 0;

	    cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start);
		if(is_B_rowmajor && use_csrmm2){
			status = cusparseScsrmm2(handle, 
		    	CUSPARSE_OPERATION_NON_TRANSPOSE,
		    	CUSPARSE_OPERATION_TRANSPOSE,
		    	M, N, K,
		    	d_csr_mat.nnz,
		    	&alpha, descr,
		    	d_csr_mat.values, d_csr_mat.rowPtr, d_csr_mat.indices,
		    	d_B, N,
		    	&beta, 
		    	d_C, M);
		}else{
			status = cusparseScsrmm(handle, 
		    	CUSPARSE_OPERATION_NON_TRANSPOSE,
		    	M, N, K,
		    	d_csr_mat.nnz,
		    	&alpha, descr,
		    	d_csr_mat.values, d_csr_mat.rowPtr, d_csr_mat.indices,
		    	d_B, K,
		    	&beta, 
		    	d_C, M);	
		}
	    
	    cudaEventRecord(stop);

	    cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapased_time, start, stop);

		cum_runtime += elapased_time;
	}

	// Averaging runs
	runtime = cum_runtime/num_iters;

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
    	fprintf(stderr, "CUSPARSE call failed with code : %d\n", status);
        exit(EXIT_FAILURE);
    }

    // Copying the result back
    gpuErrchk(cudaMemcpy(C, d_C, (M * N * sizeof(float)), cudaMemcpyDeviceToHost));

    if(is_C_rowmajor){
    	swap_memory_layout(C, M, N, false);
    }

}


int main(int argc, char const *argv[])
{
	int M,K,N;
	float sp_a;
	bool is_uniform = false;
	int NUM_ITERS = 10;

	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);

	sp_a = atof(argv[4]);

	// Generating sparse datastructures
	csr_matrix A_csr = generate_sparse_matrix(M, K, sp_a, is_uniform);
	dense_matrix B_mat = generate_dense_matrix(K, N, false);

	dense_matrix C_mat = generate_dense_matrix(M, N, false);
	for (int i = 0; i < C_mat.rows*C_mat.cols; ++i) C_mat.values[i] = 0;


	// Sparse matrix matrix multiplication call
	float cusparse_time = 0;
	cusparse_benchmark(A_csr,
					B_mat.values, B_mat.is_rowmajor,
					M, K, N,
					C_mat.values, C_mat.is_rowmajor,
					false,
					cusparse_time, NUM_ITERS);


	// VERIFICATION
	dense_matrix C_ref_mat = generate_dense_matrix(M, N, C_mat.is_rowmajor);
	for (int i = 0; i < C_ref_mat.rows*C_ref_mat.cols; ++i) C_ref_mat.values[i] = 0;

	for (int row = 0; row < A_csr.rows; ++row)
	{
		for (int col = 0; col < B_mat.cols; ++col)
		{
			float sum = 0;
			for (int id = A_csr.rowPtr[row]; id < A_csr.rowPtr[row+1]; ++id)
			{
				int k = A_csr.indices[id];
				float val = A_csr.values[id];

				if(B_mat.is_rowmajor)
					sum += val * B_mat.values[k*B_mat.cols + col];
				else
					sum += val * B_mat.values[col*B_mat.rows + k];
			}

			if(C_ref_mat.is_rowmajor)
				C_ref_mat.values[row*C_ref_mat.cols + col] = sum;
			else
				C_ref_mat.values[col*C_ref_mat.rows + row] = sum;
		}
	}

	float error = 0;
	for (int i = 0; i < C_mat.rows*C_mat.cols; ++i)
	{
		error += abs(C_mat.values[i] - C_ref_mat.values[i]);
	}

	// Timing information.
	std::cout << "Error :  " << error << std::endl;
	std::cout << "Time(ms): " << cusparse_time << std::endl;
	

	return 0;
}