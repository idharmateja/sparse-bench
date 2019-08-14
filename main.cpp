#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <chrono>
#include <iostream>
#include <iomanip>


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


// Generate sparse matrices
void populate_sparse_matrix(csr_matrix& csr_mat, bool is_uniform){

	int rows = csr_mat.rows;
	int cols = csr_mat.cols;
	int nnz = csr_mat.nnz;
	int nnz_per_row = csr_mat.nnz / csr_mat.rows;

	assert(nnz_per_row*rows == nnz);

	for (int i = 0; i < rows+1; ++i)
	{
		csr_mat.rowPtr[i] = 0;
	}


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
	

	// Populating indices and rowPtr
	int nz_id = 0;
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < cols; ++col)
		{
			int flat_id = row*cols + col;
			if(flags[flat_id] == true){
				csr_mat.indices[nz_id] = col;
				csr_mat.rowPtr[row] += 1;
				nz_id += 1;
			}
		}
	}

	// Generate rowPtr
	int sum = 0;
	for (int row = 0; row <= rows; ++row)
	{
		int temp = csr_mat.rowPtr[row];
		csr_mat.rowPtr[row] = sum;
		sum += temp;
	}

	// ALGO 0 : Random initialization
	for(int i=0; i< nnz; i++){
		csr_mat.values[i] = rand()%5 + 1;		
	}
}

void populate_dense_matrix(dense_matrix& dense_mat, int ALGO){
	int rows = dense_mat.rows;
	int cols = dense_mat.cols;

	for(int i=0; i< rows*cols; i++){
		if(ALGO == -1)
			dense_mat.values[i] = rand()%5 + 1;
		else
			dense_mat.values[i] = 0;
	}
}

void naive_sparse_dense_matmul(csr_matrix A_csr, dense_matrix B_mat, dense_matrix C_mat){
	for (int row = 0; row < A_csr.rows; ++row)
	for (int col = 0; col < B_mat.cols; ++col)
	{
		float value = 0;
		for (int id = A_csr.rowPtr[row]; id < A_csr.rowPtr[row+1]; ++id)
		{
			int col_ind = A_csr.indices[id];
			float A_val = A_csr.values[id];
			if(B_mat.is_rowmajor)
				value += A_val * B_mat.values[col_ind*B_mat.cols + col];
			else
				value += A_val * B_mat.values[col*B_mat.rows + col_ind];
		}

		if(C_mat.is_rowmajor)
			C_mat.values[row*C_mat.cols + col] = value;
		else
			C_mat.values[col*C_mat.rows + row] = value;
	}
}


#define M_ 1024
#define K_ 1024
#define N_ 1024
#define NNZ_ 1024*512

// Note : If program results in seg fault, 
// then it is probably due to stack overflow.
// Execute following command to increase stack limit.
// ulimit -s 300000000

int main(int argc, char* argv[]){
	bool is_uniform = false;
	bool debug = false;

	// Memory allocation
	float values[NNZ_];
	int indices[NNZ_];
	int rowPtr[M_+1];

	float B[M_*K_];
	float C[K_*N_];

	// Populating into corresponding data structures.
	csr_matrix A_csr;
	dense_matrix B_dense;
	dense_matrix C_dense;

	A_csr.rows = M_;
	A_csr.cols = K_;
	A_csr.nnz  = NNZ_;
	A_csr.values = values;
	A_csr.indices = indices;
	A_csr.rowPtr = rowPtr;
	populate_sparse_matrix(A_csr, is_uniform);
		
	B_dense.rows = K_;
	B_dense.cols = N_;
	B_dense.values = B;
	B_dense.is_rowmajor = false;
	populate_dense_matrix(B_dense, -1);

	C_dense.rows = M_;
	C_dense.cols = N_;
	C_dense.values = C;
	C_dense.is_rowmajor = false;
	populate_dense_matrix(C_dense, 0);
	

	if(debug){
		std::cout << "Matrix A" << std::endl;
		print_csr_matrix(A_csr);
		std::cout << "Matrix B" << std::endl;
		print_dense_matrix(B_dense);
	}

	// Calling the naive kernel
	auto start = std::chrono::high_resolution_clock::now(); 
	naive_sparse_dense_matmul(A_csr, B_dense, C_dense);
	auto stop = std::chrono::high_resolution_clock::now(); 

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	if(debug){
		std::cout << "Matrix C" << std::endl;
		print_dense_matrix(C_dense);	
	}

	std::cout << "Time taken by function: "
     << duration.count() << " microseconds" << std::endl; 
	
	return 0;

}


/*
int main(int argc, char const *argv[])
{
	int M,K,N;
	float sp_a;
	bool is_uniform = false;
	bool debug = false;

	if(argc < 5){
		std::cout << "Usage : ./cpusdmm M K N sparsity<0to1>" << std::endl;
		return 0;
	}

	// MxK is dimension of matrix A
	// KxN is dimension of matrix B
	// sp_a is sparsity in matrix A
	M = atoi(argv[1]);
	K = atoi(argv[2]);
	N = atoi(argv[3]);
	sp_a = atof(argv[4]);

	if(argc >= 6)
		debug = atoi(argv[5]);

	// Generating sparse datastructures
	csr_matrix A_csr = generate_sparse_matrix(M, K, sp_a, is_uniform);
	dense_matrix B_mat = generate_dense_matrix(K, N, true);

	dense_matrix C_mat = generate_dense_matrix(M, N, false);
	for (int i = 0; i < C_mat.rows*C_mat.cols; ++i) C_mat.values[i] = 0;

	if(debug){
		std::cout << "Matrix A" << std::endl;
		print_csr_matrix(A_csr);
		std::cout << "Matrix B" << std::endl;
		print_dense_matrix(B_mat);
	}

	// Calling the naive kernel
	auto start = std::chrono::high_resolution_clock::now(); 
	naive_sparse_dense_matmul(A_csr, B_mat, C_mat);
	auto stop = std::chrono::high_resolution_clock::now(); 

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	if(debug){
		std::cout << "Matrix C" << std::endl;
		print_dense_matrix(C_mat);	
	}

	std::cout << "Time taken by function: "
     << duration.count() << " microseconds" << std::endl; 


	return 0;
}
*/
