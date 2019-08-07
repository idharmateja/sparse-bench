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
