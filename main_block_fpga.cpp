#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef FDEBUG

typedef struct  dense_matrix
{
	int rows;
	int cols;
	float* values;
	bool is_rowmajor;
}dense_matrix;

typedef struct  bsr_matrix
{
	int rows; // Number of rows
	int cols; // Number of columns

	int bh; // Block height
	int bw; // Block width

	int nnz;  // Number of nonzero elements.
	int nnzb; // Number of nonzero blocks
	
	float* values; // stores blocks in row major fashion. Each block in column major.
	int* indices;  // [i] stores the column index of block 'i'
	int* rowBlockPtr; // [i] contains index of first element in row block 'i'
}bsr_matrix;

// Conversions between sparse matrix formats
dense_matrix convert_bsr_to_dense(bsr_matrix bsr_mat, bool in_rowmajor){
	// Initialize and allocate memory for dense matrix.
	float* values = new float[bsr_mat.rows*bsr_mat.cols]();
	
	// Convert bsr matrix to dense matrix	
	int nrb = rows / bsr_mat.bh;
	int ncb = cols / bsr_mat.bw;

	for (int rb = 0; rb < nrb; ++rb)
	{
		for (int ind = bsr_mat.rowBlockPtr[rb]; 
				 ind < bsr_mat.rowBlockPtr[rb+1]; ++ind)
		{
			int base_row = rb * bsr_mat.bh;
			int base_col = bsr_mat.indices[ind] * bsr_mat.bw;
			int base_val_ind = ind * bsr_mat.bh * bsr_mat.bw;

			// Take a block and place it in dense matrix
			for (int i = 0; i < bsr_mat.bh; ++i)
			{
				int row = base_row + i;
				for (int j = 0; j < bsr_mat.bw; ++j)
				{
					int col = base_col + j;
					int val_ind = base_val_ind + (j*bh + i);
					if(in_rowmajor)
						values[(row * bsr_mat.cols) + col] = bsr_mat.values[val_ind];
					else
						values[(col * bsr_mat.rows) + row] = bsr_mat.values[val_ind];
				}
			}
		}
	}

	// Packaging as dense matrix
	dense_matrix dense_mat;
	dense_mat.rows = bsr_mat.rows;
	dense_mat.cols = bsr_mat.cols;
	dense_mat.values = values;
	dense_mat.is_rowmajor = in_rowmajor;

	return dense_mat;
}

// print functionalities.
template <typename T>
void print_matrix(T* data, int rows, int cols, bool is_rowmajor){
	for (int i = 0; i < rows; ++i)
	{
		std::cout << "Row " << std::setw(5) << i << " :";
		for (int j = 0; j < cols; ++j)
		{
			if(is_rowmajor){
				std::cout << std::setw(2) << data[i*cols + j] << " ";
			}else{
				std::cout << std::setw(2) << data[j*rows + i] << " ";
			}
		}
		std::cout << std::endl;
	}
	//std::cout << std::endl;
}

void print_dense_matrix(dense_matrix dense_mat){
	print_matrix<float>(dense_mat.values, dense_mat.rows, dense_mat.cols, dense_mat.is_rowmajor);
}

void print_bsr_matrix(bsr_matrix bsr_mat){
	bool VERBOSE = true;

	if(VERBOSE){
		std::cout << "Rows,cols : (" << bsr_mat.rows << "," << bsr_mat.cols << ")" << std::endl;
		std::cout << "NNZ  : " << bsr_mat.nnz << std::endl;
		std::cout << "NNZB : " << bsr_mat.nnzb << std::endl;
		std::cout << "Sparsity : " << (1.0 - (bsr_mat.nnz*1.0)/(bsr_mat.rows*bsr_mat.cols))*100 << std::endl;

		std::cout << "Values  "; print_matrix<float>(bsr_mat.values,  1, bsr_mat.nnz, true);
		std::cout << "Indices "; print_matrix<int>(bsr_mat.indices, 1, bsr_mat.nnz, true);
		std::cout << "RowBlockPtr  "; print_matrix<int>(bsr_mat.rowPtr, 1, (bsr_mat.rows/bsr_mat.bh +1), true);
	}


	dense_matrix dense_mat = convert_bsr_to_dense(bsr_mat, false);
	print_dense_matrix(dense_mat);

	delete[] dense_mat.values;
}

#endif

#define M_ 4
#define K_ M_
#define N_ M_
#define BH_ 2
#define BW_ BH_
#define SPARSITY 0.5
#define NRB_ M_/BH_
#define NNZB_ int(SPARSITY*(M_/BH_)*(N_/BW_))
#define NNZ_  NNZB_*(BH_*BW_)

// Generate sparse matrices
void populate_block_sparse_matrix(int rows, int cols, int bh, int bw, int nnzb,
	float values[NNZ_], int indices[NNZB_], int rowBlockPtr[(M_/BH_)+1], bool is_uniform){

	int nrb = rows / bh;
	int ncb = cols / bw;
	int nnzb_per_row = nnzb / nrb;

	assert(nnzb_per_row*nrb == nnzb);

	for (int i = 0; i < nrb+1; ++i)
	{
		rowBlockPtr[i] = 0;
	}

	bool flags[(M_/BH_)*(K_/BW_)];

	for (int i = 0; i < (M_/BH_)*(K_/BW_); ++i)
		flags[i] = false;

	// Picking non zero blocks randomly.
	if (is_uniform)
	{
		for (int rb = 0; rb < nrb; ++rb)
		{
			int cur_nnzb_per_rb = 0;
			while(cur_nnzb_per_rb < nnzb_per_row)
			{
				int flat_id = rb*ncb + rand()%(ncb);
				if(flags[flat_id] == false){
					flags[flat_id] = true;
					cur_nnzb_per_rb += 1;
				}
			}
		}
	}else{
		int cur_nnzb_per_rb = 0;
		while(cur_nnzb_per_rb < nnz)
		{
			int flat_id = rand()%(nrb*ncb);
			if(flags[flat_id] == false){
				flags[flat_id] = true;
				cur_nnzb_per_rb += 1;
			}
		}
	}
	

	// Populating indices and rowPtr
	int nzb_id = 0;
	for (int rb = 0; rb < nrb; ++rb)
	{
		for (int cb = 0; cb < ncb; ++cb)
		{
			int flat_id = rb*ncb + cb;
			if(flags[flat_id] == true){
				indices[nzb_id] = cb;
				rowPtr[rb] += 1;
				nzb_id += 1;
			}
		}
	}

	// Generate rowBlockPtr
	int sum = 0;
	for (int rb = 0; rb <= nrb; ++rb)
	{
		int temp = rowBlockPtr[rb];
		rowBlockPtr[rb] = sum;
		sum += temp;
	}

	// ALGO 0 : Random initialization
	for(int i=0; i< nnz; i++){
		values[i] = rand()%5 + 1;		
	}
}


void naive_block_sparse_dense_matmul(int M, int K, int N, int BH, int BW,
	float values[NNZ_], int indices[NNZB_], int rowBlockPtr[NRB_+1], float B[M_*K_], float C[K_*N_]){
	// TODO
	
	/*
	for (int row = 0; row < M; ++row)
	for (int col = 0; col < N; ++col)
	{
		float value = 0;
		for (int id = rowPtr[row]; id < rowPtr[row+1]; ++id)
		{
			int col_ind = indices[id];
			float A_val = values[id];
			value += A_val * B[col*K + col_ind];
		}
		C[col*M + row] = value;
	}
	*/
}



// Note : If program results in seg fault, 
// then it is probably due to stack overflow.
// Execute following command to increase stack limit.
// ulimit -s 300000000

int main(int argc, char* argv[]){
	bool VERBOSE = false;
	bool is_uniform = false;
	bool is_rowmajor = false;

	// Memory allocation
	float values[NNZ_];
	int indices[NNZB_];
	int rowBlockPtr[NRB_+1];

	float B[K_*N_];
	float C[M_*N_];

	#ifdef FDEBUG
		srand(time(NULL));
	#endif

	// Populating into corresponding data structures.
	if(VERBOSE) std::cout << "Generating A matrix " << std::endl;
	populate_block_sparse_matrix(M_, K_, BH_, BW_, NNZB_, values, indices, rowBlockPtr, is_uniform);

	for (int i = 0; i < K_*N_; ++i)
	{
		B[i] = rand() % 5 + 1;
	}

	for (int i = 0; i < M_*N_; ++i)
	{
		C[i] = rand() % 5 + 1;
	}

	#ifdef FDEBUG

	bsr_matrix A_bsr;
	dense_matrix B_dense;
	dense_matrix C_dense;

	A_bsr.rows = M_;
	A_bsr.cols = K_;
	A_bsr.bh = BH_;
	A_bsr.bw = BW_;
	A_bsr.nnz  = NNZ_;
	A_bsr.nnzb = NNZB_;
	A_bsr.values = values;
	A_bsr.indices = indices;
	A_bsr.rowPtr = rowBlockPtr;

	B_dense.rows = K_;
	B_dense.cols = N_;
	B_dense.values = B;
	B_dense.is_rowmajor = is_rowmajor;

	C_dense.rows = M_;
	C_dense.cols = N_;
	C_dense.values = C;
	C_dense.is_rowmajor = is_rowmajor;
	
	std::cout << "Matrix A" << std::endl;
	print_bsr_matrix(A_bsr);
	std::cout << "Matrix B" << std::endl;
	print_dense_matrix(B_dense);

	#endif

    if(!is_uniform){
	    naive_block_sparse_dense_matmul(M_, K_, N_, BH_, BW_, values, indices, rowBlockPtr, B, C);
    }
    else{
    	// TODO
    	//naive_usparse_dense_matmul(M_, K_, N_, values, indices, rowPtr, B, C);
    }

	#ifdef FDEBUG
		std::cout << "Matrix C" << std::endl;
		print_dense_matrix(C_dense);	
	#endif
	
	return 0;
}
