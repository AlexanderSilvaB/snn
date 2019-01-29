#include <stdio.h>
#include <stdlib.h>
#include "snn_matrix.h"

// The base code for matrix multiplication is from fast-matrix-multiplication
// Github: https://github.com/deuxbot/fast-matrix-multiplication

#define min(a,b) (((a)<(b))?(a):(b))
void mxm(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p, int uf);

int snn_matrix_create(snn_matrix_t *mat, int rows, int cols)
{
	if(!mat)
		return SNN_FAIL;
	mat->width = cols;
	mat->height = rows;
	mat->size = rows*cols;
	mat->data = (SNN_TYPE*)malloc(mat->size * sizeof(SNN_TYPE));
	return SNN_OK;
}

int snn_matrix_invalidate(snn_matrix_t *mat)
{
	if(!mat)
		return SNN_FAIL;
	mat->width = 0;
	mat->height = 0;
	mat->size = 0;
	mat->data = NULL;
	return SNN_OK;
}

int snn_matrix_destroy(snn_matrix_t *mat)
{
	if(!mat)
		return SNN_FAIL;
	if(!mat->data)
		return SNN_FAIL;
	free(mat->data);
	mat->width = 0;
	mat->height = 0;
	mat->size = 0;
	mat->data = NULL;
	return SNN_OK;
}

void snn_matrix_fill(snn_matrix_t *mat, SNN_TYPE *data)
{
	int i;
	for(i = 0; i < mat->size; i++)
		mat->data[i] = data[i];
}

void snn_matrix_set(snn_matrix_t *mat, SNN_TYPE value)
{
	int i;
	for(i = 0; i < mat->size; i++)
		mat->data[i] = value;
}

void snn_matrix_print(snn_matrix_t *mat)
{
    int x, y, j;
    for(y = 0, j = 0; y < mat->height; y++)
    {
        for(x = 0; x < mat->width; x++, j++)
        {
            printf("%f\t", mat->data[j]);
        }
        printf("\n");
    }
}

void snn_matrix_apply(snn_matrix_t *mat, snn_act_func fn)
{
    int j;
    for(j = 0; j < mat->size; j++)
        mat->data[j] = fn(mat->data[j]);
}

void snn_matrix_apply_mult(snn_matrix_t *dst, snn_matrix_t *mat, snn_act_func fn)
{
    int j;
    for(j = 0; j < mat->size; j++)
        dst->data[j] *= fn(mat->data[j]);
}

void snn_matrix_mult(snn_matrix_t *dst, snn_matrix_t *A, snn_matrix_t *B)
{
    int m = A->height;
    int n = A->width;
    int p = B->width;
    int nn = B->height;
    //printf("(%d, %d)x(%d, %d)\n", m, n, nn, p);

    int uf; 
	
	if(B->width % 16 == 0) 		uf = 16;
	else if(B->width % 8 == 0) 	uf = 8;
	else if(B->width % 4 == 0) 	uf = 4;
	else if(B->width % 2 == 0)		uf = 2;
	else					uf = 0;
	
	//printf("Matrix dimensions %dx%d\n", m, n);
	//printf("Block size: %d\n", SNN_MATRIX_BLK_SIZE);	
	//printf("Unrolls: %d\n", uf);
	
	snn_matrix_set(dst, 0);
	mxm(dst->data, A->data, B->data, m, n, p, uf); 	
}

void snn_matrix_sub(snn_matrix_t *dst, snn_matrix_t *A, snn_matrix_t *B)
{
    int j;
    for(j = 0; j < A->size; j++)
    {
        dst->data[j] = A->data[j] - B->data[j];
    }
}

void snn_matrix_sub_scale(snn_matrix_t *dst, snn_matrix_t *A, SNN_TYPE scale)
{
    int j;
    for(j = 0; j < dst->size; j++)
    {
        dst->data[j] -= (scale * A->data[j]);
    }
}

void snn_matrix_transpose(snn_matrix_t *dst, snn_matrix_t *src)
{
    int x, y, i, j;
    for(y = 0; y < src->height; y++)
    {
        for(x = 0; x < src->width; x++)
        {
            i = y*src->width + x;
            j = x*src->height + y;
            dst->data[j] = src->data[i];
        }
    }
}

SNN_TYPE snn_matrix_mse(snn_matrix_t *A, snn_matrix_t *B)
{
    SNN_TYPE e = 0, v = 0;
    int j;
    for(j = 0; j < A->size; j++)
    {
        v = A->data[j] - B->data[j];
        e += v*v;
    }
    return e;
}

// Private
void mxm_naive(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{
    int i, j, k;

    for(i = 0; i < m; i++)
        for(j = 0; j < p; j++)
			for(k = 0; k < n; k++)
                C[p*i+j] += A[n*i+k] * B[p*k+j];			      
}

void mxm_block(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, bs = SNN_MATRIX_BLK_SIZE;

	for(ii = 0; ii < m; ii += bs)
		for(jj = 0; jj < p; jj += bs)
			for(kk = 0; kk < n; kk += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(j = jj; j < min(p, jj+bs); j++)
						for(k = kk; k < min(n, kk+bs); k++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];											
}

void mxm_block_reorder(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, bs = SNN_MATRIX_BLK_SIZE;

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += A[n*i+k] * B[p*k+j];						
				
}

void mxm_block_reorder_reuse(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, Aik, bs = SNN_MATRIX_BLK_SIZE;
	
	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j++)
							C[p*i+j] += Aik * B[p*k+j];		
					}					
}

void mxm_block_reorder_reuse_unroll_2(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, Aik, bs = SNN_MATRIX_BLK_SIZE;
	
	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=2)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
						}
					}					
}

void mxm_block_reorder_reuse_unroll_4(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, Aik, bs = SNN_MATRIX_BLK_SIZE;

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=4)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];		
						}
					}					
}

void mxm_block_reorder_reuse_unroll_8(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, Aik, bs = SNN_MATRIX_BLK_SIZE;

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=8)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];	
							C[p*i+j+4] += Aik * B[p*k+j+4];		
							C[p*i+j+5] += Aik * B[p*k+j+5];	
							C[p*i+j+6] += Aik * B[p*k+j+6];	
							C[p*i+j+7] += Aik * B[p*k+j+7];	
						}
					}					
}

void mxm_block_reorder_reuse_unroll_16(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p)
{ 
	int i, j, k, ii, jj, kk, Aik, bs = SNN_MATRIX_BLK_SIZE;

	for(ii = 0; ii < m; ii += bs)
		for(kk = 0; kk < n; kk += bs)
			for(jj = 0; jj < p; jj += bs)
				for(i = ii; i < min(m, ii+bs); i++)
					for(k = kk; k < min(n, kk+bs); k++)
					{
						Aik = A[n*i+k];
						for(j = jj; j < min(p, jj+bs); j+=16)
						{
							C[p*i+j] += Aik * B[p*k+j];
							C[p*i+j+1] += Aik * B[p*k+j+1];		
							C[p*i+j+2] += Aik * B[p*k+j+2];	
							C[p*i+j+3] += Aik * B[p*k+j+3];	
							C[p*i+j+4] += Aik * B[p*k+j+4];		
							C[p*i+j+5] += Aik * B[p*k+j+5];	
							C[p*i+j+6] += Aik * B[p*k+j+6];	
							C[p*i+j+7] += Aik * B[p*k+j+7];
							C[p*i+j+8] += Aik * B[p*k+j+8];		
							C[p*i+j+9] += Aik * B[p*k+j+9];	
							C[p*i+j+10] += Aik * B[p*k+j+10];	
							C[p*i+j+11] += Aik * B[p*k+j+11];		
							C[p*i+j+12] += Aik * B[p*k+j+12];	
							C[p*i+j+13] += Aik * B[p*k+j+13];	
							C[p*i+j+14] += Aik * B[p*k+j+14];	
							C[p*i+j+15] += Aik * B[p*k+j+15];	
						}
					}					
}

void mxm(SNN_TYPE *C, const SNN_TYPE *A, const SNN_TYPE *B, int m, int n, int p, int uf)
{
	//mxm_naive(C, A, B, m, n, p);
	//mxm_block(C, A, B, m, n, p);
	//mxm_block_reorder(C, A, B, m, n, p);
	switch(uf)
	{
		case 16:mxm_block_reorder_reuse_unroll_16(C, A, B, m, n, p);
				break;
		case 8:	mxm_block_reorder_reuse_unroll_8(C, A, B, m, n, p);
				break;
  		case 4: mxm_block_reorder_reuse_unroll_4(C, A, B, m, n, p);
				break;
  		case 2: mxm_block_reorder_reuse_unroll_2(C, A, B, m, n, p);
				break;
  		default: mxm_block_reorder_reuse(C, A, B, m, n, p);
				break;
	}
}