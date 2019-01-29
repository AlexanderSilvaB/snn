#ifndef _SNN_MATRIX_H_
#define _SNN_MATRIX_H_

#include "snn_config.h"
#include "snn_types.h"

int snn_matrix_create(snn_matrix_t *mat, int rows, int cols);
int snn_matrix_invalidate(snn_matrix_t *mat);
int snn_matrix_destroy(snn_matrix_t *mat);
void snn_matrix_fill(snn_matrix_t *mat, SNN_TYPE *data);
void snn_matrix_set(snn_matrix_t *mat, SNN_TYPE value);
void snn_matrix_print(snn_matrix_t *mat);
void snn_matrix_apply(snn_matrix_t *mat, snn_act_func fn);
void snn_matrix_apply_mult(snn_matrix_t *dst, snn_matrix_t *mat, snn_act_func fn);
void snn_matrix_mult(snn_matrix_t *dst, snn_matrix_t *A, snn_matrix_t *B);
void snn_matrix_sub(snn_matrix_t *dst, snn_matrix_t *A, snn_matrix_t *B);
void snn_matrix_sub_scale(snn_matrix_t *dst, snn_matrix_t *A, SNN_TYPE scale);
void snn_matrix_transpose(snn_matrix_t *dst, snn_matrix_t *src);
SNN_TYPE snn_matrix_mse(snn_matrix_t *A, snn_matrix_t *B);

#endif
