#ifndef _SNN_TYPES_H_
#define _SNN_TYPES_H_

#include "snn_config.h"

typedef SNN_TYPE (*snn_act_func)(SNN_TYPE x);

typedef struct
{
    SNN_TYPE *data;
    int size;
    int width, height;
}snn_matrix_t;

typedef struct
{
    char status;
    snn_act_func act;
    snn_act_func act_deriv;
    double alpha;
    int hidden_layer_size;
    int hidden_layers;
    int input_size;
    int output_size;

    snn_matrix_t *weights;
    snn_matrix_t *weights_T;
    snn_matrix_t *weights_C;
    snn_matrix_t *delta;
    int weights_length;

    snn_matrix_t *layers;
    snn_matrix_t *layers_T;
    int layers_length;

    snn_matrix_t output;

    double error;
}snn_t;

#endif
