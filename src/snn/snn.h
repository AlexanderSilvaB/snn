#ifndef _SNN_H_
#define _SNN_H_

#include "snn_config.h"
#include "snn_types.h"
#include "snn_act_funcs.h"

int snn_init(snn_t *snn, int input_size, int output_size, int hidden_layers, int hidden_layer_size);
int snn_destroy(snn_t *snn);
int snn_print(snn_t *snn);
int snn_print_weights(snn_t *snn);
int snn_print_raw(SNN_TYPE *data, int rows, int cols);
int snn_act(snn_t *snn, snn_act_func fn);
int snn_act_deriv(snn_t *snn, snn_act_func fn);
int snn_train(snn_t *snn, int size, SNN_TYPE *input, SNN_TYPE *ouput);
int snn_train_epochs(snn_t *snn, int epochs, int block_size, int size, SNN_TYPE *input, SNN_TYPE *ouput);
int snn_train_max_error_epochs(snn_t *snn, int epochs, SNN_TYPE max_error, int block_size, int size, SNN_TYPE *input, SNN_TYPE *ouput);
int snn_run(snn_t *snn, int size, SNN_TYPE *input, SNN_TYPE *ouput);

#endif
