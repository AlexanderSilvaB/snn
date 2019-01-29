#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "snn.h"
#include "snn_act_funcs.h"
#include "snn_matrix.h"

int weights_create(snn_t *snn);
int weights_destroy(snn_t *snn);
int layers_create(snn_t *snn);
int layers_destroy(snn_t *snn);
int delta_create(snn_t *snn);
int delta_destroy(snn_t *snn);
void snn_cp(SNN_TYPE *dst, SNN_TYPE *src, int size);

int snn_init(snn_t *snn, int input_size, int output_size, int hidden_layers, int hidden_layer_size)
{
    if(!snn || snn->status == SNN_VALID)
        return SNN_FAIL;
    snn->alpha = 0.2;
    snn->hidden_layers = hidden_layers;
    snn->hidden_layer_size = hidden_layer_size;
    snn->input_size = input_size;
    snn->output_size = output_size;
    snn->act = snn_act_relu;
    snn->act_deriv = snn_act_relu2deriv;

    snn->error = 0;

    snn->weights = NULL;
    snn->weights_T = NULL;
    snn->weights_C = NULL;
    snn->layers = NULL;
    snn->delta = NULL;
    snn->output.data = NULL;

    snn->status = SNN_VALID;

    if(!weights_create(snn))
    {
        weights_destroy(snn);
        snn->status = SNN_INVALID;
        return SNN_FAIL;
    }

    if(!layers_create(snn))
    {
        layers_destroy(snn);
        weights_destroy(snn);
        snn->status = SNN_INVALID;
        return SNN_FAIL;
    }
    
    if(!delta_create(snn))
    {
        delta_destroy(snn);
        layers_destroy(snn);
        weights_destroy(snn);
        snn->status = SNN_INVALID;
        return SNN_FAIL;
    }

    return SNN_OK;
}

int snn_destroy(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL;

    delta_destroy(snn);
    layers_destroy(snn);
    weights_destroy(snn);

    snn->status = SNN_INVALID;
    return SNN_OK;
}

int snn_print(snn_t *snn)
{
    if(!snn)
    {
        printf("SNN: Invalid");
        return SNN_FAIL;
    }
    if(snn->status != SNN_VALID)
    {
        printf("SNN: Not initialized");
        return SNN_FAIL;
    }

    printf("Input size: %d\n", snn->input_size);
    printf("Output size: %d\n", snn->output_size);
    printf("Hidden layers: %d\n", snn->hidden_layers);
    printf("Hidden layers size: %d\n", snn->hidden_layer_size);
    printf("Alpha: %f\n", snn->alpha);
    
    snn_print_weights(snn);
}

int snn_print_weights(snn_t *snn)
{
    if(!snn)
    {
        printf("SNN: Invalid");
        return SNN_FAIL;
    }
    if(snn->status != SNN_VALID)
    {
        printf("SNN: Not initialized");
        return SNN_FAIL;
    }

    printf("Weights: %d\n", snn->weights_length);
    
    int i;
    for(i = 0; i < snn->weights_length; i++)
    {
        printf("Weight %d->%d (%d x %d)\n", i, i+1, snn->weights[i].height, snn->weights[i].width);
        snn_matrix_print(&snn->weights[i]);
    }
    return SNN_OK;
}

int snn_print_raw(SNN_TYPE *data, int rows, int cols)
{
    int x, y, j;
    for(y = 0, j = 0; y < rows; y++)
    {
        for(x = 0; x < cols; x++, j++)
        {
            printf("%f\t", data[j]);
        }
        printf("\n");
    }
}

int snn_act(snn_t *snn, snn_act_func fn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL;
    snn->act = fn;
    return SNN_OK;
}

int snn_act_deriv(snn_t *snn, snn_act_func fn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL;
    snn->act_deriv = fn;
    return SNN_OK;
}

int snn_run(snn_t *snn, int size, SNN_TYPE *input, SNN_TYPE *ouput)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL;

    snn->error = 0;
    int y, x, j, i;
    int sz = sizeof(SNN_TYPE);

    for(i = 0; i < size; i++)
    {
        snn_cp(snn->layers[0].data, input + snn->input_size*i, snn->input_size);

        for(j = 1; j < snn->layers_length; j++)
        {
            snn_matrix_mult(&snn->layers[j], &snn->layers[j-1], &snn->weights[j-1]);
            if(j < snn->layers_length-1)
            {
                snn_matrix_apply(&snn->layers[j], snn->act);
            }
        }
        snn_cp(ouput + snn->output_size*i, snn->layers[snn->layers_length-1].data, snn->output_size);
    }
}

int snn_train(snn_t *snn, int size, SNN_TYPE *input, SNN_TYPE *ouput)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL;

    snn->error = 0;
    int y, x, j, i;

    for(i = 0; i < size; i++)
    {
        snn_cp(snn->layers[0].data, input + snn->input_size*i, snn->input_size);
        snn_cp(snn->output.data, ouput + snn->output_size*i, snn->output_size);

        for(j = 1; j < snn->layers_length; j++)
        {
            snn_matrix_mult(&snn->layers[j], &snn->layers[j-1], &snn->weights[j-1]);
            if(j < snn->layers_length-1)
            {
                snn_matrix_apply(&snn->layers[j], snn->act);
            }
        }

        snn->error += snn_matrix_mse(&snn->layers[snn->layers_length-1], &snn->output);

        j = snn->weights_length-1;
        snn_matrix_sub(&snn->delta[j], &snn->layers[snn->layers_length-1], &snn->output);
        j--;
        for(; j >= 0; j--)
        {
            snn_matrix_transpose(&snn->weights_T[j+1], &snn->weights[j+1]);
            snn_matrix_mult(&snn->delta[j], &snn->delta[j+1], &snn->weights_T[j+1]);
            snn_matrix_apply_mult(&snn->delta[j], &snn->layers[j+1], snn->act_deriv);
        }
        
        for(j = 0; j < snn->weights_length; j++)
        {
            snn_matrix_transpose(&snn->layers_T[j], &snn->layers[j]);
            snn_matrix_mult(&snn->weights_C[j], &snn->layers_T[j], &snn->delta[j]);
            snn_matrix_sub_scale(&snn->weights[j], &snn->weights_C[j], snn->alpha);
        }
        
    }
    return SNN_OK;
}

int snn_train_epochs(snn_t *snn, int epochs, int block_size, int size, SNN_TYPE *input, SNN_TYPE *ouput)
{
    int i;
    for(i = 0; i < epochs; i++)
    {
        snn_train(snn, size, input, ouput);
        if(i % block_size == 0)
            printf("Error: %f\n", snn->error);
    }
}

int snn_train_max_error_epochs(snn_t *snn, int epochs, SNN_TYPE max_error, int block_size, int size, SNN_TYPE *input, SNN_TYPE *ouput)
{
    int i;
    for(i = 0; i < epochs; i++)
    {
        snn_train(snn, size, input, ouput);
        if(i % block_size == 0)
            printf("Error: %f\n", snn->error);
        if(snn->error < max_error)
            break;
    }
}

// Private
int weights_create(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(snn->weights)
        return SNN_FAIL;

    snn->weights_length = 1 + snn->hidden_layers;
    snn->weights = (snn_matrix_t*)malloc(snn->weights_length * sizeof(snn_matrix_t));
    snn->weights_T = (snn_matrix_t*)malloc(snn->weights_length * sizeof(snn_matrix_t));
    snn->weights_C = (snn_matrix_t*)malloc(snn->weights_length * sizeof(snn_matrix_t));
    
    int width = snn->hidden_layer_size;
    int height = snn->input_size;

    int i, j;
    for(i = 0; i < snn->weights_length; i++)
    {
        if(i == snn->weights_length-1)
        {
            width = snn->output_size;
        }

        snn_matrix_create(&snn->weights[i], height, width);
        snn_matrix_create(&snn->weights_C[i], height, width);

        if(i == 0)
        {
            snn_matrix_invalidate(&snn->weights_T[i]);
        }
        else
        {
            snn_matrix_create(&snn->weights_T[i], width, height);
        }

        for(j = 0; j < snn->weights[i].size; j++)
        {
            snn->weights[i].data[j] = ((rand() % 200) - 100) / 100.0;
        }

        height = snn->hidden_layer_size;
    }

    return SNN_OK;
}

int weights_destroy(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(!snn->weights)
        return SNN_FAIL;

    int i;
    for(i = 0; i < snn->weights_length; i++)
    {
        snn_matrix_destroy(&snn->weights[i]);
        snn_matrix_destroy(&snn->weights_C[i]);
        snn_matrix_destroy(&snn->weights_T[i]);
    }
    free(snn->weights);
    free(snn->weights_T);
    free(snn->weights_C);
    
    snn->weights_length = 0;
    snn->weights = NULL;
}

int layers_create(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(snn->layers)
        return SNN_FAIL;

    snn->layers_length = 2 + snn->hidden_layers;
    snn->layers = (snn_matrix_t*)malloc(snn->layers_length * sizeof(snn_matrix_t));
    snn->layers_T = (snn_matrix_t*)malloc(snn->layers_length * sizeof(snn_matrix_t));
    
    int width = snn->input_size;
    int height = 1;

    int i, j;
    for(i = 0; i < snn->layers_length; i++)
    {
        snn_matrix_create(&snn->layers[i], height, width);

        if(i == snn->layers_length-1)
        {
            snn_matrix_invalidate(&snn->layers_T[i]);
        }
        else
        {
            snn_matrix_create(&snn->layers_T[i], width, height);
        }
        

        if(i < snn->weights_length)
            width = snn->weights[i].width;
    }

    snn_matrix_create(&snn->output, snn->layers[snn->layers_length-1].height, snn->layers[snn->layers_length-1].width);
    return SNN_OK;
}

int layers_destroy(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(!snn->layers)
        return SNN_FAIL;

    int i;
    for(i = 0; i < snn->layers_length; i++)
    {
        snn_matrix_destroy(&snn->layers[i]);
        snn_matrix_destroy(&snn->layers_T[i]);
    }
    free(snn->layers);
    free(snn->layers_T);
    
    snn->layers_length = 0;
    snn->layers = NULL;
    
    snn_matrix_destroy(&snn->output);
}

int delta_create(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(snn->delta)
        return SNN_FAIL;

    snn->delta = (snn_matrix_t*)malloc(snn->weights_length * sizeof(snn_matrix_t));
    
    int width, height;

    int i, j;
    for(i = 0; i < snn->weights_length; i++)
    {
        width = snn->layers[i+1].width;
        height = snn->layers[i+1].height;

        snn_matrix_create(&snn->delta[i], height, width);
    }

    return SNN_OK;
}

int delta_destroy(snn_t *snn)
{
    if(!snn || snn->status != SNN_VALID)
        return SNN_FAIL; 
    if(!snn->delta)
        return SNN_FAIL;

    int i;
    for(i = 0; i < snn->weights_length; i++)
    {
        snn_matrix_destroy(&snn->delta[i]);
    }
    free(snn->delta);
    
    snn->delta = NULL;
}

void snn_cp(SNN_TYPE *dst, SNN_TYPE *src, int size)
{
    int i;
    for(i = 0; i < size; i++)
        dst[i] = src[i];
}

