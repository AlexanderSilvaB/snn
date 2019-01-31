#include <stdio.h>
#include <stdlib.h>
#include <snn/snn.h>

int main(int argc, char *argv[])
{
    /*
        4-2-4
        input = output
    */

    srand(1);

    const int input_size = 4;
    const int output_size = 4;
    const int hidden_layers = 1;
    const int hidden_layer_size = 4;

    const int data_entries = 4;

    SNN_TYPE input[4*4] = 
    {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    SNN_TYPE output[4*4] = 
    {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };

    snn_t snn;
    if(!snn_init(&snn, input_size, output_size, hidden_layers, hidden_layer_size))
    {
        printf("Could not instantiate snn");
        snn_destroy(&snn);
        return SNN_FAIL;
    }

    snn_act(&snn, snn_act_bin_step);
    snn_act_deriv(&snn, snn_act_bin_step_deriv);

    printf("Training:\n");
    snn_train_epochs(&snn, 1000, 50, data_entries, input, output);
    
    printf("Structure:\n");
    snn_print(&snn);

    printf("Expected:\n");
    snn_print_raw(output, data_entries, output_size);
    snn_run(&snn, data_entries, input, output);
    printf("Output:\n");
    snn_print_raw(output, data_entries, output_size);
    

    snn_destroy(&snn);
    return SNN_FAIL;
}