#include <stdio.h>
#include <stdlib.h>
#include <snn/snn.h>

int main(int argc, char *argv[])
{
    /*
        Streetlights system
        Type:
            GREEN, YELLOW, RED (0 or 1)
        Information:
            Can walk(1), Stop(0)
    */

    srand(1);

    const int input_size = 3;
    const int output_size = 1;
    const int hidden_layers = 1;
    const int hidden_layer_size = 4;

    const int data_entries = 4;

    SNN_TYPE input[4*3] = 
    {
        1, 0, 1,
        0, 1, 1,
        0, 0, 1,
        1, 1, 1
    };

    SNN_TYPE output[4*1] = 
    {
        1,
        1,
        0,
        0
    };

    snn_t snn;
    if(!snn_init(&snn, input_size, output_size, hidden_layers, hidden_layer_size))
    {
        printf("Could not instantiate snn");
        snn_destroy(&snn);
        return SNN_FAIL;
    }

    printf("Training:\n");
    snn_train_epochs(&snn, 60, 10, data_entries, input, output);
    
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