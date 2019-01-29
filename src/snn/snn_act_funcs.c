#include "snn_act_funcs.h"

SNN_TYPE snn_act_relu(SNN_TYPE x)
{
    return (x > 0) * x;
}

SNN_TYPE snn_act_relu2deriv(SNN_TYPE x)
{
    return x > 0;
}