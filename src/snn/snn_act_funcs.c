#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "snn_act_funcs.h"

SNN_TYPE snn_abs(SNN_TYPE x)
{
    return (x > 0) ? x : -x;
}

SNN_TYPE snn_act_relu(SNN_TYPE x)
{
    return (x > 0) * x;
}

SNN_TYPE snn_act_relu_deriv(SNN_TYPE x)
{
    return x > 0;
}

SNN_TYPE snn_act_linear(SNN_TYPE x)
{
    return x;
}

SNN_TYPE snn_act_linear_deriv(SNN_TYPE x)
{
    return 1;
}

SNN_TYPE snn_act_bin_step(SNN_TYPE x)
{
    return (x >= 0) ? 1 : 0;
}

SNN_TYPE snn_act_bin_step_deriv(SNN_TYPE x)
{
    return x != 0 ? 0 : (((rand() % 200) - 100) / 100.0);
}

SNN_TYPE snn_act_sigmoid(SNN_TYPE x)
{
    return 1.0 / (1 + exp(-x));
}

SNN_TYPE snn_act_sigmoid_deriv(SNN_TYPE x)
{
    x = snn_act_sigmoid(x);
    return x*(1.0 - x);
}

SNN_TYPE snn_act_tanh(SNN_TYPE x)
{
    return tanh(x);
}

SNN_TYPE snn_act_tanh_deriv(SNN_TYPE x)
{
    x = snn_act_tanh(x);
    return (1 - x*x);
}

SNN_TYPE snn_act_atan(SNN_TYPE x)
{
    return atan(x);
}

SNN_TYPE snn_act_atan_deriv(SNN_TYPE x)
{
    return (1.0/(x*x + 1));
}

SNN_TYPE snn_act_elliot_sig(SNN_TYPE x)
{
    return x/(1 + snn_abs(x));
}

SNN_TYPE snn_act_elliot_sig_deriv(SNN_TYPE x)
{
    x = 1 + snn_abs(x);
    return 1.0/(x*x);
}

SNN_TYPE snn_act_sqnl(SNN_TYPE x)
{
    if(x > 2.0)
        return 1;
    else if(x >= 0)
        return (x - (x*x)/4);
    else if(x >= -2.0)
        return (x + (x*x)/4);
    else
        return -1;
}

SNN_TYPE snn_act_sqnl_deriv(SNN_TYPE x)
{
    return 1 + (x/2);
}

SNN_TYPE snn_act_lrelu(SNN_TYPE x)
{
    return (x < 0) ? 0.01*x : x;
}

SNN_TYPE snn_act_lrelu_deriv(SNN_TYPE x)
{
    return (x < 0) ? 0.01 : 1;
}

SNN_TYPE snn_act_selu(SNN_TYPE x)
{
    if(x < 0)
        return 1.0507*1.67326*(exp(x) - 1);
    return 1.0507*x;
}

SNN_TYPE snn_act_selu_deriv(SNN_TYPE x)
{
    if(x < 0)
        return 1.0507*1.67326*exp(x);
    return 1.0507;
}

SNN_TYPE snn_act_softplus(SNN_TYPE x)
{
    return log(1 + exp(x));
}

SNN_TYPE snn_act_softplus_deriv(SNN_TYPE x)
{
    return 1.0/(1 + exp(-x));
}

SNN_TYPE snn_act_bent_identity(SNN_TYPE x)
{
    return ((sqrt(x*x + 1) - 1)/2.0) + x;
}

SNN_TYPE snn_act_bent_identity_deriv(SNN_TYPE x)
{
    return (x/(2.0*sqrt(x*x + 1))) + 1;
}

SNN_TYPE snn_act_silu(SNN_TYPE x)
{
    return x*snn_act_sigmoid(x);
}

SNN_TYPE snn_act_silu_deriv(SNN_TYPE x)
{
    SNN_TYPE v = snn_act_silu(x);
    return v + snn_act_sigmoid(x)*(1.0 - v);
}

SNN_TYPE snn_act_sinusoid(SNN_TYPE x)
{
    return sin(x);
}

SNN_TYPE snn_act_sinusoid_deriv(SNN_TYPE x)
{
    return cos(x);
}

SNN_TYPE snn_act_sinc(SNN_TYPE x)
{
    return (x != 0) ? sin(x)/x : 1;
}

SNN_TYPE snn_act_sinc_deriv(SNN_TYPE x)
{
    return (x != 0) ? ((cos(x)/x) - (sin(x)/(x*x))) : 0;
}

SNN_TYPE snn_act_gaussian(SNN_TYPE x)
{
    return exp(-(x*x));
}

SNN_TYPE snn_act_gaussian_deriv(SNN_TYPE x)
{
    return -2.0*x*snn_act_gaussian(x);
}
