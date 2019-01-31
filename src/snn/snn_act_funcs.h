#ifndef _SNN_ACT_FUNCS_H_
#define _SNN_ACT_FUNCS_H_

#include "snn_config.h"

SNN_TYPE snn_act_relu(SNN_TYPE x);
SNN_TYPE snn_act_relu_deriv(SNN_TYPE x);

SNN_TYPE snn_act_linear(SNN_TYPE x);
SNN_TYPE snn_act_linear_deriv(SNN_TYPE x);

SNN_TYPE snn_act_bin_step(SNN_TYPE x);
SNN_TYPE snn_act_bin_step_deriv(SNN_TYPE x);

SNN_TYPE snn_act_sigmoid(SNN_TYPE x);
SNN_TYPE snn_act_sigmoid_deriv(SNN_TYPE x);

SNN_TYPE snn_act_tanh(SNN_TYPE x);
SNN_TYPE snn_act_tanh_deriv(SNN_TYPE x);

SNN_TYPE snn_act_atan(SNN_TYPE x);
SNN_TYPE snn_act_atan_deriv(SNN_TYPE x);

SNN_TYPE snn_act_elliot_sig(SNN_TYPE x);
SNN_TYPE snn_act_elliot_sig_deriv(SNN_TYPE x);

SNN_TYPE snn_act_sqnl(SNN_TYPE x);
SNN_TYPE snn_act_sqnl_deriv(SNN_TYPE x);

SNN_TYPE snn_act_lrelu(SNN_TYPE x);
SNN_TYPE snn_act_lrelu_deriv(SNN_TYPE x);

SNN_TYPE snn_act_selu(SNN_TYPE x);
SNN_TYPE snn_act_selu_deriv(SNN_TYPE x);

SNN_TYPE snn_act_softplus(SNN_TYPE x);
SNN_TYPE snn_act_softplus_deriv(SNN_TYPE x);

SNN_TYPE snn_act_bent_identity(SNN_TYPE x);
SNN_TYPE snn_act_bent_identity_deriv(SNN_TYPE x);

SNN_TYPE snn_act_silu(SNN_TYPE x);
SNN_TYPE snn_act_silu_deriv(SNN_TYPE x);

SNN_TYPE snn_act_sinusoid(SNN_TYPE x);
SNN_TYPE snn_act_sinusoid_deriv(SNN_TYPE x);

SNN_TYPE snn_act_sinc(SNN_TYPE x);
SNN_TYPE snn_act_sinc_deriv(SNN_TYPE x);

SNN_TYPE snn_act_gaussian(SNN_TYPE x);
SNN_TYPE snn_act_gaussian_deriv(SNN_TYPE x);

#endif
