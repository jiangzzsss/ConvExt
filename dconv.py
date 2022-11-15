import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.cpp_extension import load
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn.common_types import _size_2_t
from typing import Optional, Tuple
this_dir = '/mnt/cache/jiangzhen/kd0/adas-pod4.0-offlinekd/pod/zext'
dconv_cpp = load(name="dconv_cpp", sources=[str(this_dir) + "/dconv_cudnn.cpp"], extra_include_paths=[str(this_dir)],with_cuda=True)

class DCONVFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:Tensor, weight:Tensor, bias:Optional[Tensor],
                    padding:Tuple[int, ...], stride:Tuple[int, ...], dilation:Tuple[int, ...],
                    groups: int, algo:Tuple[int, ...]):
        output_ = dconv_cpp.cudnn_convolution(input, weight, padding, stride, dilation,
                                                groups, algo[0])
        variables = [input, weight]
        ctx.save_for_backward(*variables)
        ctx.params = (padding, stride, dilation, groups, algo, input.shape, weight.shape, 
                      bias if bias is not None else None)
        if(bias is not None):
            output_ += bias.view((1, -1, 1, 1))
        return output_
    
    @staticmethod
    def backward(ctx, grad_output):
        padding, stride, dilation, groups, algo, input_shape, weight_shape, bias = ctx.params 
        input, weight = ctx.saved_tensors
        input = input.detach()
        weight = weight.detach()
        # with torch.no_grad():
        grad_input = \
            dconv_cpp.cudnn_convolution_backward_input(input_shape, grad_output, weight,
                    padding, stride, dilation, groups, algo[1])
        grad_weight = \
            dconv_cpp.cudnn_convolution_backward_weight(weight_shape, grad_output, input,
                    padding, stride, dilation, groups, algo[2])
        if(bias is not None):
            grad_bias = grad_output.sum([0, 2, 3])
        else: grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None
        

class DConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        algo: Tuple[int, ...] = None, 
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        if(algo is None):
            self.algo = (0, 1, 1)
        elif(isinstance(algo, Tuple) == False):
            raise RuntimeError('algo must be a tuple of size 3! ')
        else:
            self.algo = algo
        super(DConv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return DCONVFunction.apply(input, self.weight, self.bias, 
                                   self.padding, self.stride, self.dilation,
                                   self.groups, self.algo)
