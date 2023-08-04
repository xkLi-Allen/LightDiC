import math
import torch
import numpy as np
import torch.nn.init as init

from torch import Tensor


def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    complex_linear_fns = ['c_linear', 'c_conv1d', 'c_conv2d', 'c_conv3d', 'c_conv_transpose1d', 'c_conv_transpose2d', 'c_conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid' or nonlinearity in complex_linear_fns or nonlinearity == 'c_sigmoid':
        return 1
    elif nonlinearity == 'tanh' or nonlinearity == 'c_tanh' or nonlinearity == 'mod_tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu' or nonlinearity == 'c_relu' or nonlinearity == 'z_relu' or nonlinearity == 'mod_relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu' or nonlinearity == 'c_leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu' or nonlinearity == 'c_selu':
        # Value found empirically (https://github.com/pytorch/pytorch/pull/50664)
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def complex_xavier_uniform_(tensor: Tensor, gain: float = 1.) -> Tensor:
    if tensor.is_complex():
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        sigma = gain / math.sqrt(float(fan_in + fan_out))
        magnitude = torch.from_numpy(np.random.rayleigh(scale=sigma, size=tensor.size()))
        with torch.no_grad():
            phase = tensor.real.uniform_(-math.pi, math.pi).type(tensor.type())
            phase.mul_(1j)
            phase.exp_()
            phase.mul_(magnitude)
            tensor.zero_()
            return tensor.add_(phase)
    else:
        return init.xavier_uniform_(tensor=tensor, gain=gain)

def complex_kaiming_uniform_(tensor: Tensor, a: float = 0., mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    if tensor.is_complex():
        fan = init._calculate_correct_fan(tensor, mode)
        gain = init.calculate_gain(nonlinearity, a)
        sigma = gain / math.sqrt(float(fan))
        magnitude = torch.from_numpy(np.random.rayleigh(scale=sigma, size=tensor.size()))
        with torch.no_grad():
            phase = tensor.real.uniform_(-math.pi, math.pi).type(tensor.type())
            phase.mul_(1j)
            phase.exp_()
            phase.mul_(magnitude)
            tensor.zero_()
            return tensor.add_(phase)
    else:
        return init.kaiming_uniform_(tensor=tensor, a=a, mode=mode, nonlinearity=nonlinearity)

def complex_xavier_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    if tensor.is_complex():
        fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
        sigma = gain / math.sqrt(float(fan_in + fan_out))
        magnitude = torch.from_numpy(np.random.rayleigh(scale=sigma, size=tensor.size()))
        with torch.no_grad():
            phase = tensor.real.normal_(0, sigma).type(tensor.type())
            phase.mul_(1j)
            phase.exp_()
            phase.mul_(magnitude)
            tensor.zero_()
            return tensor.add_(phase)
    else:
        return init.xavier_normal_(tensor=tensor, gain=gain)

def complex_kaiming_normal_(tensor: Tensor, a: float = 0., mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'):
    if tensor.is_complex():
        fan = init._calculate_correct_fan(tensor, mode)
        gain = init.calculate_gain(nonlinearity, a)
        sigma = gain / math.sqrt(float(fan))
        magnitude = torch.from_numpy(np.random.rayleigh(scale=sigma, size=tensor.size()))
        with torch.no_grad():
            phase = tensor.real.normal_(0, sigma).type(tensor.type())
            phase.mul_(1j)
            phase.exp_()
            phase.mul_(magnitude)
            tensor.zero_()
            return tensor.add_(phase)
    else:
        return init.kaiming_normal_(tensor=tensor, a=a, mode=mode, nonlinearity=nonlinearity)