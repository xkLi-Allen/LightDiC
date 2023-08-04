import math
import torch
import torch.nn.functional as F

from typing import Optional


Tensor = torch.Tensor

def complex_fcaller(funtional_handle, *args):
    return torch.complex(funtional_handle(args[0].real, *args[1:]), funtional_handle(args[0].imag, *args[1:]))

def d_silu(input: Tensor):
    return torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))

def c_sigmoid(input: Tensor):
    if input.is_complex():
        #return torch.complex(F.sigmoid(input.real), F.sigmoid(input.imag))
        # nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
        return torch.complex(torch.sigmoid(input.real), torch.sigmoid(input.imag))
    else:
        #return F.sigmoid(input)
        # nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.
        return torch.sigmoid(input)

def c_tanh(input: Tensor):
    if input.is_complex():
        #return torch.complex(F.tanh(input.real), F.tanh(input.imag))
        # nn.functional.tanh is deprecated. Use torch.tanh instead.
        return torch.complex(torch.tanh(input.real), torch.tanh(input.imag))
    else:
        #return F.tanh(input)
        # nn.functional.tanh is deprecated. Use torch.tanh instead.
        return torch.tanh(input)

def mod_tanh(input: Tensor, rounding_mode: str = None) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input=input, other=magnitude, rounding_mode=rounding_mode)
        #return torch.mul(F.tanh(magnitude), euler_phase).type(input.type())
        # nn.functional.tanh is deprecated. Use torch.tanh instead.
        return torch.mul(torch.tanh(magnitude), euler_phase).type(input.type())
    else:
        #return F.tanh(input)
        # nn.functional.tanh is deprecated. Use torch.tanh instead.
        return torch.tanh(input)

def hirose(input: Tensor, m: float = 1., rounding_mode: str = None, inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input, magnitude)
        if inplace:
            input = torch.mul(torch.tanh(torch.div(input=magnitude, other=torch.pow(m, 2), rounding_mode=rounding_mode)), euler_phase).type(input.type())
            return input
        else:
            hirose = torch.mul(torch.tanh(torch.div(input=magnitude, other=torch.pow(m, 2), rounding_mode=rounding_mode)), euler_phase).type(input.type())
            return hirose
    else:
        if inplace:
            input = torch.tanh(torch.div(input=input, other=torch.pow(m, 2), rounding_mode=rounding_mode)).type(input.type())
            return input
        else:
            hirose = torch.tanh(torch.div(input=input, other=torch.pow(m, 2), rounding_mode=rounding_mode)).type(input.type())
            return hirose

def siglog(input: Tensor):
    return torch.div(input, 1 + torch.abs(input))

def c_cardioid(input: Tensor):
    phase = torch.angle(input)
    return 0.5 * torch.mm(1 + torch.cos(phase), input)

def c_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.relu(input.real, inplace=inplace), F.relu(input.imag, inplace=inplace))
    else:
        return F.relu(input, inplace=inplace)

def mod_relu(input: Tensor, bias: float = -math.sqrt(2), rounding_mode: str = None, inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input=input, other=magnitude, rounding_mode=rounding_mode)
        if inplace:
            input = torch.mul(F.relu(magnitude + bias, inplace=False), euler_phase).type(input.type())
            return input
        else:
            mod_relu = torch.mul(F.relu(magnitude + bias, inplace=inplace), euler_phase).type(input.type())
            return mod_relu
    else:
        return F.relu(input, inplace=inplace)

def z_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        if inplace:
            mask = torch.zeros_like(input)
            input = torch.where(torch.angle(input) < 0, mask, input)
            input = torch.where(torch.angle(input) > (math.pi / 2), mask, input)
            return input
        else:
            mask = torch.zeros_like(input)
            z_relu = torch.where(torch.angle(input) < 0, mask, input)
            z_relu = torch.where(torch.angle(z_relu) > (math.pi / 2), mask, z_relu)
            return z_relu
    else:
        return F.relu(input, inplace=inplace)

def c_leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.leaky_relu(input=input.real, negative_slope=negative_slope, inplace=inplace), \
                            F.leaky_relu(input=input.imag, negative_slope=negative_slope, inplace=inplace))
    else:
        return F.leaky_relu(input=input, negative_slope=negative_slope, inplace=inplace)

def lip_silu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = F.silu(input=input, inplace=inplace) / 1.1
        return input
    else:
        lipsilu = F.silu(input=input, inplace=inplace) / 1.1
        return lipsilu
    
def squareplus(input: Tensor, bias: float = 4, inplace: bool = False) -> Tensor:
    if inplace:
        input = 0.5 * (input + torch.sqrt(torch.pow(input, 2) + bias))
        return input
    else:
        squareplus = 0.5 * (input + torch.sqrt(torch.pow(input, 2) + bias))
        return squareplus

def tanh_exp(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return input
    else:
        tanh_exp = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return tanh_exp

def r_gteu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        if torch.equal(input.real, torch.zeros_like(input.real)):
            return tanh_exp(input=input.imag, inplace=inplace)
        else:
            if inplace:
                input = torch.mul(input.real, torch.tanh(torch.exp(input.imag)))
                return input
            else:
                r_gteu = torch.mul(input.real, torch.tanh(torch.exp(input.imag)))
                return r_gteu
    else:
        return tanh_exp(input=input, inplace=inplace)

def mod_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.softmax(torch.abs(input), dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def mod_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.log_softmax(torch.abs(input), dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def r_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def r_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return F.log_softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)