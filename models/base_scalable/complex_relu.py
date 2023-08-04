import torch
import torch.nn as nn


class ComReLU(nn.Module):
    """The complex ReLU layer from the `MagNet: A Neural Network for Directed Graphs. <https://arxiv.org/pdf/2102.11391.pdf>`_ paper.
    """

    def __init__(self, ):
        super(ComReLU, self).__init__()

    def complex_relu(self, real: torch.FloatTensor, img: torch.FloatTensor):
        """
        Complex ReLU function.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        mask = 1.0*(real >= 0)
        return mask*real, mask*img

    def forward(self, real: torch.FloatTensor, img: torch.FloatTensor):
        """
        Making a forward pass of the complex ReLU layer.

        Arg types:
            * real, imag (PyTorch Float Tensor) - Node features.
        Return types:
            * real, imag (PyTorch Float Tensor) - Node features after complex ReLU.
        """
        real, img = self.complex_relu(real, img)
        return real, img
