import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numpy as np


################################## Generate the 2D-DCT Kernels of size NxN ##################################
def initDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    cnum = 0
    for i in range(N):
        for j in range(N):
            ivec = np.linspace(0.5 * math.pi / N * i, (N - 0.5) * math.pi / N * i, num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0.5 * math.pi / N * j, (N - 0.5) * math.pi / N * j, num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            if i==0 and j==0:
                slice = slice / N
            elif i*j==0:
                slice = slice * np.sqrt(2) / N
            else:
                slice = slice * 2.0 / N

            kernel[:,:,cnum] = slice
            cnum = cnum + 1
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (3,0,1,2))
    return kernel


################################## Generate the 2D-iDCT Kernels of size NxN ##################################
def initIDCTKernel(N):
    kernel = np.zeros((N, N, N*N))
    for i_ in range(N):
        i = N - i_ - 1
        for j_ in range(N):
            j = N - j_ - 1
            ivec = np.linspace(0, (i+0.5)*math.pi/N * (N-1), num=N)
            ivec = np.cos(ivec)
            jvec = np.linspace(0, (j+0.5)*math.pi/N * (N-1), num=N)
            jvec = np.cos(jvec)
            slice = np.outer(ivec, jvec)

            ic = np.sqrt(2.0 / N) * np.ones(N)
            ic[0] = np.sqrt(1.0 / N)
            jc = np.sqrt(2.0 / N) * np.ones(N)
            jc[0] = np.sqrt(1.0 / N)
            cmatrix = np.outer(ic, jc)

            slice = slice * cmatrix
            slice = slice.reshape((1, N*N))
            slice = slice[np.newaxis, :]
            kernel[i_, j_, :] = slice / (N * N)
    kernel = kernel[np.newaxis, :]
    kernel = np.transpose(kernel, (0,3,1,2))
    return kernel



################################## 2D-DCT Block ##################################
class DCT(nn.Module):
    def __init__(self,ksz):
        super(DCT, self).__init__()
        self.kernel_size = ksz
        in_kernel = initDCTKernel(self.kernel_size)
        in_kernel = torch.Tensor(in_kernel)
        self.in_kernel = nn.Parameter(in_kernel)
        self.in_kernel.requires_grad = False

    def forward(self, x):
        out = F.conv2d(input=x, weight=self.in_kernel, padding=self.kernel_size-1)
        return out
