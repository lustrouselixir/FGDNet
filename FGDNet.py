import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import numpy as np
from utils import initDCTKernel,initIDCTKernel
    

def conv_3x3(in_channel, out_channel, stride=1, bias=False, padding=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias)


def conv_3x3_d(in_channel, out_channel, stride=1, bias=False, padding=2, dilation=2):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding, bias=bias, dilation=dilation)


################################## FGDNet - Noise Estimation Module ##################################
class Noise_Est(nn.Module):
    def __init__(self):
        super(Noise_Est, self).__init__()

        # Parameter
        C = 32

        # Layers
        self.conv1_1 = nn.Sequential(conv_3x3(1, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(conv_3x3_d(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(conv_3x3_d(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(conv_3x3_d(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(conv_3x3_d(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(conv_3x3(C, C), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
        self.conv1_16 = conv_3x3(C, 1)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2
    

################################## FGDNet - main ##################################   
class FGDNet(nn.Module):
    def __init__(self, C1=48, C2=96):
        super(FGDNet, self).__init__()

        # Parameters
        self.kernel_size = 7
        self.channelNum = self.kernel_size*self.kernel_size
        in_kernel = initDCTKernel(self.kernel_size)
        out_kernel = initIDCTKernel(self.kernel_size)
        in_kernel = torch.Tensor(in_kernel)
        out_kernel = torch.Tensor(out_kernel)
        self.in_kernel = nn.Parameter(in_kernel)
        self.out_kernel = nn.Parameter(out_kernel)
        self.in_kernel.requires_grad = False
        self.out_kernel.requires_grad = False
        self.maxpooling = nn.MaxPool2d(2) 
        self.upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        # Encoder Layers for the Noisy Target Image
        self.conv_1t = nn.Sequential(conv_3x3(self.channelNum, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2t = nn.Sequential(conv_3x3(C1, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_3t = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_4t = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_5t = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_6t = nn.Sequential(conv_3x3(C2, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))

        # Encoder Layers for the Guidance Image
        self.conv_1g = nn.Sequential(conv_3x3(self.channelNum, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2g = nn.Sequential(conv_3x3(C1, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_3g = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_4g = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_5g = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_6g = nn.Sequential(conv_3x3(C2, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))

        # Encoder Layers for the Estimated Noise Map
        self.conv_1n = nn.Sequential(conv_3x3(self.channelNum, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2n = nn.Sequential(conv_3x3(C1, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_3n = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_4n = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_5n = nn.Sequential(conv_3x3(C2, C2), nn.ReLU(inplace=True), nn.BatchNorm2d(C2))
        self.conv_6n = nn.Sequential(conv_3x3(C2, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))

        # Decoder Layers for W_Y
        self.conv_1dt = nn.Sequential(conv_3x3(C1*3, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2dt = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_3dt = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1)) 
        self.conv_4dt = nn.Sequential(conv_3x3(C1, self.channelNum), nn.ReLU(inplace=True), nn.BatchNorm2d(self.channelNum))
        self.conv_5dt = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Decoder Layers for W_G
        self.conv_1dg = nn.Sequential(conv_3x3(C1*3, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2dg = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_3dg = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1)) 
        self.conv_4dg = nn.Sequential(conv_3x3(C1, self.channelNum), nn.ReLU(inplace=True), nn.BatchNorm2d(self.channelNum))
        self.conv_5dg = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Decoder Layers for W_N
        self.conv_1dn = nn.Sequential(conv_3x3(C1*3, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_2dn = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1))
        self.conv_3dn = nn.Sequential(conv_3x3(C1, C1), nn.ReLU(inplace=True), nn.BatchNorm2d(C1)) 
        self.conv_4dn = nn.Sequential(conv_3x3(C1, self.channelNum), nn.ReLU(inplace=True), nn.BatchNorm2d(self.channelNum))
        self.conv_5dn = nn.Sequential(conv_3x3(self.channelNum, self.channelNum), nn.Tanh())

        # Noise Estimation Moduel
        model = Noise_Est()
        model = nn.DataParallel(model, device_ids=[0]).cuda()
        self.nEst = model     

    def forward(self, target, guidance):
        # Noise Estimation
        noise = target -  self.nEst(target)

        # Frequency Decomposition
        out_t0 = F.conv2d(input=target, weight=self.in_kernel, padding=self.kernel_size-1)
        out_g0 = F.conv2d(input=guidance, weight=self.in_kernel, padding=self.kernel_size-1)
        out_n0 = F.conv2d(input=noise, weight=self.in_kernel, padding=self.kernel_size-1)

        # Guided Denoising - Encoders
        out_t = self.maxpooling(out_t0)
        out_t = self.conv_1t(out_t)
        out_t = self.maxpooling(out_t)
        out_t = self.conv_2t(out_t)
        out_t = self.conv_3t(out_t)
        out_t = self.conv_4t(out_t)
        out_t = self.conv_5t(out_t)
        out_t = self.conv_6t(out_t)

        out_g = self.maxpooling(out_g0)  
        out_g = self.conv_1g(out_g)
        out_g = self.maxpooling(out_g)
        out_g = self.conv_2g(out_g)
        out_g = self.conv_3g(out_g)
        out_g = self.conv_4g(out_g)
        out_g = self.conv_5g(out_g)
        out_g = self.conv_6g(out_g)

        out_n = self.maxpooling(out_n0)
        out_n = self.conv_1n(out_n)
        out_n = self.maxpooling(out_n)
        out_n = self.conv_2n(out_n)
        out_n = self.conv_3n(out_n)
        out_n = self.conv_4n(out_n)
        out_n = self.conv_5n(out_n)
        out_n = self.conv_6n(out_n)
        
        out_tgn = torch.cat([out_t,out_g,out_n], dim=1)

        # Guided Denoising - Decoders
        out_tgn = self.upsample(out_tgn)

        weight_t = self.conv_1dt(out_tgn)
        weight_t = self.conv_2dt(weight_t)
        weight_t = self.conv_3dt(weight_t)
        weight_t = self.conv_4dt(weight_t)
        weight_t = self.upsample(weight_t)
        weight_t = self.conv_5dt(weight_t)

        weight_g = self.conv_1dg(out_tgn)
        weight_g = self.conv_2dg(weight_g)
        weight_g = self.conv_3dg(weight_g)
        weight_g = self.conv_4dg(weight_g)
        weight_g = self.upsample(weight_g)
        weight_g = self.conv_5dg(weight_g)
        
        weight_n = self.conv_1dn(out_tgn)
        weight_n = self.conv_2dn(weight_n)
        weight_n = self.conv_3dn(weight_n)
        weight_n = self.conv_4dn(weight_n)
        weight_n = self.upsample(weight_n)
        weight_n = self.conv_5dn(weight_n)

        # Linear Regression
        out_freq = out_t0 * weight_t + out_g0 * weight_g + out_n0 * weight_n

        # Spatial Reconstruction
        out = F.conv2d(input=out_freq, weight=self.out_kernel, padding=0)

        return out, out_freq, noise
