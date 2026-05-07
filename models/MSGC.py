import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Adapter(nn.Module):

    def __init__(self, c, ffn_channel=None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                                   groups=c, bias=True, dilation=1)

    def forward(self, input):

        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)

        return x


class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)



class SpectralMLP(nn.Module):
    def __init__(self, c, expand=2):
        super().__init__()
        hidden = c * expand
        self.fc1 = nn.Linear(c, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        spec = x.mean(dim=(2, 3))          # [B, C] 光谱描述（对空间平均）
        spec = self.fc2(self.act(self.fc1(spec)))  # [B, C]
        spec = spec.view(B, C, 1, 1)
        # 不做 SE 那种纯乘权重，而是当作一个“谱偏移 / 变换”
        return x + spec

class MSGC(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
   #     self.spec_mlp = SpectralMLP(c, expand=2)

    def forward(self, inp):
        y = inp
     #   print('EBlock inp shape:', inp.shape)
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
      #  x = self.spec_mlp(y)
        x=y

        #        if self.use_adapters:
        #            return self.adapter(x)
        #        else:
        return x
