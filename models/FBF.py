import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseGaussianBlur(nn.Module):
    """深度可分离高斯平滑（每个通道独立卷积）"""
    def __init__(self, channels: int, ksize: int = 5, sigma: float = 1.0):
        super().__init__()
        assert ksize % 2 == 1, "ksize 应为奇数"
        ar = torch.arange(ksize) - (ksize // 2)
        yy, xx = torch.meshgrid(ar, ar, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        # (1,1,k,k)；前向时按通道扩展
        self.register_buffer("kernel", kernel[None, None, :, :])
        self.channels = channels
        self.ksize = ksize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        k = k.expand(self.channels, 1, self.ksize, self.ksize)  # (C,1,k,k)
        pad = self.ksize // 2
        return F.conv2d(x, k, padding=pad, groups=self.channels)
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
class FBF(nn.Module):
    """
    高频增强：y = x + alpha * (x - LPF(x))
    - lpf='pool'：使用平均池化+上采样作为低通
    - lpf='gaussian'：使用深度可分离高斯模糊作为低通
    - alpha 可学习，范围 (0, alpha_max)，可按通道设置
    - 可选 1x1+BN 对残差做尺度校准，提升稳定性
    """
    def __init__(
        self,
        channels: int,
        lpf: str = "gaussian",             # 'pool' 或 'gaussian'
        pool_scale: int = 2,            # 低通降采样倍数（pool 模式）
        interp: str = "nearest",        # 上采样方式
        gaussian_ksize: int = 3,        # 高斯核大小（gaussian 模式）
        gaussian_sigma: float = 1.0,    # 高斯 σ（gaussian 模式）
        alpha_init: float = 0.2,        # 初始增强强度
        alpha_max: float = 1.0,         # 强度上限
        per_channel_alpha: bool = True, # 是否通道独立 α
        use_residual_calib: bool = True # 是否用 1x1+BN 标定残差幅值
    ):
        super().__init__()
        self.lpf = lpf
        self.pool_scale = max(1, int(pool_scale))
        self.interp = interp
        self.alpha_max = float(alpha_max)

        if lpf == "gaussian":
            self.gaussian = DepthwiseGaussianBlur(channels, gaussian_ksize, gaussian_sigma)
        elif lpf == "pool":
            self.gaussian = None
        else:
            raise ValueError("lpf 只能是 'pool' 或 'gaussian'")

        # α 用 sigmoid 映射到 (0, alpha_max)
        logit = math.log(alpha_init / (alpha_max - alpha_init + 1e-8)) if alpha_init > 0 else -10.0
        shape = (1, channels, 1, 1) if per_channel_alpha else (1, 1, 1, 1)
        self.alpha_logit = nn.Parameter(torch.full(shape, logit))

        # 残差标定（1×1 + BN），避免把噪声/伪影放大过头
        if use_residual_calib:
            self.calib = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels)
            )
        else:
            self.calib = nn.Identity()

        self.norm = LayerNorm2d(channels)
        self.freq = FreMLP(nc=channels, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def lowpass(self, x: torch.Tensor) -> torch.Tensor:
        if self.lpf == "gaussian":
            return self.gaussian(x)
        # pool 低通：先降采样再上采样回原尺寸
        if self.pool_scale == 1:
            return x
        B, C, H, W = x.shape
        x_ds = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        x_up = F.interpolate(
            x_ds, size=(H, W), mode=self.interp,
            align_corners=False if self.interp in {"bilinear", "bicubic"} else None
        )
        return x_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 低通
        y=x
        x_step2 = self.norm(y)  # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2)  # size [B, C, H, W]
        x = y * x_freq
        x = y + x * self.gamma

        x_low = self.lowpass(x)
        # 高频残差
        r = x - x_low
        r = self.calib(r)
        # α∈(0, alpha_max)
        alpha = self.alpha_max * torch.sigmoid(self.alpha_logit)
        # 加回
        y = x + alpha * r
        return y
