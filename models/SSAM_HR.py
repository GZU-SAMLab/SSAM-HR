import os
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, to_2tuple
import torch.nn.functional as F

import torch

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
from .FBF import FBF

from .MSGC import MSGC
MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=8):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

class TIM(nn.Module):
    def __init__(self, channels_in):
        super(TIM, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in)
        self.se_depth = SqueezeAndExcitation(channels_in)
        self.alpha = nn.Parameter(torch.zeros(1, channels_in, 1, 1))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels_in, channels_in, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        # out = rgb + depth
        alpha = torch.sigmoid(self.alpha)  # [1, C, 1, 1]

        out = alpha * rgb + (1 - alpha) * depth
        out = self.fuse(out)

        # print(self.alpha)
        return out, rgb, depth
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
class FFN(nn.Module):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_dim, mid_dim=None,
                 out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = Conv2d_BN(in_dim, mid_dim, 1)
        self.fc2 = Conv2d_BN(mid_dim, out_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class LocalBlock(nn.Module):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(self, dim, hidden_dim=64, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        input = x
        x = self.dwconv(x)
        x = self.mlp(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        x = rearrange(x, 'b c h w -> b h w c')
        return x

class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv
from .SGDAMamba import SGDAMamba
class SSAM_HR(nn.Module):
    def __init__(
            self,
            embed_dim=None,
            num_classes: int = None,
            drop_rate=0.5,
            fc_drop_rate=0.,
            # checkpoint
            use_checkpoint=False,
            conv3D_channel: int = 32,
            conv3D_kernel_1: int = (5,5,5),

            dim_linear_1: int = None,
            **kwargs,
        ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models


        self.conv3d_features_1 = nn.Sequential(
            nn.Conv3d(1, out_channels=conv3D_channel, kernel_size=conv3D_kernel_1),
            nn.BatchNorm3d(conv3D_channel),
            nn.SiLU(),
        )

        self.embedding_spatial_1 = nn.Sequential(nn.Linear(conv3D_channel * dim_linear_1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()






        self.fusion = TIM(embed_dim)
        self.local=LocalBlock(dim=embed_dim, hidden_dim=int(2 * embed_dim))
        self.fbf=FBF(embed_dim, lpf="pool", gaussian_ksize=5, gaussian_sigma=1.0, alpha_init=0.3)
        self.localconv=RepDW(embed_dim)
        self.msgc =MSGC(embed_dim,extra_depth_wise = True)

        self.sgdamamba =SGDAMamba(
                hidden_dim=embed_dim,
                drop_path=0.1,
                norm_layer=nn.LayerNorm,
                ssm_d_state=16,
                ssm_rank_ratio=2.0,
                ssm_dt_rank="auto",
                ssm_act_layer=nn.SiLU,
                ssm_conv=3,
                ssm_conv_bias=True,
                ssm_drop_rate=0.0,
                ssm_simple_init=False,
                forward_type="v2",
                mlp_ratio=4.0,
                mlp_act_layer=nn.GELU,
                mlp_drop_rate=0.0,
                use_checkpoint=False,
                stage=0
            )
    def scan(self, x, scan_type=None, group_type=None):
        x = rearrange(x, 'b c t h w -> b (c t) h w')  # [10, 896, 8, 8]
        x = rearrange(x, 'b c h w -> b h w c')  # [64, 896, 8, 8]-> [10, 8, 8, 896]
        return x

    def mask_generate(self, img_size, num):
        out = []
        mask_none = torch.ones(img_size, img_size)
        for i in reversed(range(0, num - 1)):
            mask = torch.ones(img_size // (2 ** i), img_size // (2 ** i))
            mask = torch.triu(mask, diagonal=0)
            mask = torch.rot90(mask, k=1, dims=(0, 1))
            out.append(
                torch.nn.functional.pad(mask, (0, img_size - img_size // (2 ** i), 0, img_size - img_size // (2 ** i)),
                                        mode='constant', value=0))
        out.append(mask_none)
        return out

    def forward_features(self, x, inference_params=None):
        x_1 = self.conv3d_features_1(x)
        x_1 = self.scan(x_1)
        x_1 = self.embedding_spatial_1(x_1)
        x_1 = self.pos_drop(x_1)
        x_2=x_1
        x_1 = rearrange(x_1, 'b h w c -> b c h w')
        x_1 =self.msgc(x_1)
        x_1=self.localconv(x_1)
        x_1 =self.fbf(x_1)
        x_1=self.localconv(x_1)

        B3,H3,W3,C3=x_2.shape


        x_2 =self.local(x_2)
        x_2=self.sgdamamba(x_2)#B3,H3*W3,C3

        x_2=x_2.reshape(B3,H3,W3,C3).permute(0, 3, 1, 2)

        x,_,_= self.fusion(x_1, x_2)
        x = rearrange(x, 'b c h w -> b c (h w) ')  # [64, 32, 49]
        feature = x.mean(dim=2)  # [64, 32]
        return feature ##[10, 64, 96]->[10, 96]

    def forward(self, x, inference_params=None):
        feature = self.forward_features(x, inference_params)  ##[10, 192]
        x = self.head(self.head_drop(feature))   #[10, 9]
        return x, feature



