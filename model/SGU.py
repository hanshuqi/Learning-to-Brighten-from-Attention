import math
import torch.hub
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_

from .utils import RLN, WindowAttention, window_partition, window_reverse, Mlp

class Attention(nn.Module):
    def __init__(self, dim, num_heads, window_size, network_depth, use_attn):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size = window_size
        self.network_depth=network_depth
        self.use_attn = use_attn

        self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.V = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        if self.use_attn:
            self.QK = nn.Conv2d(dim, dim * 2, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)
            self.GU = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.LeakyReLU(0.2),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:  # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1 / 4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size2(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def check_size1(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (4 - h % 4) % 4
        mod_pad_w = (4 - w % 4) % 4
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X):
        B, C, H, W = X.shape
        V = self.V(X)
        if self.use_attn:
            convx = self.GU(X)

            QK = self.QK(X)
            QKV = torch.cat([QK, V], dim=1)

            # shift
            shifted_QKV = self.check_size2(QKV)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
            attn_windows = self.attn(qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            if H != shifted_out.shape[1]:
                shift_size = (self.window_size - H % self.window_size) % self.window_size
                shifted_out = shifted_out[:, shift_size:(shift_size + H), :, :]
            if W != shifted_out.shape[2]:
                shift_size = (self.window_size - W % self.window_size) % self.window_size
                shifted_out = shifted_out[:, :, shift_size:(shift_size + W), :]

            attn_out = shifted_out.permute(0, 3, 1, 2)
            conv_out = self.conv(V)
            out = [self.proj(conv_out + attn_out), convx]

        else:
            out = [self.proj(self.conv(V))]

        return out


class SGU_Block(nn.Module):
    def __init__(self, network_depth, window_size, embed_dim=192, num_heads=3,
                 mlp_ratio=0.0, use_attn=False, mlp_norm=False, norm_layer=RLN):
        super().__init__()

        #self.GU = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        #    nn.LeakyReLU(0.1), nn.Sigmoid())

        self.attn = Attention(embed_dim, num_heads, window_size=window_size,
                              network_depth=network_depth, use_attn=use_attn)
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        if self.use_attn:
            self.norm1 = norm_layer(embed_dim)
            if self.mlp_norm:
                self.norm2 = norm_layer(embed_dim)

        self.mlp = Mlp(network_depth, embed_dim, hidden_features=int(embed_dim * mlp_ratio))

    def forward(self, x):
        B_, C_, H, W = x.shape
        identity = x

        if self.use_attn:
            x, rescale, rebias = self.norm1(x)
        tmp = self.attn(x)
        if self.use_attn:
            x = tmp[0] * rescale + rebias + tmp[-1]
        else:
            x = tmp[-1]
        x = identity + x
        identity = x
        if self.use_attn and self.mlp_norm:
            x, rescale, rebias = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * rescale + rebias
        x = identity + x
        return x

class SGU_BasicLayer(nn.Module):
    def __init__(self, network_depth=3, window_size=8, embed_dim=16, num_heads=2, depth=3, mlp_ratio=2.,
                 attn_ratio=0.4, mlp_norm=False, norm_layer=RLN):
        super().__init__()
        self.dim = embed_dim
        self.depth = depth
        self.attn_ratio=attn_ratio
        attn_depth = attn_ratio * depth
        use_attns = [i >= depth - attn_depth for i in range(depth)]

        # build blocks
        self.blocks = nn.ModuleList([
            SGU_Block(network_depth, window_size, embed_dim=embed_dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, use_attn=use_attns[i], mlp_norm=mlp_norm,
                      norm_layer=norm_layer)
            for i in range(depth)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
