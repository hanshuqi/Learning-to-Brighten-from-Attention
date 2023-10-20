import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import trunc_normal_

from .utils import RLN, WindowAttention, window_partition, window_reverse, Mlp

class CrossAttention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, use_attn=False):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads
        self.window_size_m, self.window_size_s = window_size
        self.network_depth = network_depth
        self.use_attn = use_attn

        self.convm = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.convs = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        self.V_main = nn.Conv2d(dim, dim, 1)
        self.V_structure = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK_m = nn.Conv2d(dim, dim * 2, 1)
            self.QK_s = nn.Conv2d(dim, dim * 2, 1)
            self.attn_m = WindowAttention(dim, self.window_size_m, num_heads)
            self.attn_s = WindowAttention(dim, self.window_size_s, num_heads)

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

    def check_size(self, x, mtag=True):
        _, _, h, w = x.size()
        if mtag:
            mod_pad_h = (self.window_size_m - h % self.window_size_m) % self.window_size_m
            mod_pad_w = (self.window_size_m - w % self.window_size_m) % self.window_size_m
        else:
            mod_pad_h = (self.window_size_s - h % self.window_size_s) % self.window_size_s
            mod_pad_w = (self.window_size_s - w % self.window_size_s) % self.window_size_s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def check_size1(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (4 - h % 4) % 4
        mod_pad_w = (4 - w % 4) % 4
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, X, Y):
        assert X.shape == Y.shape
        B, C, H, W = X.shape
        V_m = self.V_main(X)
        V_s = self.V_structure(Y)

        if self.use_attn:
            QK_m = self.QK_m(X)
            QK_s = self.QK_s(Y)
            QKV_query_main = torch.cat([QK_m, V_s], dim=1)
            QKV_query_structure = torch.cat([QK_s, V_m], dim=1)

            # shift
            shifted_QKV_querym = self.check_size(QKV_query_main, mtag=True)
            shifted_QKV_querys = self.check_size(QKV_query_structure, mtag=False)
            Htm, Wtm = shifted_QKV_querym.shape[2:]
            Hts, Wts = shifted_QKV_querys.shape[2:]

            # partition windows
            # cross attention: query from main stream(return to main stream)
            shifted_QKV_querym = shifted_QKV_querym.permute(0, 2, 3, 1)
            qkv_querym = window_partition(shifted_QKV_querym, self.window_size_m)  # nW*B, window_size**2, C
            attn_windows_querym = self.attn_m(qkv_querym)

            # cross attention: query from structure stream(return to structure stream)
            shifted_QKV_querys = shifted_QKV_querys.permute(0, 2, 3, 1)
            qkv_querys = window_partition(shifted_QKV_querys, self.window_size_s)  # nW*B, window_size**2, C
            attn_windows_querys = self.attn_s(qkv_querys)

            # merge windows
            shifted_out_m = window_reverse(attn_windows_querym, self.window_size_m, Htm, Wtm)  # B H' W' C
            shifted_out_s = window_reverse(attn_windows_querys, self.window_size_s, Hts, Wts)  # B H' W' C

            # reverse cyclic shift
            if H != shifted_out_m.shape[1]:
                shift_size = (self.window_size_m - H % self.window_size_m) % self.window_size_m
                shifted_out_m = shifted_out_m[:, shift_size:(shift_size + H), :, :]
            if W != shifted_out_m.shape[2]:
                shift_size = (self.window_size_m - W % self.window_size_m) % self.window_size_m
                shifted_out_m = shifted_out_m[:, :, shift_size:(shift_size + W), :]

            if H != shifted_out_s.shape[1]:
                shift_size = (self.window_size_s - H % self.window_size_s) % self.window_size_s
                shifted_out_s = shifted_out_s[:, shift_size:(shift_size + H), :, :]
            if W != shifted_out_s.shape[2]:
                shift_size = (self.window_size_s - W % self.window_size_s) % self.window_size_s
                shifted_out_s = shifted_out_s[:, :, shift_size:(shift_size + W), :]

            out_m = shifted_out_m
            out_s = shifted_out_s

            attn_out_m = out_m.permute(0, 3, 1, 2)
            attn_out_s = out_s.permute(0, 3, 1, 2)

            conv_out_m = self.convm(V_m)
            conv_out_s = self.convs(V_s)

            main = [self.proj(conv_out_m + attn_out_m)]
            structure = [self.proj(conv_out_s + attn_out_s)]

        else:
            main = [self.proj(self.convm(V_m))]
            structure = [self.proj(self.convs(V_s))]

        return main, structure


class CA_Block(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=RLN, mlp_norm=False,
                 window_size=[8, 8], use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.CA = CrossAttention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              use_attn=use_attn)

        if self.use_attn:
            self.norm1_m = norm_layer(dim) if use_attn else nn.Identity()
            self.norm1_s = norm_layer(dim) if use_attn else nn.Identity()
            if self.mlp_norm:
                self.norm2_m = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
                self.norm2_s = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()

        self.mlp_m = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_s = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, m, s):
        identity_m = m
        identity_s = s
        if self.use_attn:
            m, rescale_m, rebias_m = self.norm1_m(m)
            s, rescale_s, rebias_s = self.norm1_s(s)
        m, s = self.CA(m, s)
        if self.use_attn:
            m = m[0] * rescale_m + rebias_m
            s = s[0] * rescale_s + rebias_s
        else:
            m = m[0]
            s = s[0]
        m = identity_m + m
        s = identity_s + s

        identity_m = m
        identity_s = s
        if self.use_attn and self.mlp_norm:
            m, rescale_m, rebias_m = self.norm2_m(m)
            s, rescale_s, rebias_s = self.norm2_s(s)
        m = self.mlp_m(m)
        s = self.mlp_s(s)
        if self.use_attn and self.mlp_norm:
            m = m * rescale_m + rebias_m
            s = s * rescale_s + rebias_s
        main = identity_m + m
        structure = identity_s + s
        return main, structure


class CA_BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4., norm_layer=RLN,
                 mlp_norm=False, window_size=[8, 4], attn_ratio=0.):

        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        use_attns = [i >= depth - attn_depth for i in range(depth)]


        # build blocks
        self.blocks = nn.ModuleList([
            CA_Block(network_depth=network_depth, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                 norm_layer=norm_layer, mlp_norm=mlp_norm,
                 window_size=window_size, use_attn=use_attns[i])
            for i in range(depth)])

    def forward(self, m, s):
        for blk in self.blocks:
            m, s = blk(m, s)
        return m, s
