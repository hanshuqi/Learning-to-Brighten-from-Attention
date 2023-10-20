import torch.nn as nn
import torch.nn.functional as F

from .SGU import SGU_BasicLayer
from .CA import CA_BasicLayer
from .utils import RLN, SmoothDilatedResidualBlock, ResidualBlock, SKFusion

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class UNet_base(nn.Module):
    def __init__(self, chs=(16, 32, 64, 32, 16), norm_layer=RLN, n_channels=3,
                 network_depth_sgu=[12, 12, 12], network_depth_ca=[4, 4, 4],
                 attn_ratio_sgu=[1/4, 1/2, 3/4], attn_ratio_ca=[1 / 3, 1 / 3, 1 / 3],
                 num_heads_sgu=[2, 4, 8], num_heads_ca=[1, 1, 1],
                 mlp_ratio_sgu=[2., 2., 2.], mlp_ratio_ca=[2., 2., 2.],
                 window_size_sgu=[8, 8, 8], window_size_ca=[[8, 4], [4, 8], [8, 4]],
                 mlp_norm_sgu=False, mlp_norm_ca=False):
        super(UNet_base, self).__init__()

        network_depth = sum(network_depth_sgu) + sum(network_depth_ca)
        self.CAT = nn.ModuleList()
        for idx in range(len(chs) // 2 + 1):
            blk = CA_BasicLayer(network_depth=network_depth, dim=chs[len(chs) // 2 + idx],
                                depth=network_depth_ca[idx], num_heads=num_heads_ca[idx],
                                mlp_ratio=mlp_ratio_ca[idx], norm_layer=norm_layer,
                                mlp_norm=mlp_norm_ca, window_size=window_size_ca[idx],
                                attn_ratio=attn_ratio_ca[idx])
            self.CAT.append(blk)

        self.GU = nn.ModuleList()
        for idx in range(len(chs) // 2 + 1):
            blk = SGU_BasicLayer(network_depth=network_depth, window_size=window_size_sgu[idx],
                                 embed_dim=chs[idx], num_heads=num_heads_sgu[idx],
                                 depth=network_depth_sgu[idx], mlp_ratio=mlp_ratio_sgu[idx],
                                 attn_ratio=attn_ratio_sgu[idx], mlp_norm=mlp_norm_sgu,
                                 norm_layer=norm_layer)
            self.GU.append(blk)

        self.patch_embed = nn.ModuleList([PatchEmbed(patch_size=1, in_chans=3, embed_dim=chs[0],
                                                     kernel_size=3) for _ in range(2)])
        self.down1 = nn.ModuleList([PatchEmbed(patch_size=2, in_chans=chs[0], embed_dim=chs[1])
            for _ in range(2)])
        self.down2 = nn.ModuleList([PatchEmbed(patch_size=2, in_chans=chs[1], embed_dim=chs[2])
            for _ in range(2)])
        self.patch_split1 = nn.ModuleList([PatchUnEmbed(patch_size=2, out_chans=chs[3],
            embed_dim=chs[2]) for _ in range(2)])
        self.patch_split2 = nn.ModuleList([PatchUnEmbed(patch_size=2, out_chans=chs[4],
            embed_dim=chs[3]) for _ in range(2)])
        self.sgu_fusion1 = nn.ModuleList([SKFusion(chs[0]) for _ in range(2)])
        self.sgu_fusion2 = nn.ModuleList([SKFusion(chs[1]) for _ in range(2)])
        self.sgu_fusion3 = nn.ModuleList([SKFusion(chs[2]) for _ in range(2)])
        self.fusion1 = nn.ModuleList([SKFusion(chs[3]) for _ in range(2)])
        self.fusion2 = nn.ModuleList([SKFusion(chs[4]) for _ in range(2)])
        self.final_fusion = SKFusion(n_channels)
        self.patch_unembed = nn.ModuleList([PatchUnEmbed(patch_size=1, out_chans=3,
            embed_dim=chs[4], kernel_size=3) for _ in range(2)])
        self.skip2 = nn.ModuleList([nn.Conv2d(chs[1], chs[1], 1) for _ in range(2)])
        self.skip1 = nn.ModuleList([nn.Conv2d(chs[0], chs[0], 1) for _ in range(2)])
        self.conv1 = nn.Conv2d(n_channels, chs[0], 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(chs[0], affine=True)
        self.act1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(chs[0], chs[0], 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(chs[0], affine=True)
        self.act2 = nn.LeakyReLU(0.1)
        self.res1 = SmoothDilatedResidualBlock(chs[0], dilation=2)
        self.res2 = SmoothDilatedResidualBlock(chs[0], dilation=2)
        self.res3 = SmoothDilatedResidualBlock(chs[0], dilation=2)
        self.res4 = SmoothDilatedResidualBlock(chs[0], dilation=4)
        self.res5 = SmoothDilatedResidualBlock(chs[0], dilation=4)
        self.res6 = SmoothDilatedResidualBlock(chs[0], dilation=4)
        self.res7 = ResidualBlock(chs[0], dilation=1)
        self.fusion_grad = SKFusion(chs[0], height=3)

    def check_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (4 - h % 4) % 4
        mod_pad_w = (4 - w % 4) % 4
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def gradient(self, x):
        _, _, H, W = x.size()
        y = self.act1(self.norm1(self.conv1(x)))
        y1 = self.act2(self.norm2(self.conv2(y)))

        y = self.res1(y1)
        y = self.res2(y)
        y = self.res3(y)
        y2 = self.res4(y)
        y = self.res5(y2)
        y = self.res6(y)
        y3 = self.res7(y)

        gates = self.fusion_grad([y1, y2, y3])
        xgrad = y1 * gates[:, [0], :, :] + y2 * gates[:, [1], :, :] + y3 * gates[:, [2], :, :]

        return xgrad

    def forward(self, m):
        _, _, H, W = m.shape
        m = self.check_size(m)
        s = self.gradient(x=m)

        m = self.patch_embed[0](m)
        #s = self.patch_embed[1](s)
        m_s = self.GU[0](m)
        m = self.sgu_fusion1[0]([m_s, m])
        s = self.sgu_fusion1[1]([m_s, s])
        skip1m = m
        skip1s = s
        m = self.down1[0](m)
        s = self.down1[1](s)

        m_s = self.GU[1](m)
        m = self.sgu_fusion2[0]([m_s, m])
        s = self.sgu_fusion2[1]([m_s, s])
        skip2m = m
        skip2s = s
        m = self.down2[0](m)
        s = self.down2[1](s)

        m_s = self.GU[2](m)
        m = self.sgu_fusion3[0]([m_s, m])
        s = self.sgu_fusion3[1]([m_s, s])

        m, s = self.CAT[0](m, s)
        m = self.patch_split1[0](m)
        s = self.patch_split1[1](s)
        m = self.fusion1[0]([m, self.skip2[0](skip2m)]) + m
        s = self.fusion1[1]([s, self.skip2[1](skip2s)]) + s

        m, s = self.CAT[1](m, s)
        m = self.patch_split2[0](m)
        s = self.patch_split2[1](s)
        m = self.fusion2[0]([m, self.skip1[0](skip1m)]) + m
        s = self.fusion2[1]([s, self.skip1[1](skip1s)]) + s

        m, s = self.CAT[2](m, s)
        m, s = self.patch_unembed[0](m), self.patch_unembed[1](s)
        out = self.final_fusion([m, s])

        if H != out.shape[2]:
            shift_size = (4 - H % 4) % 4
            out = out[:, :, shift_size:(shift_size + H), :]
        if W != out.shape[3]:
            shift_size = (4 - W % 4) % 4
            out = out[:, :, :, shift_size:(shift_size + W)]

        return m, s, out