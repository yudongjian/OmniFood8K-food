# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
from modules.AFPN import AFPN, AFPN222
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from transformers.models.swin.modeling_swin import SwinModel
from model.convnext1 import convnext_small



class MultiFreqFusion(nn.Module):
    """
    高频+低频融合模块（简化版 FFT）
    - 分离高低频特征
    - 高频捕捉纹理细节
    - 低频捕捉全局结构
    - 最终通道拼接输出
    """
    def __init__(self, channels, freq_threshold=0.1):
        """
        channels: 输入特征通道数
        freq_threshold: 高频频率比例，取[0,1]之间
        """
        super().__init__()
        self.channels = channels
        self.freq_threshold = freq_threshold

        # 可选卷积压缩融合特征
        self.compress = nn.Sequential(
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels*2),
            nn.ReLU(inplace=True)
        )

    def forward(self, r, d):
        B, C, H, W = r.shape

        # 1. FFT
        fr = torch.fft.fft2(r, norm="ortho")
        fd = torch.fft.fft2(d, norm="ortho")

        # 2. 构建高低频掩码
        # 低频中心区域保留
        freq_h = torch.fft.fftfreq(H, device=r.device)
        freq_w = torch.fft.fftfreq(W, device=r.device)
        fy, fx = torch.meshgrid(freq_h, freq_w, indexing='ij')
        freq_radius = torch.sqrt(fx**2 + fy**2)
        mask_low = (freq_radius < self.freq_threshold).float()
        mask_high = 1.0 - mask_low
        mask_low = mask_low.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_high = mask_high.unsqueeze(0).unsqueeze(0)

        # 3. 高频和低频分离
        r_low = torch.fft.ifft2(fr * mask_low, norm="ortho").real
        r_high = torch.fft.ifft2(fr * mask_high, norm="ortho").real
        d_low = torch.fft.ifft2(fd * mask_low, norm="ortho").real
        d_high = torch.fft.ifft2(fd * mask_high, norm="ortho").real

        # 4. 高频+低频模态融合
        fused_high = r_high + d_high
        fused_low = r_low + d_low

        # 5. 通道拼接输出
        out = torch.cat([fused_high, fused_low], dim=1)
        out = self.compress(out)
        return out

class FusionNet_3Branch_UNet_FFT(nn.Module):

    def __init__(self):
        super().__init__()

        # ---- RGB encoder ----
        self.rgb_conv1 = nn.Conv2d(128, 96, 3, padding=1)
        self.rgb_conv2 = nn.Conv2d(256, 192, 3, padding=1)
        self.rgb_conv3 = nn.Conv2d(512, 384, 3, padding=1)
        self.rgb_conv4 = nn.Conv2d(1024, 768, 3, padding=1)

        # ---- Depth encoder ----
        self.depth_conv1 = nn.Conv2d(96, 96, 3, padding=1)   #没用
        self.depth_conv2 = nn.Conv2d(192, 192, 3, padding=1)  #没用
        self.depth_conv3 = nn.Conv2d(384, 384, 3, padding=1)
        self.depth_conv4 = nn.Conv2d(768, 768, 3, padding=1)
        self.resonant1 = MultiFreqFusion(channels=96)
        self.resonant2 = MultiFreqFusion(channels=192)
        self.resonant3 = MultiFreqFusion(channels=384)  # 没用
        self.resonant4 = MultiFreqFusion(channels=768)  # 没用

    def forward(self, rgb, depth):
        B = rgb[0].shape[0]

        # Layer 1
        r1 = F.relu(self.rgb_conv1(rgb[0]))
        cat1 = self.resonant1(r1, depth[0])      # 只对低层次 做了处理

        # Layer 2
        r2 = F.relu(self.rgb_conv2(rgb[1]))
        cat2 = self.resonant2(r2, depth[1])      # 只对低层次 做了处理

        # Layer 3
        r3 = F.relu(self.rgb_conv3(rgb[2]))
        d3 = F.relu(self.depth_conv3(depth[2]))
        cat3 = torch.cat([r3, d3], dim=1)

        # Layer 4
        r4 = F.relu(self.rgb_conv4(rgb[3]))
        d4 = F.relu(self.depth_conv4(depth[3]))
        cat4 = torch.cat([r4, d4], dim=1)


        return (cat1, cat2, cat3, cat4)


