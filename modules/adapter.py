import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAdapterV4(nn.Module):
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        # ---- Step 1: 全局尺度/偏移参数 ----
        self.alpha = nn.Parameter(torch.ones(1))   # scale
        self.beta = nn.Parameter(torch.zeros(1))   # shift

        # ---- Step 2: 局部残差修正 CNN ----
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1, 3, padding=1)   # 输出残差
        )

    def forward(self, d_mono):
        # 1. 全局校正
        d_global = self.alpha * d_mono + self.beta

        # 2. 残差修正
        d_res = self.refine(d_global)

        # 3. 输出修正后的深度
        return d_global + d_res


if __name__ == "__main__":
    # 假设有 batch_size=4，单通道深度图，大小 128x128
    dummy_depth = torch.randn(4, 3, 384, 384)

    model = DepthAdapterV2(in_ch=3, base_ch=32)

    out = model(dummy_depth)

    print("输入深度图形状:", dummy_depth.shape)
    print("输出适配后深度图形状:", out.shape)


