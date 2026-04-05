import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
class AttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super(CrossAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)

    def forward(self, x1, x2):
        b, c, h, w = x1.size()
        x1 = x1.view(b, c, -1).permute(0, 2, 1)
        x2 = x2.view(b, c, -1).permute(0, 2, 1)

        attn_output, _ = self.attn(x1, x2, x2)
        attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)

        return attn_output


class GatedFusion(nn.Module):
    def __init__(self, in_channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate = self.gate(x1 + x2)
        return gate * x1 + (1 - gate) * x2


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x



# 动态通道选择模块
class ChannelMask(nn.Module):
    def __init__(self, channels):
        super(ChannelMask, self).__init__()
        # 每个通道一个可学习权重，初始化为1
        self.mask = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        # 使用 sigmoid 将权重映射到 [0,1]
        return x * torch.sigmoid(self.mask)

class FeatureFusionNetwork222_Mask(nn.Module):
    def __init__(self, dropout=0.1):
        super(FeatureFusionNetwork222_Mask, self).__init__()

        # 特征提取
        self.feature1_conv = nn.Conv2d(192, 512, kernel_size=3, stride=1, padding=1)
        self.feature1_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature2_conv = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.feature2_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature3_conv = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.feature3_pool = nn.AdaptiveAvgPool2d((12, 12))
        self.feature4_conv = nn.Conv2d(768 * 2, 512, kernel_size=3, stride=1, padding=1)

        # 融合模块
        self.cross_attn1 = CrossAttentionFusion(512)
        self.gated_fusion1 = GatedFusion(512)
        self.cross_attn2 = CrossAttentionFusion(512)
        self.gated_fusion2 = GatedFusion(512)
        self.rgb_fusion = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.semantic_fusion = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # 动态通道选择
        self.rgb_mask = ChannelMask(512)
        self.semantic_mask = ChannelMask(512)
        self.fused_mask = ChannelMask(1024)

        # 注意力 & Transformer
        self.attention = AttentionFusion(1024)

        # 输出层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x1, x2, x3, x4):
        # 特征提取
        x1 = self.feature1_pool(self.feature1_conv(x1))
        x2 = self.feature2_pool(self.feature2_conv(x2))
        x3 = self.feature3_pool(self.feature3_conv(x3))
        x4 = self.feature4_conv(x4)

        # RGB 融合
        rgb_fused = self.cross_attn1(x1, x2)
        rgb_fused = self.gated_fusion1(rgb_fused, x2)
        rgb_fused = self.rgb_fusion(rgb_fused)
        rgb_fused = self.rgb_mask(rgb_fused)  # 动态通道选择

        # 语义融合
        semantic_fused = self.cross_attn2(x3, x4)
        semantic_fused = self.gated_fusion2(semantic_fused, x4)
        semantic_fused = self.semantic_fusion(semantic_fused)
        semantic_fused = self.semantic_mask(semantic_fused)  # 动态通道选择

        # 全局融合
        fused_features = torch.cat([rgb_fused, semantic_fused], dim=1)
        fused_features = self.attention(fused_features)
        fused_features = self.fused_mask(fused_features)  # 全局融合通道选择

        # 输出
        fused_features = self.global_pool(fused_features)
        fused_features = fused_features.view(fused_features.size(0), -1)
        fused_features = self.dropout(fused_features)
        output = self.fc(fused_features)

        return output



# 测试代码（输入模拟数据）
if __name__ == "__main__":
    # 假设batch_size为4，特征的大小根据给定的要求
    batch_size = 4
    feature1 = torch.randn(batch_size, 192, 96, 96)  # RGB图像特征
    feature2 = torch.randn(batch_size, 384, 48, 48)  # RGB-D图像特征
    feature3 = torch.randn(batch_size, 768, 24, 24)  # CLIP提取的特征
    feature4 = torch.randn(batch_size, 768*2, 12, 12)  # DINO提取的特征

    # 创建网络
    model = FeatureFusionNetwork()

    # 获取输出
    output = model(feature1, feature2, feature3, feature4)

    # 输出结果
    print("输出结果：", output)
    print("输出形状：", output.shape)  # 应该是 [batch_size, 1]
