from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmdet.registry import MODELS


def BasicConv(filter_in, filter_out, kernel_size, stride=1, pad=None):
    if not pad:
        pad = (kernel_size - 1) // 2 if kernel_size else 0
    else:
        pad = pad
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU(inplace=True)),
    ]))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, filter_in, filter_out):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_out, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filter_out, filter_out, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(filter_out, momentum=0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

        # carafe
        # from mmcv.ops import CARAFEPack
        # self.upsample = nn.Sequential(
        #     BasicConv(in_channels, out_channels, 1),
        #     CARAFEPack(out_channels, scale_factor=scale_factor)
        # )

    def forward(self, x):
        x = self.upsample(x)

        return x


class Downsample_x2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x2, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 2, 2, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x4, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 4, 4, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x8(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x8, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 8, 8, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class Downsample_x16(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample_x16, self).__init__()

        self.downsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 16, 16, 0)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x


class ASFF_2(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_2, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2):
        level_1_weight_v = self.weight_level_1(input1)  # [8,*,96,96] [8,8,-,-]
        level_2_weight_v = self.weight_level_2(input2)  # [8,*,96,96] [8,8,-,-]

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)  # [8,16,96,96],[8,16,48,48]
        levels_weight = self.weight_levels(levels_weight_v)     # [8,8,*,*],[8,2,*,*]
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]
        #[8,32,96,96][8,64,48,48]
        out = self.conv(fused_out_reduced)

        return out #[8,32,96,96]


class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input1, input2, input3):
        level_1_weight_v = self.weight_level_1(input1) #[8,8,96,96][8,8,48,48][8,8,24,24]
        level_2_weight_v = self.weight_level_2(input2) #[8,8,96,96][8,8,48,48][8,8,24,24]
        level_3_weight_v = self.weight_level_3(input3) #[8,8,96,96][8,8,48,48][8,8,24,24]

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1) #[8,24,96,96][8,24,48,48][8,24,24,24]
        levels_weight = self.weight_levels(levels_weight_v) #[8,3,96,96][8,3,48,48][8,3,24,24]
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]

        out = self.conv(fused_out_reduced) #[8,32,96,96][8,64,48,48][8,128,24,24]

        return out


class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3):
        level_0_weight_v = self.weight_level_0(input0) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_1_weight_v = self.weight_level_1(input1) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_2_weight_v = self.weight_level_2(input2) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_3_weight_v = self.weight_level_3(input3) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1) #[8,32,96,96],[8,32,48,48],[8,32,24,24],[8,32,12,12]
        levels_weight = self.weight_levels(levels_weight_v) #[8,4,96,96],[8,4,48,48],,[8,8,24,24],[8,8,12,12]
        levels_weight = F.softmax(levels_weight, dim=1)  # 权重总和是1

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        out = self.conv(fused_out_reduced) #[8,32,96,96],[8,64,48,48],[8,128,24,24],[8,256,12,12]

        return out


class ASFF_5(nn.Module):
    def __init__(self, inter_dim=512):
        super(ASFF_5, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = BasicConv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_4 = BasicConv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 5, 5, kernel_size=1, stride=1, padding=0)

        self.conv = BasicConv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, input0, input1, input2, input3, input4):
        level_0_weight_v = self.weight_level_0(input0) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_1_weight_v = self.weight_level_1(input1) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_2_weight_v = self.weight_level_2(input2) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_3_weight_v = self.weight_level_3(input3) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]
        level_4_weight_v = self.weight_level_4(input4) #[8,8,96,96],[8,8,48,48],[8,8,24,24],[8,8,12,12]

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v, level_4_weight_v), 1) #[8,32,96,96],[8,32,48,48],[8,32,24,24],[8,32,12,12]
        levels_weight = self.weight_levels(levels_weight_v) #[8,4,96,96],[8,4,48,48],,[8,8,24,24],[8,8,12,12]
        levels_weight = F.softmax(levels_weight, dim=1)  # 权重总和是1

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:4, :, :] + \
                            input4 * levels_weight[:, 4:, :, :]

        out = self.conv(fused_out_reduced) #[8,32,96,96],[8,64,48,48],[8,128,24,24],[8,256,12,12]

        return out


class BlockBody(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512]):
        super(BlockBody, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )

        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]), #3x3的CBR
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),#3x3的卷积
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )

        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])
        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])
        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])
        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)

        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

    def forward(self, x):
        x0, x1, x2, x3 = x

        # 普通的1*1 卷积
        x0 = self.blocks_scalezero1(x0)     # 1x1的卷积 [8,32,96,96]
        x1 = self.blocks_scaleone1(x1)      # [8,64,48,48]
        x2 = self.blocks_scaletwo1(x2)      # [8,128,24,24]
        x3 = self.blocks_scalethree1(x3)    # [8,256,12,12]

        #   第一个融合的模块  ①首先降低x1的通道数量，然后提升x1的宽度 长度  ② 提升x0 的通道，融合x1。
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))     # 将 x1进行上采样，与x0进行连接，然后过一个ASFF   #[8,32,96,96]
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)    # 将x0进行下采样，然后与x1进行连接，然后过一个asff #[8,64,48,48]

        x0 = self.blocks_scalezero2(scalezero)  # 3x3的卷积  #[8,32,96,96]
        x1 = self.blocks_scaleone2(scaleone)     # [8,64,48,48]
        #   第二个融合的模块
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))#x0层：将x1,x2进行上采样与x0进行连接  #[8,32,96,96]
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))#x1层：将x0层进行上采样，x1不变，x2进行下采样，然后进行ASFF #[8,64,48,48]
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)#x2层：将x0,x1进行下采样，x2不变，然后经过asff  #[8,128,24,24]

        x0 = self.blocks_scalezero3(scalezero)  # 3x3的卷积 #[8,32,96,96]
        x1 = self.blocks_scaleone3(scaleone)    # [8,64,48,48]
        x2 = self.blocks_scaletwo3(scaletwo)
        #   第三个融合的模块
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))#x0层：将x1,x2,x3进行上采样,x0不变然后过ASFF  #[8,32,96,96]
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))#x1层：将x0下采样,x2,x3进行上采样过asff      #[8,64,48,48]
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))#x2层：将x0,x1进行下采样，x2不变，x3上采样过ASFF #[8,128,24,24]
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)#x3层：将x1,x2,X3进行上采样过ASFF  #[8,256,12,12]

        scalezero = self.blocks_scalezero4(scalezero)   # 3x3的卷积 #[8,32,96,96]  #wavemlp时：[4,16,96,96]
        scaleone = self.blocks_scaleone4(scaleone)  # [8,64,48,48]              #[4,32,48,48]
        scaletwo = self.blocks_scaletwo4(scaletwo)  # [8,128,24,24]             #[4,80,24,24]
        scalethree = self.blocks_scalethree4(scalethree)    # [8,256,12,12]        #[4,128,12,12]

        return scalezero, scaleone, scaletwo, scalethree

class BlockBody222(nn.Module):
    def __init__(self, channels=[32, 64, 128, 256, 512]):
        super(BlockBody222, self).__init__()

        self.blocks_scalezero1 = nn.Sequential(
            BasicConv(channels[0], channels[0], 1),
        )
        self.blocks_scaleone1 = nn.Sequential(
            BasicConv(channels[1], channels[1], 1),
        )
        self.blocks_scaletwo1 = nn.Sequential(
            BasicConv(channels[2], channels[2], 1),
        )
        self.blocks_scalethree1 = nn.Sequential(
            BasicConv(channels[3], channels[3], 1),
        )
        self.blocks_scalefour1 = nn.Sequential(
            BasicConv(channels[4], channels[4], 1),
        )


        self.downsample_scalezero1_2 = Downsample_x2(channels[0], channels[1])
        self.upsample_scaleone1_2 = Upsample(channels[1], channels[0], scale_factor=2)

        self.asff_scalezero1 = ASFF_2(inter_dim=channels[0])
        self.asff_scaleone1 = ASFF_2(inter_dim=channels[1])

        self.blocks_scalezero2 = nn.Sequential(
            BasicBlock(channels[0], channels[0]), #3x3的CBR
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone2 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),#3x3的卷积
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )

        self.downsample_scalezero2_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero2_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scaleone2_2 = Downsample_x2(channels[1], channels[2])
        self.upsample_scaleone2_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.upsample_scaletwo2_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.upsample_scaletwo2_4 = Upsample(channels[2], channels[0], scale_factor=4)

        self.asff_scalezero2 = ASFF_3(inter_dim=channels[0])
        self.asff_scaleone2 = ASFF_3(inter_dim=channels[1])
        self.asff_scaletwo2 = ASFF_3(inter_dim=channels[2])

        self.blocks_scalezero3 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone3 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo3 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        # -----------第三层
        self.downsample_scalezero3_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero3_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero3_8 = Downsample_x8(channels[0], channels[3])

        self.upsample_scaleone3_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone3_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone3_4 = Downsample_x4(channels[1], channels[3])

        self.upsample_scaletwo3_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo3_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo3_2 = Downsample_x2(channels[2], channels[3])

        self.upsample_scalethree3_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree3_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree3_2 = Upsample(channels[3], channels[2], scale_factor=2)


        self.asff_scalezero3 = ASFF_4(inter_dim=channels[0])
        self.asff_scaleone3 = ASFF_4(inter_dim=channels[1])
        self.asff_scaletwo3 = ASFF_4(inter_dim=channels[2])
        self.asff_scalethree3 = ASFF_4(inter_dim=channels[3])

        self.blocks_scalezero4 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone4 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo4 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree4 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )

        #  -------------第四层
        self.asff_scalezero4 = ASFF_5(inter_dim=channels[0])
        self.asff_scaleone4 = ASFF_5(inter_dim=channels[1])
        self.asff_scaletwo4 = ASFF_5(inter_dim=channels[2])
        self.asff_scalethree4 = ASFF_5(inter_dim=channels[3])
        self.asff_scalefour4 = ASFF_5(inter_dim=channels[4])

        self.downsample_scalezero4_2 = Downsample_x2(channels[0], channels[1])
        self.downsample_scalezero4_4 = Downsample_x4(channels[0], channels[2])
        self.downsample_scalezero4_8 = Downsample_x8(channels[0], channels[3])
        self.downsample_scalezero4_16 = Downsample_x16(channels[0], channels[4])

        self.upsample_scaleone4_2 = Upsample(channels[1], channels[0], scale_factor=2)
        self.downsample_scaleone4_2 = Downsample_x2(channels[1], channels[2])
        self.downsample_scaleone4_4 = Downsample_x4(channels[1], channels[3])
        self.downsample_scaleone4_8 = Downsample_x8(channels[1], channels[4])

        self.upsample_scaletwo4_4 = Upsample(channels[2], channels[0], scale_factor=4)
        self.upsample_scaletwo4_2 = Upsample(channels[2], channels[1], scale_factor=2)
        self.downsample_scaletwo4_2 = Downsample_x2(channels[2], channels[3])
        self.downsample_scaletwo4_4 = Downsample_x4(channels[2], channels[4])

        self.upsample_scalethree4_8 = Upsample(channels[3], channels[0], scale_factor=8)
        self.upsample_scalethree4_4 = Upsample(channels[3], channels[1], scale_factor=4)
        self.upsample_scalethree4_2 = Upsample(channels[3], channels[2], scale_factor=2)
        self.downsample_scalethree4_2 = Downsample_x2(channels[3], channels[4])

        self.upsample_scale_four4_16 = Upsample(channels[4], channels[0], scale_factor=16)
        self.upsample_scale_four4_8 = Upsample(channels[4], channels[1], scale_factor=8)
        self.upsample_scale_four4_4 = Upsample(channels[4], channels[2], scale_factor=4)
        self.upsample_scale_four4_2 = Upsample(channels[4], channels[3], scale_factor=2)

        self.blocks_scalezero5 = nn.Sequential(
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
            BasicBlock(channels[0], channels[0]),
        )
        self.blocks_scaleone5 = nn.Sequential(
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
            BasicBlock(channels[1], channels[1]),
        )
        self.blocks_scaletwo5 = nn.Sequential(
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
            BasicBlock(channels[2], channels[2]),
        )
        self.blocks_scalethree5 = nn.Sequential(
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
            BasicBlock(channels[3], channels[3]),
        )
        self.blocks_scalefour5 = nn.Sequential(
            BasicBlock(channels[4], channels[4]),
            BasicBlock(channels[4], channels[4]),
            BasicBlock(channels[4], channels[4]),
            BasicBlock(channels[4], channels[4]),
        )

    def forward(self, x):
        x0, x1, x2, x3, x4 = x

        # 普通的1*1 卷积
        x0 = self.blocks_scalezero1(x0)     # 1x1的卷积 [8,32,96,96]
        x1 = self.blocks_scaleone1(x1)      # [8,64,48,48]
        x2 = self.blocks_scaletwo1(x2)      # [8,128,24,24]
        x3 = self.blocks_scalethree1(x3)    # [8,256,12,12]
        x4 = self.blocks_scalefour1(x4)    # [8,256,12,12]

        #   第一个融合的模块  ①首先降低x1的通道数量，然后提升x1的宽度 长度  ② 提升x0 的通道，融合x1。
        scalezero = self.asff_scalezero1(x0, self.upsample_scaleone1_2(x1))     # 将 x1进行上采样，与x0进行连接，然后过一个ASFF   #[8,32,96,96]
        scaleone = self.asff_scaleone1(self.downsample_scalezero1_2(x0), x1)    # 将x0进行下采样，然后与x1进行连接，然后过一个asff #[8,64,48,48]

        x0 = self.blocks_scalezero2(scalezero)  # 3x3的卷积  #[8,32,96,96]
        x1 = self.blocks_scaleone2(scaleone)     # [8,64,48,48]

        #   第二个融合的模块
        scalezero = self.asff_scalezero2(x0, self.upsample_scaleone2_2(x1), self.upsample_scaletwo2_4(x2))#x0层：将x1,x2进行上采样与x0进行连接  #[8,32,96,96]
        scaleone = self.asff_scaleone2(self.downsample_scalezero2_2(x0), x1, self.upsample_scaletwo2_2(x2))#x1层：将x0层进行上采样，x1不变，x2进行下采样，然后进行ASFF #[8,64,48,48]
        scaletwo = self.asff_scaletwo2(self.downsample_scalezero2_4(x0), self.downsample_scaleone2_2(x1), x2)#x2层：将x0,x1进行下采样，x2不变，然后经过asff  #[8,128,24,24]

        x0 = self.blocks_scalezero3(scalezero)#3x3的卷积 #[8,32,96,96]
        x1 = self.blocks_scaleone3(scaleone) #[8,64,48,48]
        x2 = self.blocks_scaletwo3(scaletwo)

        #   第三个融合的模块
        scalezero = self.asff_scalezero3(x0, self.upsample_scaleone3_2(x1), self.upsample_scaletwo3_4(x2), self.upsample_scalethree3_8(x3))#x0层：将x1,x2,x3进行上采样,x0不变然后过ASFF  #[8,32,96,96]
        scaleone = self.asff_scaleone3(self.downsample_scalezero3_2(x0), x1, self.upsample_scaletwo3_2(x2), self.upsample_scalethree3_4(x3))#x1层：将x0下采样,x2,x3进行上采样过asff      #[8,64,48,48]
        scaletwo = self.asff_scaletwo3(self.downsample_scalezero3_4(x0), self.downsample_scaleone3_2(x1), x2, self.upsample_scalethree3_2(x3))#x2层：将x0,x1进行下采样，x2不变，x3上采样过ASFF #[8,128,24,24]
        scalethree = self.asff_scalethree3(self.downsample_scalezero3_8(x0), self.downsample_scaleone3_4(x1), self.downsample_scaletwo3_2(x2), x3)#x3层：将x1,x2,X3进行上采样过ASFF  #[8,256,12,12]

        x0 = self.blocks_scalezero4(scalezero)#3x3的卷积 #[8,32,96,96]  #wavemlp时：[4,16,96,96]
        x1 = self.blocks_scaleone4(scaleone) #[8,64,48,48]              #[4,32,48,48]
        x2 = self.blocks_scaletwo4(scaletwo) #[8,128,24,24]             #[4,80,24,24]
        x3 = self.blocks_scalethree4(scalethree)#[8,256,12,12]        #[4,128,12,12]

        #   第4个融合的模块
        scalezero = self.asff_scalezero4(x0, self.upsample_scaleone4_2(x1), self.upsample_scaletwo4_4(x2), self.upsample_scalethree4_8(x3), self.upsample_scale_four4_16(x4))#x0层：将x1,x2,x3进行上采样,x0不变然后过ASFF  #[8,32,96,96]
        scaleone = self.asff_scaleone4(self.downsample_scalezero4_2(x0), x1, self.upsample_scaletwo4_2(x2), self.upsample_scalethree4_4(x3), self.upsample_scale_four4_8(x4))#x1层：将x0下采样,x2,x3进行上采样过asff      #[8,64,48,48]

        scaletwo = self.asff_scaletwo4(self.downsample_scalezero4_4(x0), self.downsample_scaleone4_2(x1), x2, self.upsample_scalethree4_2(x3), self.upsample_scale_four4_4(x4))#x2层：将x0,x1进行下采样，x2不变，x3上采样过ASFF #[8,128,24,24]
        scalethree = self.asff_scalethree4(self.downsample_scalezero4_8(x0), self.downsample_scaleone4_4(x1), self.downsample_scaletwo4_2(x2), x3, self.upsample_scale_four4_2(x4))#x3层：将x1,x2,X3进行上采样过ASFF  #[8,256,12,12]

        scalefour = self.asff_scalefour4(self.downsample_scalezero4_16(x0), self.downsample_scaleone4_8(x1), self.downsample_scaletwo4_4(x2), self.downsample_scalethree4_2(x3), x4)#x3层：将x1,x2,X3进行上采样过ASFF  #[8,256,12,12]

        scalezero = self.blocks_scalezero5(scalezero)#3x3的卷积 #[8,32,96,96]  #wavemlp时：[4,16,96,96]
        scaleone = self.blocks_scaleone5(scaleone) #[8,64,48,48]              #[4,32,48,48]
        scaletwo = self.blocks_scaletwo5(scaletwo) #[8,128,24,24]             #[4,80,24,24]
        scalethree = self.blocks_scalethree5(scalethree)#[8,256,12,12]        #[4,128,12,12]
        scalefour = self.blocks_scalefour5(scalefour)#[8,256,12,12]        #[4,128,12,12]


        return scalezero, scaleone, scaletwo, scalethree, scalefour


# @MODELS.register_module()
class AFPN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256):
        super(AFPN, self).__init__()

        self.fp16_enabled = False

        # 下采样 8个维度
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1) #[32]  1x1的卷积进行降维 #???????????为什么除8？
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1) #[64]
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1) #[128]
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1) #[256]

        self.body = nn.Sequential(
            BlockBody([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = nn.MaxPool2d(kernel_size=1, stride=2)
        #########此处是添加一个上采样参数
        # self.upsample1=nn.Upsample(256,scale_factor=2)
        # self.upsample2=nn.Upsample(256,scale_factor=4)
        # self.upsample3=nn.Upsample(256,scale_factor=8)
        self.conv=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3 = x
        x0, x1, x2, x3 = x

        # 此处将通道数降为1\8  缩小8倍
        x0 = self.conv0(x0) #[8,32,96,96]
        x1 = self.conv1(x1) #[8,64,48,48]
        x2 = self.conv2(x2) #[8,128,24,24]
        x3 = self.conv3(x3) #[8,256,12,12]

        out0, out1, out2, out3 = self.body([x0, x1, x2, x3])  # 经过多层次的融合得到四个输出

        #   放大到 输出的维度
        out0 = self.conv00(out0)    # [8,256,96,96]       #wavemlp:[4,128,96,96]
        out1 = self.conv11(out1)    # [8,256,48,48]       #[4,128,48,48]
        out2 = self.conv22(out2)    # [8,256,24,24]       #[4,128,24,24]
        out3 = self.conv33(out3)    # [8,256,12,12]       #[4,128,12,12]

###########################此处添加最后的部分，将高层次的特征添加到低层中去，增强低层特征的表达能力。是否可以提效果？？？
        # out4 = self.conv44(out3) #[8,256,6,6]??????????????????????????????

        # out2 = out2 + self.conv(F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=False))
        # out1 = out1 + self.conv(F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False))
        # out0 = out0 + self.conv(F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False))


        return out0, out1, out2, out3

class AFPN222(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024, 2048, 4096],
                 out_channels=256):
        super(AFPN222, self).__init__()

        self.fp16_enabled = False

        # 下采样 8个维度
        self.conv0 = BasicConv(in_channels[0], in_channels[0] // 8, 1) #[32]  1x1的卷积进行降维 #???????????为什么除8？
        self.conv1 = BasicConv(in_channels[1], in_channels[1] // 8, 1) #[64]
        self.conv2 = BasicConv(in_channels[2], in_channels[2] // 8, 1) #[128]
        self.conv3 = BasicConv(in_channels[3], in_channels[3] // 8, 1) #[256]
        self.conv4 = BasicConv(in_channels[4], in_channels[4] // 8, 1) #[256]

        self.body = nn.Sequential(
            BlockBody222([in_channels[0] // 8, in_channels[1] // 8, in_channels[2] // 8, in_channels[3] // 8, in_channels[4] // 8])
        )

        self.conv00 = BasicConv(in_channels[0] // 8, out_channels, 1)
        self.conv11 = BasicConv(in_channels[1] // 8, out_channels, 1)
        self.conv22 = BasicConv(in_channels[2] // 8, out_channels, 1)
        self.conv33 = BasicConv(in_channels[3] // 8, out_channels, 1)
        self.conv44 = BasicConv(in_channels[4] // 8, out_channels, 1)

        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)
        #########此处是添加一个上采样参数
        # self.upsample1=nn.Upsample(256,scale_factor=2)
        # self.upsample2=nn.Upsample(256,scale_factor=4)
        # self.upsample3=nn.Upsample(256,scale_factor=8)
        self.conv=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1)

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x0, x1, x2, x3, x4 = x

        # 此处将通道数降为1\8  缩小8倍
        x0 = self.conv0(x0) #[8,32,96,96]
        x1 = self.conv1(x1) #[8,64,48,48]
        x2 = self.conv2(x2) #[8,128,24,24]
        x3 = self.conv3(x3) #[8,256,12,12]
        x4 = self.conv4(x4) #[8,256,12,12]

        out0, out1, out2, out3, out4 = self.body([x0, x1, x2, x3, x4])  # 经过多层次的融合得到四个输出

        #   放大到 输出的维度
        out0 = self.conv00(out0)    # [8,256,96,96]       #wavemlp:[4,128,96,96]
        out1 = self.conv11(out1)    # [8,256,48,48]       #[4,128,48,48]
        out2 = self.conv22(out2)    # [8,256,24,24]       #[4,128,24,24]
        out3 = self.conv33(out3)    # [8,256,12,12]       #[4,128,12,12]
        out4 = self.conv44(out4)    # [8,256,12,12]       #[4,128,12,12]

###########################此处添加最后的部分，将高层次的特征添加到低层中去，增强低层特征的表达能力。是否可以提效果？？？
        # out4 = self.conv44(out3) #[8,256,6,6]??????????????????????????????

        # out2 = out2 + self.conv(F.interpolate(out3, scale_factor=2, mode='bilinear', align_corners=False))
        # out1 = out1 + self.conv(F.interpolate(out2, scale_factor=2, mode='bilinear', align_corners=False))
        # out0 = out0 + self.conv(F.interpolate(out1, scale_factor=2, mode='bilinear', align_corners=False))


        return out0, out1, out2, out3, out4

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p4 = torch.randn((8, 2048, 12, 12), device=device)
    p3 = torch.randn((8, 1024, 24, 24), device=device)
    p2 = torch.randn((8, 512,48, 48), device=device)
    p1 = torch.randn((8, 256, 96, 96), device=device)
    list = tuple((p1, p2, p3, p4))
    afpn = AFPN([256, 512, 1024, 2048],256)
    afpn.to(device)
    results = afpn(list)
    cat1, cat2, cat3, cat4 = results
    print(results)
    print('debug------------')

