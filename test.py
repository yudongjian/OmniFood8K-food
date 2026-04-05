from PIL import ImageEnhance
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import csv
import random
import numpy as np
from tqdm import tqdm

from utils.utils import logtxt, check_dirs
from utils.utils_data222 import get_DataLoader

from model import dual_swin_convnext
from model.convnext1 import convnext_small
from model.myswinb import SwinTransformer
from modules.fusion import FeatureFusionNetwork222_Mask
from modules.adapter import DepthAdapterV4


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


parser = argparse.ArgumentParser(description='PyTorch Nutrition Testing')
parser.add_argument('--dataset',
                    choices=["nutrition_rgbd", "nutrition_rgb_pre_d", "nutrition8K", '11w'],
                    default='nutrition_rgb_pre_d')
parser.add_argument('--b', type=int, default=8, help='batch size')
parser.add_argument('--data_root', type=str,
                    default="/home/image1325_user/ssd_disk1/yudongjian_23/Data/nutrition5k_dataset/",
                    help="dataset root")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--ckpt', type=str, required=True, help='path to trained checkpoint, e.g. ./saved/xxx/ckpt_best.pth')
parser.add_argument('--log', type=str, default='./test_logs', help='log dir')


args = parser.parse_args()
set_seed(args)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('==> Preparing data..')

# =========================
# Build model
# =========================
net = SwinTransformer()
net2 = convnext_small(pretrained=False, in_22k=False)
net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_FFT()

pre_net1 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net2 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net3 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net4 = FeatureFusionNetwork222_Mask(dropout=0.05)
pre_net5 = FeatureFusionNetwork222_Mask(dropout=0.1)

adapter = DepthAdapterV4(in_ch=3, base_ch=32)

net = net.to(device)
net2 = net2.to(device)
net_cat = net_cat.to(device)
adapter = adapter.to(device)

pre_net1 = pre_net1.to(device)
pre_net2 = pre_net2.to(device)
pre_net3 = pre_net3.to(device)
pre_net4 = pre_net4.to(device)
pre_net5 = pre_net5.to(device)

criterion = nn.L1Loss()

# =========================
# Load checkpoint
# =========================
print('==> Loading trained checkpoint..')
ckpt = torch.load(args.ckpt, map_location=device)

net.load_state_dict(ckpt['net'], strict=False)
net2.load_state_dict(ckpt['net2'], strict=False)
adapter.load_state_dict(ckpt['adapter'], strict=False)
net_cat.load_state_dict(ckpt['net_cat'], strict=False)

pre_net1.load_state_dict(ckpt['pre_net1'], strict=False)
pre_net2.load_state_dict(ckpt['pre_net2'], strict=False)
pre_net3.load_state_dict(ckpt['pre_net3'], strict=False)
pre_net4.load_state_dict(ckpt['pre_net4'], strict=False)
pre_net5.load_state_dict(ckpt['pre_net5'], strict=False)

print(f"Loaded checkpoint from: {args.ckpt}")
if 'epoch' in ckpt:
    print(f"Checkpoint epoch: {ckpt['epoch']}")

# =========================
# DataLoader
# =========================
_, testloader = get_DataLoader(args)

log_file_path = os.path.join(args.log, "test_log.txt")
check_dirs(args.log)
logtxt(log_file_path, str(vars(args)))


def forward_once(inputs, inputs_rgbd):
    output_rgb = net(inputs)
    r0, r1, r2, r3, r4 = output_rgb

    inputs_rgbd = adapter(inputs_rgbd)
    output_hha = net2(inputs_rgbd)
    d1, d2, d3, d4 = output_hha

    outputs_feature = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])

    o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]

    outputs = [0, 0, 0, 0, 0]
    outputs[0] = pre_net1(o1, o2, o3, o4).squeeze()
    outputs[1] = pre_net2(o1, o2, o3, o4).squeeze()
    outputs[2] = pre_net3(o1, o2, o3, o4).squeeze()
    outputs[3] = pre_net4(o1, o2, o3, o4).squeeze()
    outputs[4] = pre_net5(o1, o2, o3, o4).squeeze()

    return outputs


def test():
    net.eval()
    net2.eval()
    net_cat.eval()
    adapter.eval()
    pre_net1.eval()
    pre_net2.eval()
    pre_net3.eval()
    pre_net4.eval()
    pre_net5.eval()

    calories_ae = 0
    mass_ae = 0
    fat_ae = 0
    carb_ae = 0
    protein_ae = 0

    calories_sum = 0
    mass_sum = 0
    fat_sum = 0
    carb_sum = 0
    protein_sum = 0

    csv_rows = []

    epoch_iterator = tqdm(
        testloader,
        desc="Testing...",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True
    )

    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator):
            inputs = x[0].to(device)
            dish_ids = x[1]

            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()
            inputs_rgbd = x[7].to(device)

            outputs = forward_once(inputs, inputs_rgbd)

            pred_cal = outputs[0]
            pred_mass = outputs[1]
            pred_fat = outputs[2]
            pred_carb = outputs[3]
            pred_protein = outputs[4]

            calories_ae += F.l1_loss(pred_cal, total_calories, reduction='sum').item()
            mass_ae += F.l1_loss(pred_mass, total_mass, reduction='sum').item()
            fat_ae += F.l1_loss(pred_fat, total_fat, reduction='sum').item()
            carb_ae += F.l1_loss(pred_carb, total_carb, reduction='sum').item()
            protein_ae += F.l1_loss(pred_protein, total_protein, reduction='sum').item()

            calories_sum += total_calories.sum().item()
            mass_sum += total_mass.sum().item()
            fat_sum += total_fat.sum().item()
            carb_sum += total_carb.sum().item()
            protein_sum += total_protein.sum().item()


    num_samples = len(testloader.dataset)

    calories_mae = calories_ae / num_samples
    mass_mae = mass_ae / num_samples
    fat_mae = fat_ae / num_samples
    carb_mae = carb_ae / num_samples
    protein_mae = protein_ae / num_samples

    calories_pmae = calories_ae / calories_sum
    mass_pmae = mass_ae / mass_sum
    fat_pmae = fat_ae / fat_sum
    carb_pmae = carb_ae / carb_sum
    protein_pmae = protein_ae / protein_sum

    mean_mae = (calories_mae + mass_mae + fat_mae + carb_mae + protein_mae) / 5.0
    mean_pmae = (calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae) / 5.0
    sum_pmae = calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae

    result_str = (
        "\n================ Test Results ================\n"
        f"Calories MAE  : {calories_mae:.6f}\n"
        f"Mass MAE      : {mass_mae:.6f}\n"
        f"Fat MAE       : {fat_mae:.6f}\n"
        f"Carb MAE      : {carb_mae:.6f}\n"
        f"Protein MAE   : {protein_mae:.6f}\n"
        f"Mean MAE      : {mean_mae:.6f}\n\n"
        f"Calories PMAE : {calories_pmae:.6f}\n"
        f"Mass PMAE     : {mass_pmae:.6f}\n"
        f"Fat PMAE      : {fat_pmae:.6f}\n"
        f"Carb PMAE     : {carb_pmae:.6f}\n"
        f"Protein PMAE  : {protein_pmae:.6f}\n"
        f"Mean PMAE     : {mean_pmae:.6f}\n"
        f"Sum PMAE      : {sum_pmae:.6f}\n"
        "=============================================\n"
    )

    print(result_str)
    # logtxt(log_file_path, result_str)

if __name__ == '__main__':
    test()