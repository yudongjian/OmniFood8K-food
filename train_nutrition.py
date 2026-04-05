from PIL import ImageEnhance
import torch
import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse

from utils.utils import progress_bar, load_for_transfer_learning, logtxt, check_dirs
from utils.utils_data222 import get_DataLoader
from tensorboardX import SummaryWriter
import pdb
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
import random
from model import dual_swin_convnext
from model.convnext1 import convnext_small
import torch.backends.cudnn as cudnn
from model.myswinb import SwinTransformer
from model.three_D import DynamicTaskPrioritization
from ptflops import get_model_complexity_info
from thop import profile
from thop import clever_format



def set_seed(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class PercentageLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PercentageLoss, self).__init__()
        self.eps = eps  # 防止除以0

    def forward(self, y_pred, y_true):
        percentage_error = torch.abs((y_pred - y_true) / (y_true + self.eps))
        return percentage_error.mean()


parser = argparse.ArgumentParser(description='PyTorch CDINet Training')
parser.add_argument('--dataset',
                    choices=["nutrition_rgbd", "nutrition_rgb_pre_d", "nutrition8K", '11w'], default='nutrition_rgb_pre_d')
parser.add_argument('--b', type=int, default=2, help='batch size')
parser.add_argument('--epoch', type=int, default=150, help='epoch num')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--log', '-log', type=str, help='./logs')
parser.add_argument('--data_root', type=str,
                    default="/home/image1325_user/ssd_disk1/yudongjian_23/Data/nutrition5k_dataset/", help="our dataset root")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

args = parser.parse_args()

set_seed(args)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

global net
net = SwinTransformer()
net2 = convnext_small(pretrained=False,in_22k=False)

# net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_Cat()
net_cat = dual_swin_convnext.FusionNet_3Branch_UNet_FFT()

# -------------------
from modules.fusion import FeatureFusionNetwork222_Mask
pre_net1 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net2 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net3 = FeatureFusionNetwork222_Mask(dropout=0.1)
pre_net4 = FeatureFusionNetwork222_Mask(dropout=0.05)
pre_net5 = FeatureFusionNetwork222_Mask(dropout=0.1)

task_prior = DynamicTaskPrioritization(alpha=0.3)

from modules.adapter import DepthAdapterV4
adapter = DepthAdapterV4(in_ch=3, base_ch=32)


print('==> Load checkpoint..')
swin_ckpt_path = "/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/pth/swin_base_patch4_window12_384_22k.pth"
convnext_ckpt_path = "/home/image1325_user/ssd_disk4/yudongjian_23/food-nurtrition/pth/convnext_small_22k_1k_384.pth"

swin_ckpt = torch.load(swin_ckpt_path, map_location="cpu", weights_only=True)
convnext_ckpt = torch.load(convnext_ckpt_path, map_location="cpu", weights_only=True)

net.load_state_dict(swin_ckpt["model"], strict=False)
net2.load_state_dict(convnext_ckpt["model"], strict=False)


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


optimizer = torch.optim.Adam([
    {'params': net.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # 5e-4
    {'params': net2.parameters(), 'lr': 1e-5, 'weight_decay': 1e-4},  # 5e-4

    {'params': net_cat.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},  # 5e-4

    {'params': adapter.parameters(), 'lr': 1e-4, },  # 5e-4

    {'params': pre_net1.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net2.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net3.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net4.parameters(), 'lr': 1e-4, },  # 5e-4
    {'params': pre_net5.parameters(), 'lr': 1e-4, },  # 5e-4

])

def inter_modal_alignment_loss(cat_feat):
    B, C, H, W = cat_feat.shape
    # RGB / Depth 通道拆分
    mid = C // 2
    rgb_feat = cat_feat[:, :mid]
    depth_feat = cat_feat[:, mid:]
    # 全局平均池化
    rgb_vec = F.normalize(rgb_feat.view(B, mid, -1).mean(dim=2), dim=1)
    depth_vec = F.normalize(depth_feat.view(B, mid, -1).mean(dim=2), dim=1)
    sim_matrix = torch.matmul(rgb_vec, depth_vec.t()) / 0.1
    labels = torch.arange(B, device=cat_feat.device)
    return F.cross_entropy(sim_matrix, labels)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=2e-6)

weights = []
task_losses = []
loss_ratios = []
grad_norm_losses = []

trainloader, testloader = get_DataLoader(args)

# Training
def train(epoch, net):
    global global_step
    print('\nEpoch: %d' % epoch)
    net.train()
    net2.train()
    pre_net1.train()
    pre_net2.train()
    pre_net3.train()
    pre_net4.train()
    pre_net5.train()
    
    net_cat.train()
    adapter.train()

    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(trainloader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):  # (inputs, targets,ingredient)
        '''Portion Independent Model'''

        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()
        inputs_rgbd = x[7].to(device)

        rgd_dino = x[9].to(device)

        optimizer.zero_grad()

        output_rgb = net(inputs)
        r0,r1,r2,r3,r4 = output_rgb

        inputs_rgbd = adapter(inputs_rgbd)
        output_hha = net2(inputs_rgbd)
        d1,d2,d3,d4= output_hha

        outputs_feature = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])

        #  =====  输入 4 个 进行预测   ======
        o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]
        outputs = [0, 0, 0, 0, 0]
        outputs[0] = pre_net1(o1, o2, o3, o4).squeeze()
        outputs[1] = pre_net2(o1, o2, o3, o4).squeeze()
        outputs[2] = pre_net3(o1, o2, o3, o4).squeeze()
        outputs[3] = pre_net4(o1, o2, o3, o4).squeeze()
        outputs[4] = pre_net5(o1, o2, o3, o4).squeeze()


        total_calories_loss = total_calories.shape[0] * criterion(outputs[0],total_calories) / total_calories.sum().item() if total_calories.sum().item() != 0 else criterion(outputs[0], total_calories)
        total_mass_loss = total_calories.shape[0] * criterion(outputs[1],total_mass) / total_mass.sum().item() if total_mass.sum().item() != 0 else criterion(outputs[1], total_mass)
        total_fat_loss = total_calories.shape[0] * criterion(outputs[2],total_fat) / total_fat.sum().item() if total_fat.sum().item() != 0 else criterion(outputs[2], total_fat)
        total_carb_loss = total_calories.shape[0] * criterion(outputs[3],total_carb) / total_carb.sum().item() if total_carb.sum().item() != 0 else criterion(outputs[3], total_carb)
        total_protein_loss = total_calories.shape[0] * criterion(outputs[4],total_protein) / total_protein.sum().item() if total_protein.sum().item() != 0 else criterion(outputs[4], total_protein)

        loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss

        loss_align = inter_modal_alignment_loss(o1)
        loss_align = loss_align * 0.1    # λ 权重

        k1, k2, k3, k4, k5 = task_prior.task_weights
        loss22 = k1 * total_calories_loss + k2 * total_mass_loss + k3 * total_fat_loss + \
                 k4 * total_carb_loss + k5 * total_protein_loss  + loss_align

        loss22.backward()
        optimizer.step()

        train_loss += loss.item()
        calories_loss += total_calories_loss.item()
        mass_loss += total_mass_loss.item()
        fat_loss += total_fat_loss.item()
        carb_loss += total_carb_loss.item()
        protein_loss += total_protein_loss.item()

        if (batch_idx % 100) == 0:
            print(
                "\nTraining Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %f" % (
                    epoch, train_loss / (batch_idx + 1), calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),
                    fat_loss / (batch_idx + 1), carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1),
                    optimizer.param_groups[0]['lr']))


        if (batch_idx + 1) % 100 == 0 or batch_idx + 1 == len(trainloader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                                  'Loss: {:2.5f} \t'
                                  'calorieloss: {:2.5f} \t'
                                  'massloss: {:2.5f} \t'
                                  'fatloss: {:2.5f} \t'
                                  'carbloss: {:2.5f} \t'
                                  'proteinloss: {:2.5f} \t'
                                  'loss_align: {} \t'
                                  'lr:{:2.5f}-{:2.5f}-{:2.5f}-{:2.5f}'.format(
                epoch, batch_idx + 1, len(trainloader),
                       train_loss / (batch_idx + 1),
                       calories_loss / (batch_idx + 1),
                       mass_loss / (batch_idx + 1),
                       fat_loss / (batch_idx + 1),
                       carb_loss / (batch_idx + 1),
                       protein_loss / (batch_idx + 1),
                        loss_align,
                optimizer.param_groups[0]['lr'],
                optimizer.param_groups[1]['lr'],
                optimizer.param_groups[2]['lr'],
                optimizer.param_groups[3]['lr']))

        if (batch_idx + 1) % 30 == 0 or batch_idx + 1 == len(trainloader):
            current_kpis = torch.tensor([calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),mass_loss / (batch_idx + 1),
                                         carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1)])
            task_prior.update_weights(current_kpis)
            print(task_prior.task_weights)


best_loss = 10000


def test(epoch, net):

        global best_loss
        net.eval()
        net2.eval()
        net_cat.eval()
        adapter.eval()
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

        epoch_iterator = tqdm(testloader,
                              desc="Testing... (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)
        csv_rows = []
        with torch.no_grad():
            for batch_idx, x in enumerate(epoch_iterator):  # testloader
                inputs = x[0].to(device)
                total_calories = x[2].to(device).float()
                total_mass = x[3].to(device).float()
                total_fat = x[4].to(device).float()
                total_carb = x[5].to(device).float()
                total_protein = x[6].to(device).float()
                inputs_rgbd = x[7].to(device)


                optimizer.zero_grad()
                output_rgb = net(inputs)
                r0, r1, r2, r3, r4 = output_rgb

                inputs_rgbd = adapter(inputs_rgbd)
                output_hha = net2(inputs_rgbd)
                d1, d2, d3, d4 = output_hha

                outputs_feature = net_cat([r1, r2, r3, r4], [d1, d2, d3, d4])

                #  =====  输入 4 个 进行预测   ======
                o1, o2, o3, o4 = outputs_feature[0], outputs_feature[1], outputs_feature[2], outputs_feature[3]
                outputs = [0, 0, 0, 0, 0]
                outputs[0] = pre_net1(o1, o2, o3, o4).squeeze()
                outputs[1] = pre_net2(o1, o2, o3, o4).squeeze()
                outputs[2] = pre_net3(o1, o2, o3, o4).squeeze()
                outputs[3] = pre_net4(o1, o2, o3, o4).squeeze()
                outputs[4] = pre_net5(o1, o2, o3, o4).squeeze()

                if epoch % 10 == 0:
                    #
                    for i in range(len(x[1])):  # IndexError: tuple index out of range  最后一轮的图片数量不到32，不能被batchsiz
                        dish_id = x[1][i]
                        calories = outputs[0][i]
                        mass = outputs[1][i]
                        fat = outputs[2][i]
                        carb = outputs[3][i]
                        protein = outputs[4][i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)

                calories_ae += F.l1_loss(outputs[0], total_calories, reduction='sum').item()
                mass_ae += F.l1_loss(outputs[1], total_mass, reduction='sum').item()
                fat_ae += F.l1_loss(outputs[2], total_fat, reduction='sum').item()
                carb_ae += F.l1_loss(outputs[3], total_carb, reduction='sum').item()
                protein_ae += F.l1_loss(outputs[4], total_protein, reduction='sum').item()

                calories_sum += total_calories.sum().item()
                mass_sum += total_mass.sum().item()
                fat_sum += total_fat.sum().item()
                carb_sum += total_carb.sum().item()
                protein_sum += total_protein.sum().item()

            calories_pmae = calories_ae / calories_sum
            mass_pmae = mass_ae / mass_sum
            fat_pmae = fat_ae / fat_sum
            carb_pmae = carb_ae / carb_sum
            protein_pmae = protein_ae / protein_sum

            loss_pmae = calories_pmae + mass_pmae + fat_pmae + carb_pmae + protein_pmae

            epoch_iterator.set_description(
                "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (
                    epoch, loss_pmae, calories_pmae, mass_pmae,
                    fat_pmae, carb_pmae, protein_pmae,
                    optimizer.param_groups[0]['lr'])
            )
            logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                                  'Loss: {:2.5f} \t'
                                  'calorieloss: {:2.5f} \t'
                                  'massloss: {:2.5f} \t'
                                  'fatloss: {:2.5f} \t'
                                  'carbloss: {:2.5f} \t'
                                  'proteinloss: {:2.5f} \t'
                                  'lr:{:.7f}\n'.format(
                epoch, batch_idx + 1, len(testloader),
                loss_pmae,
                calories_pmae,
                mass_pmae,
                fat_pmae,
                carb_pmae,
                protein_pmae,
                optimizer.param_groups[0]['lr']))

        if best_loss > loss_pmae:
            best_loss = loss_pmae
            print('Saving..')
            net = net.module if hasattr(net, 'module') else net
            state = {
                'net': net.state_dict(),
                'net2': net2.state_dict(),
                'adapter': adapter.state_dict(),

                'pre_net1': pre_net1.state_dict(),
                'pre_net2': pre_net2.state_dict(),
                'pre_net3': pre_net3.state_dict(),
                'pre_net4': pre_net4.state_dict(),
                'pre_net5': pre_net5.state_dict(),

                'net_cat': net_cat.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }
            global min_epoch
            min_epoch = epoch
            savepath = f"./saved/{args.log}"
            check_dirs(savepath)
            torch.save(state, os.path.join(savepath, f"ckpt_best.pth"))
            logtxt(log_file_path, "= = =  ↑ ↑ ↑ ↑ ↑ ↑ ↑ = = = ")
            logtxt(log_file_path, "======  Min  ===================")


global min_epoch

log_file_path = os.path.join(args.log, "train_log.txt")
check_dirs(args.log)
logtxt(log_file_path, str(vars(args)))

for epoch in range(0, args.epoch):
    train(epoch, net)
    test(epoch, net)
    scheduler.step()

logtxt(log_file_path, "======  Min  ===================")
logtxt(log_file_path, "====  epoch ::: {} ===================".format(min_epoch))
logtxt(log_file_path, "======  Min  ===================")


