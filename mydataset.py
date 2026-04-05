import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
# import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg

import imageio
import cv2
import pdb
import open3d as o3d

import torchvision.transforms as T


class Nutrition(Dataset):
    def __init__(self, image_path, txt_dir, transform=None):

        file = open(txt_dir, 'r')
        lines = file.readlines()
        self.images = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        # pdb.set_trace()
        for line in lines:
            image = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform

    def __getitem__(self, index):
        # img = cv2.imread(self.images[index])
        # try:
        #     # img = cv2.resize(img, (self.imsize, self.imsize))
        #     img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转PIL
        # except:
        #     print("图片有误：",self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            try:
                # lmj  RGB-D图像尺寸不同,按照不同比例缩放
                if 'realsense_overhead' in self.images[index]:
                    # pdb.set_trace()
                    self.transform.transforms[0].size = (267, 356)
                    # print(self.transform)
                img = self.transform(img)
            except:
                # print('trans_img', img)
                print('trans_img有误')
        return img, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[index]

    def __len__(self):
        return len(self.images)


dino_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])
# RGB-D
class Nutrition_RGBD(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        print(len(lines_rgb))
        print(len(lines_rgbd))
        self.images = []
        self.points = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []

        print('==============================')
        print(' 输入的 真实  真实 真实 的 深度信息')


        # pdb.set_trace()
        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            image_rgbd = line.split()[0]
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform

    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        try:
            img_rgb = cv2.imread(self.images[index])
            img_rgbd = cv2.imread(self.images_rgbd[index])
            img_points = Path(self.images_rgbd[index]).parent
            points = o3d.io.read_point_cloud(os.path.join(img_points, "processed_point_cloud.ply"))
            points_numpy = np.asarray(points.points)  # 获取 (N, 3) 点坐标
            colors_numpy = np.asarray(points.colors)  # 获取 (N, 3) 点颜色

            # 将坐标和颜色数据拼接成一个 (N, 6) 的数组
            points_and_colors = np.concatenate((points_numpy, colors_numpy), axis=1)
            # 将拼接后的数据转换为 PyTorch 张量
            points_tensor = torch.tensor(points_and_colors, dtype=torch.float32)

        except Exception as e:
            print('------------------')
            print(index)
            print(len(self.images))
            print(self.images[index])
            print(self.images_rgbd[index])
            print('------------------')
            raise Exception
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        except:
            print("图片有误：", self.images[index])

        rgb_dino = dino_transform(img_rgb)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)



        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[
            index], img_rgbd, points_tensor, rgb_dino  # 返回 2种image即可，然后再在train2.py中多一个判断，两个图片输入两次网络

    def __len__(self):
        return len(self.images)


class Nutrition_RGB_Pre_D(Dataset):
    def __init__(self, image_path, rgb_txt_dir, rgbd_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')
        file_rgbd = open(rgbd_txt_dir, 'r')
        lines_rgb = file_rgb.readlines()
        lines_rgbd = file_rgbd.readlines()
        print(len(lines_rgb))
        print(len(lines_rgbd))
        self.images = []
        self.points = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()

        print('==============================')
        print(' 输入的 预测的 深度信息')

        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg
            label = line.strip().split()[1]  # 类别 1-
            calories = line.strip().split()[2]
            mass = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            protein = line.strip().split()[6]

            self.images += [os.path.join(image_path, image_rgb)]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]
        for line in lines_rgbd:
            # image_rgbd = line.split()[0]
            image_rgbd = line.split()[0].replace('depth_color.png', 'rgb-d.png')
            self.images_rgbd += [os.path.join(image_path, image_rgbd)]

        self.transform = transform

    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        try:
            img_rgb = cv2.imread(self.images[index])
            img_rgbd = cv2.imread(self.images_rgbd[index])
            img_points = Path(self.images_rgbd[index]).parent
            points = o3d.io.read_point_cloud(os.path.join(img_points, "predict_point_cloud.ply"))
            points_numpy = np.asarray(points.points)  # 获取 (N, 3) 点坐标
            colors_numpy = np.asarray(points.colors)  # 获取 (N, 3) 点颜色

            # 将坐标和颜色数据拼接成一个 (N, 6) 的数组
            points_and_colors = np.concatenate((points_numpy, colors_numpy), axis=1)
            # 将拼接后的数据转换为 PyTorch 张量
            points_tensor = torch.tensor(points_and_colors, dtype=torch.float32)

        except Exception as e:
            print('------------------')
            print(index)
            print(len(self.images))
            print(self.images[index])
            print(self.images_rgbd[index])
            print('------------------')
            raise Exception
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        rgb_dino = dino_transform(img_rgb)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)



        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[
            index], img_rgbd, points_tensor, rgb_dino  # 返回 2种image即可，然后再在train2.py中多一个判断，两个图片输入两次网络

    def __len__(self):
        return len(self.images)




class Nutrition8k(Dataset):
    def __init__(self, image_path, rgb_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')

        lines_rgb = file_rgb.readlines()

        print(len(lines_rgb))

        self.images = []
        self.points = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()

        print('==============================')
        print(' 输入的 预测的 深度信息  -------------   8K 数据集')

        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg

            mass = line.strip().split()[1]
            calories = line.strip().split()[2]
            protein = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            label = '1'

            self.images += [os.path.join(image_path, image_rgb, 'camera_4.jpg')]  # 每张图片路径
            self.images_rgbd += [os.path.join(image_path, image_rgb, 'rgb-d.png')]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
        print(len(self.total_fat))
    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        try:
            img_rgb = cv2.imread(self.images[index])
            img_rgbd = cv2.imread(self.images_rgbd[index])
            points_tensor = 1

            if img_rgb is None:
                raise FileNotFoundError(f"图像读取失败: {img_path}")

        except Exception as e:
            print('------------------')
            print(index)
            print(len(self.images))
            print(self.images[index])
            print(self.images_rgbd[index])
            print('------------------')
            raise Exception
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        rgb_dino = dino_transform(img_rgb)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[
            index], img_rgbd, points_tensor, rgb_dino  # 返回 2种image即可，然后再在train2.py中多一个判断，两个图片输入两次网络

    def __len__(self):
        return len(self.images)


class Nutrition11w(Dataset):
    def __init__(self, image_path, rgb_txt_dir, transform=None):

        file_rgb = open(rgb_txt_dir, 'r')

        lines_rgb = file_rgb.readlines()

        print(len(lines_rgb))

        self.images = []
        self.points = []
        self.labels = []
        self.total_calories = []
        self.total_mass = []
        self.total_fat = []
        self.total_carb = []
        self.total_protein = []
        self.images_rgbd = []
        # pdb.set_trace()

        print('==============================')
        print(' 输入的 预测的 深度信息  -------------   8K 数据集')

        for line in lines_rgb:
            image_rgb = line.split()[0]  # side_angles/dish_1550862840/frames_sampled5/camera_A_frame_010.jpeg

            mass = line.strip().split()[1]
            calories = line.strip().split()[2]
            protein = line.strip().split()[3]
            fat = line.strip().split()[4]
            carb = line.strip().split()[5]
            label = '1'

            self.images += [os.path.join(image_path, image_rgb, '{}.png'.format(image_rgb.split('/')[-1]))]  # 每张图片路径
            self.images_rgbd += [os.path.join(image_path, image_rgb, 'rgb-d.png')]  # 每张图片路径
            self.labels += [str(label)]
            self.total_calories += [np.array(float(calories))]
            self.total_mass += [np.array(float(mass))]
            self.total_fat += [np.array(float(fat))]
            self.total_carb += [np.array(float(carb))]
            self.total_protein += [np.array(float(protein))]

            # pdb.set_trace()
        # self.transform_rgb = transform[0]

        self.transform = transform
        print(len(self.total_fat))
    # RGB-D  20210805
    def my_loader(path, Type):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                if Type == 3:
                    img = img.convert('RGB')
                elif Type == 1:
                    img = img.convert('L')
                return img

    def __getitem__(self, index):
        try:
            img_rgb = cv2.imread(self.images[index])
            img_rgbd = cv2.imread(self.images_rgbd[index])
            points_tensor = 1

            if img_rgb is None:
                raise FileNotFoundError(f"图像读取失败: {img_path}")

        except Exception as e:
            print('------------------')
            print(index)
            print(len(self.images))
            print(self.images[index])
            print(self.images_rgbd[index])
            print('------------------')
            raise Exception
        try:
            # img = cv2.resize(img, (self.imsize, self.imsize))
            img_rgb = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # cv2转PIL
            img_rgbd = Image.fromarray(cv2.cvtColor(img_rgbd, cv2.COLOR_BGR2RGB))  # cv2转PIL
        except:
            print("图片有误：", self.images[index])
        # 4通道
        # rgb_path, d_path = self.images[index], self.images_rgbd[index]
        # rgb_img = np.array(self.my_loader(rgb_path, 3))
        # d_img = np.array(self.my_loader(d_path, 1) )
        # d_img = np.expand_dims(d_img, axis=2) #(480, 640, 1)
        # img = np.append(rgb_img, d_img, axis=2) # (480, 640, 4)

        rgb_dino = dino_transform(img_rgb)

        if self.transform is not None:
            # img = self.transform(Image.fromarray(img))
            img_rgb = self.transform(img_rgb)
            img_rgbd = self.transform(img_rgbd)

        return img_rgb, self.labels[index], self.total_calories[index], self.total_mass[index], self.total_fat[index], self.total_carb[index], self.total_protein[
            index], img_rgbd, points_tensor, rgb_dino  # 返回 2种image即可，然后再在train2.py中多一个判断，两个图片输入两次网络

    def __len__(self):
        return len(self.images)
# 20210526


