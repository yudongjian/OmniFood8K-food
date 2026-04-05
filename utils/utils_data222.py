import logging
from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torchvision.transforms.transforms import CenterCrop
from mydataset import Nutrition_RGBD, Nutrition_RGB_Pre_D, Nutrition8k, Nutrition11w
import pdb

def get_DataLoader(args):

    if args.dataset == 'nutrition_rgbd':

        train_transform = transforms.Compose([
            # transforms.Resize((550, 550)),
            transforms.Resize((518, 518)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # 颜色的影响
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
                                    # transforms.Resize((320, 448)),
                                    # transforms.Resize((384, 384)),
                                    transforms.Resize((518, 518)),
                                    # transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])

        nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')


        nutrition_train_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgbd_train_processed.txt')
        nutrition_test_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgbd_test_processed1.txt') # depth_color.png
        nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgb_in_overhead_train_processed.txt')
        nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgb_in_overhead_test_processed1.txt') # rbg.png


        trainset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt, transform = train_transform)
        testset = Nutrition_RGBD(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform = test_transform)


    elif args.dataset == 'nutrition_rgb_pre_d':
        train_transform = transforms.Compose([
                                    # transforms.RandomRotation(degrees=(0, 180)),
                                    # transforms.Resize((320, 448)),
                                    transforms.Resize((384, 384)),
                                    # transforms.RandomHorizontalFlip(),
                                    # transforms.CenterCrop((256,256)),
                                    # transforms.ColorJitter(hue=0.05),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])
        test_transform = transforms.Compose([
                                    # transforms.Resize((320, 448)),
                                    transforms.Resize((384, 384)),
                                    # transforms.CenterCrop((256, 256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    # transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
                                    ])

        nutrition_rgbd_ims_root = os.path.join(args.data_root, 'imagery')


        nutrition_train_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgbd_train_processed.txt')
        nutrition_test_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgbd_test_processed1.txt') # depth_color.png
        nutrition_train_rgbd_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgb_in_overhead_train_processed.txt')
        nutrition_test_rgbd_txt = os.path.join(args.data_root, 'imagery', 'txt-file', 'rgb_in_overhead_test_processed1.txt') # rbg.png

        trainset = Nutrition_RGB_Pre_D(nutrition_rgbd_ims_root, nutrition_train_rgbd_txt, nutrition_train_txt, transform = train_transform)
        testset = Nutrition_RGB_Pre_D(nutrition_rgbd_ims_root, nutrition_test_rgbd_txt, nutrition_test_txt, transform = test_transform)

    elif args.dataset == 'nutrition8K':
        train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        test_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        base_path = '/home/image1325_user/ssd_disk4/yudongjian_23/Data'

        nutrition_train_txt = os.path.join(base_path, 'train_new333.txt')
        nutrition_test_txt = os.path.join(base_path, 'test_new333.txt')

        nutrition_train_path = os.path.join(base_path, '1-data')

        trainset = Nutrition8k(nutrition_train_path, nutrition_train_txt, transform=train_transform)
        testset = Nutrition8k(nutrition_train_path, nutrition_test_txt, transform=test_transform)

    elif args.dataset == '11w':
        train_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        test_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        base_path = '/home/image1325_user/ssd_disk4/yudongjian_23/Data/syn-data'
        nutrition_train_txt = os.path.join(base_path, 'train2.txt')
        nutrition_test_txt = os.path.join(base_path, 'test2.txt')

        nutrition_train_path = base_path

        trainset = Nutrition11w(nutrition_train_path, nutrition_train_txt, transform=train_transform)
        testset = Nutrition11w(nutrition_train_path, nutrition_test_txt, transform=test_transform)

    train_loader = DataLoader(trainset,
                              batch_size=args.b,
                              shuffle=True,
                              num_workers=32,
                              pin_memory=True
                              )
    test_loader = DataLoader(testset,
                             batch_size=args.b,
                             shuffle=False,
                             num_workers=32,
                             pin_memory=True
                             )

    print(len(test_loader))
    return train_loader, test_loader





