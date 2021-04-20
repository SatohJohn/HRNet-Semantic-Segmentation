# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# Referring to the implementation in 
# https://github.com/zhanghang1989/PyTorch-Encoding
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset

class PASCALContext(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=59,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(480, 480),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(PASCALContext, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        # self.label_mapping = {-1: ignore_label, 0: ignore_label, 
        #                       1: ignore_label, 2: ignore_label, 
        #                       3: ignore_label, 4: ignore_label, 
        #                       5: ignore_label, 6: ignore_label, 
        #                       7: 0, 8: 1, 9: ignore_label, 
        #                       10: ignore_label, 11: 2, 12: 3, 
        #                       13: 4, 14: ignore_label, 15: ignore_label, 
        #                       16: ignore_label, 17: 5, 18: ignore_label, 
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        #                       25: 12, 26: 13, 27: 14, 28: 15, 
        #                       29: ignore_label, 30: ignore_label, 
        #                       31: 16, 32: 17, 33: 18}

        self.label_mapping = {x: ignore_label for x in range(-1, 59)}
        self.label_mapping[28] = 15
        self.label_mapping[15] = 28

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                sample = {
                    'img': image_path[0],
                    'name': name
                }
                files.append(sample)
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                sample = {
                    'img': image_path,
                    'label': label_path,
                    'name': name
                }
                files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if 'test' in self.list_path:
            image, border_padding = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True
            )
            size = image.shape
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label_path = os.path.join(self.root, item['label'])
        label = np.array(
            Image.open(label_path).convert('P')
        )
        if self.num_classes == 59:
            label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image, border_padding = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name, border_padding

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

       
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=False)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))
