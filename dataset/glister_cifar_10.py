from cProfile import label
import os, random, time, copy
from skimage import io, transform
import numpy as np
import os.path as path
import scipy.io as sio
from scipy import misc
import matplotlib.pyplot as plt
import PIL.Image
import pickle
import skimage.transform 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
import torchvision
import argparse
import time
import math
from os import path, makedirs
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
from torchvision import datasets as datasets_torch
from torchvision import transforms
import os
from torchvision.datasets import CIFAR10,CIFAR100
import random
class CIFAR100LT(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            self.transform_tail = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.6, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else :
            self.transform = transforms.Compose([
            transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.imageList = imageList
        self.labelList = labelList
        
        list_label=labelList
        self.label_dict=dict()
        for i in range(len(list_label)):
            try:
                list_item= self.label_dict[list_label[i]]
                list_item.append(i)
                self.label_dict[list_label[i]]=list_item
            except:
                data=[]
                data.append(i)
                self.label_dict[list_label[i]]=data
        self.ori_label=dict(self.label_dict)
        self.current_set_len = len(self.labelList)
    def __len__(self):        
        return self.current_set_len

    def generate_data(self):
        for class_name in self.label_dict.keys():
            list_image=self.label_dict[class_name]
            count=len(list_image)
            remain=self.imageList[list_image[0]]
            remain=np.expand_dims(remain, axis=0)
            while(count<5000):
                for i in range(len(list_image)):
                    count+=1
                    img=self.imageList[list_image[i]]
                    img=np.expand_dims(img, axis=0)
                    print(self.imageList.shape)
                    remain=np.concatenate((remain,img),axis=0)
                    self.labelList.append(class_name)
                    if(count>=5000):
                        break
            self.imageList=np.concatenate((self.imageList,remain),axis=0)
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.fromarray(curImage)
        curImage1 = self.transform(curImage)
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if(self.set_name=="train"):
            label=self.labelList[idx]
            list_label=self.label_dict[label]
            idx_pos=random.randint(0,len(list_label)-1)
            new_pos=list_label[idx_pos]
            curImage_pos = self.imageList[new_pos]
            curImage_pos = PIL.Image.fromarray(curImage_pos)
            if(len(list_label)<=50):
                curImage_pos = self.transform_tail(curImage)
            else:
                curImage_pos = self.transform_tail(curImage)
            return [curImage1,curImage_pos],curLabel
        else:
            return curImage1,curLabel

class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01,data_type="train", rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        self.data_type=data_type
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def refresh_list_class(self):
        new_dict=dict()
        for key in self.dict_index_class_new.keys():
            new_list=self.dict_index_class_new[key]
            new_dict[key]=new_list
        self.new_dict=new_dict

    def generate_data(self):
        for class_name in self.dict_index_class.keys():
            list_image=self.dict_index_class[class_name]
            count=len(list_image)
            remain=self.data[list_image[0]]
            remain=np.expand_dims(remain, axis=0)
            while(count<5000):
                for i in range(len(list_image)):
                    count+=1
                    img=self.data[list_image[i]]
                    img=np.expand_dims(img, axis=0)
                    #print(self.data.shape)
                    remain=np.concatenate((remain,img),axis=0)
                    self.targets.append(class_name)
                    if(count>=5000):
                        break
            self.data=np.concatenate((self.data,remain[1:]),axis=0)

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.dict_index_class=dict()
        for i in range(len(self.targets)):
          try:
            self.dict_index_class[self.targets[i]].append(i)
          except:
            self.dict_index_class[self.targets[i]]=[]
            self.dict_index_class[self.targets[i]].append(i)
        if(False):
            self.generate_data()
        self.dict_index_class_new=dict()
        for i in range(len(self.targets)):
            try:
                self.dict_index_class_new[self.targets[i]].append(i)
            except:
                self.dict_index_class_new[self.targets[i]]=[]
                self.dict_index_class_new[self.targets[i]].append(i)
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def __getitem__(self, idx):   
        curImage = self.imageList[idx]
        curLabel =  np.asarray(self.labelList[idx])
        curImage = PIL.Image.fromarray(curImage.transpose(1,2,0))
        curImage1 = self.transform(curImage)
        curLabel = torch.from_numpy(curLabel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if(self.set_name=="train"):
            label=self.labelList[idx]
            list_label=self.label_dict[label]
            idx_pos=random.randint(0,len(list_label)-1)
            new_pos=list_label[idx_pos]
            curImage_pos = self.imageList[new_pos]
            curImage_pos = PIL.Image.fromarray(curImage_pos.transpose(1,2,0))
            if(len(list_label)<=50):
                curImage_pos = self.transform_tail(curImage)
            else:
                curImage_pos = self.transform_tail(curImage)
            return [curImage1,curImage_pos],curLabel
        else:
            return curImage1,curLabel

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

full_coreset = ImbalanceCIFAR10(root='data',imb_factor=0.01,data_type="train", train=True,
                                               download=True, transform=train_transform)
new_imgList, new_labelList=full_coreset.data,full_coreset.targets
label_dict=dict()
for i in range(len(new_labelList)):
    try:
        data=label_dict[new_labelList[i]]
        data.append(i)
        label_dict[new_labelList[i]]=data
    except:
        data=[]
        data.append(i)
        label_dict[new_labelList[i]]=data
total=[]
for key in label_dict.keys():
    total.append(len(label_dict[key]))
    print(len(label_dict[key]))
print(total)
# #generate coreset index
coreset_cifar_10=torch.load("coreset_cifar_10.pt")
coreset_cifar_10=coreset_cifar_10.tolist()
full_index=[int(item) for item in coreset_cifar_10]

new_imgList_coreset=new_imgList[full_index]
new_labelList_numpy=np.asarray(new_labelList)
new_label_List_coreset=list(new_labelList_numpy[full_index])
new_imgList_coreset=new_imgList[full_index]
new_labelList_numpy=np.asarray(new_labelList)
new_label_List_coreset=list(new_labelList_numpy[full_index])
setname="train"
train_coreset=CIFAR100LT(
    imageList=new_imgList_coreset, labelList=new_label_List_coreset, labelNames=[],
    set_name=setname, isAugment=setname=='train')
train_coreset_loader=DataLoader(train_coreset,
                                    batch_size=512,
                                    shuffle=True,
                                    pin_memory=False,
                                    drop_last=False,
                                    num_workers=16)            
print("length coreset")
for key in train_coreset.label_dict.keys():
    print(key,len(train_coreset.label_dict[key]))
print(train_coreset.labelList)
print(len(train_coreset.labelList))
