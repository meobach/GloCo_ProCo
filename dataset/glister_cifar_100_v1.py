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

def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num
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


def gen_imbalanced_data(img_num_per_cls, imgList, labelList):
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    new_data = []
    new_targets = []
    targets_np = np.array(labelList, dtype=np.int64)
    classes = np.unique(targets_np)
    # np.random.shuffle(classes)  # remove shuffle in the demo fair comparision
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        #np.random.shuffle(idx) # remove shuffle in the demo fair comparision
        selec_idx = idx[:the_img_num]
        new_data.append(imgList[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return (new_data, new_targets)


import random
class CIFAR100LT(Dataset):
    def __init__(self, set_name='train', imageList=[], labelList=[], labelNames=[], isAugment=True):
        self.isAugment = isAugment
        self.set_name = set_name
        self.labelNames = labelNames
        if self.set_name=='train':            
            self.transform_tail = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.8, 1.)),
                transforms.RandomHorizontalFlip(),
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
            #curImage2=self.transform(curImage)
            return [curImage1,curImage_pos],curLabel
        else:
            return curImage1,curLabel

path_to_DB = './data'
if not os.path.exists(path_to_DB): 
    os.makedirs(path_to_DB)
_ = torchvision.datasets.CIFAR100(root=path_to_DB, train=True, download=True)
nClasses=100
imb_type = 'exp' # samling long-tailed training set with an exponetially-decaying function
imb_factor = 0.01 # imbalance factor = 100 = 1/0.01
path_to_DB = path.join(path_to_DB, 'cifar-100-python')

datasets = {}
dataloaders = {}

setname = 'meta'
with open(os.path.join(path_to_DB, setname), 'rb') as obj:
    labelnames = pickle.load(obj, encoding='bytes')
    labelnames = labelnames[b'fine_label_names']
for i in range(len(labelnames)):
    labelnames[i] = labelnames[i].decode("utf-8") 
    
import math
setname = 'train'
with open(os.path.join(path_to_DB, setname), 'rb') as obj:
    DATA = pickle.load(obj, encoding='bytes')
imgList = DATA[b'data'].reshape((DATA[b'data'].shape[0],3, 32,32))
labelList = DATA[b'fine_labels']
total_num = len(labelList)
img_num_per_cls = get_img_num_per_cls(nClasses, total_num, imb_type, imb_factor)
new_imgList, new_labelList = gen_imbalanced_data(img_num_per_cls, imgList, labelList)
datasets[setname] = CIFAR100LT(
    imageList=new_imgList, labelList=new_labelList, labelNames=labelnames,
    set_name=setname, isAugment=setname=='train')
print('#examples in {}-set:'.format(setname), datasets[setname].current_set_len)
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
for key in label_dict.keys():
    print(len(label_dict[key]))

# #generate coreset img
rare_class_index=[*range(9901,10847,1)]
frequent_class_index=[343, 102, 95, 28, 391, 310, 416, 148, 50, 165, 442, 361, 70, 381, 210, 27, 231, 419, 422, 326, 55, 449, 386, 226, 256, 86, 342, 424, 160, 328, 490, 125, 472, 306, 431, 313, 263, 301, 8, 484, 413, 188, 17, 430, 353, 56, 204, 305, 269, 197, 810, 502, 884, 790, 896, 671, 668, 964, 771, 814, 759, 707, 973, 505, 956, 564, 692, 595, 879, 625, 897, 967, 588, 801, 856, 589, 723, 645, 847, 507, 605, 678, 917, 608, 572, 649, 712, 699, 735, 641, 623, 544, 512, 748, 900, 942, 521, 802, 1315, 1291, 1418, 1158, 1366, 1018, 1110, 1275, 1425, 981, 1337, 1120, 1117, 995, 1086, 1274, 1004, 1379, 1176, 1084, 1289, 1411, 1197, 1331, 1377, 997, 1125, 1323, 1181, 1322, 1153, 1375, 1207, 1159, 1324, 1261, 1106, 1297, 1236, 1340, 1142, 1397, 1338, 1112, 1095, 1184, 1502, 1864, 1652, 1683, 1459, 1627, 1599, 1547, 1785, 1662, 1792, 1587, 1769, 1814, 1721, 1469, 1718, 1786, 1706, 1608, 1784, 1749, 1592, 1659, 1561, 1856, 1524, 1693, 1687, 1804, 1566, 1556, 1457, 1586, 1482, 1865, 1739, 1533, 1826, 1818, 1558, 1858, 1611, 1661, 2208, 2171, 1947, 2017, 2140, 1932, 2273, 2256, 2185, 1949, 2058, 2021, 1917, 1927, 2262, 2131, 2248, 2090, 2035, 2157, 1955, 2259, 2116, 1990, 2224, 2161, 1884, 2197, 2032, 2006, 1939, 2233, 2045, 2174, 2252, 2194, 1953, 2008, 2069, 1880, 2177, 1967, 2558, 2668, 2373, 2343, 2467, 2570, 2651, 2288, 2567, 2355, 2439, 2600, 2498, 2593, 2502, 2451, 2308, 2649, 2330, 2383, 2526, 2322, 2659, 2284, 2429, 2393, 2495, 2515, 2540, 2412, 2319, 2320, 2302, 2607, 2368, 2444, 2631, 2432, 2450, 2625, 2750, 2892, 2901, 3035, 3041, 2726, 2867, 2996, 3030, 3045, 2916, 3011, 2763, 3046, 2970, 2785, 2798, 2831, 2847, 2990, 2884, 2804, 2846, 2830, 2872, 2954, 2967, 2968, 2994, 2765, 2914, 2839, 2803, 2684, 2682, 2829, 2930, 2882, 3351, 3171, 3255, 3397, 3318, 3389, 3340, 3069, 3088, 3221, 3259, 3107, 3215, 3168, 3100, 3191, 3333, 3250, 3317, 3355, 3291, 3274, 3303, 3359, 3237, 3220, 3310, 3066, 3338, 3059, 3292, 3401, 3116, 3155, 3212, 3186, 3289, 3717, 3431, 3578, 3546, 3523, 3471, 3577, 3651, 3426, 3726, 3500, 3565, 3561, 3482, 3652, 3697, 3441, 3626, 3488, 3458, 3517, 3520, 3714, 3693, 3719, 3491, 3747, 3613, 3439, 3430, 3650, 3674, 3646, 3555, 3658, 4055, 4038, 4041, 3947, 4061, 4031, 3821, 3925, 4000, 3861, 3887, 3808, 3845, 4013, 4084, 3767, 3954, 3942, 3983, 4011, 3869, 3862, 3955, 4066, 4079, 3855, 3896, 3901, 3885, 3995, 3799, 3943, 4060, 4163, 4183, 4295, 4202, 4276, 4110, 4194, 4294, 4367, 4096, 4358, 4307, 4380, 4375, 4376, 4215, 4292, 4326, 4283, 4274, 4267, 4103, 4299, 4142, 4165, 4263, 4388, 4216, 4379, 4262, 4341, 4344, 4612, 4566, 4674, 4455, 4472, 4593, 4473, 4405, 4403, 4406, 4636, 4461, 4544, 4443, 4631, 4519, 4517, 4694, 4476, 4533, 4404, 4666, 4438, 4695, 4600, 4584, 4634, 4419, 4496, 4464, 4926, 4767, 4723, 4702, 4947, 4874, 4960, 4941, 4746, 4798, 4778, 4837, 4906, 4850, 4788, 4902, 4802, 4879, 4730, 4729, 4984, 4894, 4762, 4897, 4875, 4704, 4842, 4796, 4937, 5072, 5186, 5222, 5131, 5133, 5157, 5090, 5259, 5123, 5189, 5029, 4987, 5119, 5006, 5121, 5247, 5167, 5071, 5215, 5051, 5002, 5039, 4997, 5224, 5016, 5220, 5243, 5153, 5324, 5352, 5418, 5368, 5468, 5452, 5387, 5503, 5495, 5404, 5484, 5377, 5290, 5299, 5279, 5451, 5411, 5328, 5267, 5513, 5443, 5336, 5312, 5454, 5446, 5476, 5694, 5645, 5578, 5737, 5607, 5524, 5621, 5677, 5647, 5573, 5752, 5596, 5671, 5756, 5589, 5649, 5689, 5719, 5522, 5765, 5552, 5531, 5727, 5717, 5764, 5815, 5966, 5954, 5989, 5930, 5908, 5782, 5816, 5933, 5861, 5991, 5896, 5805, 5890, 5958, 5893, 5837, 5998, 5899, 5934, 5774, 5862, 5924, 5798, 6097, 6030, 6133, 6007, 6125, 6069, 6162, 6146, 6154, 6079, 6150, 6195, 6045, 6036, 6228, 6134, 6219, 6187, 6074, 6120, 6063, 6018, 6209, 6403, 6289, 6242, 6261, 6388, 6270, 6264, 6283, 6421, 6265, 6417, 6378, 6348, 6344, 6257, 6404, 6315, 6318, 6410, 6374, 6243, 6407, 6580, 6585, 6491, 6649, 6632, 6548, 6559, 6600, 6636, 6598, 6611, 6468, 6618, 6568, 6451, 6562, 6571, 6521, 6626, 6625, 6599, 6747, 6801, 6728, 6699, 6689, 6773, 6690, 6669, 6702, 6763, 6839, 6722, 6831, 6849, 6670, 6751, 6836, 6790, 6772, 6721, 6908, 6901, 6971, 7007, 6978, 6875, 6931, 6924, 7017, 6874, 6932, 6963, 6943, 6855, 6936, 7010, 7008, 6973, 6866, 7183, 7157, 7133, 7099, 7199, 7213, 7080, 7126, 7112, 7185, 7216, 7174, 7148, 7098, 7162, 7143, 7089, 7215, 7262, 7263, 7287, 7217, 7379, 7372, 7293, 7329, 7337, 7237, 7344, 7357, 7281, 7366, 7284, 7341, 7354, 7260, 7488, 7508, 7547, 7476, 7522, 7472, 7457, 7466, 7500, 7543, 7438, 7408, 7460, 7435, 7405, 7518, 7411, 7648, 7588, 7705, 7558, 7677, 7674, 7562, 7552, 7598, 7625, 7627, 7559, 7668, 7658, 7671, 7576, 7725, 7834, 7818, 7789, 7796, 7847, 7843, 7827, 7751, 7833, 7716, 7722, 7769, 7798, 7768, 7877, 7951, 7960, 7900, 7954, 7914, 7977, 7993, 7887, 7976, 7881, 7991, 7879, 7893, 7928, 8060, 8005, 8022, 8130, 8114, 8032, 8121, 8024, 8113, 8082, 8059, 8132, 8025, 8057, 8239, 8234, 8256, 8261, 8137, 8163, 8189, 8190, 8251, 8169, 8215, 8202, 8156, 8302, 8347, 8376, 8340, 8315, 8276, 8326, 8280, 8279, 8381, 8313, 8353, 8365, 8459, 8494, 8440, 8478, 8403, 8458, 8469, 8496, 8483, 8418, 8420, 8465, 8535, 8545, 8517, 8540, 8554, 8566, 8595, 8527, 8510, 8604, 8534, 8611, 8704, 8647, 8631, 8619, 8663, 8661, 8717, 8660, 8653, 8652, 8618, 8780, 8814, 8808, 8810, 8797, 8747, 8722, 8779, 8734, 8733, 8768, 8909, 8920, 8853, 8863, 8850, 8838, 8829, 8913, 8896, 8911, 8987, 8930, 8943, 9002, 9013, 8985, 8970, 8963, 8961, 8974, 9045, 9021, 9077, 9032, 9050, 9023, 9087, 9098, 9100, 9116, 9110, 9165, 9163, 9111, 9108, 9161, 9118, 9135, 9207, 9215, 9216, 9237, 9189, 9264, 9234, 9244, 9257, 9306, 9291, 9299, 9335, 9307, 9346, 9341, 9315, 9385, 9407, 9402, 9367, 9365, 9375, 9384, 9396, 9427, 9442, 9447, 9456, 9433, 9477, 9455, 9551, 9524, 9533, 9491, 9498, 9499, 9532, 9566, 9611, 9613, 9593, 9568, 9586, 9559, 9660, 9678, 9665, 9629, 9668, 9667, 9666, 9719, 9721, 9694, 9732, 9683, 9725, 9795, 9782, 9786, 9758, 9771, 9785, 9849, 9817, 9819, 9824, 9802, 9812, 9856, 9866, 9877, 9880, 9896, 9861]
full_index=frequent_class_index+rare_class_index
count=0
print(new_labelList)
while(count<90):
  class_name=0
  flag1=False
  flag2=False
  for i in range(len(new_labelList)):
    #print(i)
    if(new_labelList[i]==class_name and i not in full_index):
      if(flag1==False):
        flag1=True
      else:
        if(flag2==False):
          flag2=True
        else:
          class_name+=1
          flag1=False
          flag2=False
          continue
      full_index.append(i)
      print(class_name,i)
      count+=1
      #print(count)
    if(count>89):
      break

new_imgList_coreset=new_imgList[full_index]
new_labelList_numpy=np.asarray(new_labelList)
new_label_List_coreset=list(new_labelList_numpy[full_index])
new_imgList_coreset=new_imgList[full_index]
new_labelList_numpy=np.asarray(new_labelList)
new_label_List_coreset=list(new_labelList_numpy[full_index])
train_coreset=CIFAR100LT(
    imageList=new_imgList_coreset, labelList=new_label_List_coreset, labelNames=labelnames,
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
