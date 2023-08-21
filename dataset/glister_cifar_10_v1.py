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
                transforms.RandomResizedCrop(32, scale=(0.8, 1.)),
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
rare_class_index=[*range(11515,12406,1)]
frequent_class_index=[6862, 10383, 10329, 5397, 10128, 10404, 10162, 10759, 6333, 10438, 7928, 10631, 10304, 5789, 5484, 6059, 5898, 6223, 10795, 5496, 7849, 7611, 5075, 6331, 7819, 6827, 6700, 5134, 7120, 7554, 7990, 5131, 6063, 5656, 5089, 5931, 5083, 5264, 6234, 7060, 6411, 6858, 7912, 7350, 7146, 6716, 6822, 5040, 6547, 6483, 7019, 5049, 7960, 10613, 7856, 7832, 6536, 7441, 7117, 6404, 6895, 6976, 6705, 5378, 7774, 10599, 6562, 5348, 7501, 6961, 7251, 10836, 5708, 5449, 6647, 7242, 6193, 7334, 5008, 7062, 5284, 10754, 10357, 5396, 5182, 6024, 6480, 5742, 5161, 7962, 6561, 6284, 6272, 5906, 7584, 7653, 5778, 7370, 10574, 7383, 5318, 5588, 10084, 5678, 5929, 6437, 7519, 7734, 6934, 7286, 10156, 7380, 6601, 5993, 7674, 7219, 6499, 6761, 11001, 4376, 5537, 3167, 9653, 2503, 8071, 1286, 8503, 10643, 2093, 6479, 10075, 1009, 5405, 878, 10042, 7503, 10323, 8078, 10499, 11170, 315, 10324, 11223, 8902, 10999, 10289, 2677, 11508, 3597, 10325, 187, 8608, 645, 9156, 3203, 9567, 2400, 6003, 4018, 11496, 9047, 10598, 9583, 4346, 8678, 10250, 9081, 2914, 9179, 11047, 9734, 11388, 614, 8605, 10136, 11383, 9059, 10321, 10950, 9097, 10961, 9817, 676, 10242, 9297, 2386, 6417, 8677, 10436, 9371, 10648, 4943, 9363, 10706, 11459, 10309, 2406, 10228, 9344, 11200, 8465, 10960, 8238, 2624, 8259, 11165, 1036, 8056, 2818, 8181, 4392, 8111, 11301, 4158, 9071, 10485, 8684, 11049, 7436, 4259, 8723, 3836, 9198, 11355, 1481, 6574, 10622, 4273, 11289, 8518, 10172, 8713, 1581, 9100, 1769, 8439, 11230, 8546, 10542, 9077, 9801, 11305, 10239, 8037, 4767, 10366, 657, 8595, 10088, 9008, 10024, 6609, 9700, 2836, 8311, 4627, 8209, 4982, 8883, 9841, 11453, 10678, 8962, 10435, 10934, 10394, 699, 11384, 10843, 9144, 10344, 9191, 4910, 9414, 10691, 9497, 944, 10810, 8652, 10669, 8109, 10210, 1499, 9988, 9765, 10316, 9001, 10359, 9119, 11148, 10308, 1908, 10071, 5411, 8345, 3389, 8667, 10212, 4719, 11056, 8728, 10636, 376, 10914, 8686, 3762, 8394, 10181, 9622, 11487, 8045, 11019, 338, 9520, 10112, 8637, 2662, 8261, 11410, 8041, 10649, 11256, 10090, 9233, 10416, 1155, 10130, 9089, 9975, 3441, 9370, 10675, 3153, 10379, 2166, 10587, 6828, 8779, 11266, 10426, 4465, 10360, 3163, 9814, 11216, 8643, 239, 10975, 9845, 9060, 4260, 8578, 1278, 8600, 750, 10019, 626, 8274, 10826, 217, 10993, 2363, 8567, 4834, 9572, 10845, 11235, 8937, 10235, 4025, 9837, 9940, 11402, 10140, 8878, 4166, 8079, 10365, 9359, 11101, 1800, 9407, 11027, 10552, 2696, 6728, 8432, 10978, 8762, 10654, 11245, 9726, 1457, 1597, 9873, 8777, 10583, 504, 10091, 8332, 90, 9257, 10605, 9350, 11315, 10517, 1396, 9136, 10899, 9202, 5137, 10922, 10021, 8425, 4078, 8506, 10094, 1063, 11138, 8485, 1067, 9023, 10266, 8936, 11435, 10774, 8124, 10686, 9563, 10740, 9203, 9893, 462, 10243, 8703, 10300, 1774, 9827, 11000, 9334, 11330, 1595, 9650, 3979, 8113, 10333, 1706, 11233, 9928, 2186, 10287, 11501, 10462, 1186, 7486, 3320, 9695, 713, 9392, 11260, 10687, 11295, 10707, 8632, 10840, 9355, 2199, 9905, 8702, 10772, 11308, 10493, 8042, 10824, 11110, 9917, 8898, 3629, 8159, 10593, 2940, 5379, 10976, 118, 4855, 8053, 4407, 10907, 10685, 11302, 8516, 10224, 4813, 10861, 8814, 9819, 8668, 9879, 8764, 2300, 10670, 1468, 8748, 517, 7078, 10197, 9431, 11219, 9892, 9186, 10403, 8562, 3090, 8128, 10459, 855, 11352, 9974, 9998, 11470, 8239, 11070, 8588, 2889, 10911, 9474, 4095, 8158, 5166, 10735, 3567, 8657, 10412, 9333, 5456, 9895, 11380, 194, 9548, 1169, 8499, 7942, 10799, 9539, 11354, 9330, 10351, 3295, 9894, 11201, 8391, 1160, 7184, 11082, 8227, 2526, 9660, 1202, 9741, 1735, 9204, 2358, 8970, 3551, 5741, 9338, 3663, 10047, 8887, 2911, 6178, 11358, 9153, 10791, 8242, 1911, 11114, 8504, 2555, 9675, 10157, 11092, 8633, 1223, 9393, 1393, 8613, 9875, 11067, 8257, 11040, 9852, 9715, 10910, 8265, 408, 9206, 3381, 6711, 4663, 10952, 10566, 5629, 9478, 10302, 8746, 833, 10245, 9672, 11078, 8381, 10126, 11268, 10592, 8803, 10665, 9635, 10839, 6798, 4048, 8043, 8587, 3878, 11033, 10254, 9504, 11113, 721, 8461, 528, 8840, 10905, 1489, 9637, 11013, 9647, 20, 10183, 11488, 10758, 3279, 8515, 10953, 6757, 306, 6300, 2496, 9983, 11419, 8369, 11426, 10258, 2012, 10370, 9519, 11432, 4288, 7855, 2695, 10077, 8283, 10948, 9212, 11112, 8724, 10260, 11025, 8688, 11499, 8288, 11479, 9719, 1058, 8507, 4431, 5143, 8988, 2971, 8033, 11068, 10690, 11231, 8945, 663, 8255, 6485, 9117, 10056, 206, 8298, 5928, 8771, 10347, 8383, 10656, 2843, 8730, 11309, 9240, 4599, 8410, 10963, 8823, 2313, 9115, 3229, 10828, 9267, 5859, 10886, 2077, 10699, 7860, 11251, 1507, 8330, 3294, 10072, 6560, 9770, 3789, 11180, 10488, 1924, 9558, 10198, 11190, 8508, 7744, 1469, 8968, 11469, 10013, 577, 10779, 2136, 9607, 4810, 10549, 2065, 11280, 9348, 10028, 810, 11214, 9602, 4707, 8044, 11443, 10817, 11062, 8023, 10059, 8999, 9980, 3221, 6198, 9948, 9345, 10166, 8171, 2033, 7699, 10078, 9501, 11221, 4417, 11464, 8926, 10567, 11258, 3717, 11345, 9235, 7620, 2944, 6195, 4209, 9434, 10815, 8173, 10900, 9559, 10883, 9561, 11179, 8842, 578, 10937, 8125, 511, 7846, 10808, 9649, 10524, 1237, 8935, 3779, 5918, 11413, 10185, 10935, 3527, 8100, 10154, 11281, 9028, 10039, 3579, 9041, 7148, 8352, 10222, 8536, 11509, 10305, 11208, 393, 10570, 9745, 3464, 4633, 9973, 1056, 9842, 8188, 10368, 11511, 8543, 9885, 11336, 4447, 10376, 9768, 10221, 8705, 1119, 11196, 9429, 10315, 11491, 9626, 4230, 11458, 8272, 2146, 8013, 6653, 3590, 8320, 1933, 10559, 8373, 11063, 8505, 10558, 8453, 6452, 9438, 9862, 2903, 10851, 10973, 11158, 6962, 9422, 11119, 10322, 8799, 10461, 11265, 953, 8846, 11273, 8851, 9914, 4293, 10177, 8776, 10399, 9022, 11224, 9444, 3463, 8494, 10361, 8002, 4727, 6459, 11367, 4371, 10278, 11428, 9881, 4168, 11161, 9510, 11339, 8810, 10650, 9226, 2437, 9303, 10951, 10263, 8947, 3619, 10000, 8180, 11404, 10698, 9710, 6222, 10602, 11044, 11462, 9939, 8783, 550, 8694, 11263, 10847, 9757, 4944, 10282, 8486, 10821, 8168, 11457, 8683, 9932, 8260, 11271, 4790, 9252, 4294, 8804, 11471, 2384, 5381, 11123, 2712, 10516, 5887, 11130, 8773, 232, 10132, 8440, 1881, 8101, 11467, 9918, 9484, 10453, 9279, 10225, 11346, 9966, 10970, 9286, 273, 9903, 9482, 10822, 8687, 11169, 10464, 2619, 9621, 10125, 8418, 10479, 8493, 2822, 7903, 10881, 4329, 8086, 11014, 6930, 9073, 2693, 9324, 10536, 1890, 10788, 801, 10205, 1616, 9912, 6696, 10500, 9094, 11149, 934, 10401, 11485, 9176, 11108, 10449, 10994, 8765, 5464, 3645, 10751, 2974, 10882, 8712, 10520, 1571, 9748, 11244, 8622, 3588, 11026, 10906, 8984, 10717, 8946, 9878, 523, 9483, 9972, 1234, 10979, 9976, 8630, 602, 8217, 3506, 11005, 8975, 9877, 11338, 4738, 8401, 11353, 9958, 9188, 10064, 2510, 10712, 3470, 10454, 8106, 3135, 5114, 10048, 4701, 11371, 7125, 8098, 10472, 5390, 1991, 11100, 5729, 3266, 10620, 11203, 8788, 1888, 11405, 8280, 9937, 9325, 11222, 9806, 9416, 11414, 10606, 3788, 10248, 9667, 10400, 11141, 2795, 9902, 5173, 4501, 6613, 10768, 1987, 11440, 10348, 8599, 10651, 3906, 11107, 9465, 10137, 8759, 1962, 5128, 10681, 9364, 4991, 10579, 11334, 8927, 10518, 9387, 10332, 11220, 4154, 10270, 8992, 10034, 11473, 9970, 8003, 11498, 10671, 6435, 8379, 2891, 10226, 8647, 10424, 8392, 10702, 9685, 4368, 8960, 11369, 10296, 9791, 8514, 4830, 11331, 10017, 9261, 4754, 11081, 10568, 9492, 9967, 7643, 10895, 10171, 8862, 10007, 8857, 3731, 6400, 2357, 9332, 9887, 11173, 9876]
full_index=frequent_class_index+rare_class_index

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
