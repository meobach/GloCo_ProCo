import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import random
RGB_statistics = {
    'ImageNet': {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
}


def get_data_transform(split, rgb_mean, rbg_std):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rbg_std)
        ])
    }
    return data_transforms[split]


class ImageNetLT(Dataset):
    
    def __init__(self, root, txt,type=None, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.type=type
        print(self.type)
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        print(len(self.labels))
        self.classes=[]
        self.dict_index_class=dict()
        for i in range(len(self.labels)):
          try:
            self.dict_index_class[self.labels[i]].append(i)
          except:
            self.dict_index_class[self.labels[i]]=[]
            self.dict_index_class[self.labels[i]].append(i)
        for key,val in self.dict_index_class.items():
            self.classes.append(key)
            #print(key,len(val))
        self.targets=self.labels
    def __len__(self):
        return len(self.labels)
    
    def get_cls_num_list(self):
        number_per_class=[]
        for i in range(1000):
            number_per_class.append(len(self.dict_index_class[i]))
        return number_per_class
    
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        # if self.transform is not None:
        #     sample = self.transform(sample)
        if(self.type=="train"):
            # list_index=self.dict_index_class[label]
            # list_index_pos=random.sample(list_index,3)
            # path_pos1 = self.img_path[list_index_pos[0]]
            # path_pos2 = self.img_path[list_index_pos[1]]
            # path_pos3 = self.img_path[list_index_pos[2]]
            # with open(path_pos1, 'rb') as f:
            #     pos_1 = self.transform(Image.open(f).convert('RGB'))
            # with open(path_pos2, 'rb') as f:
            #     pos_2 = self.transform(Image.open(f).convert('RGB'))
            # with open(path_pos3, 'rb') as f:
            #     pos_3 = self.transform(Image.open(f).convert('RGB'))
            sample_view1=self.transform(sample)
            sample_view2=self.transform(sample)
            return [sample_view1,sample_view2],label
        elif(self.type=="test"):
            sample_view=self.transform(sample)
            return sample_view,label
        else:
            sample_view_1=self.transform(sample)
            sample_view_2=self.transform(sample)
            return [sample_view_1,sample_view_2],label
            # pos_1=self.transform(Image.fromarray(self.data[list_index_pos[0]]))
            # pos_2=self.transform(Image.fromarray(self.data[list_index_pos[1]]))
            # pos_3=self.transform(Image.fromarray(self.data[list_index_pos[2]]))
       # return sample, label  # , index

# root = ''
# txt_train = 'tiny_train.txt'
# txt_val='tiny_val.txt'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
# augmentation_test = transforms.Compose([
#             transforms.Compose([
#                 transforms.Resize(48),
#                 transforms.CenterCrop(32),
#                 transforms.ToTensor(),
#                 normalize])])
# memory_set=ImageNetLT(root, txt_train,type="test" ,transform=augmentation_test)

# memory_loader = torch.utils.data.DataLoader(
# memory_set, batch_size=256, shuffle=False,
# num_workers=16, pin_memory=True, drop_last=True)