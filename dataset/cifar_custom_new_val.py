import random
import torchvision
import os
import pickle
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
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
        # np.random.shuffle(classes)
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
        for key in self.dict_index_class_new.keys():
            print(len(self.dict_index_class_new[key]))
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        label=self.targets[index]
        if self.transform is not None:
          if(self.data_type=="train"):
            im_1 = self.transform(img)
            list_index=self.dict_index_class_new[label]
            list_index_pos=random.sample(list_index,3)
            pos_1=self.transform(Image.fromarray(self.data[list_index_pos[0]]))
            pos_2=self.transform(Image.fromarray(self.data[list_index_pos[1]]))
            pos_3=self.transform(Image.fromarray(self.data[list_index_pos[2]]))
            im_2 = self.transform(img)
            return [im_1,im_2],[pos_1,pos_2,pos_3],label
          else:
            im_1=self.transform(img)
            return im_1,label

class ImbalanceCIFAR100(ImbalanceCIFAR10):
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


class SemiSupervisedImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10
    unlabel_size_factor = 5

    def __init__(self, root, imb_type='exp', imb_factor=1, unlabel_imb_factor=1,
                 rand_number=0, train=True, transform=None, target_transform=None, download=False):
        super(SemiSupervisedImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        # unlabeled
        self.unlabeled_data = os.path.join(root, 'ti_80M_selected.pickle')  # selected data from 80M-TI
        self.unlabeled_pseudo = os.path.join(root, 'pseudo_labeled_cifar.pickle')  # pseudo-label using model trained on imbalanced data
        self.imb_factor = imb_factor
        self.unlabel_imb_factor = unlabel_imb_factor
        self.num_per_cls_dict = dict()

        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        img_num_list_unlabeled = self.get_img_num_per_cls_unlabeled(self.cls_num, img_num_list, unlabel_imb_factor)
        self.gen_imbalanced_data(img_num_list, img_num_list_unlabeled)

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

    def get_img_num_per_cls_unlabeled(self, cls_num, labeled_img_num_list, imb_factor):
        img_unlabeled_total = np.sum(labeled_img_num_list) * self.unlabel_size_factor
        img_first_min = img_unlabeled_total // cls_num
        img_num_per_cls_unlabel = []
        for cls_idx in range(cls_num):
            num = img_first_min * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls_unlabel.append(int(num))
        factor = img_unlabeled_total / np.sum(img_num_per_cls_unlabel)
        img_num_per_cls_unlabel = [int(num * factor) for num in img_num_per_cls_unlabel]
        print(f"Unlabeled est total:\t{img_unlabeled_total}\n"
              f"After processing:\t{np.sum(img_num_per_cls_unlabel)},\t{img_num_per_cls_unlabel}")
        return img_num_per_cls_unlabel

    def gen_imbalanced_data(self, img_num_per_cls, img_num_per_cls_unlabeled):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        print(f"Labeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

        # unlabeled dataset
        print("Loading unlabeled data from %s" % self.unlabeled_data)
        with open(self.unlabeled_data, 'rb') as f:
            aux = pickle.load(f)
        aux_data = aux['data']
        aux_truth = aux['extrapolated_targets']
        print("Loading pseudo labels from %s" % self.unlabeled_pseudo)
        with open(self.unlabeled_pseudo, 'rb') as f:
            aux_targets = pickle.load(f)
        aux_targets = aux_targets['extrapolated_targets']

        for the_class, the_img_num in zip(classes, img_num_per_cls_unlabeled):
            # ground truth is only used to select samples
            idx = np.where(aux_truth == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(aux_data[selec_idx, ...])
            # append pseudo-label
            new_targets.extend(aux_targets[selec_idx])
            for pseudo_class in aux_targets[selec_idx]:
                self.num_per_cls_dict[pseudo_class] += 1
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        assert new_data.shape[0] == len(new_targets), 'Length of data & labels do not match!'
        print(f"Unlabeled data extracted:\t{len(new_targets)}")
        for i in range(self.cls_num):
            print(self.num_per_cls_dict[i])

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list