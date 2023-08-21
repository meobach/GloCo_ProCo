import torch
import torchvision
import torchvision.transforms as transforms
import models
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from dataset.imagenet_custom import *
# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 5, 0.1)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, 1000, total_top1 / total_num * 100))
    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

pretrained_path="moco_ckpt_0200.pth.tar"
pretrained=torch.load(pretrained_path)
pretrained_state_dict=pretrained["state_dict"]
print(pretrained_state_dict.keys())
# for key in pretrained_state_dict.keys():
#     print(key)
model = models.__dict__["resnet50"](num_classes=128)
dim_mlp = model.fc.weight.shape[1]
model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
model=model.cuda()
#print(pretrained["state_dict"].keys())
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
dict_ori=dict(model.state_dict())
#print(dict_ori.keys())
for key in dict_ori.keys():
    try:
        # if("fc" in key ):   
        new_key="module.encoder_q."+key
        dict_ori[key]=pretrained_state_dict[new_key]
    except:
        print(key)
    # else:
    #     print(key)
model.load_state_dict(dict_ori)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
# augmentation_train =transforms.Compose( [
#     transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
#     ], p=0.8),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])
augmentation_test = transforms.Compose([
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize])])
root="/data/ImageNet/ILSVRC/Data/CLS-LOC/"
txt_train = 'ImageNet_LT_train.txt'
txt_test='ImageNet_LT_test.txt'
memory_set=ImageNetLT(root, txt_train,type="test" ,transform=augmentation_test)
test_set=ImageNetLT(root,txt_test,type="test", transform=augmentation_test)
batch_size=256
workers=32
memory_loader = torch.utils.data.DataLoader(
memory_set, batch_size=batch_size, shuffle=False,
num_workers=workers, pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
test_set, batch_size=batch_size, shuffle=False,
num_workers=workers, pin_memory=True, drop_last=False)
test(model,memory_loader,test_loader,0)
