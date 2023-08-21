
import os
import torch
import torch.nn as nn
from scipy.special import comb
from tqdm import tqdm
import torch.nn.functional as F
# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, epoch, args):
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
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, args.epochs, total_top1 / total_num * 100))

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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features,batch_temperature,use_class_temperature, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if(use_class_temperature==True):

            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                batch_temperature)
        else:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
#this is re-center loss
criterition_center=SupConLoss(0.5)

#this function generate class temperature
def generate_temperature_comb(frequent):
    list_temperature=[]
    for keys in frequent:
        temperature=0.1+0.9*comb(len(frequent[keys]),3)/comb(len(frequent[0]),3)
        list_temperature.append(temperature)
    list_temperature_tensor=torch.Tensor(list_temperature)
    list_temperature_tensor=torch.reshape(list_temperature_tensor,(list_temperature_tensor.shape[0],1))
    return list_temperature_tensor
def generate_temperature_comb_frequent(frequent):
    list_temperature=[]
    for keys in frequent:
        #temperature=0.1+0.9*comb(len(frequent[keys]),3)/comb(len(frequent[0]),3)
        temperature=len(frequent[keys])/len(frequent[0])
        list_temperature.append(temperature)
    list_temperature_tensor=torch.Tensor(list_temperature)
    list_temperature_tensor=torch.reshape(list_temperature_tensor,(list_temperature_tensor.shape[0],1))
    return list_temperature_tensor
#calculate tensor for list coreset
def calculate_tensor_center(v1_tensor,v2_tensor,targets_tensor):
    class_list=torch.unique(targets_tensor)
    class_list=class_list.type(torch.LongTensor)
    targets_tensor=torch.reshape(targets_tensor,(targets_tensor.shape[0],))
    v1_class=dict()
    v2_class=dict()
    class_index=dict()
    for i in range(class_list.shape[0]):
        list_index_class=torch.where(targets_tensor==int(class_list[i].item()))
        element_v1=v1_tensor[list_index_class]
        element_v2=v2_tensor[list_index_class]
        v1_class[i]=element_v1
        v2_class[i]=element_v2
        class_index[i]=list_index_class
    list_center_v1=[]
    for _,val in v1_class.items():
        list_center_v1.append(torch.mean(val,axis=0).unsqueeze(0))
    center_tensor_v1=torch.cat(list_center_v1,axis=0)
    center_tensor_v1=nn.functional.normalize(center_tensor_v1, dim=1)
    list_center_v2=[]
    for _,val in v2_class.items():
        list_center_v2.append(torch.mean(val,axis=0).unsqueeze(0))
    center_tensor_v2=torch.cat(list_center_v2,axis=0)
    center_tensor_v2=nn.functional.normalize(center_tensor_v2, dim=1)
    targets_tensor=targets_tensor.type(torch.LongTensor)
    return center_tensor_v1,center_tensor_v2