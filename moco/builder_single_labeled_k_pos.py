# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F

# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        
    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)
            outcome = nn.functional.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split, 
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return nn.functional.batch_norm(
                input, self.running_mean, self.running_var, 
                self.weight, self.bias, False, self.momentum, self.eps)



class MoCo(nn.Module):
    def __init__(self, base_encoder,dim=128, K=4096, m=0.99, T=0.1, symmetric=False,mlp=True):
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.frequent=0
        self.num_positive=6
        self.class_temperature= nn.Parameter(torch.load("frequent_temperature_imagenet.pt"), requires_grad=True)
        #self.class_temperature= torch.load("frequent_temperature.pt")
        #self.class_temperature=self.class_temperature.cuda()
        # create the encoders
        print("check mlp")
        print(mlp)
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.register_buffer("queue_labels", -torch.ones(1, K).long())
        self.queue = nn.functional.normalize(self.queue, dim=0)
        

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder`
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    #this function generate class temperature
    def generate_temperature_frequent(self):
        #list_temperature=[]
        max_value=torch.max(self.class_temperature)
        min_value=torch.min(self.class_temperature)
        with torch.no_grad():
            max_value=torch.max(self.class_temperature)
        #max_value=max_value.item()
            for i in range(self.class_temperature.shape[0]):
                self.class_temperature[i]=0.05+0.1*(self.class_temperature[i]-min_value)/max_value
#             list_temperature.append(temperature)
#         list_temperature_tensor=torch.Tensor(list_temperature)
#         list_temperature_tensor=torch.reshape(list_temperature_tensor,(list_temperature_tensor.shape[0],1))
#         return list_temperature_tensor.cuda()
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys,labels):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        self.queue_labels[:, ptr:ptr + batch_size] = labels.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    def contrastive_loss(self, q, im_k,im_labels,epoch):
        im_labels = im_labels.cuda(non_blocking=True)
        #print(self.class_temperature[:5])
        # # compute query features
        # q = self.encoder_q(im_q)  # queries: NxC
        # q = nn.functional.normalize(q, dim=1)  # already normalized
        targets=torch.reshape(im_labels,(im_labels.shape[0],))
        targets=targets.type(torch.LongTensor)
        bsz = targets.shape[0]
        #got temperature for each class
        class_temperature=self.class_temperature[targets]
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # positive logits from queue
        im_labels = im_labels.contiguous().view(-1, 1)
        mask = torch.eq(im_labels, self.queue_labels.clone().detach()).float()
        mask_pos_view = torch.zeros_like(mask)
        if self.num_positive > 0:
            for i in range(self.num_positive):
                all_pos_idxs = mask.view(-1).nonzero().view(-1)
                num_pos_per_anchor = mask.sum(1)
                num_pos_cum = num_pos_per_anchor.cumsum(0).roll(1)
                num_pos_cum[0] = 0
                rand = torch.rand(mask.size(0), device=mask.device)
                idxs = ((rand * num_pos_per_anchor).floor() + num_pos_cum).long()
                idxs = idxs[num_pos_per_anchor.nonzero().view(-1)]
                sampled_pos_idxs = all_pos_idxs[idxs.view(-1)]
                mask_pos_view.view(-1)[sampled_pos_idxs] = 1
                mask.view(-1)[sampled_pos_idxs] = 0
        else:
            mask_pos_view = mask.clone()
        mask_pos_view_class = mask_pos_view.clone()
        mask_pos_view_class[:, self.queue_labels.size(1):] = 0
        mask_pos_view = torch.cat([torch.ones([mask_pos_view.shape[0], 1]).cuda(), mask_pos_view], dim=1)
        mask_pos_view_class = torch.cat([torch.ones([mask_pos_view_class.shape[0], 1]).cuda(), mask_pos_view_class], dim=1)
        # apply temperature
        #logits1=logits/class_temperature
        #logits /=self.T
        log_prob = F.normalize((logits/self.T).exp(), dim=1, p=1).log()
#         if(epoch%2==0):
        loss1 = - torch.sum((mask_pos_view_class * log_prob).sum(1) / mask_pos_view.sum(1)) / mask_pos_view.shape[0]
#         else:
        #loss2 = nn.CrossEntropyLoss().cuda()(logits/class_temperature, labels)
        loss=loss1#+ loss2)/2
        self._dequeue_and_enqueue(k,im_labels)
        
        
        return loss, q, k

    def forward(self, im1, im2,label,epoch):
#         if(self.frequent==1000):
#             self.frequent=0
#             self.generate_temperature_frequent()
#             print("update temperature")
        self.generate_temperature_frequent()
        #print(self.class_temperature[:5],self.class_temperature[-5:])
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2,label)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1,label)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            q=im1
            # q = self.encoder_q(im1)  # queries: NxC
            # q = nn.functional.normalize(q, dim=1)  # already normalized
            all_loss=[]
            all_k=[]
            for item in im2:
                if(epoch%2==0):
                    loss,_,k=self.contrastive_loss(q,item,label,epoch)
                else:
                    loss,_,k=self.contrastive_loss(q,item,label,epoch)
                all_loss.append(loss)
                all_k.append(k)
            loss=sum(all_loss)/len(all_loss)
            
            
            #loss, q, k = self.contrastive_loss(im1, im2)

        

        return loss,q,all_k

    
