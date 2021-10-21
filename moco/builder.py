# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np
import scipy
torch.random.manual_seed(0)
from sklearn import decomposition
pca = decomposition.PCA(n_components=128)
from scipy.spatial import distance
import random, os, math

#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
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
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        #print('Keys:',keys.shape)
        batch_size = keys.shape[0]
        #print("Batch size", batch_size)
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
 
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr



    @torch.no_grad()
    def _dequeue_and_enqueue_mined_keys(self, keys):
        # gather keys before updating queue
#        keys = concat_all_gather(keys)
        #print('Enque:',keys.shape)
        #batch_size = keys.shape[0]
        #print("Batch size", batch_size)
        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + 128] = keys.T
        ptr = (ptr + 128) % self.K  # move pointer
                                          
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def getHullPoints(self, q):
    
        center = q
        # torch.matmul(q, self.queue.clone().detach()).detach().sum(axis=0)
        # dist = torch.matmul(center.detach().cpu().numpy(), arr)
        
        # print(center.shape, arr.shape)
        print("Center is", center)
        # if(np.isnan(center.detach().cpu().numpy()).all()):
        #     return None

        # if(np.isnan(center).all()):
        #     return None
        # print(center.shape, arr[0].shape)
        # raise SystemError
        arr = self.queue.clone().detach().numpy().T
        print(arr[0].shape)
        #dist = [distance.euclidean(center.detach().cpu().numpy().reshape(-1), np.clip(x, np.amin(x), np.amax(x)) ) for x in arr]
        dist = [distance.euclidean(center.reshape(-1), np.clip(x, np.amin(x), np.amax(x)) ) for x in arr]
        # print(len(dist), arr.shape)    #1000, [1000,128]
        
        point_dict = {}
        for i in range(arr.shape[0]):
          point_dict[ dist[i] ] = arr[i] 
        sorted(point_dict.keys(), reverse = True)
        print(len(point_dict))  #105  
        sorted_keys = np.asarray([point_dict[k] for k in point_dict.keys()])
        
        # sorted_keys = [x for _,x in sorted(zip(dist, arr ), reverse = True)
        comp = min(128, len(point_dict))
        pca = decomposition.PCA(n_components=comp)
        #print(sorted_keys.shape)
        pca.fit(sorted_keys)
        princ_comp = pca.components_
        print(" Length PCA ",len(princ_comp))
        
        comp = len(princ_comp)
        if(comp<10):
            return None
        
        mined_negatives = np.ndarray(shape=(1024,128))
        

        for i in range(64):
          for j in range(16):
            x = center + princ_comp[j]*i*0.01     
            index = 16*(i-1) + j-1                                                         
            mined_negatives[index:index+1, :] = x
        # print(sorted_keys.shape)
        # print(sorted_keys)
        return mined_negatives






        #------------------------------------------------------------------------
        #Conic combinations generation using the PCA components
        #Sampling 64 points from the cone
        # num_conic_comb = 8
        # mined_negatives = np.ndarray(shape=(128 * num_conic_comb, 128))
        # conic_combination_matrix = np.absolute( np.ndarray(shape=(128,2)) )
        #Code for conic combinations
        # for i in range(num_conic_comb):
        #   comp1 = princ_comp[random.randint(0,comp)]
        #   comp2 = princ_comp[random.randint(0,comp)]
        #   comp3 = princ_comp[random.randint(0,comp)]

        #   vec1 = center.detach().cpu().numpy() + comp1
        #   vec2 = center.detach().cpu().numpy() + comp2*0.01
        #   vec3 = center.detach().cpu().numpy() + comp3*0.01

        #   p1 = vec1 + vec2
        #   p2 = vec1 + vec3

        #   points = np.vstack([p1,p2])
        #   mined = np.dot(conic_combination_matrix, points) 
        #   mined_negatives[i*128:(i+1)*128, :] = mined
          #-----------------------------------------------------------------------

        return mined_negatives







    def forward(self, im_q, im_k , epoch, i, gpu):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        

        if epoch > 10:
          
        #   sort_scores = torch.matmul(q, self.queue.clone().detach()).detach().sum(axis=0)
        #   sorted_keys = self.queue.clone().detach().T
        #   #sorted_keys = [x for _,x in sorted(zip(sort_scores,self.queue.clone().detach().T), reverse = True)]
        #   #---------------------------------------------------------
        #   #Sort the keys from the queue  
        #   sorted_dict = {}
        #   for i in range(sort_scores.shape[0]):
        #       sorted_dict[sort_scores[i]] = sorted_keys[i]
        #   sorted(sorted_dict.keys(), reverse = True)
        #   sorted_keys = [sorted_dict[k] for k in sorted_dict.keys()]
        #   print('Sort Done')
          #----------------------------------------------------------
          #print(type(sorted_keys)) # list
        #   sorted_ = np.zeros(shape=(1000,128))
        #   i=0
        #   for x in sorted_keys[:1000]:
        #     sorted_[i,:] = x.cpu()
        #     i+=1
          num_queries = q.shape[0]
          for j in range(num_queries):
            query = q[j:j+1,:]
            
            hull = self.getHullPoints(query)

            if hull is not None:
                chull = torch.Tensor(hull) 

            print(chull.shape)
            l_neg = torch.einsum('nc,ck->nk', [query, chull.T.clone().detach()])
            
            logits = torch.cat([logits, l_neg], dim=1)
        
        
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            # self.Q_DQ_Helper(




        



        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
