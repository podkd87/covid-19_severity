import pandas as pd
import os
import re
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
import warnings
# from pandas.api.types import CategoricalDtype
import pickle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import time
import wandb
import torch.optim as optim
import random
from util import accuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, auc, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import WeightedRandomSampler

class MyDataset(Dataset):
    def __init__(self, masking_value, data_split, padded_hosday, data_pr, prior_d):
        self.pr = data_pr
        self.masking_value = masking_value
        self._load_data = data_split
        self.pad_hosday = padded_hosday
        self.prior_d = prior_d #n일뒤 일을 예측 
        self.data = self.make_dataset()
        
    def __len__(self):
        len1 = len(self.data[0])
        return len1
    
    def _get_each_data(self, data):
        vital_set = []
        lab_set = []
        baseline = []
        outcome = []

        att_vital_set = []
        att_lab_set = []
        baseline_set = []
        each_att_lab_set = []
        each_att_vital_set = []
        each_att_base_set = []
        
        for data_num in range(int(len(data)*self.pr)):
            hos_day = data[data_num][1].shape[0]
            if data[data_num][3].size==0:
                continue

            for i in range(hos_day-self.prior_d):
                base_i = torch.tensor(data[data_num][0][:i+1], dtype=torch.float32)
                lab_i = torch.tensor(data[data_num][1][:i+1], dtype=torch.float32)
                vital_i = torch.tensor(data[data_num][2][:i+1], dtype=torch.float32)

                att_vital = (vital_i !=vital_i).all(axis=2)  # attention map: True =nan of all vital ;; False=present value
                att_vital[att_vital.all(axis=-1),0] = torch.zeros(1, dtype=torch.bool)
                each_att_vital = (vital_i !=vital_i)
                each_att_vital = each_att_vital.type(torch.float32)

                att_lab = (lab_i !=lab_i).all(axis=1) # attention map: True =nan of all vital ;; False=present value
                att_lab[att_lab.all(axis=-1),0] = torch.zeros(1, dtype=torch.bool)
                each_att_lab = (lab_i !=lab_i)
                each_att_lab = each_att_lab.type(torch.float32)

                each_att_base = (base_i !=base_i) # attention map: True =nan of all vital ;; False=present value
                each_att_base = each_att_base.type(torch.float32)

                pad_3d = (0,0,0,0,0,self.pad_hosday-(i+1)) # pad by (0, 1), (2, 1), and (3, 3) 
                pad_2d = (0,0,0,self.pad_hosday-(i+1))
                vital = F.pad(vital_i, pad_3d,"constant", self.masking_value) #until ith 
                lab = F.pad(lab_i, pad_2d, "constant", self.masking_value) #until ith 

                each_att_vital = F.pad(each_att_vital, pad_3d, "constant", 1) #1 means the loci will be masked, 3dim 

                att_vital = F.pad(att_vital, pad_2d, "constant", 1) #1 means the loci will be masked, 2dim 
                each_att_lab = F.pad(each_att_lab, pad_2d, "constant", 1) #1 means the loci will be masked, 2dim 

                att_lab = F.pad(att_lab, (0,self.pad_hosday-(i+1)), "constant", 1) #1 means the loci will be masked, 1dim 

                vital = vital.nan_to_num(self.masking_value) #fill nan with masking value
                lab = lab.nan_to_num(self.masking_value) #fill nan with masking value
                base = base_i.nan_to_num(self.masking_value) #fill nan with masking value

                vital_set.append(vital)
                lab_set.append(lab)
                baseline_set.append(base)

                att_vital_set.append(att_vital)
                each_att_vital_set.append(each_att_vital)

                att_lab_set.append(att_lab)
                each_att_lab_set.append(each_att_lab)

                each_att_base_set.append(each_att_base)


            outcome+= data[data_num][3][self.prior_d:].tolist()
        
        start = time.time()
        vital_set = torch.stack(vital_set)
        lab_set = torch.stack(lab_set)
        baseline_set = torch.stack(baseline_set)
        each_att_vital_set = torch.stack(each_att_vital_set)
        each_att_lab_set = torch.stack(each_att_lab_set)
        each_att_base_set = torch.stack(each_att_base_set)
        att_vital_set = torch.stack(att_vital_set)
        att_lab_set = torch.stack(att_lab_set)
        current = time.time()
        # print("time:", current-start, "stack complete")

        vital_set = vital_set.reshape(vital_set.size(0),vital_set.size(1)*vital_set.size(2),vital_set.size(3))
        each_att_vital_set = each_att_vital_set.reshape(each_att_vital_set.size(0),each_att_vital_set.size(1)*each_att_vital_set.size(2),each_att_vital_set.size(3))
        att_vital_set = att_vital_set.reshape(att_vital_set.size(0),att_vital_set.size(1)*att_vital_set.size(2))
        outcome = torch.tensor(outcome)
        current1 = time.time()
        # print("time:", current1-current,"dataset completely loaded")

        return (vital_set, lab_set, baseline_set, each_att_vital_set, each_att_lab_set, each_att_base_set, 
                att_vital_set, att_lab_set, outcome)
    
    def make_dataset(self):
        data = self._load_data
        # print("load completed")
        whole_data = self._get_each_data(data)
        # print("whole data is completely loaded")
        return whole_data
    
    def __getitem__(self, index):
        vital_set = self.data[0][index]
        lab_set = self.data[1][index]
        baseline_set = self.data[2][index]
        each_att_vital_set = self.data[3][index]
        each_att_lab_set = self.data[4][index]
        each_att_base_set = self.data[5][index]
        att_vital_set = self.data[6][index]
        att_lab_set = self.data[7][index]
        outcome = self.data[8][index]
        return (vital_set, lab_set, baseline_set, each_att_vital_set, each_att_lab_set, each_att_base_set, 
                att_vital_set, att_lab_set, outcome)
    
def split(file_name, random_s=None):
    with (open(file_name, "rb")) as openfile:
        data = pickle.load(openfile)
    # 해당 부분에서 stratified로 random하게 splitting을 하여 kflod validation을 할 수 있게함. 
    max_severity = list()
    for i in range(len(data)):
        max_severity.append(data[i][3].max().astype(int))
    max_severity = np.array(max_severity)
    random_list = list(range(len(data)))

    strati_train = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state = random_s)
    strati_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state = random_s)

    a = strati_train.split(random_list, max_severity)
    train, va1 = next(a)
    b = strati_test.split(va1, max_severity[va1])
    valid, test = next(b)

    # train_list = random_list[:int(len(data)*0.6)]
    # valid_list = random_list[int(len(data)*0.6):int(len(data)*0.8)]
    # test_list = random_list[int(len(data)*0.8):]


    train_set = [data[i] for i in train]
    valid_set = [data[i] for i in valid]
    test_set = [data[i] for i in test]
    return train_set, valid_set, test_set
    
class Transformer_park(nn.Module):

    def __init__(self, n_each_lab, n_embed, nhead, nhid, nlayers,  dropout=0.2, mask_on = False):
        super(Transformer_park, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.mask_on = mask_on
        self.pos_encoder = PositionalEncoding(n_embed, dropout)
        encoder_layers = TransformerEncoderLayer(n_embed, nhead, nhid, dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.initencoder = nn.Linear(n_each_lab, n_embed)
        self.n_embed = n_embed
        self.init_weights()
        if self.mask_on:
            self.init_mask_encoder = nn.Linear(n_each_lab, n_each_lab)
            self.init_mask_masking = nn.Linear(n_each_lab, n_each_lab)
        
    def init_weights(self):
        initrange = 0.1
        self.initencoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, mask, each_mask):
        if self.mask_on:
            src = self.init_mask_encoder(src) + self.init_mask_masking(each_mask)
            src = self.initencoder(src) * math.sqrt(self.n_embed)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src, src_key_padding_mask=mask)

        else:
            src = self.initencoder(src) * math.sqrt(self.n_embed)
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, dropout=0.1,  max_len = 1500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer_pretrain(nn.Module):

    def __init__(self, n_each_lab, n_embed):
        super(Transformer_pretrain, self).__init__()
        self.decoder = nn.Linear(n_embed, n_each_lab)
        
    def forward(self, src):
        output = self.decoder(src)
        return output

class base_model(nn.Module):
    def __init__(self, n_each_base, dropout=0.2, mask_on=False):
        super(base_model, self).__init__()
        self.n_each = n_each_base
        self.mask_on = mask_on
        self.dropout = dropout
        self.first = nn.Linear(self.n_each, self.n_each)
        self.second = nn.Sequential(nn.Linear(self.n_each, 256),
                                    nn.Dropout(self.dropout),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                    nn.Dropout(self.dropout),
                                     nn.ReLU(),
                                     nn.Linear(128,30))
        
    def encoder(self, base, mask):
        base_ = self.first(base)
        mask = self.first(mask)
        if self.mask_on:
            base_ = base_+ mask
        base = nn.Dropout(self.dropout)(base)
        base_ = nn.ReLU()(base)
        base_ += base
        return base_
    
    def forward(self, base, mask):
        base_ = self.encoder(base, mask)
        out = self.second(base_)
        return out
    
class base_decoder(nn.Module):
    def __init__(self, n_each_base):
        super(base_decoder, self).__init__()
        self.n_each = n_each_base
        self.decoder = nn.Sequential(nn.Linear(30,128),
                                   nn.ReLU(),
                                   nn.Linear(128,256),
                                   nn.ReLU(),
                                   nn.Linear(256, self.n_each))
    def forward(self, base):
        base_ = self.decoder(base)
        return base_
    
class hierachy_model(nn.Module):
    def __init__(self, vital_shape, lab_shape, batch_size, output_len, dropout=0.2):
        super(hierachy_model, self).__init__()
        self.batchsize = batch_size
        self.dropout = dropout
        self.vital_m = nn.Sequential(nn.Linear(vital_shape, 60),
                                    nn.Dropout(self.dropout),
                                     nn.ReLU(),
                                     )
        self.lab_m = nn.Sequential(nn.Linear(lab_shape,60),
                                   nn.Dropout(self.dropout),
                                     nn.ReLU())
        self.total_m = nn.Sequential(nn.Linear(150,60),
                                   nn.Dropout(self.dropout),
                                     nn.ReLU(),
                                     nn.Linear(60,output_len)
                                    )
    def shape_change(self, vital_tensor, lab_tensor, base_tensor):
        # if vital_tensor.shape[1]!=1000:
        #     print("vital_tensor shape is ",vital_tensor.shape)
        vital_sq = vital_tensor.transpose(0,1).reshape(self.batchsize, -1)
        lab_sq = lab_tensor.transpose(0,1).reshape(self.batchsize,-1)
        base_sq = base_tensor.squeeze()
        return vital_sq, lab_sq, base_sq
    
    def forward(self, vital_tensor, lab_tensor, base_tensor):
        vital_sq, lab_sq, base_sq = self.shape_change(vital_tensor, lab_tensor, base_tensor)
        vital_sq_out = self.vital_m(vital_sq)
        lab_sq_out = self.lab_m(lab_sq)
        total = torch.cat((vital_sq_out, lab_sq_out, base_sq),1)
        out = self.total_m(total)
        return out
    
def eval_model(result, outcome):
    result, outcome = result.cpu(), outcome.cpu()
    soft = nn.Softmax(dim=1)
    v_predict = torch.argmax(result,axis=1)
    val_acc1, val_acc2 = accuracy(result, outcome, topk=(1,2))
    val_acc1 = val_acc1.item()
    val_acc2 = val_acc2.item()
    
    one_hot_y = nn.functional.one_hot(outcome)
    roc_weight_val = roc_auc_score(one_hot_y, soft(result).detach().numpy(),  multi_class= 'ovr', average = 'weighted')
    roc_mirco_val = roc_auc_score(one_hot_y, soft(result).detach().numpy(),  multi_class= 'ovr', average = 'micro')
    return val_acc1, val_acc2, roc_weight_val, roc_mirco_val

def masking_eachvalue(input_data, pre_masked, masking_value, mask_ratio = 0.2):
    N, L, D = input_data.shape  # batch, length, dim
    if input_data.shape[1] >1: #데이터가 시계열 데이터인경우
        noise = torch.rand(N, L, D, device=input_data.device)  # noise in [0, 1]    
        len_keep = int(L * (1 - mask_ratio))

    else: #데이터가 baseline인경우
        noise = torch.rand(N, D, device=input_data.device)  # noise in [0, 1]
        len_keep = int(D * (1 - mask_ratio))
    
        # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    if input_data.shape[1] >2:
        mask = torch.ones([N, L, D], device=input_data.device)
    else:
        mask = torch.ones([N, D], device=input_data.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    if input_data.shape[1]>1:
        mask2d = mask.all(axis=-1)
        # not to make all value be masked
        exam = pre_masked.clone().detach()
        exam[mask2d==1]=True
        no_mask = exam.all(axis=-1)
        mask2d[no_mask==True]=False
        
        masked_input = input_data.clone().detach()
        masked_input[mask==True]=masking_value
    else:
        mask = mask.unsqueeze(1)
        masked_input = input_data.clone().detach()
        masked_input[mask==True]=masking_value
    return masked_input

def train(train_loader, b_model, vital_model, lab_model, decoder_vital,
          decoder_lab, decoder_base, total_model, optimizer, weight, 
          masking_value, mae, masking_ratio, i = 1):
    # """one epoch training"""
    b_model.train()
    vital_model.train()
    lab_model.train()
    total_model.train()
    decoder_vital.train()
    decoder_lab.train()
    decoder_base.train()
    total_loss = 0.

    out_t_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    mse_criter_vital = nn.MSELoss()
    mse_criter_lab = nn.MSELoss()
    mse_criter_base = nn.MSELoss()
    
    for batch, batch_data in enumerate(train_loader):
        vital_set = batch_data[0].cuda(i)
        lab_set = batch_data[1].cuda(i)
        baseline_set = batch_data[2].cuda(i)
        each_att_vital_set = batch_data[3].cuda(i)
        each_att_lab_set = batch_data[4].cuda(i)
        each_att_base_set = batch_data[5].cuda(i)
        att_vital_set = batch_data[6].cuda(i)
        att_lab_set = batch_data[7].cuda(i)
        outcome = batch_data[8]
        outcome = outcome.type(torch.LongTensor).cuda(i)

        #masked input
        if mae ==True:
            empty_mask = torch.zeros_like(baseline_set.squeeze(1))
            masked_vital = masking_eachvalue(vital_set, att_vital_set, masking_value, masking_ratio)
            masked_lab = masking_eachvalue(lab_set, att_lab_set, masking_value, masking_ratio)
            masked_base = masking_eachvalue(baseline_set, empty_mask, masking_value, masking_ratio)

            vital_tensor = vital_model(masked_vital.transpose(0,1), att_vital_set, 
                                       each_mask = each_att_vital_set.transpose(0,1))
            lab_tensor = lab_model(masked_lab.transpose(0,1), att_lab_set, 
                                   each_mask = each_att_lab_set.transpose(0,1))
            #base model은 transformer가 아니므로 transpose가 없음
            base_tensor = b_model(masked_base, each_att_base_set) 
            # print("working MAE")
        else:
            vital_tensor = vital_model(vital_set.transpose(0,1), att_vital_set, 
                                       each_mask = each_att_vital_set.transpose(0,1))
            lab_tensor = lab_model(lab_set.transpose(0,1), att_lab_set, 
                                   each_mask = each_att_lab_set.transpose(0,1))
            base_tensor = b_model(baseline_set, each_att_base_set)

        vital_out = decoder_vital(vital_tensor)  #MLM용도
        lab_out = decoder_lab(lab_tensor)    #MLM 용도
        base_out = decoder_base(base_tensor) #MLM 용도

        #compute loss
        result = total_model(vital_tensor, lab_tensor, base_tensor)
        vital_loss = mse_criter_vital(vital_out, vital_set.transpose(0,1))
        lab_loss = mse_criter_lab(lab_out, lab_set.transpose(0,1))
        base_loss = mse_criter_base(base_out, baseline_set)
        outcome_loss = criterion(result, outcome) #outcome loss는 기타 reconstruction loss와 합쳐진다. 
        
        t_loss = outcome_loss + weight * vital_loss + weight*lab_loss + weight*base_loss
        # Optimizer 
        optimizer.zero_grad()
        t_loss.backward()
        optimizer.step()
        
        total_loss += t_loss.item()
        out_t_loss += outcome_loss.item()
    
    cur_loss = total_loss/len(train_loader)
    cur_loss2 = out_t_loss/len(train_loader)
    wandb.log({"train total loss":cur_loss,
              "train outcome loss":cur_loss2})    
    return cur_loss

def validate(valid_loader, b_model, vital_model, lab_model, decoder_vital,
          decoder_lab, decoder_base, total_model, optimizer, weight, i = 1):
    # 아래의 function call에서 criterion불러옴 
    
    b_model.eval()
    vital_model.eval()
    lab_model.eval()
    total_model.eval()
    
    decoder_vital.eval()
    decoder_lab.eval()
    decoder_base.eval()
    total_loss = 0.

    out_t_loss = 0.
    criterion = torch.nn.CrossEntropyLoss()
    mse_criter_vital = nn.MSELoss()
    mse_criter_lab = nn.MSELoss()
    mse_criter_base = nn.MSELoss()
    
    for batch, batch_data in enumerate(valid_loader):
        vital_set = batch_data[0].cuda(i)
        lab_set = batch_data[1].cuda(i)
        baseline_set = batch_data[2].cuda(i)
        each_att_vital_set = batch_data[3].cuda(i)
        each_att_lab_set = batch_data[4].cuda(i)
        each_att_base_set = batch_data[5].cuda(i)
        att_vital_set = batch_data[6].cuda(i)
        att_lab_set = batch_data[7].cuda(i)
        outcome = batch_data[8]
        outcome = outcome.type(torch.LongTensor).cuda(i)

        vital_tensor = vital_model(vital_set.transpose(0,1), att_vital_set, each_mask = each_att_vital_set.transpose(0,1))
        lab_tensor = lab_model(lab_set.transpose(0,1), att_lab_set, each_mask = each_att_lab_set.transpose(0,1))
        base_tensor = b_model(baseline_set, each_att_base_set) #base model은 transformer가 아니므로 transpose가 없음

        vital_out = decoder_vital(vital_tensor)  #MLM용도
        lab_out = decoder_lab(lab_tensor)    #MLM 용도
        base_out = decoder_base(base_tensor) #MLM 용도

        #compute loss
        result = total_model(vital_tensor, lab_tensor, base_tensor)
        vital_loss = mse_criter_vital(vital_out, vital_set.transpose(0,1))
        lab_loss = mse_criter_lab(lab_out, lab_set.transpose(0,1))
        base_loss = mse_criter_base(base_out, baseline_set)
        outcome_loss = criterion(result, outcome) 
        
        t_loss = outcome_loss + weight * vital_loss + weight*lab_loss + weight*base_loss
        
        total_loss += t_loss.item()
        out_t_loss += outcome_loss.item()
        
        if batch==0:
            result_t = result.cpu().detach()
            outcome_t =outcome.cpu().detach()
        else:
            result_t = torch.cat([result.cpu(), result_t],axis=0).detach()
            outcome_t = torch.cat([outcome.cpu(), outcome_t],axis=0).detach()
    
    val_acc1, val_acc2, roc_weight_val, roc_mirco_val = eval_model(result_t, outcome_t)
    
    val_loss = total_loss/len(valid_loader)
    val_loss2 = out_t_loss/len(valid_loader)
    wandb.log({"val total loss":val_loss,
               "val outcome loss":val_loss2,
              "top1 accuracy":val_acc1,
              # "top2 accuracy":val_acc2,
              "auroc_weight":roc_weight_val,
              "auroc_micro":roc_mirco_val})    
    
    return val_loss, val_loss2, roc_weight_val

def weighted_sampling(train_dataset, valid_dataset, batch_size):
    train_label = torch.tensor([each_data[8].int().item() for each_data in train_dataset])

    sample_weight = 1/train_label.bincount()
    sample_weight/=sample_weight.sum()
    sample_weights = torch.tensor([sample_weight[i] for i in train_label])
    sample_weights = sample_weights.double()
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    valid_label = torch.tensor([each_data[8].int().item() for each_data in valid_dataset])

    val_sample_weight = 1/valid_label.bincount()
    val_sample_weight/=val_sample_weight.sum()
    val_sample_weights = torch.tensor([val_sample_weight[i] for i in valid_label])
    val_sample_weights = val_sample_weights.double()
    val_sampler = WeightedRandomSampler(val_sample_weights, len(val_sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=val_sampler, drop_last = True)
    return train_loader, valid_loader

def def_model(n_each_base, n_each_vital, n_embed, nhead, nhid, nlayers, dropout, 
              vital_shape, lab_shape, batch_size, output_len, mask_on, loaded_model):
    b_model = base_model(n_each_base, mask_on=mask_on)
    vital_model = Transformer_park(n_each_vital, n_embed, nhead, nhid, nlayers,  dropout, mask_on = mask_on )
    lab_model = Transformer_park(n_each_lab, n_embed, nhead, nhid, nlayers,  dropout, mask_on = mask_on )
    decoder_vital = Transformer_pretrain(n_each_vital, n_embed)
    decoder_lab = Transformer_pretrain(n_each_lab, n_embed)
    decoder_base = base_decoder(n_each_base)
    total_model = hierachy_model(vital_shape, lab_shape, batch_size, output_len)

    vital_model.load_state_dict(loaded_model['vital_model_dict'])
    lab_model.load_state_dict(loaded_model['lab_model_dict'])
    b_model.load_state_dict(loaded_model['b_model_dict'])
    total_model.load_state_dict(loaded_model['total_model_dict'])
    decoder_vital.load_state_dict(loaded_model['decoder_vital_dict'])
    decoder_lab.load_state_dict(loaded_model['decoder_lab_dict'])
    decoder_base.load_state_dict(loaded_model['decoder_base_dict'])
    
    vital_model.eval()
    lab_model.eval()
    b_model.eval()
    total_model.eval()
    decoder_vital.eval()
    decoder_lab.eval()
    decoder_base.eval()
    
    return vital_model, lab_model, b_model, total_model, decoder_vital, decoder_lab, decoder_base

def test_result(test_loader, vital_model, lab_model, b_model, total_model, 
                decoder_vital, decoder_lab, decoder_base):
    for batch, batch_data in enumerate(test_loader):
        vital_set = batch_data[0]
        lab_set = batch_data[1]
        baseline_set = batch_data[2]
        each_att_vital_set = batch_data[3]
        each_att_lab_set = batch_data[4]
        each_att_base_set = batch_data[5]
        att_vital_set = batch_data[6]
        att_lab_set = batch_data[7]
        outcome = batch_data[8]
        outcome = outcome.type(torch.LongTensor)

        vital_tensor = vital_model(vital_set.transpose(0,1), att_vital_set,
                                   each_mask = each_att_vital_set.transpose(0,1))
        lab_tensor = lab_model(lab_set.transpose(0,1), att_lab_set,
                               each_mask = each_att_lab_set.transpose(0,1))
        base_tensor = b_model(baseline_set, each_att_base_set)

        vital_out = decoder_vital(vital_tensor)  #MLM용도
        lab_out = decoder_lab(lab_tensor)    #MLM 용도
        base_out = decoder_base(base_tensor) #MLM 용도

        #compute loss
        result = total_model(vital_tensor, lab_tensor, base_tensor)

    return result