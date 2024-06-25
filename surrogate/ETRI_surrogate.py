import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import snntorch
from snntorch import spikegen

import matplotlib.pyplot as plt
import numpy as np
import itertools
import time
import csv
# from model import *
from src.engines import train, evaluate
from src.utils import accuracy, load_checkpoint, save_checkpoint
from data import *
from sklearn.model_selection import KFold
from spikingjelly.clock_driven import functional

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode,IFNode

from spikingjelly.clock_driven.surrogate import ATan
from spikingjelly.clock_driven.encoding import PoissonEncoder
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BCRNet')
args = parser.parse_args()


class spike_encoding(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(spike_encoding, self).__init__()
        self.conv2d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=8, stride=1, padding='same')
        self.bnorm1 = nn.BatchNorm1d(out_channels)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


  
    def forward(self, x):
        T, B, C, N1 = x.shape

        out = self.conv2d(x.flatten(0,1))
        out = self.bnorm1(out).reshape(T, B, -1, N1)
        spk=  self.lif1(out)
        
        return spk
        
class DSCONV(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        self.dim = dim
        self.proj_conv = nn.Conv1d(2, 16, kernel_size=7, stride=3, padding=3, padding_mode="zeros", bias=True)
        self.proj_bn = nn.BatchNorm1d(16)
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        self.proj_conv1 = nn.Conv1d(16, 8, kernel_size=5, stride = 2, padding=2, padding_mode="zeros", bias=True)
        self.proj_bn1 = nn.BatchNorm1d(8)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        self.fc1_conv = nn.Linear(256, 128)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Linear(128, 64)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
       

    def forward(self, x):
        T, B, C, N1 = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
  
        x = self.proj_bn(x).reshape(T, B, 16, -1).contiguous()
        
        x = self.proj_lif(x).flatten(0,1).contiguous()  
   
        x = self.proj_conv1(x)

        x = self.proj_bn1(x).reshape(T, B, 8, -1).contiguous()
        x = self.proj_lif1(x).contiguous()

        x = x.flatten(2,3)

        x = self.fc1_conv(x.flatten(0,1)).reshape(T,B,-1).contiguous()
        x = self.fc1_lif(x)

        x = self.fc2_conv(x.flatten(0,1)).reshape(T,B,-1).contiguous()
        x = self.fc2_lif(x)

        return x 
class DSCONV2(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        self.dim = dim
        self.proj_conv = nn.Conv1d(2, 8, kernel_size=7, stride=3, padding=3, padding_mode="zeros", bias=True)
        self.proj_bn = nn.BatchNorm1d(8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        self.proj_conv1 = nn.Conv1d(8, 4, kernel_size=5, stride = 2, padding=2, padding_mode="zeros", bias=True)
        self.proj_bn1 = nn.BatchNorm1d(4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

       

    def forward(self, x):
        T, B, C, N1 = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
  
        x = self.proj_bn(x).reshape(T, B, 8, -1).contiguous()
        
        x = self.proj_lif(x).flatten(0,1).contiguous()  
   
        x = self.proj_conv1(x)

        x = self.proj_bn1(x).reshape(T, B, 4, -1).contiguous()
        x = self.proj_lif1(x).contiguous()

        x = x.flatten(2,3)

        return x 
class spike_BCRNet(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=15, dim = 16,                 
                 ):
        super(spike_BCRNet, self).__init__()
        self.num_classes = num_classes    

        # self.encoding = spike_encoding(in_channels=2, out_channels=8, kernal_size=(1,8))
        self.ds = DSCONV(dim = dim)
        self.lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        # classification head
        self.head = nn.Linear(64, 15) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)


    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # T = 100
    
        T,B, _ = x.size()

        x = x.reshape(T,B,2,190)

        
        
        # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1)
        # x = self.encoding(x)
        x = self.ds(x)
        
        # x = self.head(x.mean(0))
        x = self.head(x.flatten(0,1)).reshape(T,B,-1).contiguous()
        x = self.lif1(x)

        x = x.mean(0)

        return x

class spike_BCRNet_S(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=15, dim = 8,                 
                 ):
        super(spike_BCRNet_S, self).__init__()
        self.num_classes = num_classes    

        # self.encoding = spike_encoding(in_channels=2, out_channels=8, kernal_size=(1,8))
        self.ds = DSCONV2(dim = dim)

        # classification head
        self.head = nn.Linear(128, 15) if num_classes > 0 else nn.Identity()
        self.lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.apply(self._init_weights)


    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        # T = 100
    
        T,B, _ = x.size()
        x = x.reshape(T,B,2,190)        
        
        # x = (x.unsqueeze(0)).repeat(T, 1, 1, 1)
        # x = self.encoding(x)
        x = self.ds(x)
        # print(x.size())
        # x = self.head(x.mean(0))
        x = self.head(x.flatten(0,1)).reshape(T,B,-1).contiguous()
        x = self.lif1(x)

        x = x.mean(0)

        return x

df_path    = "./dataset/df.xlsx"
dataset = BodyResponseDataset(df_path, classes=15, transform=torch.tensor)

np.random.seed(31415)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


kfold = KFold(n_splits=5, shuffle=True, random_state=42)
n_epochs = 100
T = 10
dtype = torch.float
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
criterion = nn.CrossEntropyLoss() 
t = [4,8,16,32,64]
T_VA = []
for T in t: 
    f_VA = []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        min_loss = 10
        print('--------------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler  = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        
        trainloader = torch.utils.data.DataLoader(
                        dataset, 
                        batch_size=32, sampler=train_subsampler)
        
        testloader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=32, sampler=test_subsampler)
        if args.model == 'BCRNet':
            model = spike_BCRNet(2,8,15) 
        elif args.model == 'BCRNet_S':
            model  = spike_BCRNet_S(2,8,15)
        model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs * len(trainloader))
        TL = []
        TA = []
        VL = []
        VA = []
        poisson_encoder = PoissonEncoder()
        for epoch in range(n_epochs):
            # print(model)
            model.train()
            train_loss = []
            train_accs = []
            for k,batch in enumerate(trainloader):
                t1 = time.time()
                # A batch consists of image data and corresponding labels.
                eeg,label = batch           

                eeg = eeg.to(torch.float32)

                eeg = spikegen.rate(data = eeg, num_steps=T) 
                # print(eeg.size())
                label = label.to(torch.long)
                pred = model(eeg.to(device))

                loss = criterion(pred.cpu(), label)
                functional.reset_net(model)
                optimizer.zero_grad()

                loss.backward(retain_graph=True)

                optimizer.step()
                train_loss.append(loss.item())
                acc = (pred.argmax(dim=-1) == label.to(device)).float().mean()
                

                train_accs.append(acc)            
                scheduler.step()

            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            

            TL.append(train_loss)
            TA.append(train_acc.cpu().item())
            t2 = time.time()
            print(t2-t1)
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
                
            model.eval()
            valid_loss = []
            valid_accs = []

                # Iterate the validation set by batches.
            for k,batch in enumerate(testloader):
                eeg,label = batch        

                eeg = eeg.to(torch.float32)
                eeg = spikegen.rate(data = eeg, num_steps=T) 
                label = label.to(torch.long)

        
                with torch.no_grad():
                    logits = model(eeg.to(device))            
                    loss = criterion(logits.cpu(), label)
                    functional.reset_net(model)

                acc = (logits.argmax(dim=-1) == label.to(device)).float().mean()               
                
                
                # Record the loss and accuracy.
                valid_loss.append(loss.item())
                valid_accs.append(acc)
                

            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)
            
            if valid_loss < min_loss : 
                # print("min_loss1 : ",min_loss)
                min_loss = valid_loss
                # print("min_loss2 : ",min_loss)
                torch.save(model.state_dict(), f'./trained_model/{args.model}/model_rate_ll{T}_fold{fold}.pth')
            # early_stopping(valid_loss,valid_acc, model)
            VL.append(valid_loss)
            VA.append(valid_acc.cpu().item())
            # Print the information.
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
            print('the Test Acc Best is {}, index is {}'.format(max(VA),VA.index (max(VA))))
            
            with open(f'./result/{args.model}/loss_rate_ll{T}_fold'+str(fold)+'.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([TL,VL])
            with open(f'./result/{args.model}/acc_rate_ll{T}_fold'+str(fold)+'.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows([TA,VA])
        f_VA.append(max(VA))
    print(f"Fold_best_accuracy : {f_VA}")
    T_VA.append(f_VA)
print(f"time_fold_best_accuracy : {T_VA}")