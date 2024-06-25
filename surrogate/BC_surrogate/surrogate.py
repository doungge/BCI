import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

from model.model2 import *

from sklearn.model_selection import KFold
from tqdm import tqdm
from src.prepro import CustomDataset
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns



# from src.utils import accuracy, load_checkpoint, save_checkpoint
np.random.seed(31415)
torch.manual_seed(4)


def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net
  
#   data = snn.spikegen.latency(data, num_steps=num_steps,normalize =True)
  data = snn.spikegen.rate(data, num_steps=num_steps)

  for step in range(num_steps):
    spk_out, mem_out = net(data[step])
    spk_rec.append(spk_out)
    mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

def print_batch_accuracy(data, targets, train=False):
    output, _ = forward_pass(net,num_steps,data)

    idx = output.sum(dim=0)

    idx = output.sum(dim=0).argmax(-1)

    acc = np.mean((targets == idx).detach().cpu().numpy())

    idx_np = idx.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # recall = recall_score(targets_np, idx_np)
    # precision = precision_score(targets_np, idx_np)
    # f1 = f1_score(targets_np, idx_np)

    # print("recall : ", recall)
    # print("precision : ", precision)
    # print("f1 : ", f1)


    return acc, idx_np, targets_np





data_path1='./data/fake_100_off_0.csv'
data_path2='./data/real_100_off_0.csv'
fake = pd.read_csv(data_path1)
real = pd.read_csv(data_path2)

concan_df = pd.concat([real, fake], ignore_index=True)

x,y = concan_df.drop(columns=['Var1']), concan_df['Var1']

std = StandardScaler()


dataset = CustomDataset(x.values, y.values-1) 


dtype = torch.float
device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


k_folds    = 5
num_epochs = 10
results_train = pd.DataFrame(columns=["0", "1", "2" ,"3" ,"4"], index=range(num_epochs))
results_valid = pd.DataFrame(columns=["0", "1", "2" ,"3" ,"4"], index=range(num_epochs))

kfold   = KFold(n_splits=k_folds, shuffle=True)
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
# Create DataLoaders
    print(f'FOLD {fold}')
    print('--------------------------------')
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler  = torch.utils.data.SubsetRandomSampler(test_ids)

    train_loader = torch.utils.data.DataLoader(dataset, 
                        batch_size=1000, sampler=train_subsampler)
    test_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=1000, sampler=test_subsampler)



    spike_grad = surrogate.fast_sigmoid(slope=25)
    beta = 0.5
    num_steps = 4

    net = SpikingDenseNet()
    
    net.to(device)
    net.init_weights()
 
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
    loss_fn = nn.CrossEntropyLoss() 

    test_loss_hist = []
    counter = 0
    training_acc = []
    test_acc = []
    test_acc_hist = []

    loss_hist = []
    acc_hist = []
    # torch.autograd.set_detect_anomaly(True)

    for epoch in tqdm(range(0, num_epochs), total=num_epochs):

        train_batch = len(train_loader)
        avg_loss = 0.0
        avg_acc_train = 0.0
        avg_test_acc = 0.0
        avg_test_loss= 0.0

        for i, (data, targets) in enumerate(train_loader):
            
            data = data.to(device)
            data = data.to(dtype=torch.float32)
            data = data.view(-1,1,100)
            targets = targets.to(device)
            net.train()
            spk_rec, mem_rec = forward_pass(net, num_steps, data)
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for j in range(num_steps):
                loss_val += loss_fn(spk_rec[j], targets.to(dtype=torch.long))

            # Gradient calculation + weight update
            optimizer.zero_grad()
            # loss_val.backward(retain_graph=True)
            loss_val.backward()
            optimizer.step()

            avg_loss += loss_val.item()
            # Store loss history for future plotting
            acc,  _, _ =  print_batch_accuracy(data, targets.to(dtype=torch.long), train=True)
            avg_acc_train += acc
            # Test set

            all_predictions = []
            all_targets = []
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_data = test_data.to(dtype=torch.float32)
                test_data = test_data.view(-1,1,100)
                test_targets = test_targets.to(device)


                # Test set forward pass
                test_spk, test_mem = forward_pass(net,num_steps,test_data)

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss_fn(test_spk[step], test_targets.to(dtype=torch.long))
                avg_test_loss += test_loss.item()
                test_acc, test_idx, test_target = print_batch_accuracy(test_data, test_targets.to(dtype=torch.long), train=False)
                all_predictions.extend(test_idx)
                all_targets.extend(test_target)
                avg_test_acc += test_acc
    

        loss_hist.append(avg_loss/train_batch)
        print(f"Epoch {epoch}, Iteration {i} \nAvg train acc: {(avg_acc_train/train_batch)*100:.2f}%")
        training_acc.append(avg_acc_train/train_batch)
        test_loss_hist.append(avg_test_loss/train_batch)
        avg_acc = avg_test_acc/train_batch
        test_acc_hist.append(avg_acc)
        print(f"Epoch {epoch}, Iteration {i} \nAvg test acc: {avg_acc*100:.2f}%")


        