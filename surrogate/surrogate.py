import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
import itertools

from model import *
from src.engines import train, evaluate
from src.utils import accuracy, load_checkpoint, save_checkpoint



def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
    spk_out, mem_out = net(data)
    spk_rec.append(spk_out)
    mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

def print_batch_accuracy(data, targets, train=False):
    output, _ = forward_pass(net,num_steps,data)

    _, idx = output.sum(dim=0).max(1)

    # print("targets:", targets)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc 

batch_size = 20


data_path='/data/cifar10'
num_epochs = 100
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

train = datasets.CIFAR10(".", train=True, download=True, transform=transform)
test = datasets.CIFAR10(".", train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)



spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 50

net = spiking_densenet121(3)
# print(net)
net.to(device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = nn.CrossEntropyLoss() 

test_loss_hist = []
counter = 0
training_acc = []
test_acc = []
test_acc_hist = []

loss_hist = []
acc_hist = []
torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):

    print("EPOCH : ", epoch)
    train_batch = iter(train_loader)
    avg_loss = 0.0
    avg_acc = 0.0
    avg_test_acc = 0.0
    avg_test_loss= 0.0

    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        
        net.train()
        spk_rec, mem_rec = forward_pass(net, num_steps, data)
        print(spk_rec[49])
        print(targets)
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for j in range(num_steps):
            loss_val += loss_fn(spk_rec[j], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        # loss_val.backward(retain_graph=True)
        loss_val.backward()
        optimizer.step()

        avg_loss += loss_val.item()
        # Store loss history for future plotting
        avg_acc += print_batch_accuracy(data, targets, train=True)
        # Test set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_spk, test_mem = forward_pass(net,num_steps,test_data)

            # Test set loss
            test_loss = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                test_loss += loss_fn(test_mem[step], test_targets)
            avg_test_loss += test_loss.item()
            avg_test_acc += print_batch_accuracy(test_data, test_targets, train=False)
   
            # print every 25 iterations
            if i % 10 == 0:
                acc = print_batch_accuracy(test_data, test_targets, train=False)
                print(f"Epoch {epoch}, Iteration {i} \nSingle mini batch test acc: {acc*100:.2f}%")

    loss_hist.append(avg_loss/len(train_batch))
    training_acc.append(avg_acc/len(train_batch))
    test_loss_hist.append(avg_test_loss/len(train_batch))
    avg_acc = avg_test_acc/len(train_batch)
    test_acc_hist.append(avg_acc)
    print(f"Epoch {epoch}, Iteration {i} \nAvg train acc: {avg_acc*100:.2f}%")
