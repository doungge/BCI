from collections import OrderedDict
from typing import Any, List, Tuple
import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torchsummary import summary
import numpy as np

np.random.seed(31415)
torch.manual_seed(4)


spike_grad = surrogate.fast_sigmoid()

# spike_grad = snn.surrogate.atan(alpha=2)



class SpikingDenseNet(nn.Module):
    def __init__(self, num_init_channels=2, init_weights=True):
        super().__init__()
            
        num_init_features = 8

        self.conv1 = nn.Conv1d(1, num_init_features, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm1 = nn.BatchNorm1d(num_init_features)
        self.act1 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        # self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv1d(num_init_features, num_init_features*2,kernel_size=5, padding=2, stride=2, bias=False)
        self.norm2 = nn.BatchNorm1d(num_init_features*2)
        self.act2 = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        # self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        self.classifier = nn.Linear(400, 2)
        self.class_act = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        #features = self.forward_features(x)
        # features = self.features(x)
        features = self.conv1(x)
        features = self.norm1(features)
        features = self.act1(features)

        # features = self.pool1(features)

        features = self.conv2(features)
        features = self.norm2(features)
        features = self.act2(features)

        # features = self.pool2(features)

        out = torch.flatten(features, 1)
        out = self.classifier(out)
        out, mem = self.class_act(out)


        return out, mem #final leaky membrane potential 
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, 0.0, 0.02)
                if m.bias is not None : 
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None : 
                    nn.init.constant_(m.bias, 0.0)
                
if __name__ == "__main__":
    
    tau = 2.0
    num_steps = 4
    #single_step_neuron = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
    model = SpikingDenseNet()
    data = torch.rand(1,100)
    print(summary(model,(1,100)))
    