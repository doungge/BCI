import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode,IFNode
from spikingjelly.clock_driven.surrogate import ATan
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial

class spike_encoding(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size):
        super(spike_encoding, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernal_size, stride=(1,1), padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


  
    def forward(self, x):
        T, B, C, N1,N2 = x.shape

        out = self.conv2d(x.flatten(0,1))
        out = self.bnorm1(out).reshape(T, B, -1, N1, N2)
        spk=  self.lif1(out)
        
        return spk
        
class DSCONV(nn.Module):
    def __init__(self, dim):
        super().__init__()        
        self.dim = dim
        self.proj_conv = nn.Conv1d(2, 16, kernel_size=7, stride=3, padding=0, bias=True)
        self.proj_bn = nn.BatchNorm2d(16)
        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        self.proj_conv1 = nn.Conv1d(16, 8, kernel_size=5, stride = 2, padding=0, bias=True)
        self.proj_bn1 = nn.BatchNorm2d(8)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')


        self.fc1_conv = nn.n.Linear(256, 128, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

        self.fc2_conv = nn.Linear(128, 64)
        self.fc2_bn = nn.BatchNorm1d(64)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')
       

    def forward(self, x):
        T, B, C, N1,N2 = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value
  
        x = self.proj_bn(x).reshape(T, B, -1, 1, N2).contiguous()
        x = self.proj_lif(x).flatten(0,1).contiguous()  

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, 100).contiguous()
        x = self.proj_lif1(x).contiguous()

        x = x.flatten(2,3)

        x = self.fc1_conv(x1.flatten(0,1))
        x = self.fc1_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc1_lif(x)
        x = self.drop1(x)


        x = self.fc2_conv(x.flatten(0,1))
        x = self.fc2_bn(x).reshape(T,B,C,N).contiguous()
        x = self.fc2_lif(x)

        return x 

class spike_BCRNet(nn.Module):
    def __init__(self,
                 in_channels=2, num_classes=15, dim = 16,                 
                 ):
        super(spike_EEG_model, self).__init__()
        self.num_classes = num_classes    

        self.encoding = spike_encoding(in_channels=2, out_channels=16, kernal_size=(1,8))
        self.ds = DSCONV(dim = dim)

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
        x = x.unsqueeze(1)
        B, _,_,_ = x.size()

        T = 4
        x = (x.unsqueeze(0)).repeat(T, 1, 1, 1, 1)
        x = self.encoding(x)
        x = self.ds(x)
      
        x = self.head(x.mean(0))

        return x