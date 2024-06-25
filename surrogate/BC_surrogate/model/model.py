from collections import OrderedDict
from typing import Any, List, Tuple
import copy
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

num_steps = 100
spike_grad = surrogate.fast_sigmoid()

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, 
                 norm_layer: callable, bias: bool, node: callable, **kwargs):
        super().__init__()
        
        self.drop_rate = float(drop_rate)
        output_c =  bn_size * growth_rate
        self.layer = nn.Sequential(
            norm_layer(num_input_features),
            nn.Conv1d(num_input_features,output_c, kernel_size=1, stride=1, bias=bias),
            snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
            norm_layer(output_c),
            nn.Conv1d(output_c, growth_rate, kernel_size=3, stride=1, padding=1, bias=bias),
            snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x

        x = torch.cat(prev_features, 1)
        out = self.layer(x)
        return out

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, num_input_features: int, bn_size: int, 
                 growth_rate: int, drop_rate: float, norm_layer: callable, bias: bool, 
                 node: callable = None, **kwargs):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                norm_layer=norm_layer,
                node=node,
                bias=bias,
                **kwargs
            )
            self.add_module(f"denselayer{i + 1}", layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)



class SpikingDenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_channels=2, bn_size=4, drop_rate=0, 
                 num_classes=2, init_weights=True, norm_layer: callable = None, 
                 node: callable = None, **kwargs):
        
        super().__init__()
        
        self.nz, self.numel = {}, {}
        self.out_channels = []
        
        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
            
        num_init_features = 2 * growth_rate


        self.conv = nn.Conv1d(1, num_init_features, kernel_size=7, padding=3, stride=2, bias=False)
        self.norm = nn.BatchNorm1d(num_init_features)
        self.act = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.classifier = nn.Linear(400, num_classes)
        self.class_act = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        #features = self.forward_features(x)
        x = x.to(dtype=torch.float32)
        x = x.view(-1,1,100)

        # features = self.features(x)
        features = self.conv(x)
        features = self.norm(features)
        features = self.act(features)
        features = self.pool(features)
        out = torch.flatten(features, 1) 
        out = self.classifier(out)
        out, mem = self.class_act(out)

        return out, mem #final leaky membrane potential 
    

def _densenet(
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_channels: int,
    norm_layer: callable = None, single_step_neuron: callable = None,
    **kwargs: Any,
) -> SpikingDenseNet:
    model = SpikingDenseNet(growth_rate, block_config, num_init_channels, norm_layer=norm_layer, node=single_step_neuron, **kwargs)
    return model

def spiking_densenet_custom(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs) -> SpikingDenseNet:
    r"""A spiking version of custom DenseNet model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet", growth_rate, block_config, num_init_channels, norm_layer, single_step_neuron, **kwargs)

def spiking_densenet121(num_init_channels, norm_layer: callable = None, single_step_neuron: callable = None, **kwargs) -> SpikingDenseNet:
    r"""A spiking version of Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    The required minimum input size of the model is 29x29.
    Args:
        num_init_channels (int): number of channels of the input data
        norm_layer (callable): a layer of batch norm. No batch norm if None
    """
    return _densenet("densenet121", 8, (2,3,2), num_init_channels, norm_layer, single_step_neuron, **kwargs)

if __name__ == "__main__":
    
    tau = 2.0
    num_steps = 100
    #single_step_neuron = snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True)
    model = spiking_densenet121(10)
    print(model)