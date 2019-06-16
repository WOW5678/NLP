# -*- coding: utf-8 -*-
"""
 @Time    : 2019/6/10 0010 下午 4:21
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import torch
import torch.nn as nn

class HighwayMLP(nn.Module):
    def __init__(self,input_size,gate_bias=-2,
                 activation_fun=nn.functional.relu,
                 gate_activation=nn.functional.softmax):
        super(HighwayMLP, self).__init__()
        self.activation_function=activation_fun
        self.gate_activation=gate_activation
        self.normal_layer=nn.Linear(input_size,input_size)

        self.gate_layer=nn.Linear(input_size,input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self,x):
        normal_layer_result=self.activation_function(self.normal_layer(x))
        gate_layer_result=self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal=torch.mul(normal_layer_result,gate_layer_result)
        multiplyeed_gate_and_input=torch.mul((1-gate_layer_result),x)
        return torch.add(multiplyed_gate_and_normal,multiplyeed_gate_and_input)

class HighwayCNN(nn.Module):
    def __init__(self,input_size,gate_bias=-1,activation_function=nn.functional.relu,
                 gate_activation=nn.functional.softmax):
        super(HighwayCNN, self).__init__()
        self.activation_function = activation_function
        self.gate_activation = gate_activation

        self.normal_layer = nn.Linear(input_size, input_size)

        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self,x):
        normal_layer_result = self.activation_function(self.normal_layer(x))
        gate_layer_result = self.gate_activation(self.gate_layer(x))

        multiplyed_gate_and_normal = torch.mul(normal_layer_result, gate_layer_result)
        multiplyed_gate_and_input = torch.mul((1 - gate_layer_result), x)

        return torch.add(multiplyed_gate_and_normal,
                         multiplyed_gate_and_input)
