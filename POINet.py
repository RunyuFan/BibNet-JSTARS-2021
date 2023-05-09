import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
from Vector_net import Vec_net
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
# from block import fusions


# POIS
class Pois_net(nn.Module):
    def __init__(self, in_channel,out_channel,num,dim):
        super(Pois_net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num=num
        self.out_dim=dim
        self.vec_module=nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU(inplace=True),
        )
        self.vec_net1=Vec_net(in_channel=self.out_channel,out_channel=self.out_channel)
        self.vec_net2 = Vec_net(in_channel=self.out_channel, out_channel=self.out_channel)
        self.vec_net3 = Vec_net(in_channel=self.out_channel, out_channel=self.out_channel)
        self.linear=nn.Linear(self.out_channel*self.num,self.out_dim)

    def forward(self, pois):
        x1=self.vec_module(pois)
        x1=self.vec_net1(x1)
        # x2=self.vec_net2(x1)
        x3=self.vec_net3(x1)

        x3=x3.view(-1,self.out_channel*self.num)
        return self.linear(x3)
