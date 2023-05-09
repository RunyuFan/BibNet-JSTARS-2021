import torch
import torch.nn as nn


class Vector_module3(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Vector_module3,self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.vec_module=nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.out_channel),

        )
    def forward(self,x):
        return self.vec_module(x)


class Vector_module1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Vector_module1, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vec_module = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(self.out_channel),

        )

    def forward(self, x):
        return self.vec_module(x)


class Vec_net(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Vec_net, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.vec1=Vector_module1(in_channel=self.in_channel,out_channel=self.out_channel)
        self.vec2=Vector_module3(in_channel=self.in_channel,out_channel=self.out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):

        x1=self.vec2(x)
        x2=self.vec1(x1)
        x3=self.relu(x2+x)

        return x3

