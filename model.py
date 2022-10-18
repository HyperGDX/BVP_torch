
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        elif isinstance(self.module,nn.Linear):
            b, t ,c= x.size()
            # merge batch and seq dimensions
            x_reshape = x.contiguous().view(b * t, c)
            y = self.module(x_reshape)
            # We have to reshape Y
            y = y.contiguous().view(b, t, y.size()[1])
            return y
        elif isinstance(self.module,nn.Conv2d) or isinstance(self.module,nn.MaxPool2d):
            b,t,c,h,w = x.size()
            x_reshape = x.contiguous().view(b * t, c,h,w)
            y = self.module(x_reshape)
            y = y.contiguous().view(b, t, y.size(1),y.size(2),y.size(3))
            return y
        
        elif isinstance(self.module,nn.Flatten):
            b,t,c,h,w = x.size()
            x_reshape = x.contiguous().view(b * t, c, h, w)
            y = self.module(x_reshape)
            y = y.contiguous().view(b, t, y.size(1))
            return y       

class Widar3_torch(nn.Module):
    def __init__(self,num_classes,drop_rate=0.5) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.time_conv2d1 = TimeDistributed(nn.Conv2d(1,16,5))
        self.time_pool1 = TimeDistributed(nn.MaxPool2d(2))
        self.flat1 = TimeDistributed(nn.Flatten())
        self.linear1 = TimeDistributed(nn.Linear(1024,64))
        self.linear2 = TimeDistributed(nn.Linear(64,64))
        self.gru = nn.GRU(input_size=64,hidden_size=128,batch_first=True)
        self.linear3 = nn.Linear(128,num_classes)

    def forward(self,x):
        y = self.time_conv2d1(x)
        y = F.relu(y)
        y = self.time_pool1(y)
        y = self.flat1(y)
        y = self.linear1(y)
        y = F.relu(y)
        y = F.dropout(y,self.drop_rate)
        y = self.linear2(y)
        y = F.relu(y)
        _, y = self.gru(y)
        y = y.squeeze(0)
        y = F.dropout(y,self.drop_rate)
        y = self.linear3(y)
        return y 