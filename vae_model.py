import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TimeDistributed, SeparableConv2d
from ssim import SSIM


class GRU_ENC(nn.Module):
    def __init__(self, drop_rate=0.5) -> None:
        super().__init__()
        self.drop_rate = drop_rate
        self.time_conv2d1 = TimeDistributed(nn.Conv2d(1, 8, 3))
        self.bn1 = TimeDistributed(nn.BatchNorm2d(8))
        self.time_pool1 = TimeDistributed(nn.AvgPool2d(2))
        self.time_conv2d2 = TimeDistributed(nn.Conv2d(1, 16, 3))
        self.bn2 = TimeDistributed(nn.BatchNorm2d(16))
        self.time_pool2 = TimeDistributed(nn.AvgPool2d(2))
        self.flat1 = TimeDistributed(nn.Flatten())

    def forward(self, x):
        y = x
        return y


class CAE_ENCODER(nn.Module):

    def __init__(self, latent_dim, input_shape=[32, 25, 1, 20, 20], dropout_rate=0.5):
        # 调用父类方法初始化模块的state
        super(CAE_ENCODER, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.conv1 = TimeDistributed(nn.Conv2d(1, 8, 3))  # 32,28,8,18,18
        self.bn1 = TimeDistributed(nn.BatchNorm2d(8))
        self.conv2 = TimeDistributed(nn.Conv2d(8, 16, 3))  # 32,28,16,16,16
        self.bn2 = TimeDistributed(nn.BatchNorm2d(16))
        # self.pool1 = TimeDistributed(nn.MaxPool2d(2))  # 32,28,16,8,8
        self.flat1 = TimeDistributed(nn.Flatten())
        self.linear1 = TimeDistributed(nn.LazyLinear(256))
        self.linear2 = TimeDistributed(nn.Linear(256, 64))

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = self.bn1(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.bn2(y)
        # y = self.pool1(y)
        y = self.flat1(y)
        y = self.linear1(y)
        y = F.relu(y)
        # y = F.dropout(y, self.dropout_rate)
        y = self.linear2(y)
        return y


class CAE_DECODER(nn.Module):

    def __init__(self, latent_dim, dropout_rate=0.5):
        # 调用父类方法初始化模块的state
        super(CAE_DECODER, self).__init__()

        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        # 编码器 ： [b, input_dim] => [b, z_dim]
        self.conv1 = TimeDistributed(nn.ConvTranspose2d(8, 1, 3))
        self.conv2 = TimeDistributed(nn.ConvTranspose2d(16, 8, 3))

        # self.upsample1 = TimeDistributed(nn.MaxUnpool2d(2, 1))

        self.linear1 = TimeDistributed(nn.Linear(256, 16*16*16))
        self.linear2 = TimeDistributed(nn.Linear(64, 256))

    def forward(self, x):  # 32,28,64
        y = self.linear2(x)  # 32,28,256
        y = F.relu(y)
        y = self.linear1(y)  # 32,28,1600
        y = y.contiguous().view(y.size(0), y.size(1), 16, 16, 16)  # 32,28,16,10,10
        # y = self.upsample1(y)  # 32,28,16,20,20
        y = F.relu(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.conv1(y)

        return y


class MSELoss_SEQ(nn.Module):
    def __init__(self):
        super(MSELoss_SEQ, self).__init__()

    def forward(self, x, y):
        return F.mse_loss(x[:, :, :, :, :], y[:, :, :, :, :])


if __name__ == "__main__":
    # net = CVAE_ENCODER(latent_dim=6)
    # x = torch.randn((32, 25, 1, 20, 20))
    # y = net(x)
    # net = CAE_DECODER(latent_dim=6)
    # x1 = torch.Tensor([[[[[1, 0.5], [1, 0]]]]])
    # x2 = torch.Tensor([[[[[0.5, 0], [0, 1]]]]])
    # mse = MSELoss_SEQ()
    # print(mse(x1, x2))
    x = torch.randn((32, 25, 64))
    net = CAE_DECODER(latent_dim=64)
    y = net(x)
