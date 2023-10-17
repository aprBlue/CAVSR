import torch
import torch.nn as nn
from torch.nn import functional as F

### DA
class Encoder(nn.Module):
    def __init__(self, nf=32, in_nc=3):
        super(Encoder, self).__init__()
        # [64, 128, 128]
        self.E = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        # print(x.shape)
        # fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = self.E(x).squeeze(-1).squeeze(-1)
        #fea = self.mlp(fea)

        return fea

class Ranker(nn.Module):
    def __init__(self):
        super(Ranker, self).__init__()

        self.E = Encoder()

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 128),
        )

        self.R = nn.Sequential(
            nn.LeakyReLU(0.1,True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1,True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1,True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x[:,:7]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        rep = self.E(x)
        rep = rep.view(B, T, -1) # dim=128
        rep = rep.permute(0, 2, 1)
        rep = F.avg_pool1d(rep, T)
        rep = rep.view(B, -1)
        rep = self.mlp(rep)
        #score = self.R(rep) # .squeeze(-1)

        return rep

class Ranker_256(nn.Module):
    def __init__(self):
        super(Ranker_256, self).__init__()

        self.E = Encoder(64)

        self.R = nn.Sequential(
            nn.LeakyReLU(0.1,True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1,True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1,True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1,True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x[:,:7]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        rep = self.E(x)
        rep = rep.view(B, T, -1) # dim=128
        rep = rep.permute(0, 2, 1)
        rep = F.avg_pool1d(rep, T)
        rep = rep.view(B, -1)
        return rep

class Encoder_up(nn.Module):
    def __init__(self, nf=32, in_nc=3):
        super(Encoder_up, self).__init__()
        # [64, 128, 128]
        self.E = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(0.1, True)
        )
    def forward(self, x):
        fea = self.E(x)
        return fea

class Ranker_128_up(nn.Module):
    def __init__(self):
        super(Ranker_128_up, self).__init__()

        self.E = Encoder_up(32)

        self.R = nn.Sequential(
            #nn.LeakyReLU(0.1,True),
            #nn.Linear(256, 128),
            nn.LeakyReLU(0.1,True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1,True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1,True),
            nn.Linear(32, 1),
        )

    def forward(self, x, IBP_rep):
        B, C, H, W = x.shape
        rep = self.E(x)
        rep = rep + IBP_rep.unsqueeze(-1).unsqueeze(-1)
        fea = F.interpolate(rep, size=(H, W), mode='nearest')
        #rep = rep.view(B, T, -1) 
        #rep = rep.permute(0, 2, 1)
        #rep = F.avg_pool1d(rep, T).squeeze(dim=2)
        #rep = rep.view(B, C, H, W)
        return fea

###
def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)
class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(self.channels_in, self.channels_in, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.channels_in, self.channels_out * self.kernel_size * self.kernel_size, bias=False)
        )
        self.conv = conv(channels_out, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out

class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att

class DAB(nn.Module):
    def __init__(self, n_feat, n_rep=128, kernel_size=3, reduction=8):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_rep, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_rep, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]
        return [out, x[1]]


