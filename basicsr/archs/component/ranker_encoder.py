import torch
import torch.nn as nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, nf=32, in_nc=3):
        super(Encoder, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]

        # pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # print(x.shape)
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        # fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        # fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        # fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        # fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))
        fea = self.pooling(fea)
        fea = fea.view(fea.size(0), -1)
        return fea

class Ranker(nn.Module):
    def __init__(self,ranker_ckpt):
        super(Ranker, self).__init__()

        self.E = Encoder()
        self.R = nn.Sequential(
            # nn.LeakyReLU(0.1,True),
            # nn.Linear(256,128),
            nn.LeakyReLU(0.1,True),
            nn.Linear(128,64),
            nn.LeakyReLU(0.1,True),
            nn.Linear(64,32),
            nn.LeakyReLU(0.1,True),
            nn.Linear(32,1),
        )

        checkpoint = torch.load(ranker_ckpt)
        self.load_state_dict(checkpoint)


    def forward(self, x):
        rep = self.E(x)
        score = self.R(rep)

        return score

class DA_rep(nn.Module):
    # def __init__(self, n_c, n_b, out_chanel, daEncoder_weight):
    #     super(DA_rep,self).__init__()
    def __init__(self,ranker_ckpt, nf=64, nr=128, in_nc=3):
        super(DA_rep, self).__init__()

        net = Ranker(ranker_ckpt)
        self.net = net.E


    def forward(self, x):
        # print(x.shape)
        rep = self.net(x)
        # print(rep.shape)
        return rep
