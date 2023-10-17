import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

####################
# encoder
####################
class CRF_Encoder(nn.Module):
    def __init__(self, nf=32, in_nc=3):
        super(CRF_Encoder, self).__init__()
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

        # pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # load
        checkpoint = torch.load(r"script/weights/deg_encoder.pth")
        self.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})

    def forward(self, x):
        n, c, h, w = x.shape
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = F.interpolate(fea, size=(h,w), mode='nearest')

        # fea = self.pooling(fea)
        # fea = fea.view(fea.size(0), -1)
        return fea

class IBP_Encoder(nn.Module):
    def __init__(self, ni=128, nf=128, num_class=3, ckpt=r"script/weights/ibp_encoder.pth"):
        super(IBP_Encoder, self).__init__()
        self.fc1 = nn.Linear(ni, nf)
        self.fc2 = nn.Linear(nf, nf)
        self.classcify = nn.Linear(nf, num_class)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # load
        self.load_state_dict({k.replace('module.',''):v   for k,v in torch.load(ckpt).items()})

    def forward(self, x):
        # print(x.shape)
        fea = self.lrelu(self.fc1(x))
        fea = self.lrelu(self.fc2(fea))
        pre = self.classcify(fea)

        return pre, fea


####################
# factor: beta, gamma
####################

class spatial_modulate_factor(nn.Module):
    def __init__(self, feat_nc, da_map_nc):
        super().__init__()

        ks = 3
        pw = ks // 2
        self.mlp_shared =  nn.Sequential(
            nn.Conv2d(da_map_nc, feat_nc, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(feat_nc, feat_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(feat_nc, feat_nc, kernel_size=ks, padding=pw)

    def forward(self, da_map):
        actv = self.mlp_shared(da_map)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta

class ibp_modulate_factor(nn.Module):
    def __init__(self, feat_nc, ibp_vec_nc):
        super().__init__()

        ks = 3
        self.mlp_shared = nn.Sequential(
            nn.Linear(in_features = ibp_vec_nc, out_features = feat_nc),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(in_features = feat_nc, out_features = feat_nc)
        self.mlp_beta = nn.Linear(in_features = feat_nc, out_features = feat_nc)

    def forward(self, ibp_vec):

        actv = self.mlp_shared(ibp_vec)
        gamma = self.mlp_gamma(actv)   #.view(N, C, 1, 1).expand([N, C, H, W])
        beta = self.mlp_beta(actv)     #.view(N, C, 1, 1).expand([N, C, H, W])

        # apply scale and bias
        # out = feat * (1 + gamma) + beta

        return gamma, beta
