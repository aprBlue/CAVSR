import torch
import torch.nn as nn
from torch.nn import functional as F

####################
### Encoder: crf rep
####################
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

        # pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        checkpoint = torch.load(r"script/weights/deg_encoder.pth")
        self.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})

    def forward(self, x):
        # print(x.shape)
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.pooling(fea)
        fea = fea.view(fea.size(0), -1)
        return fea



####################
### modulation
####################
from torch.autograd import Variable, Function

class asign_index(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kernel, guide_feature):
        ctx.save_for_backward(kernel, guide_feature)
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
        return torch.sum(kernel * guide_mask, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        kernel, guide_feature = ctx.saved_tensors
        guide_mask = torch.zeros_like(guide_feature).scatter_(1, guide_feature.argmax(dim=1, keepdim=True), 1).unsqueeze(2) # B x 3 x 1 x 25 x 25
        grad_kernel = grad_output.clone().unsqueeze(1) * guide_mask # B x 3 x 256 x 25 x 25
        grad_guide = grad_output.clone().unsqueeze(1) * kernel # B x 3 x 256 x 25 x 25
        grad_guide = grad_guide.sum(dim=2) # B x 3 x 25 x 25
        softmax = F.softmax(guide_feature, 1) # B x 3 x 25 x 25
        grad_guide = softmax * (grad_guide - (softmax * grad_guide).sum(dim=1, keepdim=True)) # B x 3 x 25 x 25
        return grad_kernel, grad_guide

class DA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.region_number = 4

        self.kernel = nn.Sequential(
            nn.Conv1d(channels_in, channels_out, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(channels_out, channels_out * self.region_number * self.kernel_size * self.kernel_size,
                      kernel_size=1, groups=self.region_number, bias=False)
        )
        self.conv = nn.Conv2d(channels_out, channels_out, 1, padding=(1//2), bias=True)
        self.ca = CA_layer(channels_in, channels_out, reduction)
        self.conv_guide1 = nn.Conv2d(1, self.region_number, kernel_size=kernel_size, padding=1)
        self.conv_guide2 = nn.Conv2d(1, self.region_number, kernel_size=kernel_size, padding=1)
        self.asign_index1 = asign_index.apply
        self.asign_index2 = asign_index.apply

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, q_map):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        # branch 1

        kernel = self.kernel(x[1].unsqueeze(2)).squeeze(2)
        kernel = kernel.view(b, -1, c, self.kernel_size, self.kernel_size).transpose(1, 2).contiguous()
        kernel = kernel.view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].contiguous().view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        # out = out.view(b, -1, h, w).view(b, self.region_number, -1, h, w)
        # out = out.view(-1, self.region_number, h, w).view(b, -1, self.region_number, h, w)
        out = out.view(b, -1, self.region_number, h, w)
        out = out.transpose(1, 2)
        guide_feature1 = self.conv_guide1(q_map)    # b, r, h, w
        out = self.asign_index1(out, guide_feature1)
        out = self.conv(out)

        # branch 2
        out2 = self.ca(x)   # b,r,c,h,w
        guide_feature2 = self.conv_guide2(q_map)    # b, r, h, w
        out2 = self.asign_index2(out2, guide_feature2)
        out = out + out2

        return out

class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.region_number = 4
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out * self.region_number, 1, 1, 0,
            groups=self.region_number, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()

        att = self.conv_du(x[1].unsqueeze(2).unsqueeze(3))
        att = att.view(b, -1, c, 1, 1)   # b,r,c,1,1
        out = x[0].unsqueeze(1) * att   # b,r,c,h,w

        return out

class DAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8):
        super(DAB, self).__init__()

        #self.compress = nn.Sequential(
        #    nn.Linear(128, n_feat, bias=False),
        #    nn.LeakyReLU(0.1, True),
        #    nn.Linear(n_feat, n_feat, bias=False),
        #)

        self.da_conv_1 = DA_conv(128, n_feat, kernel_size, reduction)
        self.conv_1 = nn.Conv2d(n_feat, n_feat, 3, 1, 3//2, bias=True)

        self.da_conv_2 = DA_conv(128, n_feat, kernel_size, reduction)
        self.conv_2 = nn.Conv2d(n_feat, n_feat, 3, 1, 3//2, bias=True)

        self.relu =  nn.LeakyReLU(0.1, True)

    def forward(self, x0, rep, q_map):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        #x1 = self.compress(rep)

        out = self.relu(self.da_conv_1([x0, rep], q_map))
        out = self.relu(self.conv_1(out))
        out = self.relu(self.da_conv_2([out, rep], q_map))
        out = self.conv_2(out)

        return out + x0



####################
### multi-scale
####################
class HDRO(nn.Module):
    """
    Hybrid Dilation Reconstruction Operator
    """
    def __init__(self, nf=64, out_nc=3, base_ks=3, bias=True):
        super(HDRO, self).__init__()

        self.dilation_1 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=1, dilation=1, bias=True)
        self.dilation_2 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=2, dilation=2, bias=True)
        self.dilation_3 = nn.Conv2d(nf, out_nc, 3, stride=1, padding=4, dilation=4, bias=True)
        self.conv = nn.Conv2d(out_nc*3, out_nc, base_ks, padding=(base_ks//2),stride = 1, bias=bias)
    def forward(self, fea):
        fea1 = self.dilation_1(fea)
        fea2 = self.dilation_2(fea)
        fea3 = self.dilation_3(fea)
        out_fea = self.conv(torch.cat([fea1,fea2,fea3],dim=1))

        return out_fea

