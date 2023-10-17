import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer, ResidualBlockNoBN_sft_1x1, SFTLayer_torch_1x1, SFTLayer_torch_3x3
from .component.degradation_aware import Ranker_128_up
# from .edvr_arch import PCDAlignment, TSAFusion
# from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class CAVSR(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15):    # , spynet_path=None):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        # self.spynet = SpyNet(spynet_path)

        self.encoder = Ranker_128_up()
        checkpoint = torch.load(r"./ranker.pth")
        self.encoder.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        print('encoder model loading done!')

        self.da_head_b = nn.Sequential(
            nn.Conv2d(3, num_feat, 3, 1, 1, bias=True)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

        self.da_head_f = nn.Sequential(
            nn.Conv2d(3, num_feat, 3, 1, 1, bias=True)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat*3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat*4, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Sequential(
                      nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=True),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True),
                      nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
                      )
        #self.conv_last = nn.Sequential(
        #                nn.Conv2d(num_feat, 48, 3, 1, 1),
        #                nn.PixelShuffle(4)
        #                )
        self.HR_branch = nn.Sequential(
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, 3, 3, 1, 1))

        self.HR_branch_b = nn.Sequential(
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, 3, 3, 1, 1))

        self.HR_branch_f = nn.Sequential(
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat*4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True), nn.Conv2d(num_feat, 3, 3, 1, 1))

        #self.pixel_shuffle = nn.PixelShuffle(4)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # dcn   (offset+mv)*mask
        self.deform_align_b = SecondOrderDeformableAlignment(
                    num_feat, num_feat, 3,
                    padding=1, deformable_groups=16, max_residue_magnitude=10)
        self.deform_align_f = SecondOrderDeformableAlignment(
                    num_feat, num_feat, 3,
                    padding=1, deformable_groups=16, max_residue_magnitude=10)

        num_DA_block = 10
        self.modulate_b = sft_net(num_in_ch=num_feat, num_out_ch=num_feat, rep_feat=128, num_block=num_DA_block)
        self.modulate_f = sft_net(num_in_ch=num_feat, num_out_ch=num_feat, rep_feat=128, num_block=num_DA_block)


    def forward(self, x, mvs, frame_type, IBP_rep_list, mode="train"):
        b, n, _, h, w = x.size()
        # training status of encoder must be set as eval!!!
        self.encoder.eval()
        if self.encoder.training == 'train':
            raise ValueError('training status of encoder must be set as eval')
        rep_list = []
        with torch.no_grad():
            for i in range(n):
                rep_list.append(self.encoder(x[:,i], IBP_rep_list[:,i]))
        rep = torch.stack(rep_list, dim=1)

        pre_fea_b = []
        x_b_feat = []
        for i in range(n):
            fea = self.da_head_b(x[:,i])
            pre_fea_b.append(fea)
            fea = self.modulate_b(fea, rep[:,i])
            x_b_feat.append(fea)
       
        pre_fea_b = torch.stack(pre_fea_b, dim=1)
        x_b_feat = torch.stack(x_b_feat, dim=1)

        pre_fea_f = []
        x_f_feat = []
        for i in range(n):
            fea = self.da_head_f(x[:,i])
            pre_fea_f.append(fea)
            fea = self.modulate_f(fea, rep[:,i])
            x_f_feat.append(fea)
        pre_fea_f = torch.stack(pre_fea_f, dim=1)
        x_f_feat = torch.stack(x_f_feat, dim=1)


        if mode=="train":
            one_f_pre = []
            one_b_pre = []

        # padding

        flows_forward, flows_backward = torch.chunk(mvs, 2, dim=1)
        flows_forward = flows_forward * -1
        flows_backward = flows_backward * -1
        # padded one flows in the last with zero values
        #flows_forward  =  flows_forward[:,:n-1] 
        # padded one flows in the first with zero values
        #flows_backward = flows_backward[:,1:]

        # backward branch
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        #x_b = torch.stack([x[:,i] for i in range(n)] + [x[:,-2]], dim=1)
        pre_fea_b = torch.stack([pre_fea_b[:,i] for i in range(n)] + [pre_fea_b[:,-2]], dim=1)
        x_b_feat = torch.stack([x_b_feat[:,i] for i in range(n)] + [x_b_feat[:,-2]], dim=1)
        out_b = []
        a = 0.5
        for i in range(n-1, -1, -1):
            # frms
            x_i_cur = x_b_feat[:, i, :, :, :]            # cur
            x_i_pre = x_b_feat[:, i+1, :, :, :]
            type_cur = frame_type[:, i]
            mask = torch.ones_like(type_cur) # type B
            mask =  (mask == type_cur) * 1.0
            mask = mask.view(b, 1, 1, 1)
            # hidden state & residual
            if i < n - 1:
                flow = flows_backward[:, i+1, :, :, :]
                res = x[:,i] - flow_warp(x[:,i+1], flow)
                feat_prop = self.deform_align_b(x=feat_prop, fea_cur=x_i_cur, res=res, flow=flow)
            # feature propagation
            feat_prop_ = torch.cat([x_i_cur, x_i_pre, feat_prop], dim=1)
            feat_prop_ = self.backward_trunk(feat_prop_)
            #feat_prop = feat_prop + pre_fea_b[:, i, :, :, :]
            out_b.append(feat_prop_) 
            # one-fuse reconstruction (for supervision)
            if mode == "train":
                one_b_pre.append(self.HR_branch_b(feat_prop_ + pre_fea_b[:, i, :, :, :]))
            feat_prop = mask * ((1-a) * feat_prop + a * feat_prop_) + (1-mask) * feat_prop_

        if mode == "train":
            one_b_pre = one_b_pre[::-1]

        out_b = out_b[::-1]
        out_t = []
        # forward branch
        feat_prop = out_b[0]
        #x_f = torch.stack([x[:,1]] + [x[:,i] for i in range(n)], dim=1)
        pre_fea_f = torch.stack([pre_fea_f[:,1]] + [pre_fea_f[:,i] for i in range(n)], dim=1)
        x_f_feat = torch.stack([x_f_feat[:,1]] + [x_f_feat[:,i] for i in range(n)], dim=1)
        for i in range(0, n): 
            # frms
            x_i_cur = x_f_feat[:, i+1, :, :, :]    # cur
            x_i_pre = x_f_feat[:, i, :, :, :]
            type_cur = frame_type[:, i]
            mask = torch.ones_like(type_cur)
            mask =  (mask == type_cur) * 1.0
            mask = mask.view(b, 1, 1, 1)
            # hidden state & residual
            if i > 0:
                flow = flows_forward[:, i-1, :, :, :]
                res = x[:,i] - flow_warp(x[:,i-1], flow)
                feat_prop = self.deform_align_f(x=feat_prop, fea_cur=x_i_cur, res=res, flow=flow)
            # feature propagation
            feat_prop_ = torch.cat([x_i_cur, x_i_pre, feat_prop, out_b[i]], dim=1)
            feat_prop_ = self.forward_trunk(feat_prop_)
            #feat_prop = feat_prop + pre_fea_f[:, i+1, :, :, :]
            if mode == "train":
                #out_f = feat_prop + pre_fea_f[i]
                one_f_pre.append(self.HR_branch_f(feat_prop_ + pre_fea_f[:, i+1, :, :, :]))


            # upsample
            out = torch.cat([out_b[i], feat_prop_], dim=1)
            out = self.fusion(out) + pre_fea_f[:, i+1, :, :, :]
            #out = self.conv_last(out)
            out = self.HR_branch(out)
            out_t.append(out)
            #base = F.interpolate(x_f[:, i+1, :, :, :], scale_factor=4, mode='bilinear', align_corners=False)
            #bic_lrs.append(base)
            feat_prop = mask * ((1-a) * feat_prop + a * feat_prop_) + (1-mask) * feat_prop_

        if mode == "train":
            return torch.stack(out_t, dim=1), torch.stack(one_f_pre, dim=1), torch.stack(one_b_pre, dim=1)
        else:
            return torch.stack(out_t, dim=1)

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

class sft_net(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, rep_feat=256, num_block=15):
        super().__init__()
        self.main = nn.Sequential(make_layer(ResidualBlockNoBN_sft_1x1, num_block, num_feat=num_out_ch, rep_feat=rep_feat))
        #self.sft = SFTLayer_torch_1x1(rep_feat, num_out_ch)
        self.conv2 = nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1, bias=True)
    def forward(self, fea, rep):
        x = fea.clone()
        res = self.main((fea, rep))
        #res = self.sft((res))
        res = self.conv2(res[0])
        return fea + x


from basicsr.ops.dcn import ModulatedDeformConvPack
class SecondOrderDeformableAlignment(ModulatedDeformConvPack):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.feat_conv = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        )

        self.feat_ext = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 2, 3, 1, 1)
        )

        self.init_offset()

    def init_offset(self):

        def _constant_init(module, val, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.constant_(module.weight, val)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        #_constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, x, fea_cur, res, flow):
        #feat_cur = self.feat_ext(frm_cur)
        #res = frm_cur - flow_warp(frm_pre, flow)
        res = abs(res)
        B, C, H, W = res.shape
        res = res.sum(dim=1)
        res = F.sigmoid(res).view(B, 1, H, W)

        fea_cur = self.feat_conv(fea_cur)
        aligned_feat_pre = flow_warp(x, flow)
        extra_feat = torch.cat([aligned_feat_pre, fea_cur], dim=1)
        residual = self.feat_ext(extra_feat)
        residual = residual.permute(0,2,3,1)
        res = res.permute(0,2,3,1)

        flow = flow + residual * res

        aligned_feat_pre = flow_warp(x, flow)
        return aligned_feat_pre
