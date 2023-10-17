import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class SFTLayer_torch_1x1(nn.Module):
    def __init__(self, rep_feat, num_feat):
        super(SFTLayer_torch_1x1, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(rep_feat, num_feat, 1)
        self.SFT_scale_conv1 = nn.Conv2d(num_feat, num_feat, 1)
        self.SFT_shift_conv0 = nn.Conv2d(rep_feat, num_feat, 1)
        self.SFT_shift_conv1 = nn.Conv2d(num_feat, num_feat, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: rep
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift

class SFTLayer_torch_3x3(nn.Module):
    def __init__(self, rep_feat, num_feat):
        super(SFTLayer_torch_3x3, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(rep_feat, num_feat, 3, 1, 1)
        self.SFT_scale_conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.SFT_shift_conv0 = nn.Conv2d(rep_feat, num_feat, 3, 1, 1)
        self.SFT_shift_conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: rep
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return x[0] * scale + shift

class SFTLayer_torch_new(nn.Module):
    def __init__(self, rep_feat, num_feat):
        super(SFTLayer_torch_new, self).__init__()
        self.kernel_size = 3
        self.SFT_scale_conv0 = nn.Linear(rep_feat, num_feat, bias=False)
        self.SFT_scale_conv1 = nn.Linear(num_feat, num_feat * self.kernel_size * self.kernel_size, bias=False)
        #self.SFT_shift_conv0 = nn.Linear(rep_feat, num_feat, bias=False)
        #self.SFT_shift_conv1 = nn.Linear(num_feat, num_feat, bias=False)
        self.conv = nn.Conv2d(num_feat, num_feat, 1)
    def forward(self, x):
        # x[0]: fea; x[1]: rep
        b, c, h, w = x[0].size()
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True))
        kernel = scale.view(-1, 1, self.kernel_size, self.kernel_size)
        out = F.leaky_relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2), 0.01, inplace=True)
        out = self.conv(out.view(b, -1, h, w))
        #shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True))
        return out
'''
class SFTLayer_torch_new(nn.Module):
    def __init__(self, rep_feat, num_feat):
        super(SFTLayer_torch_new, self).__init__()
        self.SFT_scale_conv0 = nn.Linear(rep_feat, num_feat)
        self.SFT_scale_conv1 = nn.Linear(num_feat, num_feat)
        self.SFT_shift_conv0 = nn.Linear(rep_feat, num_feat)
        self.SFT_shift_conv1 = nn.Linear(num_feat, num_feat)
        self.num_feat = num_feat

    def forward(self, x):
        # x[0]: fea; x[1]: rep
        b, _  = x[1].shape
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.01, inplace=True)).view(b, self.num_feat, 1, 1)
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.01, inplace=True)).view(b, self.num_feat, 1, 1)
        return x[0] * scale + shift
'''

class ResidualBlockNoBN_sft(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, rep_feat=256, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_sft, self).__init__()
        self.res_scale = res_scale
        self.sft0 = SFTLayer_torch(rep_feat, num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.sft1 = SFTLayer_torch(rep_feat, num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
         # x[0]: fea; x[1]: rep
        fea = F.relu(self.sft0(x), inplace=True)
        fea = self.conv1(fea)
        fea = F.relu(self.sft1((fea, x[1])), inplace=True)
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class ResidualBlockNoBN_sft_1x1(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, rep_feat=256, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_sft_1x1, self).__init__()
        self.res_scale = res_scale
        self.sft0 = SFTLayer_torch_1x1(rep_feat, num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.sft1 = SFTLayer_torch_1x1(rep_feat, num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
         # x[0]: fea; x[1]: rep
        fea = F.leaky_relu(self.sft0(x), 0.01, inplace=True)
        fea = self.conv1(fea)
        fea = F.leaky_relu(self.sft1((fea, x[1])), 0.01, inplace=True)
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions

class ResidualBlockNoBN_sft_3x3(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, rep_feat=256, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN_sft_3x3, self).__init__()
        self.res_scale = res_scale
        self.sft0 = SFTLayer_torch_3x3(rep_feat, num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.sft1 = SFTLayer_torch_3x3(rep_feat, num_feat)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
         # x[0]: fea; x[1]: rep
        fea = F.leaky_relu(self.sft0(x), 0.01, inplace=True)
        fea = self.conv1(fea)
        fea = F.leaky_relu(self.sft1((fea, x[1])), 0.01, inplace=True)
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


import math
from collections import OrderedDict
import torch
from torch import nn as nn
from torch.nn import functional as F

class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """

    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=16,
                out_channels=2,
                kernel_size=7,
                stride=1,
                padding=3))

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.
    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)])

        if load_path:
            state_dict = OrderedDict()
            for k, v in torch.load(load_path).items():
                k = k.replace('moduleBasic', 'basic_module')
                state_dict[k] = v
            self.load_state_dict(state_dict)

        self.register_buffer(
            'mean',
            #torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            torch.Tensor([0.406, 0.456, 0.485]).view(1, 3, 1, 1)) # BGR
        self.register_buffer(
            'std',
            #torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)) # BGR
            torch.Tensor([0.225, 0.224, 0.229]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]

        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))

        flow = ref[0].new_zeros([
            ref[0].size(0), 2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(
                input=flow,
                scale_factor=2,
                mode='bilinear',
                align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level],
                    upsampled_flow.permute(0, 2, 3, 1),
                    interp_mode='bilinear',
                    padding_mode='border'), upsampled_flow
            ], 1)) + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(
            input=ref,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)

        flow = F.interpolate(
            input=self.process(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow



to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
