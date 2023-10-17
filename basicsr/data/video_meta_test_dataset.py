import glob
from pathlib import Path
import torch
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import duf_downsample, generate_frame_indices, read_img_seq
from basicsr.utils import get_root_logger, scandir, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


from basicsr.utils.flow_util import dequantize_flow
import numpy as np

@DATASET_REGISTRY.register()
class VideoMetaTestDataset(data.Dataset):
    """Video test dataset.

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoMetaTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        if opt['name'].lower() in ['vid4', 'reds4', 'redsofficial',
                                    'vid4_crf15', 'vid4_crf25', 'vid4_crf35']:
            for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
                # get frame list for lq and gt
                subfolder_name = osp.basename(subfolder_lq)
                img_paths_lq = sorted(list(scandir(subfolder_lq, full_path=True)))
                img_paths_gt = sorted(list(scandir(subfolder_gt, full_path=True)))

                max_idx = len(img_paths_lq)
                assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                      f' and gt folders ({len(img_paths_gt)})')

                self.data_info['lq_path'].extend(img_paths_lq)
                self.data_info['gt_path'].extend(img_paths_gt)
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:
                    logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                    self.imgs_lq[subfolder_name] = read_img_seq(img_paths_lq)
                    self.imgs_gt[subfolder_name] = read_img_seq(img_paths_gt)
                else:
                    self.imgs_lq[subfolder_name] = img_paths_lq
                    self.imgs_gt[subfolder_name] = img_paths_gt
        else:
            raise ValueError(f'Non-supported video test dataset: {opt["name"]}')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]
        lq_path = self.data_info['lq_path'][index]

        select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
            img_gt = self.imgs_gt[folder][idx]
        else:
            img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
            imgs_lq = read_img_seq(img_paths_lq)
            img_gt = read_img_seq([self.imgs_gt[folder][idx]])
            img_gt.squeeze_(0)

        return {
            'lq': imgs_lq,  # (t, c, h, w)
            'gt': img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
            'lq_path': lq_path  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


# @DATASET_REGISTRY.register()
# class VideoTestVimeo90KDataset(data.Dataset):
#     """Video test dataset for Vimeo90k-Test dataset.

#     It only keeps the center frame for testing.
#     For testing datasets, there is no need to prepare LMDB files.

#     Args:
#         opt (dict): Config for train dataset. It contains the following keys:
#             dataroot_gt (str): Data root path for gt.
#             dataroot_lq (str): Data root path for lq.
#             io_backend (dict): IO backend type and other kwarg.
#             cache_data (bool): Whether to cache testing datasets.
#             name (str): Dataset name.
#             meta_info_file (str): The path to the file storing the list of test
#                 folders. If not provided, all the folders in the dataroot will
#                 be used.
#             num_frame (int): Window size for input frames.
#             padding (str): Padding mode.
#     """

#     def __init__(self, opt):
#         super(VideoTestVimeo90KDataset, self).__init__()
#         self.opt = opt
#         self.cache_data = opt['cache_data']
#         if self.cache_data:
#             raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
#         self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
#         self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
#         neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

#         # file client (io backend)
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'

#         logger = get_root_logger()
#         logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')
#         with open(opt['meta_info_file'], 'r') as fin:
#             subfolders = [line.split(' ')[0] for line in fin]
#         for idx, subfolder in enumerate(subfolders):
#             gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
#             self.data_info['gt_path'].append(gt_path)
#             lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
#             self.data_info['lq_path'].append(lq_paths)
#             self.data_info['folder'].append('vimeo90k')
#             self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
#             self.data_info['border'].append(0)

#     def __getitem__(self, index):
#         lq_path = self.data_info['lq_path'][index]
#         gt_path = self.data_info['gt_path'][index]
#         imgs_lq = read_img_seq(lq_path)
#         img_gt = read_img_seq([gt_path])
#         img_gt.squeeze_(0)

#         return {
#             'lq': imgs_lq,  # (t, c, h, w)
#             'gt': img_gt,  # (c, h, w)
#             'folder': self.data_info['folder'][index],  # folder name
#             'idx': self.data_info['idx'][index],  # e.g., 0/843
#             'border': self.data_info['border'][index],  # 0 for non-border
#             'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
#         }

#     def __len__(self):
#         return len(self.data_info['gt_path'])


# @DATASET_REGISTRY.register()
# class VideoTestDUFDataset(VideoMetaTestDataset):
#     """ Video test dataset for DUF dataset.

#     Args:
#         opt (dict): Config for train dataset.
#             Most of keys are the same as VideoTestDataset.
#             It has the following extra keys:

#             use_duf_downsampling (bool): Whether to use duf downsampling to
#                 generate low-resolution frames.
#             scale (bool): Scale, which will be added automatically.
#     """

#     def __getitem__(self, index):
#         folder = self.data_info['folder'][index]
#         idx, max_idx = self.data_info['idx'][index].split('/')
#         idx, max_idx = int(idx), int(max_idx)
#         border = self.data_info['border'][index]
#         lq_path = self.data_info['lq_path'][index]

#         select_idx = generate_frame_indices(idx, max_idx, self.opt['num_frame'], padding=self.opt['padding'])

#         if self.cache_data:
#             if self.opt['use_duf_downsampling']:
#                 # read imgs_gt to generate low-resolution frames
#                 imgs_lq = self.imgs_gt[folder].index_select(0, torch.LongTensor(select_idx))
#                 imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
#             else:
#                 imgs_lq = self.imgs_lq[folder].index_select(0, torch.LongTensor(select_idx))
#             img_gt = self.imgs_gt[folder][idx]
#         else:
#             if self.opt['use_duf_downsampling']:
#                 img_paths_lq = [self.imgs_gt[folder][i] for i in select_idx]
#                 # read imgs_gt to generate low-resolution frames
#                 imgs_lq = read_img_seq(img_paths_lq, require_mod_crop=True, scale=self.opt['scale'])
#                 imgs_lq = duf_downsample(imgs_lq, kernel_size=13, scale=self.opt['scale'])
#             else:
#                 img_paths_lq = [self.imgs_lq[folder][i] for i in select_idx]
#                 imgs_lq = read_img_seq(img_paths_lq)
#             img_gt = read_img_seq([self.imgs_gt[folder][idx]], require_mod_crop=True, scale=self.opt['scale'])
#             img_gt.squeeze_(0)

#         return {
#             'lq': imgs_lq,  # (t, c, h, w)
#             'gt': img_gt,  # (c, h, w)
#             'folder': folder,  # folder name
#             'idx': self.data_info['idx'][index],  # e.g., 0/99
#             'border': border,  # 1 for border, 0 for non-border
#             'lq_path': lq_path  # center frame
#         }


@DATASET_REGISTRY.register()
class VideoMetaRecurrentTestDataset(VideoMetaTestDataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames.

    Args:
        Same as VideoTestDataset.
        Unused opt:
            padding (str): Padding mode.

    """

    def __init__(self, opt):
        super(VideoMetaRecurrentTestDataset, self).__init__(opt)
        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

        self.meta_root = Path(opt['dataroot_meta'])

        self.meta_cache = {}

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
            imgs_gt = self.imgs_gt[folder]
        else:
            raise NotImplementedError('Without cache_data is not implemented.')

        #if folder in self.meta_cache.keys():
        #    img_mvs, frm_types =  self.meta_cache[folder]
        #else:
            # get metadata
        meta_path = self.meta_root /  f'{folder}.npz'
        frm_type, mv_f, mv_b = get_meta(meta_path)
        t = len(frm_type)

        # frame type & padding
        frm_types = []
        for i in range(t):
            if frm_type[i]=="B":
                one_hot = torch.tensor([1])
            elif frm_type[i]=="P":
                one_hot = torch.tensor([2])
                if i+1<t:
                    mv_f[i] = -1 * mv_b[i+1]
            else:
                one_hot = torch.tensor([0])

                if i+1<t:
                    mv_f[i] = -1 * mv_b[i+1]

            frm_types.append(one_hot)

        mv_f.extend(mv_b)

        img_mvs = torch.stack([torch.from_numpy(mv)
                                        for mv in mv_f],
                                dim=0)
        frm_types = torch.stack(frm_types, dim=0)

        if self.cache_data:
            self.meta_cache[folder]=(img_mvs, frm_types)

        IBP_rep = np.loadtxt('./rep.txt', dtype="float", delimiter=',') # I, B, P 3xC
        IBP_rep = torch.from_numpy(IBP_rep).float()
        IBP_rep_new = [IBP_rep[0], IBP_rep[2], IBP_rep[1]] # I, P, B
        IBP_rep = torch.stack(IBP_rep_new, dim=0)

        IBP_rep_list = []
        for i in range(len(frm_type)):
            if frm_type[i] == 'I':
                IBP_rep_list.append(IBP_rep[0])
            elif frm_type[i] == 'P':
                IBP_rep_list.append(IBP_rep[1])
            elif frm_type[i] == 'B':
                IBP_rep_list.append(IBP_rep[2])

        IBP_rep_list = torch.stack(IBP_rep_list, dim=0)
        return {
            'lq': imgs_lq,
            'gt': imgs_gt,
            'mv': img_mvs.float(), 'frm_type':frm_types,
            'folder': folder, 'IBP_rep_list':IBP_rep_list
        }

    def __len__(self):
        return len(self.folders)




def get_meta(meta_pth):
    type_dic = {0:"I",1:"P",2:"B"}

    # meta_pth = "/home/wyw/Desktop/proj/code/meta_dataset/vimeo/meta/15/c/00001/0001"
    npzfile=np.load(f'{meta_pth}')

    t,h,w = npzfile['size'].tolist()
    frm_types = ['P']*t;    frm_types[0]='I'

    mv_f = [ ];   mv_b = [ ]
    for i in range(t):
        mv_f.append(np.zeros((h,w,2)))
        mv_b.append(np.zeros((h,w,2)))

    mvs = npzfile['mvs'].tolist()
    for mv in mvs:
        framenum, source, blockx, blocky, srcx, srcy, dstx, dsty, scale, frame_type = mv
        framenum -= 1
        frm_types[framenum] = type_dic[frame_type]

        x0 = int(max(dstx - blockx/2, 0))
        y0 = int(max(dsty - blocky/2, 0))
        x1 = int(min(dstx + blockx/2, w-1))
        y1 = int(min(dsty + blocky/2, h-1))
        if source==1:
            mv_f[framenum][y0:y1,x0:x1,0] = (srcx-dstx)/scale
            mv_f[framenum][y0:y1,x0:x1,1] = (srcy-dsty)/scale
        else:
            mv_b[framenum][y0:y1,x0:x1,0] = (srcx-dstx)/scale
            mv_b[framenum][y0:y1,x0:x1,1] = (srcy-dsty)/scale


    return frm_types, mv_f, mv_b
