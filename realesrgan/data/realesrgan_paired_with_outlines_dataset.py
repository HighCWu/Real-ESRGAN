import os
import cv2
import torch
import random
import numpy as np
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class RealESRGANPairedWithOutlinesDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_ol (str): Data root path for ol.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).
            pre_scale (str): scale lq images in advance

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(RealESRGANPairedWithOutlinesDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.ol_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_ol']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.ol_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'ol']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder, self.ol_folder], ['lq', 'gt', 'ol'])
        elif 'meta_info' in self.opt and self.opt['meta_info'] is not None:
            with open(self.opt['meta_info']) as fin:
                paths = [line.strip() for line in fin]
            self.paths = []
            for path in paths:
                gt_path, lq_path, ol_path = path.split(', ')
                gt_path = os.path.join(self.gt_folder, gt_path)
                lq_path = os.path.join(self.lq_folder, lq_path)
                ol_path = os.path.join(self.ol_folder, ol_path)
                self.paths.append(dict([('gt_path', gt_path), ('lq_path', lq_path), ('ol_path', ol_path)]))
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder, self.ol_folder], ['lq', 'gt', 'ol'], self.filename_tmpl)

        self.pre_scale = opt['pre_scale'] if 'pre_scale' in opt else 1

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        pre_scale = self.pre_scale # if scale = 4 and lq is the same size as gt, it should be 0.25
        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        if lq_path != gt_path:
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
        else:
            img_lq = img_gt
        ol_path = self.paths[index]['ol_path']
        if ol_path != gt_path:
            img_bytes = self.file_client.get(ol_path, 'ol')
            img_ol = imfrombytes(img_bytes, flag='grayscale', float32=True)[...,None]
        else:
            _, w = img_gt.shape[:2]
            img_gt, img_ol = img_gt[:,:w//2], img_gt[:,w//2:]
            img_ol = 0.114 * img_ol[...,0:1] + 0.587 * img_ol[...,1:2] + 0.299 * img_ol[...,2:3] # BGR to Gray
            if lq_path == gt_path:
                img_lq = img_gt

        if pre_scale != 1:
            h, w = img_lq.shape[:2]
            rand_scale = pre_scale if random.random() < 0.5 else pre_scale * random.uniform(0.8,1)
            h2, w2 = int(h*pre_scale), int(w*pre_scale)
            h3, w3 = int(h*rand_scale), int(w*rand_scale)
            img_lq = cv2.resize(img_lq, (h3,w3), interpolation=cv2.INTER_LINEAR)
            if pre_scale != rand_scale:
                img_lq = cv2.resize(img_lq, (h2,w2), interpolation=cv2.INTER_LINEAR)

        img_gt = np.concatenate([img_gt, img_ol], -1)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, img_ol = img_gt[...,:3], img_gt[...,3:]

        if scale != 1:
            h,w = img_ol.shape[:2]
            hf = h // scale
            wf = w // scale
            img_ol = np.reshape(
                np.transpose(
                    np.reshape(
                        img_ol,
                        [hf,scale,wf,scale]
                    ),
                    [0,2,1,3]
                ),
                [hf,wf,-1]
            )

        # add noise to outlines image
        if random.random() < 0.5:
            temp = random.uniform(0.01,0.2)
            img_ol = img_ol + temp * (np.random.randn(*img_ol.shape) + 1) / 2
        if random.random() < 0.5:
            img_ol = (img_ol + random.uniform(-1.,1.)) * random.uniform(0.5,2.)
        if random.random() < 0.5:
            img_ol = img_ol * 0 + 1

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_ol = img2tensor([img_gt, img_lq, img_ol], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        img_lq = torch.cat([img_lq, img_ol], 0)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, 'ol_path': ol_path}

    def __len__(self):
        return len(self.paths)
