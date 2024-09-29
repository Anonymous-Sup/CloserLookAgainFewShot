import os
import math
import torch
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image
# from .samplers import FewshotBatchSampler, RandomSampler, ValSampler
from .sampler import FewshotSampler, RandomIdentitySampler

from .bases import ImageDataset
from .skechy_fewshot import Sketchy


import torchvision.transforms as T
from torch.utils.data import DataLoader
from timm.data.random_erasing import RandomErasing

# __factory = {
#     'market1501': Market1501,
#     'dukemtmc': DukeMTMCreID,
#     'msmt17': MSMT17,
#     'occ_duke': OCC_DukeMTMCreID,
#     'veri': VeRi,
#     'VehicleID': VehicleID,
#     'market-sketch': MarketSketch,
#     'sysu_mm01': SYSU_MM01,
#     'sksf_a': SKSF_A,
#     'celebHQ': CELAB_HQR,
#     'sketchy': Sketchy,
#     'tuberlin': TUBerlin,
#     'market-sketch-fewshot': MarketSketch_FewShot,
#     'sksf_a_fewshot': SKSF_A_FewShot,
# }

__factory = {
    'sketchy': Sketchy,
}



def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids, img_paths= zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids, img_paths


SIZE_TRAIN = [256,128]
SIZE_TEST =  [256,128]
PROB = 0.5
PADDING = 10
PIXEL_MEAN = [0.5, 0.5, 0.5]
PIXEL_STD = [0.5, 0.5, 0.5]
RE_PROB = 0.5


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=PROB),
            T.Pad(PADDING),
            T.RandomCrop(SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
            RandomErasing(probability=RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    ])

    num_workers = 8

    dataset = __factory[cfg.DATASETS](root=cfg.DATA_ROOT, config=cfg)

    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    train_loader = DataLoader(
        train_set, batch_size=cfg.DATA.TRAIN.BATCH_SIZE,
        sampler=RandomIdentitySampler(dataset.train, cfg.DATA.TRAIN.BATCH_SIZE, 4),
        num_workers=num_workers, collate_fn=train_collate_fn
    )

    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    visual_train_loader = DataLoader(
        train_set_normal, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn)


    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = DataLoader(
        query_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(
        gallery_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
    return train_loader, visual_train_loader, val_loader, query_loader, gallery_loader, len(dataset.query), num_classes, cam_num, view_num



def make_fewshot_dataloader(cfg):

    train_transforms = T.Compose([
        T.Resize(SIZE_TRAIN, interpolation=3),
        T.RandomHorizontalFlip(p=PROB),
        T.Pad(PADDING),
        T.RandomCrop(SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD),
        RandomErasing(probability=RE_PROB, mode='pixel', max_count=1, device='cpu'),
        # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
    ])

    val_transforms = T.Compose([
        T.Resize(SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)
    ])

    num_workers = 8

    dataset = __factory[cfg.DATASETS](root=cfg.DATA_ROOT, config=cfg)
    
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)

    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids


    train_loader = DataLoader(
        train_set, batch_size=cfg.FEWSHOT.NWAY*cfg.FEWSHOT.KSHOT,
        # data_source, num_classes, num_instances, num_episodes):
        sampler=FewshotSampler(dataset.train, cfg.FEWSHOT.NWAY, cfg.FEWSHOT.KSHOT, cfg.FEWSHOT.EPISODE),
        num_workers=num_workers, collate_fn=train_collate_fn
    )


    val_set = ImageDataset(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    visual_train_loader = DataLoader(
        train_set_normal, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=train_collate_fn)


    query_set = ImageDataset(dataset.query, val_transforms)
    query_loader = DataLoader(
        query_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    gallery_set = ImageDataset(dataset.gallery, val_transforms)
    gallery_loader = DataLoader(
        gallery_set, batch_size=128, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num
    return train_loader, visual_train_loader, val_loader, query_loader, gallery_loader, len(dataset.query), num_classes, cam_num, view_num