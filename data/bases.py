from PIL import Image, ImageFile, UnidentifiedImageError

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
ImageFile.LOAD_TRUNCATED_IMAGES = True

import collections
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    # while not got_img:
    #     try:
    #         img = Image.open(img_path).convert('RGB')
    #         got_img = True
    #     except IOError:
    #         print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
    #         pass
    try:
        img = Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Cannot identify image file {img_path}")
        return None
    except Exception as e:
        print(f"Error processing file {img_path}: {e}")
        return None
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []
        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery, val=None):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)
        if val is not None:
            num_val_pids, num_val_imgs, num_val_cams, num_train_views = self.get_imagedata_info(val)
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
            print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
            print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
            print("  val      | {:5d} | {:8d} | {:9d}".format(num_val_pids, num_val_imgs, num_val_cams))
            print("  ----------------------------------------")
        else:
            print("Dataset statistics:")
            print("  ----------------------------------------")
            print("  subset   | # ids | # images | # cameras")
            print("  ----------------------------------------")
            print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
            print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
            print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
            print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        img_path, pid, camid, trackid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, trackid, img_path.split('/')[-1]
    

class FEWSHOT_ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, is_few_shot=False, n_support=5, n_query=15, seed=0):
        """
        dataset: a list of (img_path, pid, camid, trackid)
        transform: image transformations
        is_few_shot: whether to use few-shot learning structure
        n_support: number of support images
        n_query: number of query images
        seed: random seed for reproducibility
        """
        self.dataset = dataset
        self.transform = transform
        self.is_few_shot = is_few_shot
        self.n_support = n_support
        self.n_query = n_query
        self._rng = np.random.RandomState(seed)
        self.class_to_indices = self._group_by_class()

    def _group_by_class(self):
        """Group dataset indices by class (pid)"""
        class_to_indices = collections.defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.dataset):
            class_to_indices[pid].append(index)
        return class_to_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.is_few_shot:
            # Regular dataset structure
            img_path, pid, camid, trackid = self.dataset[index]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return img, pid, camid, trackid, img_path.split('/')[-1]
        else:
            # Few-shot structure
           # Few-shot structure: directly use the pid from the dataset without recalculating it
            indices = self._rng.choice(self.class_to_indices[self.dataset[index][1]], self.n_support + self.n_query, replace=False)
            support_indices = indices[:self.n_support]
            query_indices = indices[self.n_support:]

            # pid = list(self.class_to_indices.keys())[index % len(self.class_to_indices)]
            # indices = self._rng.choice(self.class_to_indices[pid], self.n_support + self.n_query, replace=False)
            # support_indices = indices[:self.n_support]
            # query_indices = indices[self.n_support:]
            
            support_set = []
            query_set = []
            support_labels = []
            query_labels = []
            
            for i in support_indices:
                img_path, pid, camid, trackid = self.dataset[i]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                support_set.append(img)
                support_labels.append(pid)
            
            for i in query_indices:
                img_path, pid, camid, trackid = self.dataset[i]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                query_set.append(img)
                query_labels.append(pid)
                
            support_set = torch.stack(support_set)
            query_set = torch.stack(query_set)
            support_labels = torch.tensor(support_labels, dtype=torch.int64)
            query_labels = torch.tensor(query_labels, dtype=torch.int64)

            
            return support_set, query_set, support_labels, query_labels