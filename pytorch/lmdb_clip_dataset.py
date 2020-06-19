# code borrowed from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/dataset.py


import os
import pickle
from collections import namedtuple
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb
import numpy as np
from PIL import Image

rng = np.random.RandomState(2017) # 一定要固定 rand seed 然后用它 (确保实验可复现)


class LMDBDataset_clip_train(Dataset):
    def __init__(self, path, data_type, len_clip=9, num_width=6, transform=None):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.num_sub_video = int(txn.get('num_sub_video'.encode('utf-8')).decode('utf-8'))
            self.list_len_each_sub_video = int(txn.get('list_len_each_sub_video'.encode('utf-8')).decode('utf-8'))

        #
        self.len_clip = len_clip
        self.num_width = num_width
        self.data_type = data_type
        self.transform = transform # online transform
        # online transform based on mode (本proj没有，全部在 offline做好了transform)

    # 因为只有dataloader 才能激发 __len__(), 所以 testing mode要直接控制循环长度
    def __len__(self):
        num_clip = self.length- self.len_clip + 1
        #
        list_num_clip = [len_cur_sub_video - self.len_clip + 1
         for len_cur_sub_video in self.list_len_each_sub_video]
        assert num_clip == sum(list_num_clip), "LMDBDataset_op_train __len__ error"

        return num_clip


    def __getitem__(self, index):
        # 两级寻址: sub_vdeo -> frame_start -> clip[start, end), 取一个clip
        sub_vid = rng.randint(0, self.num_sub_video)
        cur_cid = rng.randint(0, self.list_len_each_sub_video[sub_vid] - self.len_clip)
        #
        key_list = [f'{sub_vid}-{str(cur_cid+i).zfill(self.num_width)}'.encode('utf-8')
                    for i in range(self.len_clip)]

        with self.env.begin(write=False) as txn:
            tmp_clip = []
            for key in key_list:
                sample = txn.get(key)
                sample_tensor = self.get_sample(self.data_type, sample, self.transform)
                tmp_clip.append(sample_tensor)

        return torch.stack(tmp_clip)

    def get_sample(self, data_type, sample, transform=None):
        if data_type == "rgb":
            buffer = BytesIO(sample)  # lmdb 存的是 img的二进制
            img = Image.open(buffer) # PIL可以被 torchvision.transfrom操作
        elif data_type == "op":
            img = pickle.loads(sample)  # 因为之前将 numpy做了pickle.dumps()
            img = torch.from_numpy(img)
        else:
            print("get_sample error")
            exit()
        #
        if transform:
            img = transform(img)
        return img


# 注意：train and test 确实有些code雷同，但差别也是有的，没必要再抽象个base_class
# 过分优化 和 不优化 一样是不对的，优化是为了方便少出错
# test 和 train 的3处不同:
# (1)__len__(), (2) test() 设置 cur_sub_vid, (3) __getitem__()的两级索引值
class LMDBDataset_clip_test(Dataset):
    def __init__(self, path, data_type, len_clip=9, num_width=6, transform=None):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            self.num_sub_video = int(txn.get('num_sub_video'.encode('utf-8')).decode('utf-8'))
            self.list_len_each_sub_video = int(txn.get('list_len_each_sub_video'.encode('utf-8')).decode('utf-8'))

        #
        self.len_clip = len_clip
        self.num_width = num_width
        self.data_type = data_type
        self.transform = transform # online transform
        # online transform based on mode (本proj没有，全部在 offline做好了transform)

    # 因为只有dataloader 才能激发 __len__(), 所以 testing mode要直接控制循环长度
    def __len__(self):
        num_clip = self.list_len_each_sub_video[self.cur_sub_vid] - self.len_clip + 1

        return num_clip

    def test(self, sub_vid):  # 外部调用,设置sub_vid
        self.cur_sub_vid = sub_vid

    def __getitem__(self, index):
        # 两级寻址: sub_vdeo -> frame_start -> clip[start, end), 取一个clip
        sub_vid = self.cur_sub_vid
        cur_cid = index
        #
        key_list = [f'{sub_vid}-{str(cur_cid+i).zfill(self.num_width)}'.encode('utf-8')
                    for i in range(self.len_clip)]

        with self.env.begin(write=False) as txn:
            tmp_clip = []
            for key in key_list:
                sample = txn.get(key)
                sample_tensor = self.get_sample(self.data_type, sample, self.transform)
                tmp_clip.append(sample_tensor)

        return torch.stack(tmp_clip)

    def get_sample(self, data_type, sample, transform=None):
        if data_type == "rgb":
            buffer = BytesIO(sample)  # lmdb 存的是 img的二进制
            img = Image.open(buffer) # PIL可以被 torchvision.transfrom操作
        elif data_type == "op":
            img = pickle.loads(sample)  # 因为之前将 numpy做了pickle.dumps()
            img = torch.from_numpy(img)
        else:
            print("get_sample error")
            exit()
        #
        if transform:
            img = transform(img)
        return img


class LMDBDataset_twostream_train(Dataset):
    def __init__(self, path, data_type=("rgb", "op"), len_clip=(10,9),
                 num_width=(6,6), transform=(None,None)):

        self.lmdb_rgb = LMDBDataset_clip_train(path=path[0],
               data_type=data_type[0], len_clip=len_clip[0],
               num_width=num_width[0], transform=transform[0])
        self.lmdb_op = LMDBDataset_clip_train(path=path[1],
               data_type=data_type[1], len_clip=len_clip[1],
               num_width=num_width[1], transform=transform[1])

    # 因为只有dataloader 才能激发 __len__(), 所以 testing mode要直接控制循环长度
    def __len__(self):
        assert len(self.lmdb_rgb)==len(self.lmdb_op), "LMDBDataset_twostream_train error"
        return len(self.lmdb_rgb)

    def __getitem__(self, index):
        rgb_clip_tensor = self.lmdb_rgb.__getitem__(index)
        op_clip_tensor = self.lmdb_op.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op":op_clip_tensor}


class LMDBDataset_twostream_test(Dataset):
    def __init__(self, path, data_type=("rgb", "op"), len_clip=(10, 9),
                 num_width=(6, 6), transform=(None, None)):
        self.lmdb_rgb = LMDBDataset_clip_test(path=path[0],
               data_type=data_type[0], len_clip=len_clip[0],
               num_width=num_width[0], transform=transform[0])
        self.lmdb_op = LMDBDataset_clip_test(path=path[1],
                data_type=data_type[1], len_clip=len_clip[1],
                num_width=num_width[1], transform=transform[1])

    # 因为只有dataloader 才能激发 __len__(),
    # 所以 testing mode要直接控制循环长度, 特别小心
    def __len__(self):
        assert len(self.lmdb_rgb) == len(self.lmdb_op), "LMDBDataset_twostream_train error"
        return len(self.lmdb_rgb)

    def test(self, sub_vid):  # 外部调用,设置sub_vid
        self.lmdb_rgb.test(sub_vid)
        self.lmdb_op.test(sub_vid)

    def __getitem__(self, index):
        rgb_clip_tensor = self.lmdb_rgb.__getitem__(index)
        op_clip_tensor = self.lmdb_op.__getitem__(index)

        return {"rgb": rgb_clip_tensor, "op": op_clip_tensor}
        
  # unittest, TODO
  # main
