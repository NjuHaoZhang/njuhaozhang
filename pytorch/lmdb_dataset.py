# 展示了pytorch下 如何用 lmdb 构建不同的 dataset code, lmdb的好处是读取极其快速
# code borrowed from 

# 1. 存储item为 feature map tensor, 直接用 pickle load/store
# code borrowed from https://github.com/rosinality/vq-vae-2-pytorch/blob/master/dataset.py

# ============ #
# dataset.py
# =============#
import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
import lmdb

CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])

class LMDBDataset(Dataset):
    def __init__(self, path):
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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename

# ====================== #
# dataset_loading.py
# ======================#
from dataset import LMDBDataset
dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )
for i, (top, bottom, label) in enumerate(loader):
    pass


# ============================================================================================= #


# 存储item为 图片的二进制, 用 BytesIO and Image.open() load
# code borrowed from https://github.com/rosinality/stylegan2-pytorch/edit/master/prepare_data.py


# =======================#
# create_lmdb.py
# =======================#
import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val

# image pyramid
def resize_multiple(img, sizes=(128, 256, 512, 1024),
                    resample=Image.LANCZOS, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs

# each process 的 worker (用multi_process)
def resize_worker(img_file, sizes, resample):
    i, file = img_file # idx, img_path
    img = Image.open(file)
    img = img.convert('RGB') # to rgb numpy
    out = resize_multiple(img, sizes=sizes, resample=resample)

    return i, out # i 应该是用来 做 unit_test 来 check code用的


def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024),
        resample=Image.LANCZOS, num_width=6): # 000000~999999,1 Million image
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    files = sorted(dataset.imgs, key=lambda x: x[0]) # [(img_path, label_idx),...] N个
    files = [(i, file) for i, (file, label) in enumerate(files)] # 不用label
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)): # for N img
            # pool.imap_unordered 可以看做多进程并发做 异步不保序的 map
            # 此处的imap_unordered返回 N个(i,imgs), imgs: [img_1, ...img_s]
            for size, img in zip(sizes, imgs): # for S sclae_size
                key = f'{size}-{str(i).zfill(num_width)}'.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--size', type=str, default='128,256,512,1024')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resample', type=str, default='lanczos')
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    resample_map = {'lanczos': Image.LANCZOS, 'bilinear': Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)

# ===================================================================== #


# ========================= #
# dataset.py
# ========================= #
from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
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

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes) # lmdb 存的是 img的二进制(也可以存其他格式)
        img = Image.open(buffer)
        img = self.transform(img) # online transform (因为有些offline没法做)

        return img


# =================================== #
# lmdb_loading.py
# =================================== #
from torch.utils import data

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def train(transform, args):
    '''
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    '''
    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True,
                             distributed=args.distributed),
        # 为了支持 分布式的sample, 所以手动指定了sampler
        drop_last=True,
    )
    # 注意 num_worker = 0 ,是因为 多进程读 lmdb 有问题？？？
    # 但是由于 lmdb 读取(尤其还是二进制) 极其快速，所以无须主动预取
    '''
    如果num_worker设为0，意味着每一轮迭代时，
    dataloader不再有自主加载数据到RAM这一步骤（因为没有worker了），
    而是在RAM中找batch，找不到时再加载相应的batch。缺点当然是速度更慢。
    '''

    # 用 iter_step 控制结束
    pbar = range(args.iter) # args.iter：step_num, e.g. 800000 step
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

    loader = sample_data(loader)
    real_img = next(loader) # generator 方式获得sample
    # for loader 也可以，本code这么写是为了使用 分布式 (原项目是分布式的)
