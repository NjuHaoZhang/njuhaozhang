# 我根据 vda dataset 的 两级目录总结的一份 code, 对于 pytorch 新手而言，非常规范适合学习。
# 但是由于，只有在 get item 时才真正 (1)IO: 从磁盘读取图片到内存 -> (2) cpu 执行 图片解码、图片预处理 -> (3) data cpu2gpu
# 所以 数据加载非常缓慢
# 由于 pytorch 原生的 dataloader的问题，导致这个问题更加严重。最后会导致 gpu 计算率很低
# 那么下一个代码就是从 上述的 data pipeline 着手，优化 数据加载的速度

import os,glob,sys
sys.path.append('..')
import cv2
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
# from .torch_videovision.videotransforms import (
#     video_transforms, volume_transforms)
from utils.flowlib import readFlow, flow_to_image, batch_flow_to_image
from torchvision.utils import make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from tensorboardX import SummaryWriter
from PIL import Image

from utils import utils

rng = np.random.RandomState(2017) # 一定要固定 rand seed 然后用它 (确保实验可复现)

# TODO: 这个 dataset 设计极具普适性，后面我要结合 流程图xmind, 还有UML导图，
# 来讲解这个代码的设计
# 顺便 讲下其他任务的 普适的 dataset code 设计 (要学会举一反三)
# 本质有4点：(1) 设计一个多级寻址的方案；(2) 数据预处理逻辑；(3) 测试代码(非常重要！！！)，
# (4) test 与 train phase 的核心差异

class TwoStream_Test_DS(Dataset):

    def __init__(self, video_folder, clip_length=(10,9), size=(256, 256)):
        super(TwoStream_Test_DS, self).__init__()
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = {"rgb": video_folder["rgb"], "op": video_folder["op"],}
        self.videos = {"rgb": OrderedDict(), "op": OrderedDict(),}
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10,9
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        self.transform = {
            "rgb": transforms.Compose([
                    transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)]
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size),
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False
                ]),
        }

    def _setup(self):
        videos = {
            "rgb": glob.glob(os.path.join(self.dir["rgb"], '*')),  # frames下面的若干个子目录的path：path/to/01, 02...
            "op": glob.glob(os.path.join(self.dir["op"], '*')),
        }
        for video in sorted(videos["rgb"]):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos["rgb"][sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            # {'path':all_sub_video_path(e.g. [/path/to/01, /path/to/02 ...]),
            #  'frame':all_frame_of_sub_video_path(e.g. [/path/to/01/00001.jpg, /path/to/01/00002.jpg ...]),
            # 'length':len_of_sub_video(e.g. len(01_frames), len(02_frames), ...),即子目录下帧的数目}
            # 并且将 UCSD/ped2 and ped1 的training set frame的channel都转为3了
            # 注：SIST的liu wen已经将 ped1/ped2都从tif转为jpg,并且对数据集做了处理，所以jpg格式统一一切
            self.videos["rgb"][sub_video_name]['path'] = video
            self.videos["rgb"][sub_video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos["rgb"][sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos["rgb"][sub_video_name]['length'] = \
                len(self.videos["rgb"][sub_video_name]['frame'])
        for video in sorted(videos["op"]):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos["op"][sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            self.videos["op"][sub_video_name] = {}
            self.videos["op"][sub_video_name]['path'] = video
            self.videos["op"][sub_video_name]['frame'] = glob.glob(os.path.join(video, '*.flo'))
            self.videos["op"][sub_video_name]['frame'].sort()
            self.videos["op"][sub_video_name]['length'] = \
                len(self.videos["op"][sub_video_name]['frame'])
        # self.sub_video_name_list = sorted(list(self.videos["frame"].keys()))  # 为01,02...,sub_video_name构成的list

    def test(self, sub_video_name):
        # 这个辅助函数 是 多级目录索引寻址的 核心！直接确定一级目录 (需要在主函数被显式调用)
        self.cur_sub_video_name = sub_video_name
        # 下面是核心，控制 __len__()
        self.cur_len = self.videos["rgb"][sub_video_name]['length'] - \
                       self.clip_length[0] + 1
        assert self.videos["rgb"][sub_video_name]['length'] - \
                       self.clip_length[0] + 1 == \
               self.videos["op"][sub_video_name]['length'] - \
                       self.clip_length[1] + 1
        # print(self.videos["rgb"][sub_video_name]['length'])
        # print(self.clip_length[0])
        # print("cur len", self.cur_len) # 用 assert
        # 注意 self.videos["op"][sub_video_name]['frame'] 和
        # self.videos["op"][sub_video_name]['frame'] 已经预加载了 全部 frame 的 path
        #
        # len(tensor) == tensor.size()[0] or tensor.shape[0]
        #  cur_len 求解举例： [1,2,3,4,5]以3为window_size，得到clips依次为：
        # [1,2,3], [2,3,4], [3,4,5], 即 total_len - window_size + 1 (本例：5 - 3 + 1 == 3)

    def __len__(self):
        return self.cur_len # cur_sub_video_op, len(dataset),即 num of getitem()

    def __getitem__(self, idx):
        # 本函数是核心，设计好 idx在整个 item_addr的功能，然后得到最终的 absolute_addr of item
        # (根据 idx 得到 一个 idx_list 就能得到 items)
        # idx提供frame - ids, 进而得到absolute_addr of frames 取一个clip
        #
        # (1) 一级寻址：get sub_video, 由外部调用 test() 设置
        sub_video_name = self.cur_sub_video_name
        # (2) 二级寻址：get frame_idx 并 组合得到 文件真实的绝对路径，并完成 真正加载到内存
        frame_path_list = self.videos["rgb"][sub_video_name]['frame'][idx: idx+self.clip_length[0]]
        op_path_list = self.videos["op"][sub_video_name]['frame'][idx: idx+self.clip_length[1]]
        # (3) load frame and op
        frame_clip = self._load_frames(frame_path_list)
        op_clip = self._load_ops(op_path_list)
        sample = {"rgb": frame_clip, "op": op_clip}

        return sample

    def _load_frames(self, img_list):
        all_clip = []
        for img_path in img_list:       # [start,end)
            img = cv2.imread(img_path)  # Note: output [h, w, c]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # perform transform, TODO
            if self.transform:
                # img = Image.fromarray(img)
                img = self.transform["rgb"](img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  # 合并为更高一维, 把所有的frame都合并到一个大 tensor中
        return test_clip

    def _load_ops(self, img_list):
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (self.image_height,self.image_width))
            # print(img.shape)
            if self.transform:
                img = self.transform["op"](img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中
        return test_clip


class TwoStream_Train_DS(Dataset):

    def __init__(self, video_folder, clip_length=(10,9), size=(256, 256)):
        super(TwoStream_Train_DS, self).__init__()
        # /path/to/testing, 下面 有 frames/, xxx.mat, pixel_masks/
        self.dir = {"rgb": video_folder["rgb"], "op": video_folder["op"],}
        self.videos = {"rgb": OrderedDict(), "op": OrderedDict(),}
        self.image_height = size[0]
        self.image_width = size[1]
        self.clip_length = clip_length # 直接取10,9
        self._setup()
        #
        # Other utilities by call test() in main.py
        self.cur_len = 0  # 用于 __len__(), 即 dataset length
        self.cur_sub_video_name = None  # 当前处于哪个 sub_video, 即 01, 02...
        # self.cur_sub_video_frames = None
        # transform 默认仅仅是 Totensor()
        self.transform = {
            "rgb": transforms.Compose([
                    transforms.ToPILImage(), # 必须 PIL input 才能用resize
                    transforms.Resize(size), # 所以这个api不如丢到 load里面用np做
                    transforms.ToTensor(),  # return tensor [(c,seq_len,h,w)]
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]),
            "op": transforms.Compose([
                    # transforms.Resize(size),
                    transforms.ToTensor(),# channel_nb=2, div_255=False, numpy=False
                ]),
        }

    def _setup(self):
        videos = {
            "rgb": glob.glob(os.path.join(self.dir["rgb"], '*')),  # frames下面的若干个子目录的path：path/to/01, 02...
            "op": glob.glob(os.path.join(self.dir["op"], '*')),
        }
        for video in sorted(videos["rgb"]):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos["rgb"][sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            # {'path':all_sub_video_path(e.g. [/path/to/01, /path/to/02 ...]),
            #  'frame':all_frame_of_sub_video_path(e.g. [/path/to/01/00001.jpg, /path/to/01/00002.jpg ...]),
            # 'length':len_of_sub_video(e.g. len(01_frames), len(02_frames), ...),即子目录下帧的数目}
            # 并且将 UCSD/ped2 and ped1 的training set frame的channel都转为3了
            # 注：SIST的liu wen已经将 ped1/ped2都从tif转为jpg,并且对数据集做了处理，所以jpg格式统一一切
            self.videos["rgb"][sub_video_name]['path'] = video
            self.videos["rgb"][sub_video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos["rgb"][sub_video_name]['frame'].sort()  # 一定要排序，确保order正确，在glob前面加个sorted ?
            self.videos["rgb"][sub_video_name]['length'] = \
                len(self.videos["rgb"][sub_video_name]['frame'])
        for video in sorted(videos["op"]):
            sub_video_name = video.split('/')[-1]  # 01, 02, 03... (frame和op都是相同的sub_dirs)
            self.videos["op"][sub_video_name] = {}  # self.videos是一个OrderedDict: key为01,02..., val为{},如下
            self.videos["op"][sub_video_name] = {}
            self.videos["op"][sub_video_name]['path'] = video
            self.videos["op"][sub_video_name]['frame'] = glob.glob(os.path.join(video, '*.flo'))
            self.videos["op"][sub_video_name]['frame'].sort()
            self.videos["op"][sub_video_name]['length'] = \
                len(self.videos["op"][sub_video_name]['frame'])
        # self.sub_video_name_list = sorted(list(self.videos["frame"].keys()))  # 为01,02...,sub_video_name构成的list

    def __len__(self):
        clips_num_list = self._get_clips_num_list()
        return sum(clips_num_list) # all clip_num

    def _get_clips_num_list(self):
        sub_video_name_list = sorted(list(self.videos["rgb"].keys()))
        clips_num_list = [(self.videos["rgb"][sub_video_name]['length']
                           - self.clip_length[0] + 1)
                for sub_video_name in sub_video_name_list]
        return clips_num_list

    def __getitem__(self, idx):
        # 本函数是核心，设计好 idx在整个 item_addr的功能，然后得到最终的 absolute_addr of item
        # (根据 idx 得到 一个 idx_list 就能得到 items)
        # idx提供frame - ids, 进而得到absolute_addr of frames 取一个clip
        #
        # (1) 一级寻址：get sub_video, 由外部调用 test() 设置
        # sub_video_name = self.cur_sub_video_name
        sub_video_name_list = sorted(list(self.videos["rgb"].keys()))
        sub_vid = rng.randint(0, len(sub_video_name_list))
        sub_video_name = sub_video_name_list[sub_vid]
        # (2) 二级寻址：get frame_idx 并 组合得到 文件真实的绝对路径，并完成 真正加载到内存
        cur_cid = rng.randint(0,
              self.videos["rgb"][sub_video_name]['length'] - self.clip_length[0])
        # randint: [`low`, `high`)
        frame_path_list = self.videos["rgb"][sub_video_name]['frame'][cur_cid:
                            cur_cid+self.clip_length[0]]
        op_path_list = self.videos["op"][sub_video_name]['frame'][cur_cid:
                            cur_cid+self.clip_length[1]]
        # (3) load frame and op
        frame_clip = self._load_frames(frame_path_list)
        op_clip = self._load_ops(op_path_list)
        sample = {"rgb": frame_clip, "op": op_clip}

        return sample

    def _load_frames(self, img_list):
        all_clip = []
        for img_path in img_list:       # [start,end)
            img = cv2.imread(img_path)  # Note: output [h, w, c]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # perform transform, TODO
            if self.transform:
                # img = Image.fromarray(img)
                img = self.transform["rgb"](img)
            all_clip.append(img)  # list
        test_clip = torch.stack(all_clip)  # 合并为更高一维, 把所有的frame都合并到一个大 tensor中
        return test_clip

    def _load_ops(self, img_list):
        img_clip = []
        for img_path in img_list:
            img = readFlow(img_path)  # Note: output [h, w, c]
            # print(img.shape)
            img = cv2.resize(img, (self.image_height,self.image_width))
            # print(img.shape)
            if self.transform:
                img = self.transform["op"](img)
            # img = torch.tensor(img).view(0,2,1)
            # print(img.shape)
            img_clip.append(img)  # list
        test_clip = torch.stack(img_clip)  # 合并为更高一维的tensor, 把所有的frame都合并到一个大 tensor中
        return test_clip


# ============================================================ #
# unit test for TwoStream_Test_DS
class test_TwoStream_Test_DS():
    # 测试 子目录
    def test_1(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            dataset.test(sub_video_name)
            print(dataset.cur_sub_video_name)

    # 测试 一些数目 是否正确 (for len(dataset)方式)
    def test_2(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            print(sub_video_name)
            dataset.test(sub_video_name)
            print(len(dataset),'--', dataset.clip_length, '--', dataset.cur_len)
            # for idx, sample in enumerate(dataset): # enumerate 似乎有问题?TODO
            for idx in range(len(dataset)):
                sample = dataset[idx]
                print(idx) # 1429 in avenue-01, since 1438-9+1 = 1430
                print(sample["rgb"].shape, '--', sample["op"].shape)
            #
            # 测试 iter_num (没有batch,等于frame_num; 有batch, 等于 ceil(frame_num/batch_size))
            assert len(dataset) == idx + 1, "len(dataset) != idx+1"
            # 测试 rgb and op num
            frame_num = dataset.videos["rgb"][sub_video_name]['length']
            op_num = dataset.videos["op"][sub_video_name]['length']
            assert frame_num == op_num + 1, "frame_num != op_num + 1"
            # 测试 clip_num
            c1 = frame_num - dataset.clip_length[0] + 1
            c2 = op_num - dataset.clip_length[1] + 1
            assert c1 == c2 and c1 == len(dataset), "error, c1==c2 and c1==len(dataset)"
            #
            break  # 只处理第一个sub_video 即可
        print("exit succ")

    # 测试 一些数目 是否正确 (dataloader方式)
    def test_3(self, dataset, all_sub_video_name_list):
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            print(sub_video_name)
            dataset.test(sub_video_name)
            print(len(dataset),'--', dataset.clip_length, '--', dataset.cur_len)
            batch_size = 1
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
            for idx, sample in enumerate(loader): # enumerate 似乎有问题?TODO
            # for idx in range(len(dataset)):
                # sample = dataset[idx]
                print(idx) # 1429 in avenue-01, since 1438-9+1 = 1430
                print(sample["rgb"].shape, '--', sample["op"].shape)
            #
            # 测试 iter_num (没有batch,等于frame_num; 有batch, 等于 ceil(frame_num/batch_size))
            assert len(dataset) == idx + 1, "len(dataset) != idx+1"
            # 测试 rgb and op num
            frame_num = dataset.videos["rgb"][sub_video_name]['length']
            op_num = dataset.videos["op"][sub_video_name]['length']
            assert frame_num == op_num + 1, "frame_num != op_num + 1"
            # 测试 clip_num
            c1 = frame_num - dataset.clip_length[0] + 1
            c2 = op_num - dataset.clip_length[1] + 1
            assert c1 == c2 and c1 == len(dataset), "error, c1==c2 and c1==len(dataset)"
            #
            break  # 只处理第一个sub_video 即可
        print("exit succ")

    # 测试 单个 sample & vis (vis 非常完美，这套code可以继承下去)
    def test_4(self, dataset, all_sub_video_name_list,writer):

        vis_info = "vis_test_4"
        for video_id, sub_video_name in enumerate(all_sub_video_name_list):
            dataset.test(sub_video_name)
            #
            if video_id == 0:
                batch_size = 1
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                for idx, sample in enumerate(loader):
                    rgb = sample["rgb"][0] # 因为 它是 (b,t,c,h,w)
                    op = sample["op"][0] # 默认情况下 batch 是分别施加到 rgb and op
                    if idx==0: # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape) # rgb, [10,3,256,256]
                        print(op.shape) # op, [9,2,256,256]
                        print(rgb.min(), rgb.max()) # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

                    if idx==len(dataset)-1: # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

            if video_id == len(all_sub_video_name_list)-1:
                batch_size = 1
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                for idx, sample in enumerate(loader):
                    rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
                    op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
                    if idx == 0:  # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb":rgb, "op":op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)

                    if idx == len(dataset) - 1:  # 第一个和最后一个clip
                        # print(sample)
                        print(rgb.shape)  # rgb, [10,3,256,256]
                        print(op.shape)  # op, [9,2,256,256]
                        print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
                        print(op.min(), op.max())  # 测 op value
                        # vis to compare with gt
                        sample = {"rgb": rgb, "op": op}
                        vis_load_gt(dataset, sub_video_name, idx,
                                    sample, writer, vis_info)


# unit test for TwoStream_Train_DS
class test_TwoStream_Train_DS():
    # 测试 子目录/
    def test_1(self, dataset):
        batch_size = 1
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print(rgb.shape)  # rgb, [10,3,256,256]
            print(op.shape)  # op, [9,2,256,256]
            print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
            print(op.min(), op.max())  # 测 op value
            #
            print(len(dataset)) # num_all_clip
            print(dataset._get_clips_num_list(), sum(dataset._get_clips_num_list()))
            break

    # 测试 iter_num (ceil(num_all_clip/batch_size))
    def test_2(self, dataset):
        batch_size = 32
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print(rgb.shape)  # rgb, [10,3,256,256]
            print(op.shape)  # op, [9,2,256,256]
            print(rgb.min(), rgb.max())  # 测 rgb value:[-1,1]
            print(op.min(), op.max())  # 测 op value
            #
            print(len(dataset)) # num_all_clip
            print(dataset._get_clips_num_list(), sum(dataset._get_clips_num_list()))
            print("iter: ",idx)
        assert idx + 1 == torch.ceil(len(dataset) / batch_size)

    # 测试 单个 sample & vis (vis 非常完美，这套code可以继承下去)
    def test_3(self, dataset, writer):

        vis_info = "vis_train_test_4"
        batch_size = 32
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print("iter: ", idx)
            if idx==0: # 第一个和最后一个clip
                # vis to compare with gt
                sample = {"rgb": rgb, "op": op}
                vis_load(sample, writer, vis_info, idx)

            if idx==len(loader)-1: # 第一个和最后一个clip
                sample = {"rgb": rgb, "op": op}
                vis_load(sample, writer, vis_info, idx)

        assert idx + 1 == torch.ceil(len(dataset) / batch_size)

    def test_shanghaitech_bug(self,dataset):
        batch_size = 4
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
        for idx, sample in enumerate(loader):
            rgb = sample["rgb"][0]  # 因为 它是 (b,t,c,h,w)
            op = sample["op"][0]  # 默认情况下 batch 是分别施加到 rgb and op
            print("iter: ", idx)
# ============================================================ #
# train
def vis_load(sample, writer, vis_info, idx): # (t,c,h,w)
    seq_len = sample["rgb"].size()[0]
    grid_rgb = get_vis_tensor(sample["rgb"], "rgb", seq_len)
    writer.add_image(vis_info+"/rgb_iter=_{}".format(idx), grid_rgb)
    seq_len = sample["op"].size()[0]
    grid_op = get_vis_tensor(sample["op"], "op", seq_len)
    writer.add_image(vis_info + "/op_iter=_{}".format(idx), grid_op)

# test
def vis_load_gt(dataset, sub_video_name, idx,
                sample, writer, vis_info):
    #
    rgb_list = dataset.videos["rgb"][sub_video_name]['frame'][idx:
    idx + dataset.clip_length[0]]
    op_list = dataset.videos["op"][sub_video_name]['frame'][idx:
    idx + dataset.clip_length[1]]
    rgb, op = dataset._load_frames(rgb_list), dataset._load_ops(op_list)
    # transform
    # rgb, op = transform["rgb"](rgb), transform["op"](op)
    rgb_load_gt = torch.cat([sample["rgb"], rgb], 0)
    op_load_gt = torch.cat([sample["op"], op], 0)
    #
    seq_len = sample["rgb"].size()[0]
    grid_rgb = get_vis_tensor(rgb_load_gt, "rgb", seq_len)
    writer.add_image(vis_info+"/rgb_{}_{}".format(sub_video_name, idx), grid_rgb)
    seq_len = sample["op"].size()[0]
    grid_op = get_vis_tensor(op_load_gt, "op", seq_len)
    writer.add_image(vis_info + "/op_{}_{}".format(sub_video_name, idx), grid_op)

def get_vis_tensor(vis_tensor, dataset_type, nrow):
    if dataset_type == "rgb": # or dataset_type == "optical_flow":
        grid = make_grid(vis_tensor, nrow=nrow, normalize=True, range=(-1, 1))  # normalize, (-1,1) -> (0,1)
    elif dataset_type == "op":
        flow_batch = vis_tensor.permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()  # [b, h, w, 2]
        flow_vis_batch = batch_flow_to_image(flow_batch)  # [b, h, w, 3]
        tensor = torch.from_numpy(flow_vis_batch)  # [b, h, w, c]
        tensor = tensor.permute(0, 3, 1, 2)  # [b, c, h, w]
        grid = make_grid(tensor, nrow=nrow)  # (0,1), 无需 normalize
    else:
        grid = None
        print("dataset_type error ! ")
        exit()
    return grid

# =========================================================== #
if __name__ == '__main__':
    # test TwoStream_Test_DS
    def test_test_TwoStream_Test_DS():
        #
        sum_path = utils.get_dir("/p300/test_TwoStream_Test_DS")
        writer = SummaryWriter(log_dir=sum_path)
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/testing/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/testing/frames/flow".format(dataset_name))  #
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Test_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Test_DS = test_TwoStream_Test_DS()
        # test_1
        # t_test_TwoStream_Test_DS.test_1(dataset, all_sub_video_name_list)
        # test_2
        # t_test_TwoStream_Test_DS.test_2(dataset, all_sub_video_name_list)
        # test_3
        # t_test_TwoStream_Test_DS.test_3(dataset, all_sub_video_name_list)
        #
        # test_4 : 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Test_DS.test_4(dataset, all_sub_video_name_list, writer)
    # test_test_TwoStream_Test_DS()

    # test TwoStream_Train_DS
    def test_test_TwoStream_Train_DS():
        #
        sum_path = utils.get_dir("/p300/test_TwoStream_Train_DS")
        writer = SummaryWriter(log_dir=sum_path)
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "avenue"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/training/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/training/frames/flow".format(dataset_name))  #
        # print(path_rgb)
        # print(path_optical_flow)
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Train_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Train_DS = test_TwoStream_Train_DS()
        # test_1
        # t_test_TwoStream_Train_DS.test_1(dataset)
        # test_2
        # t_test_TwoStream_Train_DS.test_2(dataset)
        #
        # test_3: 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Train_DS.test_3(dataset, writer)
    # test_test_TwoStream_Train_DS()

    def test_test_TwoStream_Train_DS_shanghaitech():
        #
        dataset_root = "/p300/dataset"  # universial, in p300
        dataset_name = "shanghaitech"  # 其实应该用 toy dataset 来做 unit test
        path_rgb = os.path.join(dataset_root, "{}/training/frames".format(dataset_name))  #
        path_optical_flow = os.path.join(dataset_root, "{}/optical_flow/training/frames/flow".format(dataset_name))  #
        # print(path_rgb)
        # print(path_optical_flow)
        video_folder = {"rgb": path_rgb, "op": path_optical_flow, }
        dataset = TwoStream_Train_DS(video_folder)
        all_sub_video_name_list = sorted(list(dataset.videos["rgb"].keys()))
        # print(all_sub_video_name_list)
        #
        t_test_TwoStream_Train_DS = test_TwoStream_Train_DS()
        # test_1
        # t_test_TwoStream_Train_DS.test_1(dataset)
        # test_2
        # t_test_TwoStream_Train_DS.test_2(dataset)
        #
        # test_3: 真正的 unittest, 使用各种 assert传入 gt, 不报错即为 通过
        #
        t_test_TwoStream_Train_DS.test_shanghaitech_bug(dataset)
    test_test_TwoStream_Train_DS_shanghaitech()
