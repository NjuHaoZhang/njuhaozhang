'''
@Author: your name
@Date: 2020-06-10 23:44:11
@LastEditTime: 2020-06-10 23:47:08
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \undefinede:\Skills\Github\personal_code_repository\pytorch\vqvae_code_3.py
'''


# 一个很暴力的 Twostream_VQVAE: 结合网络图解释得更清晰，再开个 ipyub 讲解这个



import sys
sys.path.append("./")

import torch
from torch import nn
from torch.nn import functional as F

from model.vqvae import (
    get_VQVAE_frame, get_VQVAE_optical_flow,
    get_VQVAE_seq2image_frame, get_VQVAE_seq2image_optical_flow,
)
from model.bridge_net import get_frame2op, get_op2frame

import ipdb
# ====================================================================================================== #
# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch
# by HaoZhang: Borrowed from https://github.com/jeffreyyihuang/two-stream-action-recognition
# ====================================================================================================== #
# transfrom based on vq_index (bridge_net中操作的对象是 vq_index)
class TwostreamVQVAE_v1(nn.Module):

    '''
    VQVAE_frame, VQVAE_optical_flow
    and
    bridge_net暂时使用 FCN_VGG, 只使用 frame2op branch
    '''

    def __init__(self, VQVAE_frame, VQVAE_optical_flow, bridge_frame2op, bridge_op2frame):
        super().__init__()

        self.VQVAE_frame = VQVAE_frame
        self.VQVAE_optical_flow = VQVAE_optical_flow
        self.bridge_frame2op = bridge_frame2op
        self.bridge_op2frame = bridge_op2frame

    def forward(self, x_list):

        frame, optical_flow = x_list

        # direc_rec
        frame_direct_rec, op_direct_rec = self.direct_rec(frame, optical_flow)
        # frame2op
        direction = "frame2op"
        frame2op_transform_rec = self.single_direction_transform_rec(frame, direction)
        # op2frame
        direction = "op2frame"
        # ipdb.set_trace()
        op2frame_transform_rec = self.single_direction_transform_rec(optical_flow, direction)
        # cycle_consistency
        frame_rec_cycle, op_rec_cycle =  self.cycle_consistency_rec(frame, optical_flow)

        return frame_direct_rec, op_direct_rec, frame2op_transform_rec , \
               op2frame_transform_rec, frame_rec_cycle, op_rec_cycle

    def cycle_consistency_rec(self, frame, optical_flow):
        direction = "frame2op"
        op_rec = self.single_direction_transform_rec(frame, direction)
        direction = "op2frame"
        frame_rec_cycle = self.single_direction_transform_rec(op_rec, direction)
        #
        direction = "op2frame"
        frame_rec = self.single_direction_transform_rec(optical_flow, direction)
        direction = "frame2op"
        op_rec_cycle = self.single_direction_transform_rec(frame_rec, direction)
        return frame_rec_cycle, op_rec_cycle

    def direct_rec(self, frame, optical_flow):
        # path_3: directly reconstruction (for baseline and compared group)
        frame_direct_rec, _ = self.VQVAE_frame(frame)
        op_direct_rec, _ = self.VQVAE_optical_flow(optical_flow)
        return frame_direct_rec, op_direct_rec

    def single_direction_transform_rec(self, tensor, direction):
        # path: frame -> op_rec -> frame_rec
        # (1) extract vq_index of frame
        # 下面这个两个分支都是可以直接运行的，所以dict 直接 return result
        encoding = {
            "frame2op": self.VQVAE_frame.encode,
            "op2frame": self.VQVAE_optical_flow.encode,
        }
        _, _, _, id_t, id_b = encoding.get(direction, None)(tensor)
        # ========================================================================================== #
        # (2) frame2op, based on vq_index
        id_t = id_t.unsqueeze(0).float()
        id_b = id_b.unsqueeze(0).float()
        transform = {
            "frame2op": self.bridge_frame2op,
            "op2frame": self.bridge_op2frame,
        }
        id_transform_list = transform.get(direction, None)([id_t, id_b])
        id_t_transform, id_b_transform = parse_transform_ret(id_transform_list)  # (b,C,h,w) -> (b,1,h,w), argmax
        # print("id_op size: ", id_t_op.size(), id_b_op.size())
        id_t_transform = id_t_transform.squeeze(0)
        id_b_transform = id_b_transform.squeeze(0)
        # ========================================================================================= #
        # (3) decode vq_op_index to get op_rec
        # dict会真正试运行每个item, 而下面两个item只能正确运行一个，需改 dict return func_name
        decoding = {
            "frame2op": self.VQVAE_optical_flow.decode_code,
            "op2frame": self.VQVAE_frame.decode_code,
        }
        decoded_sample = decoding.get(direction,None)(id_t_transform, id_b_transform) # dict.get 更安全
        transform_rec = decoded_sample.clamp(-1, 1)

        return transform_rec

# transfrom based on vq_value
class TwostreamVQVAE(nn.Module):

    '''
    VQVAE_frame, VQVAE_optical_flow
    and
    bridge_net暂时使用 FCN_VGG, 只使用 frame2op branch
    '''

    def __init__(self, VQVAE_frame, VQVAE_optical_flow, bridge_frame2op, bridge_op2frame):
        super().__init__()

        self.VQVAE_frame = VQVAE_frame
        self.VQVAE_optical_flow = VQVAE_optical_flow
        self.bridge_frame2op = bridge_frame2op
        self.bridge_op2frame = bridge_op2frame

    def forward(self, x_list, memory_list):

        frame, optical_flow = x_list
        rgb_memory, op_memory = memory_list

        # direc_rec
        frame_direct_rec, op_direct_rec = self.direct_rec(frame, optical_flow) # [2, 3, 256, 256],[2, 2, 256, 256]
        # frame2op
        direction = "frame2op"
        frame2op_transform_rec = self.single_direction_transform_rec(frame, direction, rgb_memory, op_memory) # [2, 2, 256, 256]
        # op2frame
        direction = "op2frame"
        # ipdb.set_trace()
        op2frame_transform_rec = self.single_direction_transform_rec(optical_flow, direction, rgb_memory, op_memory) # [2, 3, 256, 256]
        # cycle_consistency
        frame_rec_cycle, op_rec_cycle =  self.cycle_consistency_rec(frame, optical_flow, rgb_memory, op_memory)
        # 无法做 cycle_consistency, since twostream is seq2frame, not seq2seq

        # post-processing
        # for frame, clamp(-1,1)
        # frame_direct_rec, op2frame_transform_rec, frame_rec_cycle = \
        # frame_direct_rec.clamp(-1,1), op2frame_transform_rec.clamp(-1,1), frame_rec_cycle.clamp(-1,1)
        # # for op, nothing to do
        return frame_direct_rec, op_direct_rec, \
               frame2op_transform_rec , op2frame_transform_rec, frame_rec_cycle, op_rec_cycle


    def cycle_consistency_rec_v1(self, frame, optical_flow):
        direction = "frame2op"
        op_rec = self.single_direction_transform_rec(frame, direction) # [b, 3, 256, 256]
        direction = "op2frame"
        frame_rec_cycle = self.single_direction_transform_rec(op_rec, direction)
        #
        direction = "op2frame"
        frame_rec = self.single_direction_transform_rec(optical_flow, direction)
        direction = "frame2op"
        op_rec_cycle = self.single_direction_transform_rec(frame_rec, direction)
        return frame_rec_cycle, op_rec_cycle

    def cycle_consistency_rec(self, frame, optical_flow, rgb_memory, op_memory):

        def cycle_rec(input_type, tensor, rgb_memory, op_memory):
            encoding = {
                "frame": self.VQVAE_frame.encode,
                "op": self.VQVAE_optical_flow.encode,
            }
            quant_b, _, _ = encoding.get(input_type, None)(tensor)  # [b, 64, 64, 64]
            # ========================================================================================== #
            if input_type == "frame":
                quant_b = self.bridge_frame2op(quant_b, op_memory)
                quant_b = self.bridge_op2frame(quant_b, rgb_memory)
            else:
                quant_b = self.bridge_op2frame(quant_b, rgb_memory)
                quant_b = self.bridge_frame2op(quant_b, op_memory)
            # ========================================================================================= #
            # (3) decode vq_op_index to get op_rec
            # dict会真正试运行每个item, 而下面两个item只能正确运行一个，需改 dict return func_name
            decoding = {
                "frame": self.VQVAE_frame.decode,
                "op": self.VQVAE_optical_flow.decode,
            }
            decoded_sample = decoding.get(input_type, None)(quant_b)  # [b,c,h,w]
            cycle_rec = decoded_sample
            # vis frame and cycle_rec
            # ipdb.set_trace() todo
            return cycle_rec

        input_type = "frame"
        frame_cycle_rec = cycle_rec(input_type, frame, rgb_memory, op_memory)
        input_type = "op"
        op_cycle_rec = cycle_rec(input_type, optical_flow, rgb_memory, op_memory)

        return frame_cycle_rec, op_cycle_rec

    def direct_rec(self, frame, optical_flow):
        # path_3: directly reconstruction (for baseline and compared group)
        frame_direct_rec, _ = self.VQVAE_frame(frame)
        op_direct_rec, _ = self.VQVAE_optical_flow(optical_flow)
        return frame_direct_rec, op_direct_rec # [b, c, h, w]

    def single_direction_transform_rec(self, tensor, direction, rgb_memory, op_memory):
        # path: frame -> op_rec -> frame_rec
        # (1) extract vq_index of frame
        # 下面这个两个分支都是可以直接运行的，所以dict 直接 return result
        encoding = {
            "frame2op": self.VQVAE_frame.encode,
            "op2frame": self.VQVAE_optical_flow.encode,
        }
        quant_b, _, _ = encoding.get(direction, None)(tensor) # [b, 64, 64, 64]
        # ========================================================================================== #
        # (2) frame2op, based on vq_value
        transform = {
            "frame2op": self.bridge_frame2op,
            "op2frame": self.bridge_op2frame,
        }
        memory_mapping = {
            "frame2op": op_memory,
            "op2frame": rgb_memory,
        }
        memory = memory_mapping[direction]
        quant_b = transform.get(direction, None)(quant_b, memory) # [b, 64, 64, 64]
        # ========================================================================================= #
        # (3) decode vq_op_index to get op_rec
        # dict会真正试运行每个item, 而下面两个item只能正确运行一个，需改 dict return func_name
        decoding = {
            "frame2op": self.VQVAE_optical_flow.decode,
            "op2frame": self.VQVAE_frame.decode,
        }
        decoded_sample = decoding.get(direction,None)(quant_b) # [b,c,h,w]
        transform_rec = decoded_sample

        return transform_rec

def get_two_stream(embed_dim=64, frame_n_emb=64, op_n_emb=64):
    # [0] for frame, [1] for op
    VQVAE_frame = get_VQVAE_seq2image_frame(embed_dim=embed_dim, n_embed=frame_n_emb)
    VQVAE_optical_flow = get_VQVAE_seq2image_optical_flow(embed_dim=embed_dim, n_embed=op_n_emb)
    bridge_frame2op = get_frame2op(in_channel=embed_dim, out_channel=op_n_emb)
    bridge_op2frame = get_op2frame(in_channel=embed_dim, out_channel=frame_n_emb)
    two_stream = TwostreamVQVAE(VQVAE_frame, VQVAE_optical_flow, bridge_frame2op, bridge_op2frame)
    return two_stream


# ======================================================================================================= #
# unit test
# ======================================

# for twostream v1

def get_two_stream_v1():
    # [0] for frame, [1] for op
    VQVAE_frame = get_VQVAE_frame()
    VQVAE_optical_flow = get_VQVAE_optical_flow()
    bridge_frame2op = get_frame2op()
    bridge_op2frame = get_op2frame()
    two_stream = TwostreamVQVAE_v1(VQVAE_frame, VQVAE_optical_flow, bridge_frame2op, bridge_op2frame)
    return two_stream


def parse_transform_ret(id_op_pred): # redirect to util.py
    id_t_op, id_b_op = id_op_pred
    _, id_t_op_pred = torch.max(id_t_op, dim=1, keepdim=True)  # [b,1,h,w]
    _, id_b_op_pred = torch.max(id_b_op, dim=1, keepdim=True)  # [b,1,h,w]
    return id_t_op_pred, id_b_op_pred

def test_get_two_stream_v1():
    net = get_two_stream()
    bridge = net.bridge_frame2op
    print("net: ", bridge)
    # top, bottom = bridge[0], bridge[1]
    # print("top: ", top)
    # print("bottom: ", bottom)
    for param in bridge.parameters():
        print(type(param.data), param.size())
    #
    bridge = net.bridge_op2frame
    print("net: ", bridge)
    # top, bottom = bridge[0], bridge[1]
    # print("top: ", top)
    # print("bottom: ", bottom)
    for param in bridge.parameters():
        print(type(param.data), param.size())
    #
    # 预估参数量
    from torchsummaryX import summary
    x = torch.randn(1, 1, 32, 32)  # [b,1,h,w]
    y = torch.randn(1, 1, 64, 64)
    summary(net.bridge_frame2op, [x,y])
    print("ok \n\n")
    #
    summary(net.bridge_op2frame, [x,y])
    print("ok \n\n")
    #
    x = torch.randn(1, 3, 256, 256)
    y = torch.randn(1, 2, 256, 256) # 当前是 2-channel
    summary(net, x,y) # 测试 net的 forward
    print("ok \n\n")

def filter_params_v1(top_ckpt_path, bottom_ckpt_path, device):
    from collections import OrderedDict

    top_data = torch.load(top_ckpt_path, map_location=device)
    bottom_data = torch.load(bottom_ckpt_path, map_location=device)
    # print(top_data.keys()) # 看model.params的keys，然后制定filter logic
    print("top_data.keys(): ", type(top_data.keys())) #  <class 'odict_keys'>, 施加list()变为 list
    print("top_data: ", type(top_data))  # <class 'collections.OrderedDict'>
    fuse_data = OrderedDict()
    for key, val in top_data.items():
        if key.startswith('network.0'):
            fuse_data[key] = val
    for key, val in bottom_data.items():
        if key.startswith('network.1'):
            fuse_data[key] = val
    # fuse_data = torch.tensor(fuse_data, device=device)

    return fuse_data

def test_twostream_load_ckpt_frame2op_v1():
    import os
    device = torch.device("cuda:2")
    # model
    model = get_two_stream()
    model.to(device).eval()
    # model_ckpt
    load_root = "/p300/model_run_result/mmcmp_cvpr2020_v4/train_vqvae2"
    load_frame_ckpt_name = "n_embed_64_avenue_frame/checkpoint/vqvae_epoch_560.pt"
    load_op_ckpt_name = "n_embed_32_avenue_optical_flow/checkpoint/vqvae_epoch_560.pt"
    load_frame_ckpt_path = os.path.join(load_root, load_frame_ckpt_name)
    load_op_ckpt_path = os.path.join(load_root, load_op_ckpt_name)
    # 暂时先简单写死，修改 todo
    load_bridge_ckpt_root = {
        "top_frame2op": "/p300/model_run_result/mmcmp_cvpr2020_v5/train_BridgeNet_use_vqvae_epoch_560/FCN_VGG_16/"
               "LR=_0.001_lam_frame2op=1.0_0.0_op_n_emb=32/avenue/time=1573029119_gpu=5/save_model_ckpt/epoch_590.pth",
        "bottom_frame2op": "/p300/model_run_result/mmcmp_cvpr2020_v5/train_BridgeNet_use_vqvae_epoch_560/FCN_VGG_16/"
                  "LR=_0.001_lam_frame2op=0.0_1.0_op_n_emb=32/avenue/time=1573029119_gpu=6/save_model_ckpt/epoch_590.pth",
    }
    # self.load_ckpt_path 描述的是 bridge_net的ckpt, 而frame_vqvae and op_vqvae的ckpt是固定的: 560_epoch
    model.VQVAE_frame.load_state_dict(torch.load(load_frame_ckpt_path, map_location=device))
    model.VQVAE_optical_flow.load_state_dict(torch.load(load_op_ckpt_path, map_location=device))
    # bridge_net 需要解析 ckpt
    top = load_bridge_ckpt_root["top_frame2op"]
    bottom = load_bridge_ckpt_root["bottom_frame2op"]
    fuse_data = filter_params(top, bottom, device)
    model.bridge_frame2op.load_state_dict(fuse_data)
    print("model: ", model)
    print("OK\n\n")

def test_twostream_load_ckpt_op2frame_v1():
    import os
    device = torch.device("cuda:3")
    # model
    model = get_two_stream()
    model.to(device).eval()
    # model_ckpt
    #
    load_bridge_ckpt_root = {
        "top_op2frame": "/p300/model_run_result/mmcmp_cvpr2020_v5/train_BridgeNet_use_vqvae_epoch_560/FCN_VGG_16/"
            "LR=_0.001_lam_op2frame=1.0_0.0_op_n_emb=32/avenue/time=1573378422_gpu=1/save_model_ckpt/epoch_150.pth",
        "bottom_op2frame": "/p300/model_run_result/mmcmp_cvpr2020_v5/train_BridgeNet_use_vqvae_epoch_560/FCN_VGG_16/"
            "direction=op2frame_LR=_0.001_lam=0.0_1.0_op_n_emb=32/avenue/time=1573458359_gpu=9/save_model_ckpt/epoch_170.pth",
    }
    top = load_bridge_ckpt_root["top_op2frame"]
    bottom = load_bridge_ckpt_root["bottom_op2frame"]
    # debug
    # keys = torch.load(top).keys()
    # print("keys: ", keys)
    #
    fuse_data = filter_params(top, bottom, device)
    print("fuse_data: ", fuse_data)
    model.bridge_op2frame.load_state_dict(fuse_data)
    #
    print("model: ", model)
    print("OK\n\n")


# for twostream
def test_get_two_stream():
    net = get_two_stream()
    bridge = net.bridge_frame2op
    print("net: ", bridge)
    for param in bridge.parameters():
        print(type(param.data), param.size())
    #
    bridge = net.bridge_op2frame
    print("net: ", bridge)
    for param in bridge.parameters():
        print(type(param.data), param.size())
    #
    # 预估参数量
    from torchsummaryX import summary
    x = torch.randn(2, 64, 64, 64)
    summary(net.bridge_frame2op, x)
    print("ok \n\n")
    #
    summary(net.bridge_op2frame, x)
    print("ok \n\n")
    #
    b, t, c, h, w = 2, 9, 3, 256, 256
    frame = torch.randn(b, t, c, h, w)
    b, t, c, h, w = 2, 8, 2, 256, 256
    op = torch.randn(b, t, c, h, w)
    summary(net, [frame,op]) # 测试 net的 forward
    #
    out_list = net([frame,op])
    for out in out_list:
        print(out.size())



# ======================================================================================================= #
if __name__ == '__main__':
    # for twostream_v1
    # test_get_two_stream_v1()
    # test_twostream_load_ckpt_frame2op_v1()
    # test_twostream_load_ckpt_op2frame_v1() # 注意：op2frame: top:fcn_vgg19, bottom: fcn_resnet34

    # ---------------------------------------------------------------------------------------
    # for twostream
    test_get_two_stream()
    # test_twostream_load_ckpt_frame2op()
    # test_twostream_load_ckpt_op2frame() # 注意：op2frame: top:fcn_vgg19, bottom: fcn_resnet34



