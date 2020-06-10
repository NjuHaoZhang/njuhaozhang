'''
修改了 vqvae2: (1) residual connection; (2) top-k item
'''

class Quantize_topk(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, k=1):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.k = k

        embed = torch.randn(dim, n_embed)
        self.register_buffer('embed', embed) # buffer 支持持久化保持 Quantize的数据(save to file)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, input):

        # ======== lookup dictionary ================================================================== #
        # input: (b,h,w,c)
        flatten = input.reshape(-1, self.dim) # (b,h,w,c) -> (num, c)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        ) # (num, n_embed), num个pixel vector(c-dim) 对 n_embed 个 vector(c-dim) 的 距离矩阵
        _, embed_ind = (-dist).max(1) # (num,1), num 个 pixel, 每个pixel 对应一个最近邻e_{idx}的idx
        # (num, 1) -->[one_hot] (num, n_embed)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

        # ========== 不影响 dict optimization 仅用于(1) out vq_value to decoder,========== #
        # (2)copy grad in decoder to encoder
        _, embed_ind_topk = (-dist).topk(self.k, dim=1) # (num, k) 而不是 (num,1)
        embed_ind_topk = embed_ind_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        # (num, k)->(b,h,w,k), next to embedding
        #
        quantize_topk = self.embed_code(embed_ind_topk) # [b, h, w, k, emb_dim]
        quantize_topk = quantize_topk.view(input.shape[0], input.shape[1], input.shape[2], -1)
        assert quantize_topk.shape[-1] == self.k * self.dim
        # ================================================================================= #

        # ============= exponential moving average to update vq_dict ====================== #
        # 所以并没有 training 的过程，是直接用 closed-form solution 直接迭代的算 codebook item的
        # 不和具体的 item selection 的过程 发生联系，所以  top-k 不会影响 codebook 的 优化
        # so 这个是好是坏？？？TODO check !!! (从实验看问题不大，先做完再来纠结)
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            ) # 计算 cluster_size (moving average)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum) # 计算 cluster_data
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0) # data / size ==> avg
            self.embed.data.copy_(embed_normalized)
        # ================================================================================= #

        # ============ (1) calculate latent_loss to update encoder, ======================= #
        # (2) copy grad in decoder to encoder (skip vq_module, 图中那条红线)
        input_topk = input.repeat(1,1,1,self.k) # (b,h,w,k*emb_dim) to support quantize_topk
        # repeat input channel 来支持 top-k，这套代码唯一的修改方法
        diff = (quantize_topk.detach() - input_topk).pow(2).mean() # detach() 即  stop_gradient
        # diff 这个 loss bp时不优化 quantize_topk 及其的前驱结点，即不优化 codebook的 item
        # 仅优化 input_topk，即 原文 loss 公式的 term 3: 让 encoder(x) item 靠近 最近的 code in codebook
        # 那 原文 loss 公式的 term 2 呢？ 答：见上面的 moving average, 直接用 closed-from solution 代替 loss bp 优化了
        # loss 公式的 term 1 呢？答：那个不是 VQ 部分，所以不在本类代码中实现
        quantize_topk = input_topk + (quantize_topk - input_topk).detach() # copy grad of quan to encoder
        # 另外，
        # quantize_topk的fusion params 交给 decoder来学习
        # ================================================================================= #

        return quantize_topk, diff, embed_ind

    def embed_code(self, embed_id):
        # self.embed 表示当前 Quantize 所存储的 memory ( shape is [emb_dim, n_emb] )
        # embed_id: [b,h,w,k], self.embed: [emb_dim, n_emb]
        return F.embedding(embed_id, self.embed.transpose(0, 1)) # 等价于拿着idx去查表，获取对应的item(emb_dim维vector)
        # 具体用法待查， todo

# 加 top-k vq(z)
class enc_quan_dec_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_topk, self).__init__()
        self.enc = nn.Conv2d(in_c, embed_dim, 1) # 1x1 conv for depth_control
        self.quantize = Quantize_topk(dim=embed_dim, n_embed=n_embed, k=k)
        self.dec = nn.Conv2d(embed_dim*k, in_c, 1) # 1x1 conv for depth_control

    def forward(self,x):
        x = self.enc(x).permute(0, 2, 3, 1) # (b,c,h,w) -> (b, h, w, c)
        quantize, diff, embed_ind = self.quantize(x) # quantize: (b, h, w, c*k)
        quantize = quantize.permute(0, 3, 1, 2) # (b, h, w, c*k) -> (b,c*k,h,w)
        diff = diff.unsqueeze(0)
        x = self.dec(quantize)
        return x, diff, embed_ind

# 加 res_connect: fusion: (z, vq(z))
class enc_quan_dec_res_topk(nn.Module):
    def __init__(self, in_c, embed_dim, n_embed, k=1):
        super(enc_quan_dec_res_topk, self).__init__()
        self.quan = enc_quan_dec_topk(in_c, embed_dim, n_embed, k=k)

    def forward(self,x):
        out, diff, embed_ind = self.quan(x)
        out += x
        return out, diff, embed_ind

class VQVAE_topk(nn.Module):
    # 还可以加上 res, 
    # 这个 VQVAE_topk 还没写完，TODO
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4) # 4x donwsapling
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2) # 2x
        # self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1) # ? F=1，减小计算量？bottle_neck ? yes. depth减小了
        self.quantize_t = enc_quan_dec_res_topk(embed_dim, n_embed) # top_level quantize
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        ) # 2x upsampling, [top_e] to [2x top_e]
        # self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1) # embed_dim是来自 [2x top_e] 的 concat
        self.quantize_b = enc_quan_dec_res_topk(embed_dim, n_embed) # bottom_level quantize
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        ) # 2x upsampling,
        self.dec = Decoder(
            embed_dim + embed_dim, # concat ?
            out_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        ) # 4x upsampling,
        # 再结合 self.upsample_t， 等于 8x, i.e. [top_e] to [8x top_e]

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input) # enc_and_quant
        dec = self.decode(quant_t, quant_b) # dec

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input) # 4x downsampling (bottom-level)
        enc_t = self.enc_t(enc_b) # 再2x i.e. 1/8 of input (top-level)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1) # F=1，通过降低depth减小计算量(w/h不变); c,h,w->h,w,c
        quant_t, diff_t, id_t = self.quantize_t(quant_t) # vq (类似k-means)
        quant_t = quant_t.permute(0, 3, 1, 2) # shape 和 enc_t 一致
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t) # (w,h) 调整为 与enc_b一致
        enc_b = torch.cat([dec_t, enc_b], 1) # spatial wise concat (ch_1+ch_2)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1) # 先将depth降至 emb_memory
        quant_b, diff_b, id_b = self.quantize_b(quant_b) # 再做 quantize
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t) # 先 upsample(2x) 到 shape of quant_b
        quant = torch.cat([upsample_t, quant_b], 1) # 再 concat then decode
        dec = self.dec(quant) # 4x upsample

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec


# 处理 clip: (1) channel-wise concat 最简单，先试下; (2) C2D-C3D; 先用C2D提特征
    # (3)LSTM尽量不用太慢太难调