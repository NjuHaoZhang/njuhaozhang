'''
解读 VQVAE-2
'''

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

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
        )
        # print("dist.size(): ", dist.size())
        # 注:
        # (1) 全部用法见 pytorch 文档
        # 此处用法：对 (num, c) 每个 elem 做 elem^2 操作，return (num, c)
        # 详细解释：(num, c).pow(2) 即 tensor.pow() 等效于 torch.pow(tensor, 2), 即 out[i] = in[i]^{2}
        # 而此处 in[i]为 c_dim vector, 即 each elem of vector 都执行 elem^2
        #
        # (2) sum(1, keepdim=True), so return (num,1)
        #
        # (3)
        # a @ b 即矩阵乘法，so flatten @ self.embed 即  (num,c)x(c,n_emb)=>(num, n_emb)
        #
        # so dist的本质，看形式我联想到: x^2 -2*x*y + y^2, 其实是求 MSE(flatten, self.embed),
        # 总共 num(i.e. b x h x w)个 pixel, 每个 pixel 计算到memeory中每个elem(总共n_emb个elem)的distance,
        # so return (num, n_emb)

        _, embed_ind = (-dist).max(1)
        # print("embed_ind.size() in (-dist).max(1): ", embed_ind.size()) # (num,)
        # 注:
        # tensor.max()等价于 torch.max(tensor), return max_value, max_index,
        # axes==1, -dist, so 等效于求min in axes=1, so return (num,)
        # 注: 没有keep_dim，所以dim_1被干掉了
        # why not dist.min(1) directly ? todo
        #
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype) # (num, 1) -->[one_hot] (num, n_embed)
        # print("embed_onehot: ", embed_onehot.size())    # one_hot for training, (num, n_embed)

        #
        embed_ind = embed_ind.view(*input.shape[:-1]) # (num,) -> (b,h,w), since num==b*h*w
        # print("embed_ind.size() in view(*input.shape[:-1]): ", embed_ind.size()) # [b, h, w]
        # 注：input.shape[:-1] 即 (b,h,w)
        # *input: * collects all the positional arguments in a tuple.
        # so view(view(*input.shape[:-1])) == view(b,h,w), 这种用法细查，todo
        # return [b, h, w]
        #
        quantize = self.embed_code(embed_ind) #
        # print("quantize in self.embed_code(embed_ind): ", quantize.size()) # [b, h, w, emb_dim]

        # ============= exponential moving average ======================================================== #
        # where is stop gradient operator ? todo, 答: detach()
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean() # detach() 即  stop_gradient
        quantize = input + (quantize - input).detach()
        # 两次 detach(), 将 原图分为三个子图，然后各自train,but 正向传播没有断开

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        # self.embed 表示当前 Quantize 所存储的 memory ( shape is [emb_dim, n_emb] )
        # embed_id: [b,h,w], self.embed: [emb_dim, n_emb]
        return F.embedding(embed_id, self.embed.transpose(0, 1)) # 等价于拿着idx去查表，获取对应的item(emb_dim维vector)
        # 具体用法待查， todo


# w,h,c 均不改变
class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input # vilia res_block

        return out

# c->1/2 or 不变(if out_ch==in_ch)，w(h)->1/4 or 1/2 (w,h变小主要是为了减少计算量，次要是denoise)
class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        # c->1/2，w(h)->1/4
        if stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]
        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        # c,h,w最终均不变，仅仅提特征
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks) # 串行连接这些 block

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        #  w(h) 不变, 将 channel由emb_dim还原为encoder的输出channel
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        # w(h) 不变
        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # upsampling(ConvTranspose)
        if stride == 8:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        out_channel, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        elif stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )
        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        # extend只能接受一个list，并将list中每个元素逐一加入到目标list尾部；append接受任意类型的数据，并将这个数据整体追加到目标list尾部
        # e.g. [1,2,3].extend([4,5,6]) == [1,2,3,4,5,6],  [1,2,3].append([4,5,6]) == [1,2,3,[4,5,6]]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# 注：相比dalao原code, 做了一些微小改动,但主体还是follow他的
class VQVAE(nn.Module):
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
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1) # ? F=1，减小计算量？bottle_neck ? yes. depth减小了
        self.quantize_t = Quantize(embed_dim, n_embed) # top_level quantize
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        ) # 2x upsampling, [top_e] to [2x top_e]
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1) # embed_dim是来自 [2x top_e] 的 concat
        self.quantize_b = Quantize(embed_dim, n_embed) # bottom_level quantize
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