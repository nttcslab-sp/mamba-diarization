# See the code original license https://github.com/nttcslab-sp/EEND-vector-clustering/blob/main/LICENCE
# All content below is copied from http://kishin-gitlab.cslab.kecl.ntt.co.jp/tawara/torchEEND/blob/chime8/eend/pytorch_backend/transformer.py

# Copyright (c) 2023 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved

# This code is inspired by the following code.
# https://github.com/hitachi-speech/EEND/blob/master/eend/chainer_backend/transformer.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler


class NoamScheduler(_LRScheduler):
    """ learning rate scheduler used in the transformer
    See https://arxiv.org/pdf/1706.03762.pdf
    lrate = d_model**(-0.5) * \
            min(step_num**(-0.5), step_num*warmup_steps**(-1.5))
    Scaling factor is implemented as in
        http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    """

    def __init__(self, optimizer, d_model, warmup_steps, tot_step, scale, last_iter_for_resume=-1, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.tot_step = tot_step
        self.scale = scale
        self.flag = 0
        self.last_iter_for_resume = last_iter_for_resume
        super(NoamScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        self.last_epoch = max(1, self.last_epoch)
        if self.flag == 0 and self.last_iter_for_resume != -1:
            self.last_epoch = self.last_iter_for_resume
            self.flag += 1
        elif self.flag == 1 and self.last_iter_for_resume != -1:
            self.last_epoch = self.last_iter_for_resume + 1
            self.flag += 1
        step_num = self.last_epoch
        val = self.scale * self.d_model ** (-0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))

        return [base_lr / base_lr * val for base_lr in self.base_lrs]


class MultiHeadSelfAttention(nn.Module):
    """Multi head "self" attention layer"""

    def __init__(self, n_units, h=8, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.linearQ = nn.Linear(n_units, n_units)
        self.linearK = nn.Linear(n_units, n_units)
        self.linearV = nn.Linear(n_units, n_units)
        self.linearO = nn.Linear(n_units, n_units)
        self.d_k = n_units // h
        self.h = h
        self.dropout = nn.Dropout(p=dropout_rate)
        # attention for plot
        self.att = None

    def forward(self, x, batch_size):
        # x: (BT, F)
        q: torch.Tensor = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k: torch.Tensor = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v: torch.Tensor = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)  # (B, T, h, d_k)

        scores = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1)) / np.sqrt(self.d_k)
        # scores: (B, h, T, T) = (B, h, T, d_k) x (B, h, d_k, T)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        # x: (B, h, T, d_k) = (B, h, T, T) x (B, h, T, d_k)
        x = torch.matmul(p_att, v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(-1, self.h * self.d_k)

        return self.linearO(x)


class MultiHeadCoAttention(MultiHeadSelfAttention):
    """Multi head "co-" attention layer"""

    def __init__(self, n_units, h=8, dropout_rate=0.1):
        super().__init__(n_units, h, dropout_rate)

    def forward(self, x, batch_size):
        # x: (BT, F) or (B, C, T, F)
        n_x_shape = len(x.shape)
        if n_x_shape == 2:
            _d_k = self.d_k
            q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)  # (B, T, h, d_k)
            k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
            v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        elif n_x_shape == 4:
            nch = x.shape[1]
            _d_k = self.d_k * nch
            q = self.linearQ(x).transpose(1, 2).reshape(batch_size, -1, self.h, _d_k)  # (B, T, h, _d_k)
            k = self.linearK(x).transpose(1, 2).reshape(batch_size, -1, self.h, _d_k)  # (B, T, h, _d_k)
            v = self.linearV(x).reshape(batch_size, nch, -1, self.h, self.d_k)  # (B, C, T, h, d_k)

        # scores: (B, h, T, T) = (B, h, T, d_k) x (B, h, d_k, T)
        scores = torch.matmul(q.transpose(1, 2), k.permute(0, 2, 3, 1)) / np.sqrt(_d_k)
        self.att = F.softmax(scores, dim=3)
        p_att = self.dropout(self.att)
        if n_x_shape == 4:
            # reshape: (B, h, T, T) -> (B, C, h, T, T)
            p_att = p_att.unsqueeze(1).expand(-1, nch, -1, -1, -1)
        # x: (B, C, h, T, d_k) = (B, C, h, T, T) x (B, C, h, T, d_k)
        x = torch.matmul(p_att, v.transpose(-3, -2))
        x = x.transpose(-3, -2)  # (B, C, T, h, d_k)
        if n_x_shape == 2:
            x = x.reshape(-1, self.h * self.d_k)
        elif n_x_shape == 4:
            x = x.reshape(batch_size, nch, -1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed-forward layer"""

    def __init__(self, n_units, d_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(n_units, d_units)
        self.linear2 = nn.Linear(d_units, n_units)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Positional encoding function"""

    def __init__(self, n_units, dropout_rate, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        positions = np.arange(0, max_len, dtype="f")[:, None]
        dens = np.exp(np.arange(0, n_units, 2, dtype="f") * -(np.log(10000.0) / n_units))
        self.enc = np.zeros((max_len, n_units), dtype="f")
        self.enc[:, ::2] = np.sin(positions * dens)
        self.enc[:, 1::2] = np.cos(positions * dens)
        self.scale = np.sqrt(n_units)

    def forward(self, x):
        x = x * self.scale + self.xp.array(self.enc[:, : x.shape[1]])
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    """
    Transformer block for TransformerEncoder/CoAttentionEncoder
    """

    def __init__(self, n_units, e_units=2048, h=8, dropout_rate=0.1, att_type="self"):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(n_units)
        if att_type == "self":
            self.self_att = MultiHeadSelfAttention(n_units, h, dropout_rate)
        elif att_type == "co":
            self.self_att = MultiHeadCoAttention(n_units, h, dropout_rate)
        self.lnorm2 = nn.LayerNorm(n_units)
        self.ff = PositionwiseFeedForward(n_units, e_units, dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # layer normalization
        e = self.lnorm1(x)
        # co-attention
        s = self.self_att(e, x.shape[0])
        # residual
        e = e + self.dropout(s)
        # layer normalization (Eq. (17) in https://arxiv.org/pdf/2210.03459.pdf)
        e = self.lnorm2(e)
        # positionwise feed-forward
        s = self.ff(e)
        # residual
        e = e + self.dropout(s)
        return e


class TransformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_units, e_units=2048, h=8, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.linear_in = nn.Linear(idim, n_units)
        self.lnorm_in = nn.LayerNorm(n_units)
        self.pos_enc = PositionalEncoding(n_units, dropout_rate, 5000)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        for i in range(n_layers):
            setattr(self, "{}{:d}".format("lnorm1_", i), nn.LayerNorm(n_units))
            setattr(self, "{}{:d}".format("self_att_", i), MultiHeadSelfAttention(n_units, h, dropout_rate))
            setattr(self, "{}{:d}".format("lnorm2_", i), nn.LayerNorm(n_units))
            setattr(self, "{}{:d}".format("ff_", i), PositionwiseFeedForward(n_units, e_units, dropout_rate))
        """
        self.transformers = nn.Sequential(
                *[TransformerEncoderBlock(
                    n_units, e_units, h, dropout_rate, att_type='self'
                    ) for i in range(n_layers)]
                )
        """
        self.lnorm_out = nn.LayerNorm(n_units)

    def forward(self, x):
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, "{}{:d}".format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, "{}{:d}".format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + self.dropout(s)
            # layer normalization
            e = getattr(self, "{}{:d}".format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, "{}{:d}".format("ff_", i))(e)
            # residual
            e = e + self.dropout(s)
        """
        # Encoder stack
        e = self.transformers(e)
        """
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)


class ChannelDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        if p < 0.0 or p > 1.0:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, x):
        """
        input:
            x: (B, C, T, F) - shaped tensor
        output:
            x: (B, C, T, F) - shaped tensor that dropped (C-1) channels with prob. p
        """
        if self.training:
            assert len(x.shape) == 4
            nbat, nch, nlen, ndim = x.shape
            _prob = torch.rand(nbat)
            remain_idx = torch.randint(nch, (1,))
            mask = torch.ones(x.shape).to(x.dtype).to(x.device)
            mask[_prob > self.p] = 0
            mask[:, remain_idx] = 1
            x = x * mask
        return x


class CoAttentionEncoder(TransformerEncoder):
    def __init__(self, idim, n_layers, n_units, e_units=2048, h=8, dropout_rate=0.1, channel_dropout_rate=0.1):
        super().__init__(idim, n_layers, n_units, e_units, h, dropout_rate)
        for i in range(n_layers):
            setattr(self, "{}{:d}".format("self_att_", i), MultiHeadCoAttention(n_units, h, dropout_rate))
        self.channel_dropout = ChannelDropout(channel_dropout_rate)

    def forward(self, x):
        # x: (B, C, T, F) .. batch, ch, time, melfreq
        x = self.channel_dropout(x)
        e = self.linear_in(x)

        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, "{}{:d}".format("lnorm1_", i))(e)
            # co-attention
            s = getattr(self, "{}{:d}".format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + self.dropout(s)
            # layer normalization (Eq. (17) in https://arxiv.org/pdf/2210.03459.pdf)
            e = getattr(self, "{}{:d}".format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, "{}{:d}".format("ff_", i))(e)
            # residual
            e = e + self.dropout(s)
        """
        # Encoder stack
        e = self.transformers(e)
        """
        # final layer normalization
        # output: (B, C, T, F)
        e = self.lnorm_out(e)
        # channel averaging (Eq. (16)): (B, C, T, F) -> (B, T, F)
        e = torch.mean(e, dim=1)
        return e.reshape(-1, e.shape[-1])  # return (BT, F)
