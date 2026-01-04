import math
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask1=None, mask2=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask1 is not None and mask2 is not None:
            # print('Attention', attn.shape, 'Mask', mask1.shape)
            # print('Attention', attn.shape, 'Mask', mask2.shape)
            # # print(mask)
            attn = attn.masked_fill_(mask1, 1e-9) # Fills elements of att with 1e-9 where mask is True.
            attn = attn.masked_fill_(mask2, 1e-9) # Fills elements of att with 1e-9 where mask is True.
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=128, d_ff=2048,proj_drop=0.2):
        super(PoswiseFeedForwardNet, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            #nn.Conv1d(d_model, d_model*2,kernel_size=1, stride=1, bias=False),
            nn.Dropout(proj_drop),
            nn.GELU(),
            nn.Linear(d_ff, d_model,bias=False))
            #nn.Conv1d(d_model*2, d_model,kernel_size=1, stride=1, bias=False))

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.layer_norm((output + residual))  # [batch_size, seq_len, d_model]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_emb_q, d_emb_v, d_k=512, d_v=1024, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_emb_q, self.d_emb_v, self.n_head = d_emb_q, d_emb_v, n_head
        self.d_k = d_k if d_k is not None else d_emb_q
        self.d_v = d_v if d_v is not None else d_emb_v

        assert self.d_k % n_head == 0, 'Error from MultiHeadAttention: self.d_k % n_head should be zero.'
        assert self.d_v % n_head == 0, 'Error from MultiHeadAttention: self.d_v % n_head should be zero.'

        self.w_q = nn.Linear(d_emb_q, self.d_k,bias=False)  # you can also try nn.Conv1d(d_emb_q, self.d_k, 3, 1, 1, bias=False)     d_emb_q=296    self.d_k=32
        self.w_k = nn.Linear(d_emb_v, self.d_k, bias=False)  # d_emb_v=296    self.d_k=32
        self.w_v = nn.Linear(d_emb_v, self.d_v, bias=False)  # d_emb_v=296    self.d_k=32
        self.fc = nn.Linear(self.d_v, d_emb_q, bias=False)

        self.keep_q_k = nn.Conv1d(self.d_k, self.d_k,kernel_size=1,stride=1,bias=False)  # self.d_k=32 self.d_k=32
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_emb_q, eps=1e-6)

        #input_Q （2，5，512） attn_mask （2，5，5）

    def forward(self, q, k, v, mask1=None, mask2=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        assert len_k == len_v, 'len_k should be equal with len_v.'

        residual = q

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k // n_head)  # Original 避免最后一个batch只有27而报错
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k // n_head)

        # q = self.keep_q_k(self.w_q(q).permute(0,2,1)).permute(0,2,1).view(sz_b, len_q, n_head, d_k // n_head)
        # k = self.keep_q_k(self.w_k(k).permute(0,2,1)).permute(0,2,1).view(sz_b, len_k, n_head, d_k // n_head)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v // n_head)

        # Transpose for attention dot product: b x n x l x d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # For head axis broadcasting.
            mask2 = mask2.unsqueeze(1).unsqueeze(2)  # For head axis broadcasting.

        # result b x n x lq x (dv/n)
        result, attn = self.attention(q, k, v, mask1=mask1,
                                      mask2=mask2)  # result.shape=([2, 8, 48, 4])  attn.shape=torch.Size([2, 8, 48, 48])

        # Transpose to move the head dimension back: b x l x n x (dv/n)
        # Combine the last two dimensions to concatenate all the heads together: b x l x (dv)
        result = result.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # result.shape=([2, 48, 32])
        # b x l x (d_model)
        result = self.dropout(self.fc(result))  # result.shape=([2, 48, 140])

        result += residual  # result = result.masked_fill(gate_ < 0, 0)

        result = self.layer_norm(result)
        result = result.masked_fill(torch.isnan(result), 0.0)

        return result, attn


class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_position=100):
        super(PositionEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(int(pos_i)) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [1,N,d]

    def forward(self, x):
        # x [B,N,d]
        # print(x.shape ,self.pos_table[:, :x.size(1)].shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class my_encoder(nn.Module):
    def __init__(self,n_head, d_emb_q, d_emb_v, d_k=512, d_v=1024, dropout=0.1, d_ff=128,n_position=2048):
        super(my_encoder, self).__init__()
        if d_emb_v is None:
            d_emb_v = d_emb_q
        self.multi = MultiHeadAttention(d_emb_q=d_emb_q, d_emb_v=d_emb_v, d_k=d_k, d_v=d_v, n_head=n_head, dropout=dropout)
        self.pos = PoswiseFeedForwardNet(d_model=d_emb_q,d_ff=d_ff,proj_drop=dropout)

    def forward(self, seq1, seq2=None):

        if seq2 is not None:
            seq1 = seq1.masked_fill(torch.isnan(seq1),0.0)
            seq2 = seq2.masked_fill(torch.isnan(seq2),0.0)
            att_out, attn = self.multi(seq1, seq2, seq2)
        else:
            seq1 = seq1.masked_fill(torch.isnan(seq1),0.0)
            att_out, attn = self.multi(seq1, seq1, seq1)
        encoder_out = self.pos(att_out)
        return encoder_out, attn


