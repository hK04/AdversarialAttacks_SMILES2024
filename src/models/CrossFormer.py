import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import numpy as np
import pandas as pd

from .utils import Activation
from .base_model import BaseModel

from src.models.CrossFormer_src.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from src.models.CrossFormer_src.Embed import PatchEmbedding
from src.models.CrossFormer_src.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from src.models.CrossFormer_src.PatchTST import FlattenHead


from math import ceil


class CrossFormer(BaseModel):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(self, activation_type: str = "sigmoid", **kwargs):
        super(CrossFormer, self).__init__()
        class Config:
            def __init__(self, dictionary):
                for k, v in dictionary.items():
                    setattr(self, k, v)
        configs = Config(kwargs)
                    
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (configs.e_layers - 1)))
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(configs.d_model, self.seg_len, self.seg_len, self.pad_in_len - configs.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model))
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(configs, 1 if l is 0 else self.win_size, configs.d_model, configs.n_heads, configs.d_ff,
                            1, configs.dropout,
                            self.in_seg_num if l is 0 else ceil(self.in_seg_num / self.win_size ** l), configs.factor
                            ) for l in range(configs.e_layers)
            ]
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
        self.head_nf * configs.enc_in, 1)
        self.final_activaton = Activation(activation_type)

    def forward(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return self.final_activaton(output)