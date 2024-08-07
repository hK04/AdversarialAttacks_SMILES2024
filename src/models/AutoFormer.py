from .base_model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.AutoFormer_src.layers.Embed import DataEmbedding_wo_pos, TokenEmbedding, PositionalEmbedding
from src.models.AutoFormer_src.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from src.models.AutoFormer_src.layers.Autoformer_EncDec import Encoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class AutoFormer(BaseModel):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(
        self, 
        c_in=1, 
        moving_avg=25, 
        d_model=128, 
        factor=3,
        n_heads=4,
        dropout=0.05,
        d_ff=2048,
        activation='gelu',
        e_layers=4,
        n_classes=1,
    ):
        super(AutoFormer, self).__init__()
        # Decomp
        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model, max_len=5000)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False, 
                            factor, 
                            attention_dropout=dropout, 
                            output_attention=False
                        ),
                        d_model, 
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

        # classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
            nn.Sigmoid(),
        )        

    def forward(self, x):  
        x0 = x
        b_size = len(x)

        x = self.enc_embedding(x)
        cls_tokens = self.cls_token.repeat(b_size, 1, 1)
        x += self.position_embedding(x)

        x, attn = self.encoder(x)

        x = self.mlp_head(x)

        return x
