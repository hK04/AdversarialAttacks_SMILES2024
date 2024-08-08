import torch
from torch import nn
import numpy as np
import pandas as pd

from .utils import Activation
from .base_model import BaseModel

from src.models.FEDformer_src.layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from src.models.FEDformer_src.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from src.models.FEDformer_src.layers.FourierCorrelation import FourierBlock, FourierCrossAttention
from src.models.FEDformer_src.layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from src.models.FEDformer_src.layers.Autoformer_EncDec import Encoder, EncoderLayer, my_Layernorm, series_decomp
from src.models.FEDformer_src.timefeatures import time_features

class FEDformer(BaseModel):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, activation_type: str = "sigmoid", **kwargs):
        super(FEDformer, self).__init__()
        class Config:
            def __init__(self, dictionary):
                for k, v in dictionary.items():
                    setattr(self, k, v)
                    
        configs = Config(kwargs)

        self.seq_len = configs.seq_len

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes

        # Decomp
        self.decomp = series_decomp(configs.moving_avg)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                        configs.dropout)

        if self.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model,
                                                    L=1,
                                                    base='legendre')
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=self.modes,
                                            mode_select_method=self.mode_select)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,  # instead of multi-head attention in transformer
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) 
                for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        
        self.act = Activation(configs.activation)
        self.final_activaton = Activation(activation_type)
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, 1)

    def forward(self, x_enc, x_mark_enc=None):
        if x_mark_enc is None:
            x_mark_enc = torch.tensor(time_features(pd.date_range('2018-04-24 00:00:00', '2018-04-24 08:19:00', periods=500), '1h'), dtype=torch.float32).transpose(1, 0)
            x_mark_enc = x_mark_enc.to(x_enc.device)
            
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        output = self.dropout(enc_out)
        #output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return self.final_activaton(output)