"""
MambaMIL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def init_mamba_module(type, layer=None,d_model=256):
        
    if type == "Transformer":
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,  # 设定多头注意力的头数
            dim_feedforward=2 * d_model,  # FFN隐藏层维度
            dropout=0.1, 
            activation="relu",
            batch_first=True  # 使输入输出维度为 (batch, seq_len, d_model)
        )
        model = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 设定Transformer层数
    else:
        raise NotImplementedError("Mamba [{}] is not implemented".format(type))

    return model

class SSM_intra(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=1,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            cfg=None,
            type="Mamba",
            # ======================
            **kwargs,
    ):
        super(SSM_intra, self).__init__()
        factory_kwargs = {"device": None, "dtype": None}
        self.d_model = d_model
        self.d_inner = int(ssm_ratio* self.d_model)
        # in proj =======================================
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_p = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # Mamba =========================================
        self.mamba_t = init_mamba_module(type,d_model=d_model)
        self.mamba_p = init_mamba_module(type,d_model=d_model)
        # out proj ======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    def ssm_stage_1(self, t, p):
        t = self.mamba_t(t)
        p = self.mamba_p(p)
        return torch.cat([t, p], dim=1)

    def forward(self, t, p):
        # input shape (B, N, -1)
        B, N, D = t.shape

        xz_p = self.in_proj_p(p)
        xz_t = self.in_proj_t(t)
        # print(xz_p.shape)
        # print(xz_t.shape)
        x_p, z_p = xz_p.chunk(2, dim=-1)
        x_t, z_t = xz_t.chunk(2, dim=-1)
        # print(x_p.shape)
        # print(z_p.shape)

        z = torch.cat([z_t, z_p], dim=1)
        y = self.ssm_stage_1(x_t, x_p)
        y = y * F.silu(z)

        out = self.dropout(self.out_proj(y))
        new_t = out[:, :N]
        new_p = out[:, N:]
        return new_p+p, new_t+t
    

class SSM_inter(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=1,
            dt_rank="auto",
            # dwconv ===============
            # d_conv=-1, # < 2 means no conv
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            softmax_version=False,
            # ======================
            type="Mamba",
            cfg=None,
            # ======================
            **kwargs,
    ):  
        super(SSM_inter, self).__init__()
        self.d_model = d_model
        self.d_inner = int(ssm_ratio* self.d_model)
        factory_kwargs = {"device": None, "dtype": None}
        # in proj =======================================
        self.in_proj_p = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_t = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # Mamba =========================================
        self.mamba_fusion = init_mamba_module(type,d_model=d_model)

        # out proj =======================================
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    
    def ssm_stage_2(self, p, t, ):
        x_for = torch.cat([p, t], dim=1)
        y_for = self.mamba_fusion(x_for)        
        return y_for

    def forward(self, t, p):
        B, N, D = t.shape

        xz_p = self.in_proj_p(p)
        xz_t = self.in_proj_t(t)

        x_p, z_p = xz_p.chunk(2, dim=-1)
        x_t, z_t = xz_t.chunk(2, dim=-1)

        z = torch.cat([z_t, z_p], dim=1)
        y = self.ssm_stage_2(x_t, x_p)
        y = y * F.silu(z)

        out = self.dropout(self.out_proj(y))
        new_t = out[:, :N]
        new_p = out[:, N:]
        return new_t+t, new_p+p

class MM_SS2D(nn.Module):
    def __init__(self, d_model, cfg=None,dt_rank = "auto",d_state = 16,type_layer="Mamba", **kwargs):
        super(MM_SS2D, self).__init__()
        self.SSM_intra = SSM_intra(d_model=d_model, d_state=d_state, cfg=cfg,dt_rank=dt_rank,type=type_layer)
        self.SSM_inter = SSM_inter(d_model=d_model, d_state=d_state, cfg=cfg,dt_rank=dt_rank,type=type_layer)

    def forward(self, p, t, **kwargs):
        p, t = self.SSM_intra(p, t)
        p, t = self.SSM_inter(p, t)
        return p, t

