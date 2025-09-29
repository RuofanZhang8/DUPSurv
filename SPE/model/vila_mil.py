# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import math
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F

# 通用路径导入
from .model_utils import *

logger = logging.getLogger(__name__)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class ViLa_MIL_Model(nn.Module):
    def __init__(self, config, text_encoder, num_classes=3):
        super(ViLa_MIL_Model, self).__init__()

        self.text_encoder = text_encoder
        self.norm = nn.LayerNorm(config['input_size'])

        self.cross_attention_image = nn.MultiheadAttention(
            embed_dim=config['input_size'], num_heads=1, batch_first=True
        )
        self.cross_attention_text = nn.MultiheadAttention(
            embed_dim=config['input_size'], num_heads=1, batch_first=True
        )
        self.cross_attention_tissue = nn.MultiheadAttention(
            embed_dim=config['input_size'], num_heads=1, batch_first=True
        )

        self.learnable_image_center = nn.Parameter(
            torch.Tensor(1, config['prototype_number_image'], config['input_size'])
        )
        self.learnable_text_center = nn.Parameter(
            torch.Tensor(1, config['prototype_number_text'], config['input_size'])
        )

        trunc_normal_(self.learnable_image_center, std=.02)
        trunc_normal_(self.learnable_text_center, std=.02)
        self.set_learnable_text_encoder(learnable=False)

    def forward(self, input_ids_report, embedding_tissue, ft_wsi, guassian_prototype):
        input_ids_report = input_ids_report.to(self.text_encoder.device)
        embedding_report = self.text_encoder(input_ids_report, attention_mask=None)[0]

        # Image prototype attention
        compents, _ = self.cross_attention_image(
            self.learnable_image_center, ft_wsi, ft_wsi
        )
        compents_image = self.norm(compents + self.learnable_image_center)

        # Text prototype attention
        embedding_report = embedding_report.view(1, -1, 768)
        compents, _ = self.cross_attention_text(
            self.learnable_text_center, embedding_report, embedding_report
        )
        compents_text = self.norm(compents + self.learnable_text_center)

        image_report_prototype = torch.cat(
            (compents_text, compents_image, guassian_prototype), dim=1
        )

        # Tissue prototype attention
        embedding_tissue = embedding_tissue.view(1, -1, 768)
        compents, _ = self.cross_attention_tissue(
            embedding_tissue, image_report_prototype, image_report_prototype
        )
        embedding_tissue = embedding_tissue + compents

        return embedding_tissue

    def set_learnable_text_encoder(self, learnable=False):
        self.text_encoder.requires_grad = learnable

    @torch.no_grad()
    def forward_emb(self, input_ids_report, embedding_tissue, ft_wsi, guassian_prototype=None):
        input_ids_report = input_ids_report.to(self.text_encoder.device)
        embedding_report = self.text_encoder(input_ids_report, attention_mask=None)[0]

        compents, _ = self.cross_attention_image(
            self.learnable_image_center, ft_wsi, ft_wsi
        )
        compents_image = self.norm(compents + self.learnable_image_center)

        embedding_report = embedding_report.view(1, -1, 768)
        compents, _ = self.cross_attention_text(
            self.learnable_text_center, embedding_report, embedding_report
        )
        compents_text = self.norm(compents + self.learnable_text_center)

        image_report_prototype = torch.cat((compents_text, compents_image), dim=1)

        return image_report_prototype
