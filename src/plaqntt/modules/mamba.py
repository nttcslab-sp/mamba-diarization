# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from pyannote.audio.utils.params import merge_dict


class MambaBlockV2(nn.Module):
    """
    Parametrized bidirectional Mamba block from SPMamba https://github.com/JusperLee/SPMamba/blob/main/look2hear/models/SPMamba.py.
    Under Apache License 2.0 (not provided in the original repository).
    """

    def __init__(self, in_channels, n_layer=1, d_state=16, d_conv=4, expand=4, rmsnorm_eps=1e-5, bidirectional=False):
        super(MambaBlockV2, self).__init__()
        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            self.forward_blocks.append(
                Block(
                    in_channels,
                    mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                    norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                    fused_add_norm=False,
                )
            )
        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                self.backward_blocks.append(
                    Block(
                        in_channels,
                        mixer_cls=partial(Mamba, layer_idx=i, d_state=d_state, d_conv=d_conv, expand=expand),
                        norm_cls=partial(RMSNorm, eps=rmsnorm_eps),
                        fused_add_norm=False,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input):
        for_residual = None
        forward_f = input.clone()
        for block in self.forward_blocks:
            forward_f, for_residual = block(forward_f, for_residual, inference_params=None)
        residual = (forward_f + for_residual) if for_residual is not None else forward_f

        if hasattr(self, "backward_blocks"):
            back_residual = None
            backward_f = torch.flip(input, [1])
            for block in self.backward_blocks:
                backward_f, back_residual = block(backward_f, back_residual, inference_params=None)
            back_residual = (backward_f + back_residual) if back_residual is not None else backward_f

            back_residual = torch.flip(back_residual, [1])
            residual = torch.cat([residual, back_residual], -1)

        return residual
