# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import math
from functools import partial
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn
from mamba_ssm.utils.generation import InferenceParams
from torch import Tensor


BiMergingStrategy = Literal["concat", "add", "mul"]


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mixer_cls: Callable,
        mlp_cls: Optional[Callable] = None,
        norm_cls: Callable = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        """Adapted from 'Block' in mamba_ssm.modules.block, adapted to support additional mixers such as LTSMs (https://github.com/state-spaces/mamba/).
        Original license: Apache License 2.0 (https://github.com/state-spaces/mamba/blob/main/LICENSE)
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer (-> Add -> LN -> MLP), returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).

        Parameters
        ----------
        dim : int
            Input dimension D
        mixer_cls : Callable
            Class for the 'mixer' to use (e.g. Mamba)
        mlp_cls : Callable
            Class for MLP module to use (e.g. GatedMLP), by default None
        norm_cls : Callable, optional
            Class for the normalization to use, by default nn.LayerNorm
        fused_add_norm : bool, optional
            Fuse add and norm into one call, by default False
        residual_in_fp32 : bool, optional
            Force residual to be in dtype torch.float32, by default False
        """
        super().__init__()

        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if mlp_cls is not None and mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,
        residual: Optional[Tensor] = None,
        inference_params: Optional[InferenceParams] = None,
        **mixer_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pass the input through the encoder layer.

        Parameters
        ----------
        hidden_states : Tensor
            the sequence to the encoder layer
        residual : Optional[Tensor], optional
            hidden_states = Mixer(LN(residual)), by default None
        inference_params : InferenceParams, optional
            inference params, by default None

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple of tensors (hidden_states, residual)
        """
        # -- First half of the network: add + norm + mixer
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )
            residual: torch.Tensor
        # Only pass inference_params if they are provided, bc only Mamba supports it
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        mixer_return = self.mixer(hidden_states, **mixer_kwargs)
        # Discard all but the first output (LSTMs can return multiple, while Mamba returns exactly one)
        if isinstance(mixer_return, tuple):
            hidden_states = mixer_return[0]
        elif isinstance(mixer_return, torch.Tensor):
            hidden_states = mixer_return
        else:
            raise RuntimeError(f"Unexpected mixer forward value type: {type(mixer_return)}")

        # Second half of the network if MLP is specified: add + norm + mlp
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm),
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class ResidualBlockSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        block_cls: type,
        block_params: dict,
        norm_cls: Optional[type] = RMSNorm,
        norm_params: Optional[dict] = None,
        mlp_cls: Optional[type] = None,
        n_layer: int = 1,
        bidirectional: bool = False,
        bidirectional_merging: BiMergingStrategy = "concat",
        fused_add_norm: bool = False,
        pass_layer_idx: Optional[str] = None,
    ):
        """Parametrized (bidirectional) sequence of residual blocks.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        block_cls : type
            Core class of the residual block. e.g `mamba_ssm.modules.mamba_simple.Mamba`
        block_params : dict
            kwargs to instantiate the block_cls
        norm_cls : type, optional
            Class of the normalization, by default `RMSNorm`
        norm_params : Optional[dict], optional
            kwargs to instantiate the normalization, by default None
        n_layer : int, optional
            Number of blocks/layers to chain together, by default 1
        bidirectional : bool, optional
            Should it be bidirectional (a separate second sequence process the flipped input), by default False
        bidirectional_merging : Literal['concat', 'add', 'mul'], optional
            How to merge the bidirectional outputs. Only effective if `bidirection is True`.
            Will influence the number of features (`N*2` if `concat`, `N` if `add`) in the output vector.
            By default 'concat'
        fused_add_norm : bool, optional
            Fuse the add and norm operations, by default False
        pass_layer_idx : Optional[str], optional
            Name/key of the block_params kwarg containing the index of the layer
            If left to None will try to autodetect and not pass it if not found, by default None
        """
        super(ResidualBlockSequence, self).__init__()

        if bidirectional_merging not in ["concat", "add", "mul"]:
            raise ValueError(f"Invalid bidirectional_merging: {bidirectional_merging}")
        self.bidirectional_merging = bidirectional_merging

        # Instantiate default norm params
        if norm_params is None:
            norm_params = {}

        if pass_layer_idx is None:
            if block_cls == Mamba:
                pass_layer_idx = "layer_idx"

        self.forward_blocks = nn.ModuleList([])
        for i in range(n_layer):
            this_block_params = block_params.copy()
            if pass_layer_idx is not None:
                this_block_params[pass_layer_idx] = i
            self.forward_blocks.append(
                ResBlock(
                    in_channels,
                    mixer_cls=partial(block_cls, **this_block_params),
                    norm_cls=partial(norm_cls, **norm_params),
                    mlp_cls=mlp_cls,
                    fused_add_norm=fused_add_norm,
                )
            )

        if bidirectional:
            self.backward_blocks = nn.ModuleList([])
            for i in range(n_layer):
                this_block_params = block_params.copy()
                if pass_layer_idx is not None:
                    this_block_params[pass_layer_idx] = i
                self.backward_blocks.append(
                    ResBlock(
                        in_channels,
                        mixer_cls=partial(block_cls, **this_block_params),
                        norm_cls=partial(norm_cls, **norm_params),
                        mlp_cls=mlp_cls,
                        fused_add_norm=fused_add_norm,
                    )
                )

        self.apply(partial(_init_weights, n_layer=n_layer))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor
            Input vector `(BATCH, SEQUENCE, N)`-shaped

        Returns
        -------
        torch.Tensor
            Output vector `(BATCH, SEQUENCE, N2)`-shaped, where `N2 = N * 2 if (bidirectional and bidirectional_merging=='concat') else N`
        """

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
            if self.bidirectional_merging == "concat":
                residual = torch.cat([residual, back_residual], -1)
            elif self.bidirectional_merging == "add":
                residual += back_residual
            else:
                residual = torch.mul(residual, back_residual)

        return residual
