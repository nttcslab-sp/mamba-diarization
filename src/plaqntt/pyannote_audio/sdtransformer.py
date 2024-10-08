# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import contextlib
from functools import lru_cache
from typing import Optional, TypedDict, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)

from plaqntt.modules.transformer_eendvc import TransformerEncoder


class TrEncParams(TypedDict):
    n_layers: int
    """Number of transformer encoder layers."""

    n_units: int
    """Number of features inside the Encoder (after a linear projection)"""

    e_units: int
    """Number of features for the PositionwiseFeedForward (n_units -> e_units (w/ dropout) -> n_units)"""

    dropout_rate: float
    """MHSA and PositionWiseFeedForward Dropout rate."""

    h: int
    """Number of heads in the MultiHeadSelfAttention."""


def build_blocks_from_cfg(params: TrEncParams, in_features) -> tuple[nn.Module, int]:
    conformamba = TransformerEncoder(
        idim=in_features,
        n_layers=params["n_layers"],
        n_units=params["n_units"],
        e_units=params["e_units"],
        h=params["h"],
        dropout_rate=params["dropout_rate"],
    )

    return conformamba, params["n_units"]


class SdTransformer(Model):
    """Segmentation model based on transformer encoder.

    wav2vec > TransformerEncoder > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec : dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer : int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    trencoder : TrEncParams, optional
        Configuration of the Transformer Encoder.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    TRENC_DEFAULTS: TrEncParams = {
        "n_layers": 6,
        "n_units": 256,
        "e_units": 2048,
        "dropout_rate": 0.1,
        "h": 8,
    }

    def __init__(
        self,
        wav2vec: Optional[Union[dict, str]] = None,
        wav2vec_layer: int = -1,
        trenc: Optional[TrEncParams] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if wav2vec is None:
            wav2vec = self.WAV2VEC_DEFAULTS

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz.")
                wav2vec_dim: int = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()

            # `wav2vec` is a path to a self-supervised representation checkpoint
            else:
                _checkpoint = torch.load(wav2vec)
                wav2vec = _checkpoint.pop("config")
                self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                state_dict = _checkpoint.pop("state_dict")
                self.wav2vec.load_state_dict(state_dict)
                wav2vec_dim: int = wav2vec["encoder_embed_dim"]
                wav2vec_num_layers = wav2vec["encoder_num_layers"]

        # `wav2vec` is a config dictionary understood by `wav2vec2_model`
        # this branch is typically used by Model.from_pretrained(...)
        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim: int = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]
        else:
            raise ValueError("Expected `wav2vec` to be a string, a dictionary, or a path to a checkpoint.")

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(data=torch.ones(wav2vec_num_layers), requires_grad=True)

        trenc: TrEncParams = merge_dict(self.TRENC_DEFAULTS, trenc)  # type: ignore
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("wav2vec", "wav2vec_layer", "trenc", "linear")

        # Initialize the blocks
        print(f"{dict(self.hparams.trenc)}")  # type: ignore
        self.trenc, trenc_outdim = build_blocks_from_cfg(trenc, wav2vec_dim)

        if linear["num_layers"] < 1:
            return

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        trenc_outdim,
                    ]
                    + [self.hparams.linear["hidden_size"]] * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.freeze_wavlm = True

    @property
    def dimension(self) -> int:
        """Dimension of output"""
        if isinstance(self.specifications, tuple):
            raise ValueError("SSeRiouSS does not support multi-tasking.")

        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (2 if self.hparams.lstm["bidirectional"] else 1)
            raise NotImplementedError("Not implemented for SdMambaV2")

        self.classifier = nn.Linear(in_features, self.dimension)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        """Compute number of output frames

        Parameters
        ----------
        num_samples : int
            Number of input samples.

        Returns
        -------
        num_frames : int
            Number of output frames.
        """

        num_frames = num_samples
        for conv_layer in self.wav2vec.feature_extractor.conv_layers:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )

        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        """Compute size of receptive field

        Parameters
        ----------
        num_frames : int, optional
            Number of frames in the output signal

        Returns
        -------
        receptive_field_size : int
            Receptive field size.
        """

        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        """Compute center of receptive field

        Parameters
        ----------
        frame : int, optional
            Frame index

        Returns
        -------
        receptive_field_center : int
            Index of receptive field center.
        """
        receptive_field_center = frame
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        num_layers = None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer

        with torch.no_grad() if self.freeze_wavlm else contextlib.nullcontext():
            outputs, _ = self.wav2vec.extract_features(waveforms.squeeze(1), num_layers=num_layers)

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.wav2vec_weights, dim=0)
        else:
            outputs = outputs[-1]

        # because the Encoder outputs (batch*frames, features) we have to reshape
        n_batches, n_frames, _ = outputs.shape
        outputs = self.trenc(outputs).reshape(n_batches, n_frames, -1)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
