# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024
"""Generalistic Residual blocks-based Speaker Diarization Segmentation models.
"""


import contextlib
from functools import lru_cache
from typing import Iterable, Literal, NotRequired, Optional, TypedDict, Union
from copy import deepcopy

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core.utils.generators import pairwise
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)
from pyannote.audio.utils.powerset import Powerset

from plaqntt.modules.resblock import ResidualBlockSequence, BiMergingStrategy

try:
    from xLSTM.block import xLSTMBlock
except:
    xLSTMBlock = None

from pyannote.audio.models.blocks.sincnet import SincNet


class ResBlockParams(TypedDict):
    n_layer: int
    bidirectional: bool
    bidirectional_merging: BiMergingStrategy
    inproj: str
    outproj: str


class BlockParams(TypedDict):
    res: ResBlockParams
    block: dict


RESBLOCK_DEFAULT_PARAMS: ResBlockParams = {
    "n_layer": 1,
    "bidirectional": True,
    "bidirectional_merging": "concat",
    "inproj": "x1.0",  # linear projection to multiply the number of input features
    "outproj": "x1.0",  # 1.0 means projecting to keep half the features in case of bidirectional==True !
}
BLOCKS_CLASSES = {
    "mamba": Mamba,
    "mamba2": Mamba2,
    "xlstm": xLSTMBlock,
    "lstm": nn.LSTM,
}
BLOCKS_BLOCKS_DEFAULTS = {
    "mamba": {
        "d_state": 16,  # state dimension
        "d_conv": 4,  # convolution dimension
        "expand": 2,  # expansion factor
        "dt_rank": "auto",
    },
    "mamba2": {
        "d_state": 128,
        "d_conv": 4,
        "expand": 2,
        "headdim": 64,
        # d_ssm: Optional[int] = None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        # ngroups: int = 1,
        # A_init_range: tuple[float, float] = (1, 16),
        # D_has_hdim: bool = False,
        # rmsnorm: bool = True,
        # norm_before_gate: bool = False,
        # dt_min: float = 0.001,
        # dt_max: float = 0.1,
        # dt_init_floor: float = 1e-4,
        # dt_limit: tuple[float, float] = (0.0, float("inf")),
        # bias: bool = False,
        # conv_bias: bool = True,
    },
    "xlstm": {
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "lstm_type": "slstm",
    },
    "lstm": {
        "hidden_size": 128,
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": False,
        "batch_first": True,
    },
}


def parse_inoutproj(value, features: int) -> int:
    if isinstance(value, str):
        value = value.lower()
        if value.startswith("x"):
            return round(features * float(value[1:]))
        elif value.startswith("c"):
            return int(value[1:])
        else:
            raise ValueError(f"Invalid value: {value}")
    elif isinstance(value, int):
        print("USING DANGEROUS PARSING")
        if value > 2:
            return value
    elif isinstance(value, float):
        print("USING DANGEROUS PARSING")
        if value > 2:
            return int(value)
        elif value >= 0 and value <= 1:
            return round(features * value)
    raise ValueError(f"Invalid value type: {value} of type {type(value)}")


def build_blocks_from_cfg(
    blocks: dict[str, BlockParams],
    input_features: int,
    return_module: Literal["sequential", "modulelist", "list"] = "modulelist",
) -> tuple[nn.Module, int]:
    nnblocks_list: list[nn.Module] = []

    # First sanitize and fill missing keys
    for block_id, block_params in blocks.items():
        block_type = block_id.rsplit("@")[-1]
        if "block" not in block_params:
            block_params["block"] = {}
        if "res" not in block_params:
            block_params["res"] = {}  # type: ignore
        block_params["block"] = merge_dict(BLOCKS_BLOCKS_DEFAULTS[block_type], block_params["block"])
        block_params["res"] = merge_dict(RESBLOCK_DEFAULT_PARAMS, block_params["res"])  # type: ignore
        blocks[block_id] = block_params  # not sure if this is necessary, idr how dicts work

    # Then build the blocks
    last_block_features: int = input_features
    for block_key in sorted(blocks.keys(), key=lambda x: int(x.split("@")[0])):
        block_type = block_key.rsplit("@")[-1]
        block_params: BlockParams = blocks[block_key]

        print(f"-- Block {block_key}, input features: {last_block_features}, cfg: {block_params} --")
        stage_modules = []
        # Compute n of input featuresm create linear layer if needed
        block_input_features = parse_inoutproj(block_params["res"]["inproj"], last_block_features)
        if block_input_features != last_block_features:
            stage_modules.append(nn.Linear(last_block_features, block_input_features))
            print(f"Input linear layer added: {last_block_features} -> {block_input_features}")
        # Create the central residual block and append it
        resblock = ResidualBlockSequence(
            in_channels=block_input_features,
            block_cls=BLOCKS_CLASSES[block_type],
            block_params=block_params["block"],
            n_layer=block_params["res"]["n_layer"],
            bidirectional=block_params["res"]["bidirectional"],
            bidirectional_merging=block_params["res"]["bidirectional_merging"],
        )
        stage_modules.append(resblock)
        # evaluate its output size
        with torch.no_grad():
            gpu = torch.device("cuda")
            resblock.to(gpu)
            res_output_features: int = resblock(torch.randn(1, 512, block_input_features, device=gpu)).shape[-1]

        # bidirectional_multiplier: float = 2.0 if block_params["res"]["bidirectional"] is True else 1.0
        # res_output_features: int = round(block_input_features * bidirectional_multiplier)
        print(f"{block_type} block added: {block_input_features} -> {res_output_features}")
        # Compute n of output features, create linear layer if needed
        block_output_features = parse_inoutproj(block_params["res"]["outproj"], block_input_features)

        if block_output_features != res_output_features:
            stage_modules.append(nn.Linear(res_output_features, block_output_features))
            print(f"Output linear layer added: {res_output_features} -> {block_output_features}")
        # Append the block to the list and prepare for next
        nnblocks_list.append(nn.Sequential(*stage_modules))
        last_block_features = block_output_features
    # The mamba part is just the sequential call of all stages
    if return_module == "sequential":
        return nn.Sequential(*nnblocks_list), last_block_features
    elif return_module == "modulelist":
        return nn.ModuleList(nnblocks_list), last_block_features
    elif return_module == "list":
        return nnblocks_list, last_block_features
    else:
        raise ValueError(f"Invalid return_module: {return_module}")


class MultiActivation(nn.Module):
    def __init__(self, activations: list[tuple[int, nn.Module]]):
        super().__init__()
        self.activations = nn.ModuleList([act for _, act in activations])
        self.lengths = [idx for idx, _ in activations]

        # compute starting indices from lengths
        self.start_indices = []
        start_idx = 0
        for idx, act in activations:
            self.start_indices.append(start_idx)
            start_idx += idx

    def forward(self, x) -> torch.Tensor:
        outputs = []
        for idx, length, act in zip(self.start_indices, self.lengths, self.activations):
            outputs.append(act(x[..., idx : idx + length]))
        return torch.cat(outputs, dim=-1)


class SdResBlocks(Model):
    """Residual blocks based Speaker Diarization Segmentation model.

    wav2vec > Residual blocks > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    wav2vec: dict or str, optional
        Defaults to "WAVLM_BASE".
    wav2vec_layer: int, optional
        Index of layer to use as input to the LSTM.
        Defaults (-1) to use average of all layers (with learnable weights).
    blocks : dict, optional
        Dictionary describing the sequence of blocks.
        Keys should be str formatted "ID@TYPE" where ID is the block sequence index and type its block type.
        Values are dictionaries with content:
        'block': block_kwargs_dict
        'res': res_kwargs_dict
        Supported block types are lstm, xlstm and mamba.
        Defaults to {"0@lstm": {}}, i.e. a single LSTM block.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    WAV2VEC_DEFAULTS = "WAVLM_BASE"
    RESBLOCKS_DEFAULTS: dict[str, BlockParams] = {"0@lstm": {}}  # type: ignore
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2, "force_ps_n_ml": False}

    def __init__(
        self,
        wav2vec: Union[dict, str] = None,
        wav2vec_layer: int = -1,
        blocks: Optional[dict[str, BlockParams]] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if isinstance(wav2vec, str):
            # `wav2vec` is one of the supported pipelines from torchaudio (e.g. "WAVLM_BASE")
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz.")
                wav2vec_dim = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()
                # WAVLM_LARGE has a wav2vec.model.(representations/...) structure instead of wav2vec.(.../..)
                if hasattr(self.wav2vec, "model"):
                    self.wav2vec = self.wav2vec.model

            # `wav2vec` is a path to a self-supervised representation checkpoint
            else:
                _checkpoint = torch.load(wav2vec)
                wav2vec = _checkpoint.pop("config")
                self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
                state_dict = _checkpoint.pop("state_dict")
                self.wav2vec.load_state_dict(state_dict)
                wav2vec_dim = wav2vec["encoder_embed_dim"]
                wav2vec_num_layers = wav2vec["encoder_num_layers"]

        # `wav2vec` is a config dictionary understood by `wav2vec2_model`
        # this branch is typically used by Model.from_pretrained(...)
        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(data=torch.ones(wav2vec_num_layers), requires_grad=True)

        if blocks is None:
            blocks = self.RESBLOCKS_DEFAULTS

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("wav2vec", "wav2vec_layer", "blocks", "linear")

        # Initialize the blocks
        print(f"{dict(self.hparams.blocks)}")
        self.blocks, last_block_features = build_blocks_from_cfg(blocks, wav2vec_dim)

        if linear["num_layers"] < 1:
            return

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        last_block_features,
                    ]
                    + [self.hparams.linear["hidden_size"]] * self.hparams.linear["num_layers"]
                )
            ]
        )

        self.multi_output_mode = ["ml", "ps"]
        self.has_multi_output = self.hparams["linear"]["force_ps_n_ml"]
        self.default_forward_mode = "one"
        self.default_forward_outputs = ["ps"]

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

        if not self.has_multi_output:
            self.classifier = nn.Linear(in_features, self.dimension)
            self.activation = self.default_activation()
        else:
            self.classifier = nn.Linear(
                in_features, self.specifications.num_powerset_classes + len(self.specifications.classes)
            )
            self.activation = MultiActivation(
                [
                    (self.specifications.num_powerset_classes, nn.LogSoftmax(dim=-1)),
                    (len(self.specifications.classes), nn.Sigmoid()),
                ]
            )

        if self.specifications.powerset:
            self.powerset = Powerset(len(self.specifications.classes), self.specifications.powerset_max_classes)

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

    def forward(
        self,
        waveforms: torch.Tensor,
        multi_output_mode: Literal["concat", "dict", "one", "mean"] = None,
        multi_output: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        if multi_output_mode is None:
            multi_output_mode = self.default_forward_mode
        if multi_output is None:
            multi_output = self.default_forward_outputs
        if multi_output_mode == "one" and len(multi_output) > 1:
            raise ValueError("Cannot use multi_output_mode 'one' with multiple outputs")

        num_layers = None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer

        with torch.no_grad() if self.freeze_wavlm else contextlib.nullcontext():
            outputs, _ = self.wav2vec.extract_features(waveforms.squeeze(1), num_layers=num_layers)

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.wav2vec_weights, dim=0)
        else:
            outputs = outputs[-1]
        outputs: torch.Tensor

        for block in self.blocks:
            outputs = block(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        outputs = self.classifier(outputs)
        outputs = self.activation(outputs)

        if self.has_multi_output:
            outputs_dict = {
                "ps": None,
                "ml": None,
            }
            if "ps" in self.multi_output_mode:
                outputs_dict["ps"] = outputs[:, :, : self.specifications.num_powerset_classes]
            if "ml" in self.multi_output_mode:
                outputs_dict["ml"] = outputs[:, :, self.specifications.num_powerset_classes :]

            if multi_output_mode == "dict":
                return outputs_dict
            elif multi_output_mode == "concat":
                return torch.cat(tensors=list(outputs_dict.values()), dim=-1)
            elif multi_output_mode == "one":
                return outputs_dict[multi_output[0]]
            else:
                raise ValueError(f"Invalid multi_output_mode: {multi_output_mode}")
        else:
            return outputs

    # def tensor_to_multioutput_dict(self, tensor: torch.Tensor, output_mode=None):
    #     if output_mode is None:
    #         output_mode = self.multi_output_mode

    #     res = {
    #         "ps": None,
    #         "ml": None,
    #     }

    #     last_class_idx = 0
    #     if "ps" in output_mode:
    #         res['ps'] = tensor[:,:,:self.specifications.num_powerset_classes]
    #         last_class_idx += self.specifications.num_powerset_classes
    #     if "ml" in output_mode:
    #         res["ml"] = tensor[:,:,last_class_idx:last_class_idx+len(self.specifications.classes)]

    #     return res


class SdResBlocksSincnet(Model):
    """Residual blocks based Speaker Diarization Segmentation model.

    wav2vec > Residual blocks > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet: dict or str, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    blocks : dict, optional
        Dictionary describing the sequence of blocks.
        Keys should be str formatted "ID@TYPE" where ID is the block sequence index and type its block type.
        Values are dictionaries with content:
        'block': block_kwargs_dict
        'res': res_kwargs_dict
        Supported block types are lstm, xlstm and mamba.
        Defaults to {"0@lstm": {}}, i.e. a single LSTM block.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}
    RESBLOCKS_DEFAULTS: dict[str, BlockParams] = {"0@lstm": {}}  # type: ignore
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: Optional[Union[dict, str]] = None,
        blocks: Optional[dict[str, BlockParams]] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if isinstance(sincnet, str):
            self.sincnet = torch.load(sincnet)
            sincnet = {
                "stride": self.sincnet.stride,
                # 'sample_rate': self.sincnet.sample_rate,
            }
            sincnet_features = getattr(self.sincnet, "n_filters_out", 60)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        if blocks is None:
            blocks = self.RESBLOCKS_DEFAULTS

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("sincnet", "blocks", "linear")

        # If sincnet has not already been loaded from pretrained model
        if not hasattr(self, "sincnet"):
            kernel_size = self.hparams.sincnet.get("kernel_size", None)
            if isinstance(kernel_size, Iterable):
                sincnet_kwargs = deepcopy(self.hparams.sincnet)
                del sincnet_kwargs["kernel_size"]
                kernel_size = sorted(kernel_size)

                self.sincnet = nn.ModuleList(
                    [SincNet(kernel_size=n, padding=(n - kernel_size[0]) // 2, **sincnet_kwargs) for n in kernel_size]
                )
                sincnet_features = sum([sinc.n_filters_out for sinc in self.sincnet])
            else:
                self.sincnet = SincNet(**self.hparams.sincnet)
                sincnet_features = getattr(self.sincnet, "n_filters_out", 60)

        # Initialize the blocks
        print(f"{dict(self.hparams.blocks)}")
        self.blocks, last_block_features = build_blocks_from_cfg(blocks, sincnet_features)

        if linear["num_layers"] < 1:
            return

        self.linear = nn.ModuleList(
            [
                nn.Linear(in_features, out_features)
                for in_features, out_features in pairwise(
                    [
                        last_block_features,
                    ]
                    + [self.hparams.linear["hidden_size"]] * self.hparams.linear["num_layers"]
                )
            ]
        )

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
        """Compute number of output frames for a given number of input samples

        Parameters
        ----------
        num_samples : int
            Number of input samples

        Returns
        -------
        num_frames : int
            Number of output frames
        """
        if isinstance(self.sincnet, nn.ModuleList):
            return self.sincnet[-1].num_frames(num_samples)
        else:
            return self.sincnet.num_frames(num_samples)

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
        if isinstance(self.sincnet, nn.ModuleList):
            return self.sincnet[-1].receptive_field_size(num_frames=num_frames)
        else:
            return self.sincnet.receptive_field_size(num_frames=num_frames)

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

        if isinstance(self.sincnet, nn.ModuleList):
            return self.sincnet[-1].receptive_field_center(frame=frame)
        else:
            return self.sincnet.receptive_field_center(frame=frame)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        if isinstance(self.sincnet, nn.ModuleList):
            outputs = torch.cat([sincnet(waveforms) for sincnet in self.sincnet], dim=1)
        else:
            outputs = self.sincnet(waveforms)
        outputs = rearrange(outputs, "batch feature frame -> batch frame feature")

        for block in self.blocks:
            outputs = block(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
