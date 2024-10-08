# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import datetime
import math
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Self, Text, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature, Timeline
from pytorch_lightning.utilities.memory import is_oom_error

from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.inference import Inference
from pyannote.audio.core.model import Model, Specifications
from pyannote.audio.core.task import Resolution
from pyannote.audio.utils.multi_task import map_with_specifications
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.utils.reproducibility import fix_reproducibility


class InferenceV2(Inference):
    """Inference with more arguments for _call / infer: do_conversion, do_aggregation, do_ndarray_conversion
    Also the logic is easier to read I think...
    """

    def __init__(
        self,
        model: Union[Model, Text, Path],
        window: Text = "sliding",
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        device: Optional[torch.device] = None,
        batch_size: int = 32,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__(
            model=model,
            window=window,
            duration=duration,
            step=step,
            pre_aggregation_hook=pre_aggregation_hook,
            device=device,
            batch_size=batch_size,
            use_auth_token=use_auth_token,
            skip_aggregation=False,
            skip_conversion=False,
        )
        # last two arguments are not used by the pipeline
        # get rid of them for good measure
        del self.skip_aggregation
        del self.skip_conversion

    def _get_specifications_iterable(self):
        if isinstance(self.model.specifications, Specifications):
            return [self.model.specifications]
        else:
            return self.model.specifications

    def has_conversion(self) -> bool:
        return not isinstance(self.conversion, nn.Identity)

    def apply_conversion(
        self, outputs: Union[List[torch.Tensor], List[np.ndarray]]
    ) -> Union[List[torch.Tensor], List[np.ndarray]]:
        """Apply the conversion (to multilabel) to a list of tensor or numpy arrays.
        Has no effect if the model outputs multilabel arrays.

        Parameters
        ----------
        outputs : Union[List[torch.Tensor], List[np.ndarray]]
            (B,F,features) shaped inputs to be converted.

        Returns
        -------
        Union[List[torch.Tensor], List[np.ndarray]]
            List of (B,F,spk_count) shaped outputs.
        """
        result: Union[List[torch.Tensor], List[np.ndarray]] = []

        for output in outputs:
            if isinstance(output, torch.Tensor):
                self.conversion.to(device=output.device)
                result.append(self.conversion(output))
            elif isinstance(output, np.ndarray):
                self.conversion.to(device="cpu")
                result.append(self.conversion(torch.from_numpy(output)).numpy())
            else:
                raise ValueError(f"Unsupported output type: {type(output).__name__}")
        self.conversion.to(device=self.device)
        return result

    def apply_ndarray_conversion(self, outputs: List[torch.Tensor]) -> List[np.ndarray]:
        return [output.cpu().detach().numpy() for output in outputs]

    def apply_aggregation(self, outputs):
        raise NotImplementedError("TODO")

    def infer(
        self,
        chunks: torch.Tensor,
        do_conversion: bool = True,
        do_ndarray_conversion: bool = True,
    ) -> Tuple[np.ndarray]:
        """Forward pass. Supports skipping powerset conversion and ndarray conversion."""

        with torch.inference_mode():
            try:
                outputs = self.model(chunks.to(self.device))
                if isinstance(outputs, torch.Tensor):
                    outputs = [outputs]
            except RuntimeError as exception:
                if is_oom_error(exception):
                    raise MemoryError(
                        f"batch_size ({self.batch_size: d}) is probably too large. "
                        f"Try with a smaller value until memory error disappears."
                    )
                else:
                    raise exception

        conv: nn.Module = self.conversion if do_conversion else nn.Identity()
        ndarray_conv = lambda x: x.cpu().detach().numpy() if do_ndarray_conversion else x

        return [ndarray_conv(conv(output)) for output in outputs]

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        do_conversion: bool = True,
        do_aggregation: bool = True,
        do_ndarray_conversion: bool = True,
        uem: Optional[Timeline] = None,
        hook: Optional[Callable] = None,
    ) -> List[SlidingWindowFeature]:
        """Slide model on a waveform

        Parameters
        ----------
        waveform: (num_channels, num_samples) torch.Tensor
            Waveform.
        sample_rate : int
            Sample rate.
        hook: Optional[Callable]
            When a callable is provided, it is called everytime a batch is
            processed with two keyword arguments:
            - `completed`: the number of chunks that have been processed so far
            - `total`: the total number of chunks

        Returns
        -------
        output : List[SlidingWindowFeature]
            Model output. Shape is (num_chunks, dimension) for chunk-level tasks,
            and (num_frames, dimension) for frame-level tasks.
        """

        window_size: int = self.model.audio.get_num_samples(self.duration)
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape

        frames: List[SlidingWindow] = []
        for specs in self._get_specifications_iterable():
            if specs.resolution == Resolution.CHUNK:
                frames.append(SlidingWindow(start=0.0, duration=self.duration, step=self.step))
            else:
                frames.append(self.model.receptive_field)

        # prepare complete chunks
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0

        # prepare last incomplete chunk
        has_last_chunk = (num_samples < window_size) or (num_samples - window_size) % step_size > 0
        if has_last_chunk:
            # pad last chunk with zeros
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))

        # One output list per specification (= number of elements in the model output tuple)
        outputs: List[List[np.ndarray]] = [list() for _ in frames]

        def __append_batch(batch) -> None:
            for i, b in enumerate(batch):
                outputs[i].append(b)

        if hook is not None:
            hook(completed=0, total=num_chunks + has_last_chunk)

        # slide over audio chunks in batch
        sw_chunks_start, sw_chunks_end = 0.0, num_samples / sample_rate
        first_chunk_idx, last_chunk_idx = 0, num_chunks - 1
        # if uem is passed, only process what's inside the extent
        # TODO: support more fine-grained UEMs (ie with gaps)
        if uem is not None:
            first_chunk_idx = math.floor(uem.extent().start / self.step)
            # TODO: i think if its '(uem.extent().end - self.duration) / self.step' then the min is not necessary anymore
            last_chunk_idx = min(math.floor(uem.extent().end / self.step), num_chunks - 1)
            sw_chunks_start = first_chunk_idx * self.step
            if has_last_chunk and last_chunk_idx == num_chunks - 1:
                sw_chunks_end = num_samples / sample_rate
            else:
                sw_chunks_end = last_chunk_idx * self.step + self.duration

            if uem.extent().end > num_samples / sample_rate:
                print(
                    f"!!! WARNING: UEM extent {uem.extent()} is longer than the audio file duration {num_samples / sample_rate} !!!",
                    flush=True,
                )
            # print(f'Using batches {first_chunk_idx} to {last_chunk_idx} out of {num_chunks}. From UEM {[s for s in uem.segments_list_]} ({sw_chunks_start} - {sw_chunks_end})', flush=True)

        for c in np.arange(first_chunk_idx, last_chunk_idx + 1, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]
            # print(f'>>> {datetime.datetime.now().strftime("%H:%M:%S")}] Processing batch {c} to {c + self.batch_size} shape={batch.shape}', flush=True)

            # Compute outputs and append them to their respective lists
            batch_outputs: Union[np.ndarray, Tuple[np.ndarray]] = self.infer(
                batch, do_conversion=do_conversion, do_ndarray_conversion=do_ndarray_conversion
            )
            __append_batch(batch_outputs)

            if hook is not None:
                hook(completed=c + self.batch_size, total=num_chunks + has_last_chunk)

        # process orphan last chunk (if we process the file to the end)
        if has_last_chunk and last_chunk_idx == num_chunks - 1:
            last_outputs = self.infer(
                last_chunk[None], do_conversion=do_conversion, do_ndarray_conversion=do_ndarray_conversion
            )
            __append_batch(last_outputs)

            if hook is not None:
                hook(
                    completed=num_chunks + has_last_chunk,
                    total=num_chunks + has_last_chunk,
                )

        if do_ndarray_conversion:
            outputs: List[np.ndarray] = list(map(np.vstack, outputs))
        else:
            outputs: List[torch.Tensor] = list(map(torch.vstack, outputs))

        result: List[SlidingWindowFeature] = []
        for i_out, i_frames, i_specs in zip(outputs, frames, self._get_specifications_iterable()):
            i_out: Union[np.ndarray, torch.Tensor]
            i_frames: SlidingWindow
            i_specs: Specifications
            # skip aggregation when requested,
            # or when model outputs just one vector per chunk
            # or when model is permutation-invariant (and not post-processed)
            if (
                not do_aggregation
                or i_specs.resolution == Resolution.CHUNK
                or (i_specs.permutation_invariant and self.pre_aggregation_hook is None)
            ):
                i_frames = SlidingWindow(start=sw_chunks_start, duration=self.duration, step=self.step)
                result.append(SlidingWindowFeature(i_out, i_frames))
            else:
                if self.pre_aggregation_hook is not None:
                    i_out = self.pre_aggregation_hook(i_out)

                aggregated = self.aggregate(
                    SlidingWindowFeature(
                        i_out,
                        SlidingWindow(start=sw_chunks_start, duration=self.duration, step=self.step),
                    ),
                    i_frames,
                    warm_up=self.warm_up,
                    hamming=True,
                    missing=0.0,
                )

                # remove padding that was added to last chunk
                # if has_last_chunk:
                aggregated = aggregated.crop(Segment(sw_chunks_start, sw_chunks_end), mode="loose", return_data=False)

                result.append(aggregated)
        return result

    def __call__(
        self,
        file: AudioFile,
        force_return_list: bool = False,
        do_conversion: bool = True,
        do_aggregation: bool = True,
        do_ndarray_conversion: bool = True,
        use_uem: bool = False,
        hook: Optional[Callable] = None,
    ) -> Union[List[SlidingWindowFeature], List[np.ndarray]]:
        """Run inference on a whole file.
        Supports disabling conversion, aggregation and ndarray conversion.
        """

        fix_reproducibility(self.device)

        waveform, sample_rate = self.model.audio(file)

        uem: Optional[Timeline] = None
        if use_uem:
            uem = file["annotated"]

        # Compute
        if self.window == "sliding":
            result = self.slide(
                waveform,
                sample_rate,
                do_conversion=do_conversion,
                do_ndarray_conversion=do_ndarray_conversion,
                do_aggregation=do_aggregation,
                uem=uem,
                hook=hook,
            )
        else:
            outputs: Tuple[np.ndarray] = self.infer(
                waveform[None], do_conversion=do_conversion, do_ndarray_conversion=do_ndarray_conversion
            )
            result = [o[0] for o in outputs]
        # Return
        if not force_return_list and len(result) == 1:
            return result[0]
        else:
            return result
