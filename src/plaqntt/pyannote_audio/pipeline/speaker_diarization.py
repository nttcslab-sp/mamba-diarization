# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import functools
from typing import Optional, Text, Union

import numpy as np
from pyannote.core import SlidingWindowFeature

from pyannote.audio.pipelines.utils import (
    PipelineModel,
)
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization as SpeakerDiarizationV1

from plaqntt.pyannote_audio.core.inference2 import InferenceV2


class SpeakerDiarizationV2(SpeakerDiarizationV1):
    """Changes the usual SpeakerDiarization pipeline to use the new InferenceV2 class"""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation@2022.07",
        segmentation_step: float = 0.1,
        embedding: PipelineModel = "speechbrain/spkrec-ecapa-voxceleb@5c0be3875fda05e81f3c004ed8c7c06be308de1e",
        embedding_exclude_overlap: bool = False,
        clustering: str = "AgglomerativeClustering",
        embedding_batch_size: int = 1,
        segmentation_batch_size: int = 1,
        use_uem: bool = False,
        der_variant: Optional[dict] = None,
        use_auth_token: Union[Text, None] = None,
    ):
        super().__init__(
            segmentation=segmentation,
            segmentation_step=segmentation_step,
            embedding=embedding,
            embedding_exclude_overlap=embedding_exclude_overlap,
            clustering=clustering,
            embedding_batch_size=embedding_batch_size,
            segmentation_batch_size=segmentation_batch_size,
            der_variant=der_variant,
            use_auth_token=use_auth_token,
        )

        self.use_uem = use_uem

        # override the parent inference
        self._segmentation = InferenceV2(
            self._segmentation.model,
            duration=self._segmentation.duration,
            step=self._segmentation.step,
            batch_size=segmentation_batch_size,
        )

    def get_segmentations(self, file, hook=None) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile
        hook : Optional[Callable]

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """

        if hook is not None:
            hook = functools.partial(hook, "segmentation", None)

        if self.training:
            # Apply model / retrieve existing segmentations
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(
                    file, do_conversion=False, do_aggregation=False, hook=hook, use_uem=self.use_uem
                )
                file[self.CACHED_SEGMENTATION] = segmentations
            # Apply conversion if needed
            needs_conversion = self._segmentation.has_conversion()

            if needs_conversion:
                segmentations = SlidingWindowFeature(
                    data=self._segmentation.apply_conversion([segmentations.data])[0],
                    sliding_window=segmentations.sliding_window,
                    labels=segmentations.labels,
                )
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file, hook=hook, use_uem=self.use_uem)

        return segmentations

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        # This method is taken straight from the parent class.
        # We only fix the num_cluster assignment to not break under
        # edge cases.

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = max(0, np.max(hard_clusters) + 1)  # << Fixed line
        clustered_segmentations = np.nan * np.zeros((num_chunks, num_frames, num_clusters))

        for c, (cluster, (chunk, segmentation)) in enumerate(zip(hard_clusters, segmentations)):
            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(segmentation[:, cluster == k], axis=1)

        clustered_segmentations = SlidingWindowFeature(clustered_segmentations, segmentations.sliding_window)

        return self.to_diarization(clustered_segmentations, count)
