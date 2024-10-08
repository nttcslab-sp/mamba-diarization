# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024
"""Messy specialized DER computation script."""

import argparse
from pathlib import Path
import re
from typing import Optional
import zipfile
import numpy as np
from pyannote.database import registry, ProtocolFile
from pyannote.database.util import load_rttm
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio.utils.powerset import Powerset
from pyannote.audio.torchmetrics.audio import (
    DiarizationErrorRate as DiarizationErrorRateTorchMetrics,
    OptimalDiarizationErrorRate as OptimalDiarizationErrorRateTorchMetrics,
)
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.torchmetrics.functional.audio.diarization_error_rate import _der_compute
from plaqntt.utils.pyannote_database import protocol_fullname_to_name_subset
import torch
from torchmetrics import Metric
import tqdm
import pandas as pd
import yaml
from plaqntt.pyannote_audio.utils.permutation import match_speaker_count
from torchmetrics.classification import MulticlassCalibrationError, BinaryCalibrationError


LOCAL_THRESHOLDS_FOLDER = "local_thresholds"


def _get_filterable_metric_from_file(file: ProtocolFile, metric: str) -> float:
    if metric == "duration":
        return file["annotated"].duration()
    else:
        raise ValueError(f"Unknown filterable metric {metric}")


def _get_filterable_criterion_fn(criterion: str) -> callable:
    if criterion == "gt":
        return lambda x, y: x > y
    elif criterion == "lt":
        return lambda x, y: x < y
    elif criterion == "geq":
        return lambda x, y: x >= y
    elif criterion == "leq":
        return lambda x, y: x <= y
    elif criterion == "eq":
        return lambda x, y: x == y
    elif criterion == "neq":
        return lambda x, y: x != y
    raise ValueError(f"Unknown criterion {criterion}")


def test_match_filter(file: ProtocolFile, filters: list[str]) -> bool:
    for identifier, criterion, value in zip(filters[0::3], filters[1::3], filters[2::3]):
        metric = _get_filterable_metric_from_file(file, identifier)
        criterion_fn = _get_filterable_criterion_fn(criterion)
        try:
            value = float(value)
        except ValueError:
            pass
        if not criterion_fn(metric, value):
            print(f"Filter {identifier} {criterion} {value} failed for file {file['uri']} (value={metric})")
            return False

    return True


def str_to_comment(s: str, comment_prefix: str = "#\t") -> str:
    return "\n".join([f"{comment_prefix}{l}" for l in s.split("\n")]) + "\n"


def get_localder_thresholds_ders(metric):
    ders = _der_compute(
        metric.FalseAlarm,
        metric.MissedDetection,
        metric.SpeakerConfusion,
        metric.speech_total,
    )
    return metric.threshold, ders


def compute_segmentation_error_rate(
    protocolfiles,
    inference_zip_path,
    n_thresholds=51,
) -> tuple[dict[str, Metric], dict[str, dict[str, Metric]]]:
    threshold_search_space = torch.linspace(0.0, 1.0, n_thresholds)

    empty_time_tensors_count = 0
    empty_spk_tensors_count = 0
    seen_tensors = 0

    der = OptimalDiarizationErrorRateTorchMetrics(threshold=threshold_search_space.clone())
    ece = None

    uri_to_dermetric: dict[str, OptimalDiarizationErrorRateTorchMetrics] = {}
    uri_to_ecemetric: dict[str, Metric] = {}

    for file in tqdm.tqdm(protocolfiles, desc="Computing local DER"):
        # Open zipfile and retrieve the 'meta' file and the npz file containing all data (especially segmentations and uem_t)
        with zipfile.ZipFile(inference_zip_path, "r") as zipf:
            with zipf.open(file["uri"] + "/ndarrays.npz", "r") as f:
                meta = yaml.load(zipf.open(file["uri"] + "/meta.yaml", "r"), Loader=yaml.FullLoader)
                npz = np.load(f)
                predswf = SlidingWindowFeature(
                    data=npz["segmentations"],
                    sliding_window=SlidingWindow(**{k: v for k, v in meta["segmentation_sw"].items() if k != "labels"}),
                )
                uem_t: torch.Tensor = torch.from_numpy(npz["segmentation_uem_t"])
                resolution = SlidingWindow(
                    **{k: v for k, v in meta["seg_model"]["receptive_field"].items() if k != "labels"}
                )
                specs_dict: dict = meta["seg_model"]["specifications"]
                n_speakers: int = len(specs_dict["classes"])
                # resolution = predswf.sliding_window.duration / predswf.data.shape[1]

        is_powerset: bool = specs_dict["powerset"] is True and not (
            "-psml_double" in inference_zip_path and "/ml" in inference_zip_path
        )

        # Prepare conversion functions and calibration error depending on output type
        if is_powerset:
            ps = Powerset(len(specs_dict["classes"]), specs_dict["powerset_max_classes"])
            if ps.num_powerset_classes != specs_dict["num_powerset_classes"]:
                raise RuntimeError(
                    f"Something went wrong with the powerset conversion. {ps.num_powerset_classes} classes vs {specs_dict['num_powerset_classes']} expected."
                )
            to_multilabel = lambda x: ps.to_multilabel(x)
            file_ece = MulticlassCalibrationError(num_classes=ps.num_powerset_classes)
        else:
            to_multilabel = lambda x: x
            file_ece = BinaryCalibrationError()

        # Init and save the metric specific to this file
        fileder = OptimalDiarizationErrorRateTorchMetrics(threshold=threshold_search_space.clone())
        uri_to_dermetric[file["uri"]] = fileder
        if ece is None:
            ece = file_ece.clone()
        uri_to_ecemetric[file["uri"]] = file_ece

        ref: Annotation = file["annotation"]
        # Iterate and compute metrics on each chunk
        seen_tensors += predswf.data.shape[0]
        for i, (predswffeat, predswfsegment) in enumerate(predswf.iterfeatures(window=True)):
            uem_t_i = uem_t[i, :, 0]  # (num_frames,) bool

            # Get tensor discretized version of the annotation and UEM for this chunk
            discretized = ref.discretize(
                predswfsegment,
                resolution=resolution,
                duration=predswf.sliding_window.duration,
            ).data
            if discretized.shape[0] >= uem_t_i.shape[0] and discretized.shape[0] <= uem_t_i.shape[0] + 1:
                discretized = discretized[: uem_t_i.shape[0]]
            else:
                raise RuntimeError(
                    f"REF too short or long for {file['uri']}: {discretized.shape[0]=} vs {uem_t_i.shape[1]=}"
                )

            # Get torch tensor of the predictions for this chunk
            # (num_frames, num_classes) ndarray to (1, num_frames_inuem, num_classes) tensors
            predswffeat_t = (torch.from_numpy(predswffeat)[uem_t_i])[None, ...].float()
            ref_t = torch.from_numpy(discretized)[uem_t_i][None, ...].float()

            # if the segmentations are not in multilabel space, convert to it
            preds_ml_raw = to_multilabel(predswffeat_t)

            # convert to same shape (1, num_frames, num_spk) for both tensors
            ref_ml, preds_ml = match_speaker_count(ref_t, preds_ml_raw)
            if ref_ml.shape[1] == 0:
                empty_time_tensors_count += 1
                continue

            # Obtain `ref_t_padded` of size (1, num_frames, num_speaker_model) padded/cropped the number of spk
            # such that the most talkative speakers are kept,
            ref_t_padded = torch.zeros((1, ref_t.shape[1], n_speakers))
            most_active_idx = torch.argsort(-torch.sum(ref_t[0], dim=0), dim=0)[: min(n_speakers, ref_t.shape[2])]
            ref_t_padded[:, :, torch.arange(0, most_active_idx.shape[0])] = ref_t[:, :, most_active_idx]
            ref_t_padded, _ = permutate(preds_ml_raw, ref_t_padded)

            # Compute the ECE
            if is_powerset:
                ref_ps = ps.to_powerset(ref_t_padded).argmax(dim=-1)
                rawpreds_prob = predswffeat_t.exp()  # logprob to prob
                summedprobs = rawpreds_prob.sum(dim=-1)
                if not torch.isclose(summedprobs, torch.ones_like(summedprobs)).all():
                    raise RuntimeError(f"Probabilities do not sum to 1: {summedprobs}")
                ece.update(rawpreds_prob[0], ref_ps[0])
                file_ece.update(rawpreds_prob[0], ref_ps[0])
            else:
                # bin_pred = (predswffeat_t.flatten() > 0.5)  # bool pos/neg prediction
                # bin_ref = (ref_t_padded.flatten() > 0.5)    # bool pos/neg reference
                # bin_correct = (bin_pred == bin_ref).float() # float correct (1.0) / incorrect (0.0)
                # bin_conf = 0.5 + torch.abs(predswffeat_t.flatten() - 0.5)   # float conf in [0.5, 1.0]

                # commented out is a way to really evaluate only the confidence regardless of the validity of the prediction
                # ece.update(preds_ml_raw.flatten(), ref_t_padded.flatten())
                # file_ece.update(preds_ml_raw.flatten(), ref_t_padded.flatten())

                # prob of all correct labels at the same time
                bin_ref = ref_t_padded > 0.5  # bool (B,F,C) pos/neg reference
                bin_pred = predswffeat_t > 0.5  # bool (B,F,C) pos/neg prediction
                conf_t = 0.5 + torch.abs(predswffeat_t - 0.5)  # conf for each label (B,F,C)
                conf_t2 = torch.prod(conf_t, dim=-1)  # conf for all labels at the same time (prod) (B,F)
                correct = torch.all(
                    bin_ref == bin_pred, dim=-1
                ).float()  # 1 if all labels are correctly predicted (B,F)

                ece.update(conf_t2.flatten(), correct.flatten())
                file_ece.update(conf_t2.flatten(), correct.flatten())

            # Compute DER if there are active speakers in the tensor
            if ref_ml.shape[2] != 0:
                ref_ml, _ = permutate(preds_ml, ref_ml)
                der.update(
                    preds_ml.permute(0, 2, 1),
                    ref_t.permute(0, 2, 1),
                )
                fileder.update(
                    preds_ml.permute(0, 2, 1),
                    ref_t.permute(0, 2, 1),
                )
            else:
                empty_spk_tensors_count += 1

    # log some info
    if empty_time_tensors_count > 0:
        print(f"Skipped {empty_time_tensors_count} tensors with empty time dim (out of {seen_tensors})")
    if empty_spk_tensors_count > 0:
        print(f"Skipped {empty_spk_tensors_count} tensors with empty speaker dim (out of {seen_tensors})")

    global_metrics = {
        "der": der,
        "ece": ece,
    }
    per_file_metrics = {
        uri: {
            "der": uri_to_dermetric[uri],
            "ece": uri_to_ecemetric[uri],
        }
        for uri in uri_to_dermetric
    }
    return global_metrics, per_file_metrics


def get_der_components_dict(
    der: OptimalDiarizationErrorRateTorchMetrics, best_threshold: Optional[int] = None
) -> tuple[dict[str, float], int]:
    if best_threshold is None:
        dercomponents, best_threshold = der.compute_components(return_optimal_threshold_idx=True)
    else:
        localder = (
            der.FalseAlarm[best_threshold] + der.MissedDetection[best_threshold] + der.SpeakerConfusion[best_threshold]
        ) / der.speech_total.item()
        dercomponents = (
            der.FalseAlarm[best_threshold] / der.speech_total.item(),
            der.MissedDetection[best_threshold] / der.speech_total.item(),
            der.SpeakerConfusion[best_threshold] / der.speech_total.item(),
            localder,
        )
    dercomponents_list: list[float] = [v.item() * 100 for v in dercomponents]
    local_fa, local_miss, local_conf, local_der = dercomponents_list
    return {
        "local_fa": local_fa / 100 * der.speech_total.item(),
        "local_fa%": local_fa,
        "local_miss": local_miss / 100 * der.speech_total.item(),
        "local_miss%": local_miss,
        "local_conf": local_conf / 100 * der.speech_total.item(),
        "local_conf%": local_conf,
        "local_der%": local_der,
        "local_total": der.speech_total.item(),
    }, best_threshold


def main():
    parser = argparse.ArgumentParser(description="Specialized der computation script")
    parser.add_argument(
        "rttm_root",
        type=str,
        help="Path to the root folder containing the rttm files (subfolders should be for variants eg oracle)",
    )
    parser.add_argument("pplinfzip", type=str, help="Path to the .pplinf.zip file")
    parser.add_argument("--protocols", type=str, nargs="+", help="pyannote.database protocols to use")
    parser.add_argument("--output-folder", "-o", required=True, type=str, default=None, help="Path to output folder")
    parser.add_argument("--filter", type=str, default=[], nargs="+", help="Filtering criterions eg 'duration>=100'")
    parser.add_argument(
        "--db-yml",
        type=str,
        default="/home/aplaquet/work58/databases/database.yml",
        help="Path to database configuration file",
    )
    parser.add_argument(
        "--localder-n-thresholds",
        type=int,
        default=51,
        help="Number of thresholds to use for local DER computation optimal threshold search",
    )
    parser.add_argument(
        "--store-local-thresholds",
        action="store_true",
        help="Store the local der (thresholds/der) tuples in a subfolder",
    )
    parser.add_argument(
        "--collars",
        type=float,
        nargs="+",
        default=[],
        help="Collar values to use for additional DER computation",
    )

    args = parser.parse_args()

    rttm_root_path = Path(args.rttm_root)

    # maps a subfolder to a list of contained rttm paths
    rttm_filepaths: dict[str, list[str]] = {}
    for rttm in rttm_root_path.rglob("*.rttm"):
        relpath_parent = str(rttm.relative_to(rttm_root_path).parent)
        if relpath_parent not in rttm_filepaths:
            rttm_filepaths[relpath_parent] = []
        rttm_filepaths[relpath_parent].append(str(rttm))

    # maps a subfolder to its URI:annotation mapping
    rttms: dict[str, dict[str, Annotation]] = {}
    for subfolder, rttm_paths in rttm_filepaths.items():
        rttms_subfolder: dict[str, Annotation] = {}
        for rttmpath in rttm_paths:
            loaded_rttms = load_rttm(rttmpath)
            for uri, ann in loaded_rttms.items():
                if uri not in rttms_subfolder:
                    rttms_subfolder[uri] = ann
                else:
                    raise RuntimeError(f"Found duplicate prediction for URI {uri}")
        rttms[subfolder] = rttms_subfolder

    registry.load_database(args.db_yml)

    for protocol_fullname in args.protocols:
        reports: dict[str, pd.DataFrame] = {}

        pname, subset = protocol_fullname_to_name_subset(protocol_fullname, None)
        protocol = registry.get_protocol(pname, preprocessors={})
        files = list(getattr(protocol, subset)())

        # If the user specified a filter, apply it
        if args.filter is not None and len(args.filter) > 0:
            args.filter = [f.strip() for f in args.filter]
            files = [f for f in files if test_match_filter(f, args.filter)]

        # Compute the global DER for all subfolder using RTTMs
        for subfolder, rttms_subfolder in rttms.items():
            if subset is None:
                print(f"!!! Could not extract subset from protocol name {protocol_fullname}, skipping")
                continue

            # Initialize all DER / DER variants (eg collar)
            metrics = {"": DiarizationErrorRate()}
            for collar in args.collars:
                metrics[f"collar_{collar}"] = DiarizationErrorRate(collar=collar)
            # Compute
            for file in tqdm.tqdm(files, desc=f"Computing DER on {protocol_fullname} ({subfolder})"):
                if file["uri"] not in rttms_subfolder:
                    raise RuntimeError(f"No prediction for URI {file['uri']}")
                pred = rttms_subfolder[file["uri"]]
                ref = file["annotation"]
                uem = file["annotated"]
                for metric in metrics.values():
                    metric(ref, pred, uem=uem)

            # Create the report dataframe for this subfolder
            def sanitize_report(r: pd.DataFrame) -> None:
                r.columns = ["".join(c) for c in r.columns.to_flat_index()]

            report_agg: pd.DataFrame = metrics[""].report()
            sanitize_report(report_agg)
            # aggregate other reports to it if
            for metric_id, metric in metrics.items():
                if metric_id == "":
                    continue

                report = metric.report()
                sanitize_report(report)
                print(report.columns)
                if metric_id.startswith("collar_"):
                    report_agg[f"{metric_id}_diarization error rate%"] = report["diarization error rate%"]

            report_agg = report_agg.add_prefix((subfolder + "_" if subfolder != "." else ""))
            reports[subfolder] = report_agg

        # merge all reports into a single one
        report = pd.concat(reports.values(), axis=1)

        # compute the local DER and store it into the report dataframe.
        local_metrics, local_metrics_filewise = compute_segmentation_error_rate(
            protocolfiles=files, inference_zip_path=args.pplinfzip, n_thresholds=args.localder_n_thresholds
        )
        local_der = local_metrics["der"]

        local_overall_der_comps, best_datasetwise_thr_idx = get_der_components_dict(local_der)
        local_overall_der_comps["ece"] = -42
        for k in local_overall_der_comps.keys():
            report[k] = -1

        for uri in report.index:
            if uri == "TOTAL":
                localder_components, _ = get_der_components_dict(local_der, best_datasetwise_thr_idx)
                localder_components["ece"] = local_metrics["ece"].compute().item()
            else:
                localder_components, _ = get_der_components_dict(
                    local_metrics_filewise[uri]["der"], best_datasetwise_thr_idx
                )
                localder_components["ece"] = local_metrics_filewise[uri]["ece"].compute().item()
            for k, v in localder_components.items():
                report.at[uri, k] = v

        # by the way, store the local thresholds / ders if needed
        if args.store_local_thresholds:
            local_thresholds_filepath = Path(args.output_folder) / LOCAL_THRESHOLDS_FOLDER / f"{protocol_fullname}.csv"
            local_thresholds_filepath.parent.mkdir(parents=True, exist_ok=True)

            thresholds, ders_thresholds = get_localder_thresholds_ders(local_der)
            local_thresholds_df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "der": ders_thresholds,
                }
            )
            with open(local_thresholds_filepath, "w") as f:
                f.write(
                    str_to_comment(
                        f"Best threshold: {thresholds[torch.argmin(ders_thresholds)]}\nBest DER: {torch.min(ders_thresholds)}\n"
                    )
                )
                local_thresholds_df.to_csv(f, index=False)

        # Save the report dataframe
        report_path = Path(args.output_folder) / f"{protocol_fullname}.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(str_to_comment(report.to_markdown()))

            report.to_csv(f)


if __name__ == "__main__":
    main()
