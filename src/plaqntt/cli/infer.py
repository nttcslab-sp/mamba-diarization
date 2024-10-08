# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024
"""Does the tuning and inference.
The tuning generates a .yaml file with the best hyperparameters.
The inference generates the RTTMs and a zip containing embeddings and raw segmentation outputs.

To get metrics, please use the cli command `pntt-der` from `plaqntt` python package."""

import argparse
import itertools
import math
import os
import re
import socket
from typing import Optional
import warnings
import zipfile
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from plaqntt.utils.argparse_types import strtobool
from plaqntt.utils.pyannote_database import protocol_fullname_to_name_subset
from pyannote.audio.pipelines.clustering import OracleClustering

# from pyannote.audio.pipelines.speaker_diarization import (
#     SpeakerDiarization as SpeakerDiarizationPipeline,
# )
from plaqntt.pyannote_audio.pipeline.speaker_diarization import (
    SpeakerDiarizationV2 as SpeakerDiarizationPipeline,
)
from pyannote.audio.torchmetrics.audio import (
    # DiarizationErrorRate as DiarizationErrorRateTorchMetrics,
    OptimalDiarizationErrorRate as OptimalDiarizationErrorRateTorchMetrics,
)
from pyannote.audio.core.task import Specifications
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.audio import Model
from pyannote.database import FileFinder, registry, ProtocolFile
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.pipeline import Optimizer
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
from omegaconf import OmegaConf
from powerset_calibration.utils.pyannote_core import timeline_subtimeline


# old parameters
DEFAULT_PARAMS = {
    5.0: {
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 12,
            "threshold": 0.7045654963945799,
        },
        "segmentation": {
            "min_duration_off": 0.0,
            "threshold": 0.5,
        },
    }
}

# new better warm start
# DEFAULT_PARAMS = {
#     5.0 : {
#         "clustering": {
#             "method": "centroid",
#             "min_cluster_size": 9,
#             "threshold": 0.6920270315817946,
#         },
#         "segmentation": {
#             "min_duration_off": 0.0,
#             "threshold": 0.5,
#         },
#     },
#     10.0: {
#         "clustering": {
#             "method": "centroid",
#             "min_cluster_size": 7,
#             "threshold": 0.6826890519795143,
#         },
#         "segmentation": {
#             "min_duration_off": 0.0,
#             "threshold": 0.5,
#         },
#     },
#     30.0: {
#         "clustering": {
#             "method": "centroid",
#             "min_cluster_size": 6,
#             "threshold": 0.6887997720379344,
#         },
#         "segmentation": {
#             "min_duration_off": 0.0,
#             "threshold": 0.5,
#         },
#     },
#     50.0: {
#         "clustering": {
#             "method": "centroid",
#             "min_cluster_size": 6,
#             "threshold": 0.6931483859517644,
#         },
#         "segmentation": {
#             "min_duration_off": 0.0,
#             "threshold": 0.5,
#         },
#     },
# }


def update_dict(d, dotpath, value, cast=False):
    keys = dotpath.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if keys[-1] in d and cast:
        value = type(d[keys[-1]])(value)
    d[keys[-1]] = value


def get_default_params(specifications: Specifications, clustering: str) -> dict:
    closest_duration = min(DEFAULT_PARAMS.keys(), key=lambda x: abs(x - specifications.duration))
    params = deepcopy(DEFAULT_PARAMS[closest_duration])
    print(f"Using default params for duration {closest_duration}")
    if specifications.powerset:
        params["segmentation"].pop("threshold")
    if clustering == "AgglomerativeClusteringV2":
        params["clustering"] |= {
            "mcs_factor": 0.1,
        }
    return params


def str_to_comment(s: str, comment_prefix: str = "#\t") -> str:
    return "\n".join([f"{comment_prefix}{l}" for l in s.split("\n")]) + "\n"


def sliding_window_to_dict(sw: SlidingWindow) -> dict[str, float]:
    return {
        "start": sw.start,
        "duration": sw.duration,
        "step": sw.step,
        "end": sw.end,
    }


def specifications_to_dict(specs: Specifications) -> dict:
    return {
        "classes": specs.classes,
        "duration": specs.duration,
        "min_duration": specs.min_duration,
        "num_powerset_classes": specs.num_powerset_classes if specs.powerset else -1,
        "permutation_invariant": specs.permutation_invariant,
        "powerset": specs.powerset,
        "powerset_max_classes": specs.powerset_max_classes,
        "warm_up": list(specs.warm_up or []),
        "problem": str(specs.problem),
        "resolution": str(specs.resolution),
    }


def find_ckpt_exname(ckpt_path: str) -> str:
    split = ckpt_path.split("/")
    split.reverse()
    for s in split:
        if re.match(r"w\d+.+", s) is not None:
            return s
    raise ValueError(f"couldnt find exname in {ckpt_path}")


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


def load_previous_evaluated_inference(
    rttm_folder, do_oracle: bool = True
) -> tuple[dict[str, Annotation], dict[str, Annotation]]:
    predictions: dict[str, Annotation] = {}
    predictions_oracle: dict[str, Annotation] = {}
    # Load existing data
    if rttm_folder is not None:
        # load regular rttms
        for rttmfile in Path(rttm_folder).glob("*.rttm"):
            predictions.update(load_rttm(rttmfile))
        # load oracle rttms
        if do_oracle:
            for rttmfile in (Path(rttm_folder) / "oracle").glob("*.rttm"):
                predictions_oracle.update(load_rttm(rttmfile))
            print(f"Loaded {len(predictions)} already computed RTTMs (and {len(predictions_oracle)} oracle ones)")
            # Remove RTTMs that are not duplicated between oracle and regular
            not_in_common = set(predictions.keys()) ^ set(predictions_oracle.keys())
            if len(not_in_common) != 0:
                print(f"DIFFERENT AMOUT OF ORACLE RTTMs, DISCARDING THOSE NOT IN COMMON ({not_in_common})")
                for uri in not_in_common:
                    if uri in predictions:
                        del predictions[uri]
                    if uri in predictions_oracle:
                        del predictions_oracle[uri]

            oracle_regular_rttm_differ = len(predictions) == 0 and len(predictions_oracle) == 0
            for uri in list(predictions.keys()):
                if predictions[uri] != predictions_oracle[uri]:
                    oracle_regular_rttm_differ = True
                    break
            if not oracle_regular_rttm_differ:
                raise RuntimeError(f"Same predictions for oracle and normal RTTM in all URIS ???")
    return predictions, predictions_oracle


def save_to_pplinf_zip(path: Path, uri: str, file_npz: dict, yml_meta: dict):
    with zipfile.ZipFile(path, "a") as zipf:
        zipfolder = uri
        with zipf.open(zipfolder + "/ndarrays.npz", "w", force_zip64=True) as f:
            np.savez_compressed(f, **file_npz)
        with zipf.open(zipfolder + "/meta.yaml", "w") as f:
            f.write(yaml.dump(data=yml_meta).encode())


def apply_on_files(
    pipeline,
    files,
    exname,
    protocolname,
    output_folder,
    metadata=None,
    superset_protocolname=None,
    do_oracle=True,
    force_recompute=False,
):
    og_output_folder = output_folder
    output_folder = Path(og_output_folder) / protocolname
    output_folder.mkdir(parents=True, exist_ok=True)
    if pipeline is not None:
        pipeline.training = True

    if superset_protocolname is None:
        loaded_folder: Path = output_folder
        loaded_protocol = protocolname
    else:
        print(f"Going to use pre-computed data from superset {superset_protocolname}")
        loaded_folder: Path = Path(og_output_folder) / superset_protocolname
        loaded_protocol: str = superset_protocolname

    if force_recompute:
        predictions: dict[str, Annotation] = {}
        predictions_oracle: dict[str, Annotation] = {}
    else:
        predictions, predictions_oracle = load_previous_evaluated_inference(loaded_folder, do_oracle)

    inference_zip_path = Path(loaded_folder).parent / f"{loaded_protocol}.pplinf.zip"
    for file in (pbar := tqdm.tqdm(files)):
        uri = file["uri"]

        # Load previously computed segmentation (normal & oracle)
        if uri in predictions:
            pred: Annotation = predictions[uri]
            pred_oracle: Annotation = predictions_oracle.get(uri, None)
            pbar.set_postfix_str(f"Using cached {file['uri']}")
        # Compute segmentation (normal & oracle)
        elif superset_protocolname is None:
            # apply finetuned pipeline with normal embeddings
            pbar.set_postfix_str(f"Computing {file['uri']} ({file['database']})")
            pred: Annotation = pipeline(file)
            predictions[uri] = pred

            # apply finetuned pipeline with oracle clustering and save its RTTM too
            # HACK-EY, be aware it might break one day
            if do_oracle:
                pbar.set_postfix_str(f"Computing {file['uri']} (oracle version)")
                pipeline.__clustering_old = pipeline.clustering
                pipeline.__klustering_old = pipeline.klustering
                pipeline.klustering = "OracleClustering"
                pipeline.clustering = OracleClustering()
                pred_oracle, _ = pipeline(file, return_embeddings=True)
                pipeline.clustering = pipeline.__clustering_old
                pipeline.klustering = pipeline.__klustering_old

                predictions_oracle[uri] = pred_oracle

            # get local UEM
            uem = file["annotated"]
            num_chunks, num_frames, num_classes = file["training_cache/segmentation"].data.shape
            uem_t: np.ndarray = np.zeros((num_chunks, num_frames, 1), dtype=bool)
            for i in range(num_chunks):
                extent = file["training_cache/segmentation"].sliding_window[i]
                uem_t_i = (
                    uem.support()
                    .to_annotation()
                    .rename_labels(generator=itertools.cycle(["uem"]))
                    .discretize(
                        support=extent,
                        resolution=pipeline._segmentation.model.receptive_field,
                        duration=pipeline._segmentation.duration,
                        labels=["uem"],
                    )
                    .data
                    > 0.5
                )
                if uem_t_i.shape[0] >= uem_t.shape[1] and uem_t_i.shape[0] <= uem_t.shape[1] + 1:
                    uem_t[i] = uem_t_i[: uem_t.shape[1]]
                else:
                    raise RuntimeError(
                        f"UEM too short or long for {file['uri']}: {uem_t_i.shape[0]=} vs {uem_t.shape[1]=}"
                    )

            # Save inference to zip file
            save_to_pplinf_zip(
                path=inference_zip_path,
                uri=uri,
                file_npz={
                    "segmentations": file["training_cache/segmentation"].data,
                    "embeddings": file["training_cache/embeddings"]["embeddings"],
                    "segmentation_uem_t": uem_t,
                },
                yml_meta={
                    "uri": uri,
                    "exname": exname,
                    "protocol": protocolname,
                    "savedate": str(pd.Timestamp.now()),
                    "labels": file["training_cache/segmentation"].labels,
                    "segmentation_sw": {
                        **sliding_window_to_dict(file["training_cache/segmentation"].sliding_window),
                    },
                    "seg_model": {
                        "receptive_field": sliding_window_to_dict(pipeline._segmentation.model.receptive_field),
                        "specifications": specifications_to_dict(pipeline._segmentation.model.specifications),
                    },
                    "pipeline_params": {
                        "embedding": pipeline.embedding,
                        "clustering": pipeline.klustering,
                        "embedding_batch_size": pipeline.embedding_batch_size,
                        "segmentation_batch_size": pipeline.segmentation_batch_size,
                        "embedding_exclude_overlap": pipeline.embedding_exclude_overlap,
                        "segmentation_step": pipeline.segmentation_step,
                    },
                },
            )
            # save RTTMs
            file_rrtm_outpath = Path(output_folder) / f"{file['uri']}.rttm"
            with open(file_rrtm_outpath, "w") as f:
                pred.write_rttm(f)

            if do_oracle:
                file_oraclerrtm_outpath = Path(output_folder) / "oracle" / f"{file['uri']}.rttm"
                file_oraclerrtm_outpath.parent.mkdir(parents=True, exist_ok=True)
                with open(file_oraclerrtm_outpath, "w") as f:
                    pred_oracle.write_rttm(f)
        else:
            needed_uris = ",".join([f2["uri"] for f2 in files])
            available_uris = ",".join(list(predictions.keys()))
            raise RuntimeError(
                f"Superset protocol doesnt include all URIs needed ! Needed: {needed_uris}\nAvailable:{available_uris}"
            )


def interleave_anylengths(*iterables):
    from itertools import chain, zip_longest

    return chain.from_iterable((e for e in t if e is not None) for t in zip_longest(*iterables))


def tune_or_load_best_params(
    pipeline,
    protocol,
    default_params: dict,
    best_params_file: Path,
    patience=50,
    max_iterations=999999999,
    force_tuning: bool = False,
    opt_sampler="TPESampler",
    opt_pruner=None,
    save_tmp_best: bool = True,
    min_file_duration: float = 0.0,
    max_set_duration: float = math.inf,
    pruner_warmup_criterion: str = "domain_representation",
    pruner_warmup_value: Optional[float] = None,
    subsets: list[str] = ["development"],
    optuna_db: Optional[str] = None,
) -> dict:
    if not force_tuning and best_params_file.exists():
        return yaml.load(best_params_file.open("r"), Loader=yaml.FullLoader)

    SEED = 0
    rnd = random.Random(SEED)

    # Instantiate pipeline parameters
    # pipeline.instantiate(default_params)
    frozen = deepcopy(default_params)
    del frozen["clustering"]["min_cluster_size"]
    del frozen["clustering"]["threshold"]
    pipeline.freeze(frozen)

    # prepare files
    files_all = []
    for subset in subsets:
        files_all += list(getattr(protocol, subset)())
    # probably overkill but lets make files unique just incase train and dev share files
    files_all = list({f["uri"]: f for f in files_all}.values())

    # shuffle and filter files
    rnd.shuffle(files_all)  # probably useless because we interleave but let's do it anyway
    db_file_mapping: dict[str, list] = {}
    for f in files_all:
        if f["annotated"].duration() >= min_file_duration:
            if f["database"] not in db_file_mapping:
                db_file_mapping[f["database"]] = []
            db_file_mapping[f["database"]].append(f)
        else:
            continue
    files = list(interleave_anylengths(*db_file_mapping.values()))  # files = interleaved files from each db
    print(
        f"Selected {len(files)}/{len(files_all)} (from {len(db_file_mapping)} DBs) files for tuning ({sum([f['annotated'].duration() for f in files])/60:.1f} / {sum([f['annotated'].duration() for f in files_all])/60:.1f} min)"
    )

    # cut the files so that it fits the max duration
    total_duration_in_files = sum([f["annotated"].duration() for f in files])
    if total_duration_in_files > max_set_duration:
        per_file_duration = max_set_duration / len(files)
        if per_file_duration < min_file_duration:
            reduced_file_count = int(max_set_duration / min_file_duration)
            print(
                f"!!) Cut files to match target duration of {max_set_duration/60:.2f} min. File count reduced from {len(files)} to {reduced_file_count}"
            )
            files = files[-reduced_file_count:]
            per_file_duration = max_set_duration / len(files)
        time_debt = 0.0

        edited_files: dict[str, ProtocolFile] = {}
        duration_sorted_files = sorted(files, key=lambda f: f["annotated"].duration(), reverse=False)  # longest first
        target_durations = []
        # compute the new durations for each file
        for f in duration_sorted_files:
            f_duration = f["annotated"].duration()
            f_duration_new = min(f_duration, per_file_duration + time_debt)
            time_debt += per_file_duration - f_duration_new
            target_durations.append(f_duration_new)
        if time_debt > 0.0:
            raise RuntimeError(f"Time debt of {time_debt} left after cutting files, something is wrong")
        # cut the timelines and create the new files
        for f, new_duration in zip(duration_sorted_files, target_durations):
            new_f = deepcopy(f)
            tl: Timeline = new_f["annotated"]
            new_duration_ratio = new_duration / f["annotated"].duration()
            start_pos = rnd.uniform(0.0, 1.0 - new_duration_ratio)
            new_f["annotated"] = timeline_subtimeline(tl, start_pos, start_pos + new_duration_ratio, min_duration=0.0)
            edited_files[f["uri"]] = new_f
        # replace the original 'files' variable, keep the original shuffled/interleaved order
        files = [edited_files[f["uri"]] for f in files]
        print(
            f"Cut files to match target duration of {max_set_duration/60:.2f} min : {sum([f['annotated'].duration() for f in files])/60:.1f}"
        )
        print(f'  Old file durations (min): {[f["annotated"].duration()/60 for f in duration_sorted_files]}')
        print(f'  New file durations (min): {[f["annotated"].duration()/60 for f in edited_files.values()]}')

    if sum([f["annotated"].duration() for f in files]) > max_set_duration * 1.001:
        raise RuntimeError(f"Failed to cut files to fit the max duration {max_set_duration}...")

    # Prepare the sampler
    startup_trials = 10
    if opt_sampler == "TPESampler":
        opt_sampler = TPESampler(
            multivariate=True,
            warn_independent_sampling=True,
            seed=SEED,
            n_startup_trials=startup_trials,
        )
    if opt_pruner == "MedianPruner":
        if pruner_warmup_criterion == "domain_representation":
            pruner_warmup_value = pruner_warmup_value if pruner_warmup_value is not None else 2.0
            warmup_steps = round(pruner_warmup_value * (len(db_file_mapping)))
        elif pruner_warmup_criterion == "fixed":
            pruner_warmup_value = pruner_warmup_value if pruner_warmup_value is not None else 1.0
            warmup_steps = round(pruner_warmup_value)
        elif pruner_warmup_criterion == "file_ratio":
            pruner_warmup_value = pruner_warmup_value if pruner_warmup_value is not None else 0.2
            warmup_steps = round(pruner_warmup_value * len(files))
        elif pruner_warmup_criterion == "duration":
            if pruner_warmup_value is None:
                raise ValueError("pruner_warmup_value must be set for criterion 'duration'")
            tmp_total_duration = 0
            file_idx_over_duration = 0
            for i, f in enumerate(files):
                tmp_total_duration += f["annotated"].duration()
                if tmp_total_duration >= pruner_warmup_value:
                    file_idx_over_duration = i
                    break
            warmup_steps = file_idx_over_duration
        else:
            raise ValueError(f"Unknown pruner_warmup_criterion: {pruner_warmup_criterion}")
        warmup_steps = min(len(files), warmup_steps)
        opt_pruner = MedianPruner(n_warmup_steps=warmup_steps, n_startup_trials=startup_trials // 2)
        print(
            f"Using MedianPruner with {warmup_steps} warmup steps (criterion={pruner_warmup_criterion}, value={pruner_warmup_value})"
        )
    elif opt_pruner == "Hyperband":
        # untested
        opt_pruner = HyperbandPruner(
            min_resource=len(db_file_mapping),
            max_resource=len(files),
            reduction_factor=2,
        )

    # Create the optimizer
    optimizer = Optimizer(
        pipeline=pipeline,
        sampler=opt_sampler,
        pruner=opt_pruner,
        db=optuna_db,
        seed=SEED,
    )
    iterations = optimizer.tune_iter(
        files,
        warm_start=default_params,
        show_progress=False,
    )
    print(iterations, type(iterations))

    last_best_savepath = None
    last_best = (999, None)  # best (iteration, loss) tuple
    with tqdm.tqdm(total=min(patience, max_iterations), desc="Tuning ...") as pbar:
        for i, iteration in enumerate(iterations):
            # progress bar
            pbar_postfix = {
                "it": i,
                "loss": f'{iteration["loss"]:.6f}',
            }
            # If this iteration is better than what we had
            if i == 0 or iteration["loss"] < last_best[1]:
                last_best = (i, iteration["loss"])
                # print(
                #     f"[{i}] New best: {iteration['params']['clustering']['threshold']} / {iteration['params']['clustering']['min_cluster_size']} -> {iteration['loss']:.3f}"
                # )
                pbar.reset(total=min(patience, max_iterations - i))
                if last_best_savepath is not None:
                    last_best_savepath.unlink(missing_ok=True)
                if save_tmp_best:
                    last_best_savepath = best_params_file.parent / f"{best_params_file.stem}@{i}.yaml"
                    last_best_savepath.parent.mkdir(parents=True, exist_ok=True)
                    yaml.dump(iteration["params"], last_best_savepath.open("w"))
                # display it in the progress bar
                pbar_postfix |= {
                    "clustering.threshold": iteration["params"]["clustering"]["threshold"],
                    "clustering.min_cluster_size": iteration["params"]["clustering"]["min_cluster_size"],
                }
            # Progress bar update
            pbar.set_postfix(pbar_postfix, refresh=True)
            pbar.update(1)
            # stop if needed
            if i - last_best[0] > patience or i > max_iterations:
                print(f"Stopping at iteration {i} (patience={i-last_best[0]}/{patience}) (max it={max_iterations})")
                break
    print(f"Best clustering : {optimizer.best_params}")
    print(f"Took {i} iterations. Saving to {best_params_file}")

    best_params_file.parent.mkdir(parents=True, exist_ok=True)
    yaml.dump(optimizer.best_params, best_params_file.open("w"))

    if last_best_savepath is not None:
        last_best_savepath.unlink(missing_ok=True)

    return optimizer.best_params


def main():

    parser = argparse.ArgumentParser()

    def add_arg(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    add_arg(
        "--phases",
        type=str,
        default=["eval"],
        nargs="+",
        help="Phases to run",
        choices=["tune", "eval"],
    )
    add_arg(
        "--seg-ckpts",
        type=str,
        default=None,
        help="path to the segmentation ckpts",
        required=True,
        nargs="+",
    )
    add_arg(
        "--protocols",
        type=str,
        default=None,
        help="protocols to compute stats on",
        required=True,
        nargs="+",
    )
    add_arg(
        "--emb-model",
        type=str,
        default="pyannote/wespeaker-voxceleb-resnet34-LM",
        help="embedding model",
    )
    add_arg(
        "--clustering",
        type=str,
        default="AgglomerativeClustering",
        help="clustering method",
    )
    add_arg(
        "--output-folder",
        type=str,
        default="evals",
        help="evaluation output folder",
    )
    add_arg(
        "--subfolder",
        type=str,
        default="",
        help="subfolder (hierarchy) in the output folder (ie will look like outfolder/exname/subfolder/)",
    )
    add_arg("--batch-size-seg", type=int, default=32, help="batch size")
    add_arg("--batch-size-emb", type=int, default=32, help="batch size")
    add_arg(
        "--override-segmentation-length",
        type=float,
        default=None,
        help="Override the length of the segmentation model. Leave to None to use the model's default",
    )
    add_arg(
        "--seg-step",
        type=float,
        default=0.5,
        help="segmentation step ratio in ]0.0, 1.0]",
    )
    add_arg(
        "--embedding-exclude-overlap",
        type=strtobool,
        default=True,
        help="exclude overlap in embeddings",
    )
    add_arg(
        "--hparams",
        type=str,
        default=None,
        help="hyperparameters file. If left empty, will tune. If set to DEFAULT, will use default hyperparameters.",
    )
    add_arg(
        "--hparams-override",
        type=str,
        default=[],
        nargs="+",
        help="tuples of dothpaths=value to override in hyperparameters eg clustering.threshold=0.55",
    )
    add_arg(
        "--tuning-subsets",
        type=str,
        nargs="+",
        default=["development"],
        help="subsets to use for tuning",
        choices=[
            "development",
            "train",
        ],  # no 'test' because ... let's not trust the user (me)
    )
    add_arg(
        "--tuning-min-file-duration",
        type=float,
        default=0.0,
        help="Files in the dev set under this duration will be ignored for tuning (useful to skip very short files that wouldn't test the clustering much or at all)",
    )
    add_arg(
        "--tuning-max-set-duration",
        type=float,
        default=math.inf,
        help="Max total duration of the tuning set size. File length will be cut to fit this duration",
    )
    add_arg("--tuning-patience", type=int, default=50, help="tuning patience")
    add_arg("--tuning-max-iterations", type=int, default=999999999, help="tuning iterations")
    add_arg(
        "--tuning-step",
        type=float,
        default=0.5,
        help="segmentation step ratio for tuning, in ]0.0, 1.0]",
    )
    add_arg(
        "--tuning-sampler",
        type=str,
        default="TPESampler",
        help="optuna sampler",
        choices=["TPESampler", "RandomSampler"],
    )
    add_arg(
        "--tuning-pruner",
        type=str,
        default="MedianPruner",
        help="optuna pruner",
        choices=["MedianPruner", "SuccessiveHalvingPruner"],
    )
    add_arg(
        "--tuning-pruner-warmup-criterion",
        type=str,
        default="domain_representation",
        help="How to decide how many warmup steps to use for the pruner",
        choices=["domain_representation", "fixed", "file_ratio", "duration"],
    )
    add_arg(
        "--tuning-pruner-warmup-value",
        type=float,
        default=None,
        help="Meaning of this value depends on the criterion",
    )
    add_arg(
        "--use-superset-data",
        type=str,
        default=None,
        help="name of protocol in same output folder to reuse rttms/predictions from previous experiments. Will force enable --no-pipeline",
    )
    add_arg(
        "--no-pipeline",
        type=strtobool,
        default=False,
        help="Disable the pipeline, at your own risks !",
    )
    add_arg(
        "--no-oracle",
        type=strtobool,
        default=False,
        help="Disable oracle clustering second pass",
    )
    add_arg(
        "--force-recompute",
        type=strtobool,
        default=False,
        help="Ignore precomputed RTTMS and overwrite them",
    )
    add_arg(
        "--multi-output-mode",
        type=str,
        default="error",
        help="How to merge the multiple outputs of the segmentation model",
        choices=["error", "mean", "one"],
    )
    add_arg(
        "--multi-output-outputs",
        type=str,
        nargs="+",
        default=["ml"],
        choices=["ml", "ps"],
    )

    args = parser.parse_args()
    print(args)

    print(f"--------------------")
    print(f"-- hostname: {socket.gethostname()}")
    print(f"-- PID: {os.getpid()}")
    print(f"--------------------")

    if args.use_superset_data:
        args.no_pipeline = True

    registry.load_database("/home/aplaquet/work58/databases/database.yml")

    for seg_model, protocol_name in itertools.product(args.seg_ckpts, args.protocols):
        exname = find_ckpt_exname(seg_model)
        print(f"---- {seg_model} on {protocol_name} (detected exname: {exname}) ----")

        output_folder = Path(args.output_folder) / exname
        if args.subfolder is not None and args.subfolder.strip() != "":
            output_folder = output_folder / args.subfolder
        print(f"Experiment folder: {output_folder}")

        protocol_name, subset = protocol_fullname_to_name_subset(protocol_name, "test")
        protocol = registry.get_protocol(protocol_name, preprocessors={"audio": FileFinder()})

        def ppl_maker(tuning: bool = False):
            _seg = Model.from_pretrained(seg_model)
            if args.override_segmentation_length is not None:
                _seg.specifications.duration = args.override_segmentation_length
            # terrible code
            _seg.default_forward_mode = args.multi_output_mode
            _seg.default_forward_outputs = args.multi_output_outputs

            _ppl = SpeakerDiarizationPipeline(
                segmentation=_seg,
                embedding=args.emb_model,
                clustering=args.clustering,
                embedding_batch_size=args.batch_size_emb,
                segmentation_batch_size=args.batch_size_seg,
                embedding_exclude_overlap=args.embedding_exclude_overlap,
                segmentation_step=args.tuning_step if tuning else args.seg_step,
                use_uem=tuning,
            )
            _ppl.to(torch.device("cuda"))
            return _ppl

        # --- get best pipeline hyperparams (tuning or loading from file)
        ppl = ppl_maker(True)
        best_params = get_default_params(ppl._segmentation.model.specifications, args.clustering)
        tuned_params_file = output_folder / "hparams" / f"{protocol_name}.yaml"
        if args.hparams == "DEFAULT":
            print("Using default hyperparameters")
        elif args.hparams == "TUNED":
            print(f"Using tuned hyperparameters at {tuned_params_file}")
            best_params = yaml.load(tuned_params_file.open("r"), Loader=yaml.FullLoader)
        elif args.hparams is not None:
            print(f"Loading hyperparameters from {args.hparams}")
            best_params = yaml.load(open(args.hparams, "r"), Loader=yaml.FullLoader)
        if "tune" in args.phases:
            print("Tuning hyperparameters")
            best_params: dict = tune_or_load_best_params(
                ppl,
                protocol,
                default_params=best_params,
                best_params_file=tuned_params_file,
                force_tuning=False,
                opt_sampler=args.tuning_sampler,
                opt_pruner=args.tuning_pruner,
                patience=args.tuning_patience,
                max_iterations=args.tuning_max_iterations,
                min_file_duration=args.tuning_min_file_duration,
                max_set_duration=args.tuning_max_set_duration,
                pruner_warmup_criterion=args.tuning_pruner_warmup_criterion,
                pruner_warmup_value=args.tuning_pruner_warmup_value,
                subsets=args.tuning_subsets,
            )
        print(f"Found best hyperparameters: {best_params}")

        # --- compute the RTTMs of the pipeline
        if "eval" in args.phases:
            if not args.no_pipeline:
                ppl = ppl_maker(False)
                if (
                    "threshold" not in best_params["segmentation"]
                    and not ppl._segmentation.model.specifications.powerset
                ):
                    best_params["segmentation"]["threshold"] = 0.5

                for override in args.hparams_override:
                    dotpath, value = override.split("=", 1)
                    update_dict(best_params, dotpath, value, cast=True)

                ppl.instantiate(best_params)
                print(f"Params: {ppl.parameters()}\nParams instantiated: {ppl.parameters(instantiated=True)}")
            else:
                ppl = None
                print("Skipped instantiating the pipeline")

            apply_on_files(
                ppl,
                list(getattr(protocol, subset)()),
                exname=exname,
                protocolname=protocol_name,
                output_folder=output_folder,
                metadata=best_params,
                superset_protocolname=args.use_superset_data,
                do_oracle=not args.no_oracle,
                force_recompute=args.force_recompute,
            )
            print(f"---- {seg_model} on {protocol_name} done ----")
    print("All done !")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()
