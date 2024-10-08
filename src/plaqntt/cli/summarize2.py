# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024
"""Script to summarize the results of multiple experiments in multiple folder in a csv format."""

import argparse
import glob
from pathlib import Path
from typing import TypedDict

import pandas as pd
from plaqntt.utils.argparse_types import strtobool
from natsort import natsorted


def get_total_row(df: pd.DataFrame):
    total_rows = df[df["item"] == "TOTAL"]
    if len(total_rows) != 1:
        raise RuntimeError(f"Expected 1 row with item='TOTAL', found {len(total_rows)}")
    return total_rows.iloc[0]


def get_str_conversion(df: pd.DataFrame, format: str, floatfmt=None):
    if format == "md":
        return df.to_markdown(index=False, tablefmt="github", floatfmt=floatfmt)
    elif format == "csv":
        return df.to_csv(index=False)
    else:
        raise ValueError(f"Unknown format {format}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--evals-folder", type=str, default="evals", help="path to the evals folder")
    parser.add_argument("--subfolder_filter", type=str, default=None, help="filter for subfolder name", nargs="+")
    parser.add_argument(
        "--output-format", "-f", type=str, nargs="+", default=["md"], help="format", choices=["csv", "md"]
    )
    parser.add_argument("--output-directory", "-o", type=str, default=None, help="output directory for the reports")
    parser.add_argument("--sort", "-s", type=str, nargs="+", default=["exname"], help="columns to sort by")
    parser.add_argument("--md-floatfmt", type=str, default=".3f", help="float format for markdown")
    parser.add_argument(
        "--filter-path-include",
        type=str,
        nargs="+",
        default=[],
        help="Paths must include all these strings to be considered",
    )
    parser.add_argument(
        "--filter-path-exclude",
        type=str,
        nargs="+",
        default=[],
        help="Paths must exclude all these strings to be considered",
    )
    parser.add_argument(
        "--columns",
        type=str,
        nargs="+",
        help="columns to include in the output",
        default=[
            "exname",
            "subfolder",
            "segder%",
            "der%",
            "fa%",
            "miss%",
            "conf%",
            "oracle_der%",
            "oracle_conf%",
        ],
    )
    parser.add_argument("--do-merged", action="store_true", default=False, help="do merged csv")
    parser.add_argument("--aggregate-all", type=strtobool, default=False, help="aggregate all reports (advised !)")
    args = parser.parse_args()

    reports: dict[str, dict[tuple[str, str], dict]] = {}

    evals_folder = Path(args.evals_folder)
    if not evals_folder.exists():
        raise FileNotFoundError(f"Folder {evals_folder} does not exist")

    for report in evals_folder.glob("**/*.csv"):
        if args.filter_path_include is not None and not all([s in str(report) for s in args.filter_path_include]):
            continue
        if args.filter_path_exclude is not None and any([s in str(report) for s in args.filter_path_exclude]):
            continue
        if report.parent.name == "local_thresholds":
            continue
        report = str(report)
        if args.do_merged != Path(report).name.startswith("merged_"):
            print(f"Skipping {report} due to the merge option")
            continue
        if "@oracle" in report.lower():
            print(f"Skipping {report} (oracle report)")
            continue

        split_path = report.split("/")
        dataset = split_path[-1]
        exname = split_path[1]
        subfolder = "_".join(split_path[2:-1])
        if args.subfolder_filter is not None and subfolder not in args.subfolder_filter:
            continue

        # print(report)
        if dataset not in reports:
            reports[dataset] = {}

        try:
            reportdf = pd.read_csv(report, comment="#")
        except pd.errors.EmptyDataError:
            print(f"Skipping {report} due to empty data")
            continue
        reportdf.columns = reportdf.columns.str.replace("diarization error rate", "der")
        reportdf.columns = reportdf.columns.str.replace("false alarm", "fa")
        reportdf.columns = reportdf.columns.str.replace("missed detection", "miss")
        reportdf.columns = reportdf.columns.str.replace("confusion", "conf")

        total_row = get_total_row(reportdf)

        stats = {
            "exname": exname,
            "dataset": dataset,
        }
        for variant in ["", "oracle_", "local_"]:
            stats |= {
                f"{variant}der%": total_row[f"{variant}der%"],
                f"{variant}fa%": total_row[f"{variant}fa%"],
                f"{variant}miss%": total_row[f"{variant}miss%"],
                f"{variant}conf%": total_row[f"{variant}conf%"],
            }
        if "ece" in reportdf.columns:
            stats["ece"] = total_row["ece"]

        if subfolder != "":
            stats["subfolder"] = subfolder
        reports[dataset][(exname, subfolder)] = stats

    report_dataframes: dict[str, pd.DataFrame] = {}
    for dataset, dataset_reports in reports.items():
        # print(f"Dataset: {dataset}")
        rows = []
        for (exname, subfolder), stats in dataset_reports.items():
            row = []
            for column in args.columns:
                if column in stats:
                    row.append(stats[column])
                else:
                    print(f"missing column {column} in {exname}, {subfolder}. Available: {stats.keys()}")
                    row.append(None)
            rows.append(row)
        report_df = pd.DataFrame(rows, columns=args.columns)
        report_df.sort_values(by=args.sort, inplace=True)
        report_dataframes[dataset] = report_df

    aggregated_report: pd.DataFrame = None
    for dataset, report_df in report_dataframes.items():
        output_strings: dict[str, str] = {
            format: get_str_conversion(report_df, format, args.md_floatfmt) for format in args.output_format
        }

        if args.output_directory is not None:
            for format, outstr in output_strings.items():
                output_path = Path(args.output_directory) / f"{dataset}.{format}"
                with open(output_path, "w") as f:
                    f.write(outstr)

        print(f"{dataset}\n" + output_strings[args.output_format[0]])
        if args.aggregate_all:
            to_agg = report_df.copy()
            to_agg["dataset"] = dataset
            if aggregated_report is None:
                aggregated_report = to_agg
            else:
                aggregated_report = pd.concat([aggregated_report, to_agg], ignore_index=True)

    if args.aggregate_all:
        print("------------ AGGREGATED --------------")
        aggregated_lines: dict[str, list[str]] = {}  # maps format to lines
        for dataset_idx, dataset in enumerate(aggregated_report["dataset"].unique()):
            dataset_df: pd.DataFrame = aggregated_report[aggregated_report["dataset"] == dataset].copy()
            dataset_df = dataset_df[[col for col in dataset_df.columns if col in args.columns]]
            dataset_df.sort_values(by=args.sort, inplace=True)
            if dataset_idx != 0:
                dataset_df = dataset_df.drop(["exname", "subfolder"], axis=1)

            output_strings: dict[str, str] = {
                format: get_str_conversion(dataset_df, format, args.md_floatfmt) for format in args.output_format
            }
            for format, outstr in output_strings.items():
                lines = outstr.split("\n")
                if format not in aggregated_lines:
                    aggregated_lines[format] = lines
                else:
                    for i, (l1, l2) in enumerate(zip(aggregated_lines[format], lines)):
                        aggregated_lines[format][i] = l1 + "," + l2
        # here the -3 = -2 (exname,subfolder) -1 (dataset)
        print(
            ",,"
            + ",".join([dataset + "," * (len(args.columns) - 3) for dataset in aggregated_report["dataset"].unique()])
        )
        print("\n".join(aggregated_lines[args.output_format[0]]))


if __name__ == "__main__":
    main()
