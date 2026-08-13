# Copyright 2026 OPPO. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import pandas as pd
import argparse
import sys
import numpy as np
import re

# ================= 配置区域 =================

# 配置每个Benchmark对应的文件后缀、读取方式和需要提取的列
BENCHMARK_CONFIG = {
    "AMBER": {
        "suffix": "_AMBER_score.csv",
        "type": "csv",
        "main_metric": "Avg ACC",
        "columns_map": {
            "Attribute": "Attr", 
            "Existence": "Exist", 
            "Relation": "Rel", 
            "Avg ACC": "Avg"
        }
    },
    "HallusionBench": {
        "suffix": "_HallusionBench_score.csv",
        "type": "csv",
        "main_metric": "avg",
        "filter_key": "split",
        "filter_val": "Overall",
        "columns_map": {
            "avg": "Avg",
            "aAcc": "aAcc",
            "fAcc": "fAcc",
            "qAcc": "qAcc"
        }
    },
    "POPE": {
        "suffix": "_POPE_score.csv",
        "type": "csv",
        "main_metric": "acc",
        "filter_key": "split",
        "filter_val": "Overall",
        "columns_map": {
            "acc": "Acc",
            "F1": "F1" 
        }
    },
    "CRPE_RELATION": {
        "suffix": "_CRPE_RELATION_score.json",
        "type": "json",
        "main_metric": "total",
        "columns_map": {"total": "Score"}
    },
    "CRPE_EXIST": {
        "suffix": "_CRPE_EXIST_score.json",
        "type": "json",
        "main_metric": "total",
        "columns_map": {"total": "Score"}
    },
    "R-Bench-Dis": {
        "suffix": "_R-Bench-Dis_acc.csv",
        "type": "csv",
        "main_metric": "Overall",
        "filter_key": "split",
        "filter_val": "dis",
        "columns_map": {"Overall": "Score"}
    },
    "R-Bench-Ref": {
        "suffix": "_R-Bench-Ref_acc.csv",
        "type": "csv",
        "main_metric": "Overall",
        "filter_key": "split",
        "filter_val": "ref",
        "columns_map": {"Overall": "Score"}
    },
    "BLINK": {
        "suffix": "_BLINK_acc.csv",
        "type": "csv",
        "main_metric": "Overall",
        "filter_key": "split",
        "filter_val": "none",
        "columns_map": {"Overall": "Score"}
    },
    "MMVP": {
        "suffix": "_MMVP_acc.csv",
        "type": "csv",
        "main_metric": "Overall",
        "filter_key": "split",
        "filter_val": "none",
        "columns_map": {"Overall": "Score"}
    },
    "MMStar": {
        "suffix": "_MMStar_acc.csv",
        "type": "csv",
        "main_metric": "Overall",
        "filter_key": "split",
        "filter_val": "none",
        "columns_map": {"Overall": "Score"}
    }
}

BENCHMARK_GROUPS = {
    "CRPE": ["CRPE_RELATION", "CRPE_EXIST"],
    "R-Bench": ["R-Bench-Dis", "R-Bench-Ref"],
}

PAPER_TABLE2_COLUMNS = [
    ("Meta", "Method"),
    ("Meta", "Role"),
    ("Meta", "Base Model"),
    ("Overall", "Score"),
    ("HallusionBench", "aAcc"),
    ("HallusionBench", "fAcc"),
    ("HallusionBench", "qAcc"),
    ("AMBER", "Attr"),
    ("AMBER", "Exist"),
    ("AMBER", "Rel"),
    ("CRPE", "Exist"),
    ("CRPE", "Rel"),
    ("R-Bench", "Dis"),
    ("R-Bench", "Ref"),
    ("BLINK", "Score"),
]

# ================= 工具函数 =================

def format_score(val):
    """将分数统一转换为100分制，保留2位小数"""
    if val is None:
        return None
    try:
        f_val = float(val)
        if f_val <= 1.0 and f_val != 0.0: 
             f_val *= 100
        return round(f_val, 2)
    except ValueError:
        return None

def find_file(model_dir, suffix):
    search_path = os.path.join(model_dir, f"*{suffix}")
    files = glob.glob(search_path)
    return files[0] if files else None


def strip_timestamp_suffix(name):
    return re.sub(r"_20\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$", "", name)


def parse_model_identity(model_dir, label=None):
    if os.path.basename(os.path.normpath(model_dir)).startswith("T202"):
        eval_dir = os.path.dirname(os.path.normpath(model_dir))
    else:
        eval_dir = model_dir

    eval_dir, base_model = os.path.split(os.path.normpath(eval_dir))
    eval_dir, _ = os.path.split(eval_dir)
    train_dir, method_name = os.path.split(eval_dir)
    train_type = os.path.basename(train_dir)

    if label is not None:
        return base_model or "Unknown", label

    method_name = strip_timestamp_suffix(method_name)
    if not method_name:
        method_name = "Default"
    else:
        method_name = f"({train_type}) {method_name}"

    return base_model or "Unknown", method_name


def group_score(bench_main_scores, members):
    scores = []
    for member in members:
        score = bench_main_scores.get(member)
        if score is None:
            return None
        scores.append(score)
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def collect_benchmark_data(model_dir, requested_benchmarks):
    bench_main_scores = {}
    bench_results_map = {}
    missing_any_benchmark = False

    for bench in requested_benchmarks:
        metrics, main_score = extract_metrics(model_dir, bench)
        if metrics is None:
            missing_any_benchmark = True
            continue

        bench_main_scores[bench] = main_score
        bench_results_map[bench] = metrics

    return bench_main_scores, bench_results_map, missing_any_benchmark


def compute_global_score(bench_main_scores, requested_benchmarks):
    final_scores_for_avg = []
    processed_benchmarks = set()

    for _, sub_tasks in BENCHMARK_GROUPS.items():
        current_group_scores = []
        found_group_member = False

        for sub in sub_tasks:
            if sub in requested_benchmarks:
                found_group_member = True
                processed_benchmarks.add(sub)
                score = bench_main_scores.get(sub)
                if score is not None:
                    current_group_scores.append(score)

        if found_group_member and current_group_scores:
            final_scores_for_avg.append(sum(current_group_scores) / len(current_group_scores))

    for bench, score in bench_main_scores.items():
        if bench not in processed_benchmarks and score is not None:
            final_scores_for_avg.append(score)

    if not final_scores_for_avg:
        return "-"

    return round(sum(final_scores_for_avg) / len(final_scores_for_avg), 2)


def get_metric(metrics_by_bench, bench_name, raw_key):
    metrics = metrics_by_bench.get(bench_name)
    if not metrics:
        return "-"
    value = metrics.get(raw_key)
    return "-" if value is None else value


def fill_hallusion_avg(results):
    if results.get("avg") is not None:
        return

    values = [results.get(key) for key in ("aAcc", "fAcc", "qAcc")]
    if all(value is not None for value in values):
        results["avg"] = round(sum(values) / len(values), 2)


def extract_metrics(model_dir, benchmark_name):
    cfg = BENCHMARK_CONFIG.get(benchmark_name)
    if not cfg:
        return None, None

    file_path = find_file(model_dir, cfg["suffix"])
    if not file_path:
        return None, None

    results = {}
    raw_columns = list(cfg["columns_map"].keys())
    
    try:
        if cfg["type"] == "csv":
            df = pd.read_csv(file_path)
            if "filter_key" in cfg:
                row = df[df.iloc[:, 0] == cfg["filter_val"]]
                if row.empty:
                    if cfg["filter_key"] not in df.columns:
                        row = df.iloc[0]
                    else:
                        return None, None
                else:
                    row = row.iloc[0]
            else:
                row = df.iloc[0]

            for col in raw_columns:
                if col == "F1" and benchmark_name == "POPE":
                    continue 
                if col in row:
                    results[col] = format_score(row[col])
                elif col in df.columns:
                    results[col] = format_score(df[col].iloc[0])

            if benchmark_name == "POPE":
                p = format_score(row.get("precision", 0)) or 0
                r = format_score(row.get("recall", 0)) or 0
                if (p + r) > 0:
                    results["F1"] = round(2 * (p * r) / (p + r), 2)
                else:
                    results["F1"] = 0.0
                results["acc"] = format_score(row.get("acc"))

            if benchmark_name == "HallusionBench":
                fill_hallusion_avg(results)

        elif cfg["type"] == "json":
            with open(file_path, 'r') as f:
                data = json.load(f)
            for col in raw_columns:
                results[col] = format_score(data.get(col))
        
        main_score = results.get(cfg["main_metric"])
        if benchmark_name == "POPE":
             main_score = results.get("acc")

        return results, main_score

    except Exception as e:
        return None, None

def build_paper_main_dataframe(args):
    columns = pd.MultiIndex.from_tuples(PAPER_TABLE2_COLUMNS)
    rows = []

    for idx, model_dir in enumerate(args.dirs):
        label = args.labels[idx] if args.labels is not None else None
        role = args.roles[idx] if args.roles is not None else "-"
        base_model, method_name = parse_model_identity(model_dir, label=label)
        bench_main_scores, metrics_by_bench, missing_any_benchmark = collect_benchmark_data(model_dir, args.benchmarks)

        overall = "-" if missing_any_benchmark else compute_global_score(bench_main_scores, args.benchmarks)
        row = {
            ("Meta", "Method"): method_name,
            ("Meta", "Role"): role,
            ("Meta", "Base Model"): base_model,
            ("Overall", "Score"): overall,
            ("HallusionBench", "aAcc"): get_metric(metrics_by_bench, "HallusionBench", "aAcc"),
            ("HallusionBench", "fAcc"): get_metric(metrics_by_bench, "HallusionBench", "fAcc"),
            ("HallusionBench", "qAcc"): get_metric(metrics_by_bench, "HallusionBench", "qAcc"),
            ("AMBER", "Attr"): get_metric(metrics_by_bench, "AMBER", "Attribute"),
            ("AMBER", "Exist"): get_metric(metrics_by_bench, "AMBER", "Existence"),
            ("AMBER", "Rel"): get_metric(metrics_by_bench, "AMBER", "Relation"),
            ("CRPE", "Exist"): get_metric(metrics_by_bench, "CRPE_EXIST", "total"),
            ("CRPE", "Rel"): get_metric(metrics_by_bench, "CRPE_RELATION", "total"),
            ("R-Bench", "Dis"): get_metric(metrics_by_bench, "R-Bench-Dis", "Overall"),
            ("R-Bench", "Ref"): get_metric(metrics_by_bench, "R-Bench-Ref", "Overall"),
            ("BLINK", "Score"): get_metric(metrics_by_bench, "BLINK", "Overall"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.reindex(columns=columns)


def build_detailed_dataframe(args):
    header_tuples = [("Overall", "Global Score")]
    for bench in args.benchmarks:
        cfg = BENCHMARK_CONFIG[bench]
        for metric_display_name in cfg["columns_map"].values():
            header_tuples.append((bench, metric_display_name))

    columns = pd.MultiIndex.from_tuples(header_tuples)
    rows_data = []
    index_names = []
    last_base_model = None

    for idx, model_dir in enumerate(args.dirs):
        label = args.labels[idx] if args.labels is not None else None
        base_model, method_name = parse_model_identity(model_dir, label=label)

        if base_model != last_base_model:
            empty_row = [np.nan] * len(header_tuples)
            rows_data.append(empty_row)
            index_names.append(f"--- {base_model} ---")
            last_base_model = base_model

        current_row = []
        bench_main_scores, metrics_by_bench, missing_any_benchmark = collect_benchmark_data(model_dir, args.benchmarks)
        bench_results_map = {}

        for bench, metrics in metrics_by_bench.items():
            cfg = BENCHMARK_CONFIG[bench]
            for raw_col, display_col in cfg["columns_map"].items():
                bench_results_map[(bench, display_col)] = metrics.get(raw_col, "-")

        if missing_any_benchmark:
            global_score = "-"
        else:
            global_score = compute_global_score(bench_main_scores, args.benchmarks)

        current_row.append(global_score)

        for i in range(1, len(header_tuples)):
            bench_name, metric_name = header_tuples[i]
            val = bench_results_map.get((bench_name, metric_name), "-")
            current_row.append(val)

        rows_data.append(current_row)
        index_names.append(method_name)

    df = pd.DataFrame(rows_data, index=index_names, columns=columns)
    df.index.name = "Method"
    return df


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Evaluation Aggregator")
    parser.add_argument("--dirs", nargs='+', required=True, help="List of model directories")
    parser.add_argument("--labels", nargs='+', help="Optional display labels aligned with --dirs")
    parser.add_argument("--roles", nargs='+', help="Optional role labels aligned with --dirs")
    parser.add_argument("--benchmarks", nargs='+', required=True,
                        choices=BENCHMARK_CONFIG.keys(),
                        help="List of benchmarks to include")
    parser.add_argument("--output", type=str, help="Path to save the output CSV")
    parser.add_argument(
        "--table-format",
        choices=["paper-main", "detailed"],
        default="paper-main",
        help="paper-main aligns the default output with Table 2's benchmark metrics; detailed keeps the repository's raw benchmark sub-metrics.",
    )

    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.dirs):
        parser.error("--labels must have the same number of entries as --dirs")
    if args.roles is not None and len(args.roles) != len(args.dirs):
        parser.error("--roles must have the same number of entries as --dirs")

    if args.table_format == "paper-main":
        df = build_paper_main_dataframe(args)
    else:
        df = build_detailed_dataframe(args)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.multi_sparse', False)

    print("\n" + "="*40 + " Evaluation Results " + "="*40)
    if args.table_format == "paper-main":
        print(df.to_string(index=False, justify='center', na_rep=""))
    else:
        print(df.to_string(justify='center', na_rep=""))
    print("="*100 + "\n")

    if args.output:
        if args.table_format == "paper-main":
            df.to_csv(args.output, index=False)
        else:
            df.to_csv(args.output, na_rep="")
        print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
