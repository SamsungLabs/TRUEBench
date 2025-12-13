import argparse
import glob
import json
import csv
import os
from collections import defaultdict
from itertools import product


def create_stats(target_dir):
    headers = [
        "Model Name",
        "Overall",
        "Content Generation",
        "Editing",
        "Data Analysis",
        "Reasoning",
        "Hallucination",
        "Safety",
        "Repetition",
        "Summarization",
        "Translation",
        "Single-Turn",
        "Multi-Turn",
    ]

    rows = []

    output_file = os.path.join(target_dir, "stats_cat.csv")

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)

        json_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
        json_files = sorted(json_files)

        for file in json_files:
            cnt = {}
            for header in headers:
                if header != "Model Name":
                    cnt[header + " total"] = 0
                    cnt[header + " passed"] = 0

            if "_TRUEBench-v" in os.path.basename(file):
                model_name = os.path.basename(file).split("_TRUEBench-v")[0]
            elif "_eval_result.jsonl" in os.path.basename(file):
                model_name = os.path.basename(file).split("_eval_result.jsonl")[0]
            else:
                model_name = os.path.basename(file).split(".jsonl")[0]

            row = [model_name]

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    cnt["Overall total"] += 1
                    cnt[data["category"] + " total"] += 1

                    if data["category"] != "Multi-Turn":
                        cnt["Single-Turn total"] += 1

                    if data["pass"] == True:
                        cnt["Overall passed"] += 1
                        cnt[data["category"] + " passed"] += 1
                        if data["category"] != "Multi-Turn":
                            cnt["Single-Turn passed"] += 1

            for header in headers:
                if header != "Model Name":
                    if cnt[header + " total"] == 0:
                        row.append(0)
                    else:
                        row.append(
                            round(
                                (cnt[header + " passed"] / cnt[header + " total"])
                                * 100,
                                2,
                            )
                        )

            writer.writerow(row)
            rows.append(row)

    return headers, rows


def create_stats_lang(target_dir):
    headers = [
        "Model Name",
        "Overall",
        "KO",
        "EN",
        "JA",
        "ZH",
        "PL",
        "DE",
        "PT",
        "ES",
        "FR",
        "IT",
        "RU",
        "VI",
    ]
    rows = []

    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, "stats_lang.csv")

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)

        json_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
        json_files = sorted(json_files)

        for file in json_files:
            cnt = {}
            for header in headers:
                if header != "Model Name":
                    cnt[header + " total"] = 0
                    cnt[header + " passed"] = 0

            if "_TRUEBench-v" in os.path.basename(file):
                model_name = os.path.basename(file).split("_TRUEBench-v")[0]
            elif "_eval_result.jsonl" in os.path.basename(file):
                model_name = os.path.basename(file).split("_eval_result.jsonl")[0]
            else:
                model_name = os.path.basename(file).split(".jsonl")[0]

            row = [model_name]

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)

                    lang_key = data["language"]

                    cnt["Overall total"] += 1
                    if lang_key in headers:
                        cnt[lang_key + " total"] += 1

                    if data["pass"] == True:
                        cnt["Overall passed"] += 1
                        if lang_key in headers:
                            cnt[lang_key + " passed"] += 1

            for header in headers:
                if header != "Model Name":
                    if cnt[header + " total"] == 0:
                        row.append(0)
                    else:
                        row.append(
                            round(
                                (cnt[header + " passed"] / cnt[header + " total"])
                                * 100,
                                2,
                            )
                        )

            writer.writerow(row)
            rows.append(row)

    return headers, rows


def create_usage(target_dir):
    headers = [
        "Model Name",
        "inference_input_tokens",
        "inference_think_tokens",
        "inference_response_tokens",
        "inference_total_tokens",
        "judge_input_tokens",
        "judge_think_tokens",
        "judge_response_tokens",
        "judge_total_tokens",
        "total_tokens",
    ]
    run_types = ["inference", "judge"]
    token_types = ["input", "think", "response"]
    rows = []

    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, "usage.csv")

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(headers)

        json_files = glob.glob(os.path.join(target_dir, "*.jsonl"))
        json_files = sorted(json_files)

        for file in json_files:
            tokens = defaultdict(int)
            model_name = ""

            if "_TRUEBench-v" in os.path.basename(file):
                model_name = os.path.basename(file).split("_TRUEBench-v")[0]
            elif "_eval_result.jsonl" in os.path.basename(file):
                model_name = os.path.basename(file).split("_eval_result.jsonl")[0]
            else:
                model_name = os.path.basename(file).split(".jsonl")[0]

            row = [model_name]

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    for run_type, token_type in product(run_types, token_types):
                        key = f"{run_type}_{token_type}_tokens"
                        run_total_key = f"{run_type}_total_tokens"
                        total_key = "total_tokens"

                        tokens_sum = sum(data[key])
                        tokens[key] += tokens_sum
                        tokens[run_total_key] += tokens_sum
                        tokens[total_key] += tokens_sum

            row += [tokens[header] for header in headers if header != "Model Name"]

            writer.writerow(row)
            rows.append(row)

    return headers, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    cat_headers, cat_data = create_stats(args.target_dir)
    lang_headers, lang_data = create_stats_lang(args.target_dir)
    usage_headers, usage_data = create_usage(args.target_dir)

    for cat_scores, lang_scores, token_counts in zip(cat_data, lang_data, usage_data):
        score_cat = dict(zip(cat_headers, cat_scores))
        score_lang = dict(zip(lang_headers, lang_scores))
        usage = dict(zip(usage_headers, token_counts))

        model_name, overall = score_cat["Model Name"], score_cat["Overall"]
        for remove_key in ["Model Name", "Overall"]:
            score_cat.pop(remove_key, None)
            score_lang.pop(remove_key, None)
            usage.pop(remove_key, None)

        stats = {
            "TRUEBench": {
                "overall": overall,
                "score_cat": score_cat,
                "score_lang": score_lang,
                "usage": usage,
            }
        }

        with open(
            args.target_dir + "/" + model_name + ".json", encoding="utf-8", mode="w"
        ) as out_f:
            json.dump(stats, out_f, ensure_ascii=False, indent=4)
