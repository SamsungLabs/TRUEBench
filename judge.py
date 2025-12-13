import json
import argparse
import os

import polars as pl

from pathlib import Path
from tqdm import tqdm

from prompts.judge_prompt import (
    judge_prompt_system,
    judge_prompt_user,
    judge_prompt_user_multiturn,
)
from utils import get_model_configs, create_directory_if_not_exists
from inference_adaptor.openai_adaptor import OpenaiAdaptor
from inference_adaptor.vertexai_adaptor import VertexaiAdaptor
from inference_adaptor.anthropic_vertexai_adaptor import AnthropicVertexaiAdaptor


def load_inference_result(path):
    if path.suffix == ".jsonl":
        return pl.read_ndjson(str(path))
    else:
        return pl.read_ndjson(str(path) + ".jsonl")


def build_criteria(criteria):
    if isinstance(criteria, str):
        return criteria.replace("\n\n", "\n").strip()
    elif isinstance(criteria, list):
        criteria_str = ""
        for idx, criterion in enumerate(criteria):
            criteria_str += f"{idx+1}. {criterion}\n"
        return criteria_str.replace("\n\n", "\n").strip()
    else:
        print(f"Invalid criteria type : {type(criteria)}, criteria : {criteria}")
        return ""


def build_judge_prompt_singleturn(criteria, instruction, response):
    criteria = build_criteria(criteria)
    return {
        "role": ["system", "user"],
        "input": [
            judge_prompt_system,
            judge_prompt_user.replace("___CRITERIA___", criteria)
            .replace("___INSTRUCTION___", instruction)
            .replace("___RESPONSE___", response),
        ],
    }


def build_judge_prompt_multiturn(convs, criteria, instruction, response):
    pre_convs = ""
    for _instruction, _response in convs:
        pre_convs += f"User: {_instruction}\nAssistant: {_response}\n"
    criteria = build_criteria(criteria)
    return {
        "role": ["system", "user"],
        "input": [
            judge_prompt_system,
            judge_prompt_user_multiturn.replace("___CONVERSATIONS___", pre_convs)
            .replace("___CRITERIA___", criteria)
            .replace("___INSTRUCTION___", instruction)
            .replace("___RESPONSE___", response),
        ],
    }


def parse_score(line):
    content = line.split("```json")[-1]
    content = content.split("```")[0]
    try:
        data = json.loads(content)
        return (True, data)
    except:
        return (False, None)


def get_score(line):
    flag, data = parse_score(line)
    if flag:
        labels = []
        for _key in data:
            if data[_key].strip().lower() == "pass":
                labels.append(True)
            elif data[_key].strip().lower() == "fail":
                labels.append(False)
            else:
                return {"result": False, "type": "Parsing Error", "labels": []}
        if sum(labels) == len(labels):
            return {"result": True, "type": "Pass", "labels": labels}
        else:
            return {"result": False, "type": "Fail", "labels": labels}
    else:
        return {"result": False, "type": "Parsing Error", "labels": []}


def vote_judges(judge0_parsed, judge1_parsed, judge2_parsed):
    PARSING_ERROR = "Parsing Error"
    is_parsing_errors = [
        judge0_parsed["type"] == PARSING_ERROR,
        judge1_parsed["type"] == PARSING_ERROR,
        judge2_parsed["type"] == PARSING_ERROR,
    ]
    false_indexes = [i for i, val in enumerate(is_parsing_errors) if not val]

    if len(false_indexes) == 0:
        return False, "All the three judge got Parsing Error"
    elif len(false_indexes) < 3:
        target = [judge0_parsed, judge1_parsed, judge2_parsed][false_indexes[0]]
        return (
            target["result"],
            "Parsing Error is occured judge. Evaluate score with judge"
            + str(false_indexes[0])
            + ".\n",
        )

    labels0 = judge0_parsed["labels"]
    labels1 = judge1_parsed["labels"]
    labels2 = judge2_parsed["labels"]
    if len(labels0) != len(labels1) or len(labels0) != len(labels2):
        return (
            judge0_parsed["result"],
            "Label count mismatch. Evaluate score with judge0",
        )

    failed_indexes = []
    for index, (label0, label1, label2) in enumerate(zip(labels0, labels1, labels2)):
        voting = int(label0) + int(label1) + int(label2)
        if voting < 2:
            failed_indexes.append(index)

    if len(failed_indexes) == 0:
        return True, "Pass"
    else:
        fail_nums = ""
        for failed_index in failed_indexes:
            if fail_nums == "":
                fail_nums = str(failed_index)
            else:
                fail_nums = fail_nums + "," + str(failed_index + 1)
        return False, "Failed Criteria " + fail_nums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="eval_results/")
    args = parser.parse_args()

    print(args.eval_file)
    args.eval_file = args.eval_file.replace("\\", "/")

    model_configs = get_model_configs(args.config)
    if model_configs["serving_type"] == "vertexai":
        inference_adaptor = VertexaiAdaptor(model_configs)
    elif model_configs["serving_type"] == "anthropic_vertexai":
        inference_adaptor = AnthropicVertexaiAdaptor(model_configs)
    else:
        inference_adaptor = OpenaiAdaptor(model_configs)
    output_path = args.output_path

    script_dir = Path(__file__).resolve().parent
    eval_file = (script_dir / args.eval_file).resolve()
    output_path = (script_dir / args.output_path).resolve()

    df = load_inference_result(eval_file)

    create_directory_if_not_exists(output_path)

    eval_filename = eval_file.name.removesuffix(".jsonl")

    output_file = os.path.join(output_path, eval_filename + "_eval_result.jsonl")

    batch = []
    criteria_warned = False
    for line in tqdm(df.iter_rows(named=True)):
        convs = []
        for criteria, instruction, response in zip(
            line["criteria"], line["input"], line["response"]
        ):
            if (
                isinstance(criteria, str)
                and criteria[:2] == '["'
                and criteria[-2:] == '"]'
                and not criteria_warned
            ):
                criteria_warned = True
                print(
                    "Warning : Criteria seems to be mix of string and list, handling as string"
                )
            if len(convs) < 1:
                prompt = build_judge_prompt_singleturn(criteria, instruction, response)
            else:
                prompt = build_judge_prompt_multiturn(
                    convs, criteria, instruction, response
                )
            convs.append((instruction, response))
            batch.append(prompt)

    api_responses = inference_adaptor.inference(batch)
    inference_adaptor.terminate()

    for line in tqdm(df.iter_rows(named=True)):
        is_passed = True
        judges = []
        judge_parseds = []
        vote_logs = []
        judge_input_tokens = []
        judge_think_tokens = []
        judge_response_tokens = []
        judge_elapsed_time = []
        for criteria in line["criteria"]:
            api_response = api_responses.pop(0)
            judge = api_response["response"][-1]
            judge_elapsed_time.append(api_response["elapsed_time"][-1])
            judge_input_tokens.append(api_response["input_tokens"][-1])
            judge_think_tokens.append(api_response["think_tokens"][-1])
            judge_response_tokens.append(api_response["response_tokens"][-1])
            judge_parsed = get_score(judge)

            if judge_parsed["result"] is False:
                is_passed = False

            judges.append(judge)
            judge_parseds.append(judge_parsed)

        dt = {
            "index": line["index"],
            "category": line["category"],
            "language": line["language"],
            "sub_category": line["sub_category"],
            "turns": line["turns"],
            "criteria": line["criteria"],
            "input": line["input"],
            "response": line["response"],
            "think": line["think"],
            "inference_elapsed_time": line["elapsed_time"],
            "inference_input_tokens": line.get("input_tokens", []),
            "inference_think_tokens": line.get("think_tokens", []),
            "inference_response_tokens": line.get("response_tokens", []),
            "judge_elapsed_time": judge_elapsed_time,
            "judge_input_tokens": judge_input_tokens,
            "judge_think_tokens": judge_think_tokens,
            "judge_response_tokens": judge_response_tokens,
            "judge": judges,
            "judge_parsed": judge_parseds,
            "vote_logs": vote_logs,
            "pass": is_passed,
        }

        with open(output_file, "a", encoding="utf-8") as fo:
            json.dump(dt, fo, ensure_ascii=False)
            fo.write("\n")
