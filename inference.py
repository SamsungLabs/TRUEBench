import argparse
import jsonlines
import json
from pathlib import Path
from utils import get_model_configs, create_directory_if_not_exists

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--inference_adaptor", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--sample_cnt", type=int, default=-1)
    parser.add_argument("--output_path", type=str, default="results/")
    args = parser.parse_args()

    output_path = args.output_path
    dataset_path = Path(args.dataset_path)
    if dataset_path.suffix == ".jsonl":
        dataset_path = dataset_path.with_suffix("")
    sample_cnt = args.sample_cnt
    model_configs = get_model_configs(args.config)
    create_directory_if_not_exists(output_path)

    if args.inference_adaptor == "vllm":
        from inference_adaptor.vllm_adaptor import VllmAdaptor

        inference_adaptor = VllmAdaptor(model_configs)
    elif args.inference_adaptor == "openai":
        from inference_adaptor.openai_adaptor import OpenaiAdaptor

        inference_adaptor = OpenaiAdaptor(model_configs)
    elif args.inference_adaptor == "vertexai":
        from inference_adaptor.vertexai_adaptor import VertexaiAdaptor

        inference_adaptor = VertexaiAdaptor(model_configs)
    elif args.inference_adaptor == "anthropic_vertexai":
        from inference_adaptor.anthropic_vertexai_adaptor import (
            AnthropicVertexaiAdaptor,
        )

        inference_adaptor = AnthropicVertexaiAdaptor(model_configs)

    queue = []
    with jsonlines.open(f"{str(dataset_path)}.jsonl") as in_f:
        for input_obj in in_f:
            input_obj["role"] = ["user" for _ in input_obj["input"]]

            queue.append(input_obj)
            if len(queue) == sample_cnt:
                break

    print(len(queue))

    outputs = inference_adaptor.inference(queue)

    for output in outputs:
        output.pop("role", None)

    inference_adaptor.terminate()
    sorted_outputs = sorted(outputs, key=lambda x: x["index"])

    output_file = output_path + "/" + args.config + "_" + dataset_path.name
    with open(output_file + ".jsonl", encoding="utf-8", mode="w") as out_f:
        for item in sorted_outputs:
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("*" * 50)
    print("done")
    print("*" * 50)
