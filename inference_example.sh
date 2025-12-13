#!/bin/bash

python inference.py --inference_adaptor vllm --config vllm-Qwen3-8B --dataset_path dataset/TRUEBench-v0.6.1 --output_path results
