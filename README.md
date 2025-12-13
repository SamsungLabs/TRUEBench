# TRUEBench (v0.6.1)
TRUEBench is a novel benchmark designed to evaluate LLM-based productivity assistants in real-world scenarios. It addresses limitations in existing benchmarks by incorporating 12 languages, nuanced implicit constraints, and dynamic multi-turn dialogues with context switches. TRUEBench evaluates LLMs across 10 categories.

*We will continue to update TRUEBench version with the ongoing advancements in NEW Language Models.

## Prerequisites
- Install **Python 3.10.12 or later**
- Install **uv** 
  ```bash
  pip install uv
  ```

## Quick Start

### Set up
- Clone & enter the project:
  ```bash
  git clone https://github.com/SamsungLabs/TRUEBench.git
  cd TRUEBench
  ```
- Create and activate a virtual environment:
  ```bash
  uv venv .venv
  source .venv/bin/activate
  ```
- Install dependencies:
  ```bash
  uv pip sync requirements.lock
  ```

### Prepare configuration files
Refer to [Inference Adaptor Configuration Guide](docs/inference_adaptor_configuration_guide.md)

### Inference
Inference dataset with:
```
python inference.py --config {config_filename} --inference_adaptor {vllm/openai/vertexai/anthropic_vertexai} --dataset_path {dataset_path}
```
1. inference_adaptor: Select an inference adaptor. Supported options are vllm, openai, vertexai, or anthropic_vertexai.
  - vllm: Supports local inference for open models (example config: `"configs/vllm-Qwen3-8B.json"`, `"configs/vllm-Qwen3-8B-think.json"`).
  - openai: Supports OpenAI API (example config: `"configs/azure_openai-gpt-o3.json"`, `"configs/azure_openai-gpt-5.json"`, `"openai-Qwen3-32B.json"`, `"openai-Qwen3-32B-think.json"`).
  - vertexai: Supports VertexAI API (example config: `"configs/vertexai-gemini-2.5-flash.json"`, `"configs/vertexai-gemini-3-pro-preview.json"`).
  - anthropic_vertexai: Supports Anthropic VertexAI API (example config: `"configs/anthropic_vertexai-claude-haiku-4.5.json"`, `"configs/anthropic_vertexai-claude-haiku-4.5-think.json"`).
2. config: Path to a model configuration file in the `"configs/"` folder.
3. dataset_path: Path to the evaluation dataset.
4. sample_cnt: Number of sample to inference (default option make to inference all TC).
5. output_path: Running this command generates results at `"{output_path}/{config_name}_{dataset_name}.jsonl"` (default output path is `"results"`)

### Judge
Judge inference results with:
```
python judge.py --config {config_filename} --eval_file {eval_filename} --output_path {output_path}
```
1. config: Path to a judge model configuration file from `"configs/"`. Judge model should be set with openai adaptor.
2. eval_file: Model output file to evaluate.
3. output_path: Folder to save evaluation results (default output path is `"eval_results"`).

Judge Model is recommended to use the gpt-5 2025-08-07 model with default sampling params.

### Get Scores
Get scores from eval_results with:
```
python get_score.py --target_dir eval_results 
```
1. target_dir: Directory containing evaluation results (default: eval_results).
Outputs stats.csv and stats_lang.csv in the target directory.
