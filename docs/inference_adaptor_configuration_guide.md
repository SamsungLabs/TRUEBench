
# Inference Adaptor Configuration Guide

## Overview
The `inference_adaptor` parameter determines the backend for model inference. Supported options are:
- **openai**: For integration with OpenAI REST API (e.g., using `configs/azure_openai-gpt-o3.json`, `configs/azure_openai-gpt-5.json`, `openai-Qwen3-32B.json`, `openai-Qwen3-32B-think.json`).
- **vllm**: For local inference of open models (e.g., using `configs/vllm-Qwen3-8B.json`, `configs/vllm-Qwen3-8B-think.json`).
- **vertexai**: For integration with Google Vertex AI (e.g., using `configs/vertexai-gemini-2.5-flash.json`, `configs/vertexai-gemini-3-pro-preview.json`).
- **anthropic_vertexai**: For integration with Anthropic models via Google Vertex AI (e.g., using `configs/anthropic_vertexai-claude-haiku-4.5.json`, `configs/anthropic_vertexai-claude-haiku-4.5-think.json`).

## Configuration for OpenAI Adaptor
When using `openai`, configure your settings via a JSON file (e.g., `azure_openai-gpt-o3.json`, `openai-Qwen3-32B.json`). The adaptor utilizes OpenAI's chat_completion with an asynchronous client for non-blocking request handling.

### Configuration Fields
| Field | Description |
| --- | --- |
| serving_type | Service provider (e.g., `"azure"` or `"openai"`).  |
| model_name | Name of the model to use (e.g., `"o3"`, `"Qwen/Qwen3-32B"`).  |
| semaphore_max_count | Maximum concurrent requests allowed.  |
| api_version | Version of the API to use (e.g., `"2025-03-01-preview"`).  |
| base_url | Base URL for the API endpoint |
| api_key | Authentication key |
| sampling_params | Parameters controlling text generation (e.g., max_completion_tokens). Add other sampling_params (e.g., `"temperature"`, `"top_p"`) as needed.  |
| chat_template_kwargs | Additional parameters used to customize the chat template for text generation |
| tokenizer_path | Path to the tokenizer used for splitting the completion_tokens into think_tokens and response_tokens when the serving engine does not supply completion_tokens_detail |
| response_prefix | Fixed string added only when Think mode is enabled (e.g., `"</think>"`). If you serve the model with `"reasoning_parser"` enabled, the tags are automatically extracted, so you should not also set `"response_prefix"`. |

#### Sampling Parameters (`sampling_params`)
Azure Serving Type: We support Azure's chat completions and reasoning models. For details:
- Refer to [Work with chat completions models](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/chatgpt).
- For reasoning models, refer to [Azure OpenAI reasoning models](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning?tabs=python-secure%2Cpy)
OpenAI Serving Type: We support OpenAI's chat completion parameters.
- Refer to https://platform.openai.com/docs/api-reference/chat/create

### Example
example for azure serving type.
```json
{
    "serving_type": "azure",
    "model_name": "o3",
    "semaphore_max_count": 32,
    "api_version": "2025-03-01-preview",
    "base_url": "your-base-url",
    "api_key": "your-api-key",
    "sampling_params": {
        "max_completion_tokens": 65536
    }
}
```

example for openai serving type.
```json
{
    "serving_type": "openai",
    "model_name": "Qwen/Qwen3-32B",
    "semaphore_max_count": 32,
    "base_url": "your-base-url",
    "api_key": "your-api-key",
    "sampling_params": {
        "max_tokens": 32768,
        "top_p": 0.95,
        "temperature": 0.3
    },
    "chat_template_kwargs": {
        "enable_thinking": true
    },
    "tokenizer_path": "Qwen/Qwen3-32B",
    "response_prefix": "</think>"
}
```

## Configuration for VLLM Adaptor
When using `vllm`, configure your settings via a JSON file (e.g., `vllm-Qwen3-8B.json`, `vllm-Qwen3-8B-think.json`). The adaptor utilizes VLLM's local inference engine for high-performance model serving.

### Configuration Fields
| Field | Description |
| --- | --- |
| serving_type | Must be set to `"vllm"` for local inference. |
| model_name | Name identifier for the model (e.g., `"Qwen3-8B"`). |
| model_path | **Absolute path** to the local model directory. |
| max_user_input_tokens | Maximum tokens allowed for user input (truncates longer inputs). |
| torch_dtype | Torch data type for model weights (e.g., `"bfloat16"`). |
| serving_params | Hardware/resource optimization parameters. Add other serving_params as needed. |
| sampling_params | Parameters controlling text generation (e.g., `"max_tokens"`, `"temperature"`). Add other sampling_params as needed. |
| enable_thinking | Boolean Parameter indicating whether the tokenizer should apply the “thinking” format or not. |
| response_prefix | Fixed string added only when Think mode is enabled (e.g., `"</think>"`). |

#### Serving Parameters (`serving_params`)
We support vLLM's default serving parameters. For details, refer to the https://docs.vllm.ai/en/v0.10.2/api/vllm/index.html#vllm.LLM

#### Sampling Parameters (`sampling_params`)
We support vLLM's default sampling parameters. For details, refer to the https://docs.vllm.ai/en/v0.10.2/api/vllm/index.html#vllm.SamplingParams

### Example
```json
{
    "serving_type": "vllm",
    "model_name": "Qwen3-8B",
    "model_path": "",
    "max_user_input_tokens": 16384,
    "torch_dtype": "bfloat16",
    "serving_params": {
        "max_seq_len_to_capture": 32768,
        "gpu_memory_utilization": 0.90,
        "swap_space": 16,
        "seed": 42
    },
    "sampling_params": {
        "max_tokens": 32768,
        "top_p": 0.95,
        "temperature": 0.3
    },
    "enable_thinking": true,
    "response_prefix": "</think>"
}

```

## Configuration for Vertex AI Adaptor
When using `vertexai`, configure your settings via a JSON file (e.g., `vertexai-gemini-2.5-flash.json`, `configs/vertexai-gemini-3-pro-preview.json`). The adaptor utilizes Google's Vertex AI for model inference with asynchronous request handling.

### Configuration Fields
| Field | Description |
| --- | --- |
| serving_type | Must be set to `"vertexai"` for Google Vertex AI integration. |
| model_name | Name of the Vertex AI model to use (e.g., `"gemini-2.5-flash"`, `"gemini-3-pro-preview"`). |
| project_id | Google Cloud project ID where the Vertex AI API is enabled. |
| location | Vertex AI region/location (default: `"global"`). |
| semaphore_max_count | Maximum concurrent requests allowed. |
| credentials_path | Path to Google Cloud service account credentials JSON file. |
| sampling_params | Parameters controlling text generation. |

#### Sampling Parameters (`sampling_params`)
We support Vertex AI's generation parameters. For details:
- Refer to [GenerationConfig documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GenerationConfig)
- Some reasoning models(e.g., `"gemini-3-pro-preview"`) require to set `thinking_level` parameter (e.g., `"low"`, `"high"`).

### Example
example for basic Vertex AI model.
```json
{
    "serving_type": "vertexai",
    "model_name": "gemini-2.5-flash",
    "project_id": "your-project-id",
    "semaphore_max_count": 32,
    "credentials_path": "/your/credentials/path/credentials.json"
}
```

example for Vertex AI thinking model.
```json
{
    "serving_type": "vertexai",
    "model_name": "gemini-3-pro-preview",
    "project_id": "your-project-id",
    "semaphore_max_count": 32,
    "credentials_path": "/your/credentials/path/credentials.json",
    "sampling_params": {
        "thinking_level": "high"
    }
}
```

## Configuration for Anthropic Vertex AI Adaptor
When using `anthropic_vertexai`, configure your settings via a JSON file (e.g., `anthropic_vertexai-claude-haiku-4.5.json`, `configs/anthropic_vertexai-claude-haiku-4.5-think.json`). The adaptor utilizes Anthropic's models served through Google Vertex AI with asynchronous request handling and streaming support.

### Configuration Fields
| Field | Description |
| --- | --- |
| serving_type | Must be set to `"anthropic_vertexai"` for Anthropic models via Vertex AI integration. |
| model_name | Name of the Anthropic model to use (e.g., `"claude-haiku-4-5@20251001"`). |
| project_id | Google Cloud project ID where the Vertex AI API is enabled. |
| location | Vertex AI region/location (default: `"global"`). |
| semaphore_max_count | Maximum concurrent requests allowed (default: 16). |
| credentials_path | Path to Google Cloud service account credentials JSON file. |
| sampling_params | Parameters controlling text generation. |

#### Sampling Parameters (`sampling_params`)
We support Anthropic's message creation parameters via Vertex AI. For details:
- Refer to https://platform.claude.com/docs/en/api/messages/create
- For thinking models, you can enable thinking mode with the `thinking` parameter.

### Example
example for basic Anthropic model.
```json
{
    "serving_type": "anthropic_vertexai",
    "model_name": "claude-haiku-4-5@20251001",
    "location": "global",
    "project_id": "your-project-id",
    "semaphore_max_count": 32,
    "credentials_path": "/your/credentials/path/credentials.json",
    "sampling_params": {
        "max_tokens": 64000
    }
}
```

example for Anthropic thinking model.
```json
{
    "serving_type": "anthropic_vertexai",
    "model_name": "claude-haiku-4-5@20251001",
    "location": "global",
    "project_id": "your-project-id",
    "semaphore_max_count": 32,
    "credentials_path": "/your/credentials/path/credentials.json",
    "sampling_params": {
        "max_tokens": 64000,
        "thinking": {
            "type": "enabled",
            "budget_tokens": 32000
        }
    }
}
```
