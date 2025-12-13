import torch

from inference_adaptor.base_adaptor import BaseAdaptor
from vllm import LLM, SamplingParams


class VllmAdaptor(BaseAdaptor):
    def __init__(self, model_configs):
        self.max_user_input_tokens = model_configs["max_user_input_tokens"]
        serving_params = {
            "model": model_configs["model_path"],
            "tensor_parallel_size": torch.cuda.device_count(),
            "served_model_name": model_configs["model_name"],
            "enable_chunked_prefill": True,
            "trust_remote_code": True,
        }
        if "torch_dtype" in model_configs:
            model_configs["serving_params"]["dtype"] = self._str_to_torch_dtype(
                model_configs["torch_dtype"]
            )
        serving_params |= model_configs["serving_params"]

        self.llm = LLM(**serving_params)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(**model_configs["sampling_params"])
        self.enable_thinking = model_configs.get("enable_thinking", True)
        self.response_prefix = model_configs.get("response_prefix", "")

    def terminate(self):
        print("terminate VLLM")

    def inference_turn(self, batch):
        prompt_token_ids = [
            self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self.enable_thinking,
            )
            for message in batch
        ]

        truncated_prompt_token_ids = self._truncate_center(prompt_token_ids)

        responses = self.llm.generate(
            truncated_prompt_token_ids, sampling_params=self.sampling_params
        )
        raw_responses = []
        for idx, response in enumerate(responses):
            response_text = response.outputs[0].text
            if self.response_prefix and self.response_prefix in response_text:
                think_text, _, response_text = response_text.partition(
                    self.response_prefix
                )
                think_text, response_text = think_text.strip(), response_text.strip()
            else:
                think_text, response_text = "", response_text.strip()

            input_tokens = len(truncated_prompt_token_ids[idx])
            think_tokens = len(
                self.tokenizer.encode(think_text, add_special_tokens=False)
            )
            response_tokens = len(
                self.tokenizer.encode(response_text, add_special_tokens=False)
            )
            raw_responses.append(
                {
                    "response": response_text,
                    "think": think_text,
                    "elapsed_time": -1,
                    "input_tokens": input_tokens,
                    "think_tokens": think_tokens,
                    "response_tokens": response_tokens,
                }
            )
        return raw_responses

    def inference(self, batch):
        queue = self.initialize_batch(batch)
        outputs = []
        while len(queue) > 0:
            singleturn_batch = []
            items = []
            next_queue = []
            for item in queue:
                if len(item["input"]) == len(item["response"]):
                    outputs.append(item)
                else:
                    turn = len(item["response"])
                    item["accumulated_conversations"].append(
                        {"role": item["role"][turn], "content": item["input"][turn]}
                    )

                    if item["role"][turn] == "system":
                        item["response"].append("")
                        item["think"].append("")
                        item["input_tokens"].append(0)
                        item["think_tokens"].append(0)
                        item["response_tokens"].append(0)
                        item["elapsed_time"].append(-1)
                        next_queue.append(item)

                    else:
                        items.append(item)
                        singleturn_batch.append(item["accumulated_conversations"])

            response_objs = self.inference_turn(singleturn_batch)

            for response_obj, item in zip(response_objs, items):
                item["response"].append(response_obj["response"])
                item["think"].append(response_obj["think"])
                item["input_tokens"].append(response_obj["input_tokens"])
                item["think_tokens"].append(response_obj["think_tokens"])
                item["response_tokens"].append(response_obj["response_tokens"])
                item["accumulated_conversations"].append(
                    {"role": "assistant", "content": response_obj["response"]}
                )
                item["elapsed_time"].append(response_obj["elapsed_time"])
                next_queue.append(item)

            queue = next_queue
        return outputs

    def _str_to_torch_dtype(self, s):
        dtype = getattr(torch, s, None)
        if isinstance(dtype, torch.dtype):
            return dtype
        raise ValueError(f"Unknonw torch dtype: {s}")

    def _truncate_center(self, prompt_token_ids):
        half = int(self.max_user_input_tokens // 2)

        truncated_prompt_token_ids = []
        for prompt_token_id in prompt_token_ids:
            if len(prompt_token_id) > self.max_user_input_tokens:
                truncated_prompt_token_id = (
                    prompt_token_id[:half] + prompt_token_id[-half:]
                )
                truncated_prompt_token_ids.append(truncated_prompt_token_id)
            else:
                truncated_prompt_token_ids.append(prompt_token_id)

        return truncated_prompt_token_ids
