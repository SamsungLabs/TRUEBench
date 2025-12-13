import asyncio
import openai
import re
import time

from inference_adaptor.base_adaptor import BaseAdaptor
from transformers import AutoTokenizer


class OpenaiAdaptor(BaseAdaptor):
    def __init__(self, model_configs):
        self.count = 0
        self.lock = asyncio.Lock()
        self.serving_type = model_configs["serving_type"]
        self.model_name = model_configs["model_name"]
        self.response_prefix = model_configs.get("response_prefix", "")
        self.sampling_params = model_configs.get("sampling_params", {})
        self.semaphore_cnt = model_configs.get("semaphore_max_count", 16)
        self.tokenizer_path = model_configs.get("tokenizer_path", "")
        self.tokenizer = None
        if self.tokenizer_path != "":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path, trust_remote_code=True
            )

        if "chat_template_kwargs" in model_configs:
            self.chat_template_kwargs = model_configs["chat_template_kwargs"]
        else:
            self.chat_template_kwargs = None

        if self.serving_type == "azure":
            if "api_version" in model_configs:
                api_version = model_configs["api_version"]
            else:
                api_version = "2025-03-01-preview"

            self.client = openai.AsyncAzureOpenAI(
                azure_endpoint=model_configs["base_url"],
                api_key=model_configs["api_key"],
                api_version=api_version,
                timeout=300.0,
                max_retries=4,
            )
        elif self.serving_type == "openai":
            self.client = openai.AsyncOpenAI(
                base_url=model_configs["base_url"],
                api_key=model_configs["api_key"],
            )
        else:
            raise ValueError(f"Unsupported serving type: {self.serving_type}")

    def terminate(self):
        print("terminate OpenAI Adaptor")

    async def print_count(self):
        async with self.lock:
            self.count += 1
            print(f"{self.count} tasks are done")

    async def send_request(self, request):
        MAX_RETRY = 5
        response = ""
        think = ""
        input_tokens = 0
        think_tokens = 0
        response_tokens = 0
        elapsed_time = 99999999
        for retry_cnt in range(MAX_RETRY):
            try:
                start_time = time.time()
                api_response = await self.client.chat.completions.create(**request)
                elapsed_time = time.time() - start_time
                response = api_response.choices[0].message.content
                think = ""
                if hasattr(api_response.choices[0].message, "reasoning_content"):
                    think = api_response.choices[0].message.reasoning_content

                usage = api_response.usage
                details = getattr(usage, "completion_tokens_details", None)
                think_tokens = getattr(details, "reasoning_tokens", 0)
                response_tokens = usage.completion_tokens - think_tokens
                input_tokens = usage.prompt_tokens

                error_pattern = r"^Error\s+code:\s+\d{3}\s+-.*"
                if re.match(error_pattern, response):
                    print("Error is occurred: ", response[:80])
                    print("retry...", retry_cnt + 1)
                else:
                    break

            except Exception as e:
                response = f"{e}"
                elapsed_time = -1

        if self.tokenizer != None:
            think_tokens = len(self.tokenizer.encode(think, add_special_tokens=False))
            response_tokens = len(
                self.tokenizer.encode(response, add_special_tokens=False)
            )

        return {
            "response": response,
            "think": think,
            "elapsed_time": elapsed_time,
            "input_tokens": input_tokens,
            "think_tokens": think_tokens,
            "response_tokens": response_tokens,
        }

    async def process_request(self, semaphore, request):
        async with semaphore:
            if len(request["role"]) != len(request["input"]):
                print("Malformed input : length of role and input mismatch")
            for role, message in zip(request["role"], request["input"]):
                request["accumulated_conversations"].append(
                    {"role": role, "content": message}
                )
                if role == "system":
                    request["response"].append("")
                    request["think"].append("")
                    request["input_tokens"].append(0)
                    request["think_tokens"].append(0)
                    request["response_tokens"].append(0)
                    request["elapsed_time"].append(0)
                else:
                    completion_request = {
                        "model": self.model_name,
                        "messages": request["accumulated_conversations"],
                    }
                    completion_request |= self.sampling_params
                    response = await self.send_request(completion_request)

                    response_text = response["response"]
                    if response_text == None:
                        response_text = "error"

                    request["accumulated_conversations"].append(
                        {"role": "assistant", "content": response_text}
                    )
                    request["response"].append(response_text)
                    request["think"].append(response["think"])
                    request["input_tokens"].append(response["input_tokens"])
                    request["think_tokens"].append(response["think_tokens"])
                    request["response_tokens"].append(response["response_tokens"])
                    request["elapsed_time"].append(response["elapsed_time"])

            await self.print_count()
        return request

    async def generate(self, request_list):
        semaphore = asyncio.Semaphore(self.semaphore_cnt)
        tasks = [self.process_request(semaphore, request) for request in request_list]
        responses = await asyncio.gather(*tasks)
        return responses

    def inference(self, batch):
        initialized_batch = self.initialize_batch(batch)
        output = asyncio.run(self.generate(initialized_batch))
        return output
