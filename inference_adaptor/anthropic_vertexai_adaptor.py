import asyncio
import re
import time
import os

from anthropic import AsyncAnthropicVertex
from inference_adaptor.base_adaptor import BaseAdaptor


class AnthropicVertexaiAdaptor(BaseAdaptor):
    def __init__(self, model_configs):
        self.count = 0
        self.lock = asyncio.Lock()
        self.model_name = model_configs["model_name"]
        self.project_id = model_configs["project_id"]
        self.location = model_configs.get("location", "global")
        self.semaphore_cnt = model_configs.get("semaphore_max_count", 16)
        self.sampling_params = model_configs.get("sampling_params", {})

        if self.project_id == "your-project-id":
            raise ValueError("please set proper project id")

        # Set up Google Application Credentials if provided
        if "credentials_path" in model_configs:

            if (
                model_configs["credentials_path"]
                == "/your/credentials/path/credentials.json"
            ):
                raise ValueError("please set proper credentials path")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = model_configs[
                "credentials_path"
            ]

    def terminate(self):
        print("terminate Anthropic Vertexai Adaptor")

    async def print_count(self):
        async with self.lock:
            self.count += 1
            print(f"{self.count} tasks are done")

    async def send_request(self, client, request):
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
                response = ""
                think = ""
                async with client.messages.stream(**request) as stream:
                    async for text in stream.text_stream:
                        response += text
                    api_response = await stream.get_final_message()
                elapsed_time = time.time() - start_time
                if api_response.content[0].type == "thinking":
                    think = api_response.content[0].thinking

                usage = api_response.usage
                input_tokens = usage.input_tokens
                response_tokens = usage.output_tokens
                # anthropic vertexai don't give us think token count
                think_tokens = 0

                error_pattern = r"^Error\s+code:\s+\d{3}\s+-.*"
                if re.match(error_pattern, response):
                    print("Error is occurred: ", response[:80])
                    print("retry...", retry_cnt + 1)
                else:
                    break

            except Exception as e:
                response = f"{e}"
                elapsed_time = -1
                if retry_cnt == MAX_RETRY - 1:
                    print(f"Max retries reached for Anthropic Vertex AI request: {e}")
                else:
                    print(
                        f"Anthropic Vertex AI request failed (attempt {retry_cnt + 1}/{MAX_RETRY}): {type(e)}: {e}"
                    )

        return {
            "response": response,
            "think": think,
            "elapsed_time": elapsed_time,
            "input_tokens": input_tokens,
            "think_tokens": think_tokens,
            "response_tokens": response_tokens,
        }

    async def process_request(self, semaphore, client, request):
        async with semaphore:
            if len(request["role"]) != len(request["input"]):
                print("Malformed input : length of role and input mismatch")
            system_prompts = [
                msg
                for role, msg in zip(request["role"], request["input"])
                if role == "system"
            ]
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
                        "messages": [
                            conv
                            for conv in request["accumulated_conversations"]
                            if conv["role"] != "system"
                        ],
                    }
                    if len(system_prompts) > 0:
                        completion_request["system"] = "\n\n".join(system_prompts)
                    completion_request |= self.sampling_params
                    response = await self.send_request(client, completion_request)

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
        client = AsyncAnthropicVertex(project_id=self.project_id, region=self.location)
        semaphore = asyncio.Semaphore(self.semaphore_cnt)
        tasks = [
            self.process_request(semaphore, client, request) for request in request_list
        ]
        responses = await asyncio.gather(*tasks)
        await client.close()
        return responses

    def inference(self, batch):
        initialized_batch = self.initialize_batch(batch)
        output = asyncio.run(self.generate(initialized_batch))
        return output
