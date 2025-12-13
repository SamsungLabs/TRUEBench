import asyncio
import os
import time
from google import genai
from google.genai import types

from inference_adaptor.base_adaptor import BaseAdaptor


class VertexaiAdaptor(BaseAdaptor):
    def __init__(self, model_configs):
        self.count = 0
        self.lock = asyncio.Lock()
        self.model_name = model_configs["model_name"]
        self.project_id = model_configs["project_id"]
        self.location = model_configs.get("location", "global")
        self.semaphore_cnt = model_configs.get("semaphore_max_count", 16)
        self.sampling_params = self._init_sampling_params(
            model_configs.get("sampling_params", {})
        )

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
        print("terminate Vertex AI Adaptor")

    def _init_sampling_params(self, sampling_params):
        if "thinking_config" in sampling_params and isinstance(
            sampling_params["thinking_config"], dict
        ):
            thinking_config = types.ThinkingConfig(**sampling_params["thinking_config"])
            sampling_params["thinking_config"] = thinking_config
        return sampling_params

    def create_context(self, client, system_prompts: list[str]):
        if system_prompts:
            generation_config = types.GenerateContentConfig(
                system_instruction=system_prompts, **self.sampling_params
            )
        else:
            generation_config = types.GenerateContentConfig(**self.sampling_params)

        return client.chats.create(
            model=self.model_name,
            config=generation_config,
        )

    async def print_count(self):
        async with self.lock:
            self.count += 1
            print(f"{self.count} tasks are done")

    def create_fallback_response(self, request, error_message):
        request["response"].append(error_message)
        while len(request["response"]) < len(request["input"]):
            request["response"].append(f"Error on previous turns : {error_message}")
        request["think"] += [""] * (len(request["input"]) - len(request["think"]))
        request["input_tokens"] += [0] * (
            len(request["input"]) - len(request["input_tokens"])
        )
        request["think_tokens"] += [0] * (
            len(request["input"]) - len(request["think_tokens"])
        )
        request["response_tokens"] += [0] * (
            len(request["input"]) - len(request["response_tokens"])
        )
        request["elapsed_time"] += [-1] * (
            len(request["input"]) - len(request["elapsed_time"])
        )

        return request

    def reset_response(self, request):
        request["response"] = []
        request["think"] = []
        request["accumulated_conversations"] = []
        request["input_tokens"] = []
        request["think_tokens"] = []
        request["response_tokens"] = []
        request["elapsed_time"] = []
        return request

    async def process_request(self, semaphore, request, client):
        async with semaphore:
            if len(request["role"]) != len(request["input"]):
                print("Malformed input : length of role and input mismatch")

            system_prompts = [
                msg
                for role, msg in zip(request["role"], request["input"])
                if role == "system"
            ]

            MAX_RETRY = 5
            for retry_cnt in range(MAX_RETRY):
                try:
                    context = self.create_context(client, system_prompts=system_prompts)
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
                            start_time = time.time()
                            SEND_MESSAGE_TIMEOUT = 5 * 60  # 20 minutes

                            api_response = await asyncio.wait_for(
                                context.send_message(message),
                                SEND_MESSAGE_TIMEOUT,
                            )

                            response_text = api_response.text
                            elapsed_time = time.time() - start_time

                            # Extract usage metadata
                            if (
                                hasattr(api_response, "usage_metadata")
                                and api_response.usage_metadata
                            ):
                                usage = api_response.usage_metadata
                                input_tokens = getattr(usage, "prompt_token_count", 0)
                                think_tokens = getattr(usage, "thoughts_token_count", 0)
                                if not think_tokens:
                                    think_tokens = 0
                                response_tokens = getattr(
                                    usage, "candidates_token_count", 0
                                )
                                if not response_tokens:
                                    response_tokens = 0
                            else:
                                input_tokens = 0
                                response_tokens = 0
                                think_tokens = 0

                            request["accumulated_conversations"].append(
                                {"role": "assistant", "content": response_text}
                            )
                            request["response"].append(response_text)
                            request["think"].append("")
                            request["input_tokens"].append(input_tokens)
                            request["think_tokens"].append(think_tokens)
                            request["response_tokens"].append(response_tokens)
                            request["elapsed_time"].append(elapsed_time)

                    break

                except Exception as e:
                    error_message = f"Exception occured : {e}"
                    if retry_cnt == MAX_RETRY - 1:
                        print(f"Max retries reached for Vertex AI request: {e}")
                        request = self.create_fallback_response(request, error_message)
                    else:
                        print(
                            f"Vertex AI request failed (attempt {retry_cnt + 1}/{MAX_RETRY}): {e}"
                        )
                        request = self.reset_response(request)

            await self.print_count()
        return request

    async def generate(self, request_list):
        VERTEXAI_TIMEOUT = (
            15 * 60 * 1000
        )  # 15 minutes, maximum 75 minutes when 5 tries all timed out
        # Initialize Vertex AI client
        client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            http_options=types.HttpOptions(timeout=VERTEXAI_TIMEOUT),
        ).aio

        semaphore = asyncio.Semaphore(self.semaphore_cnt)
        tasks = [
            self.process_request(semaphore, request, client) for request in request_list
        ]
        responses = await asyncio.gather(*tasks)
        await client.aclose()
        return responses

    def inference(self, batch):
        initialized_batch = self.initialize_batch(batch)
        output = asyncio.run(self.generate(initialized_batch))
        return output
