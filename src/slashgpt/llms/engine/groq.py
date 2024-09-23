from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, AsyncGenerator

import tiktoken  # for counting tokens

from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_error

from groq import Groq, AsyncGroq

if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


class LLMEngineGroq(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("GROQ_API_KEY environment variable is missing from .env")
            sys.exit()

        self.client = Groq(api_key=key)
        self.async_client = AsyncGroq(api_key=key)

        # Override default openai endpoint for custom-hosted models
        api_base = llm_model.get_api_base()
        if api_base:
            self.async_client.api_base = api_base

        return

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        # For now, we are trying to force as much determinism as possible
        temperature = 0.00000000001
        top_p = 0.00000000001
        max_tokens = 2048
        model_name = self.llm_model.name()
        manifest.functions()
        stream = manifest.stream()
        num_completions = manifest.num_completions()
        manifest.images()
        # max_tokens = manifest.max_tokens()

        # TODO: parse each message to see if it contains an image URL
        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "n": num_completions,
            "max_tokens": max_tokens,
        }

        stream_keys = ["model", "stream", "messages", "temperature", "top_p", "max_tokens"]
        stream_params = {k: params[k] for k in stream_keys}

        collected_messages = []
        stream = await self.async_client.chat.completions.create(**stream_params)
        async for chunk in stream:
            resp = chunk.choices[0].delta.content
            collected_messages.append(resp)
            yield resp

    def __num_tokens(self, text: str):
        model_name = self.llm_model.name()
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def is_within_budget(self, text: str, verbose: bool = False):
        token_budget = self.llm_model.max_token() - 500
        return self.__num_tokens(text) <= token_budget
