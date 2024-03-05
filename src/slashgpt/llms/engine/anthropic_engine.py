from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, AsyncGenerator

from anthropic import AsyncAnthropic
import tiktoken  # for counting tokens

from slashgpt.function.function_call import FunctionCall
from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_debug, print_error


if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


class LLMEngineAnthropic(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("ANTHROPIC_API_KEY environment variable is missing from .env")
            sys.exit()

        self.client = AsyncAnthropic(api_key=key)

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        model_name = self.llm_model.name()
        functions = manifest.functions()
        max_tokens = manifest.max_tokens()

        # Parse the format to account for Anthropic API format
        system_prompt = None
        processed_messages = None
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content")
            if message.get("role") == "user":
                processed_messages = message

        params = {
            "model": model_name,
            "messages": [processed_messages],
            "system": system_prompt,
            "max_tokens": max_tokens,
        }
        if functions:
            raise NotImplementedError
            # params["functions"] = functions

        async with self.client.messages.stream(**params) as stream:
            async for chunk in stream.text_stream:
                yield chunk

    def __num_tokens(self, text: str):
        model_name = self.llm_model.name()
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def is_within_budget(self, text: str, verbose: bool = False):
        token_budget = self.llm_model.max_token() - 500
        return self.__num_tokens(text) <= token_budget
