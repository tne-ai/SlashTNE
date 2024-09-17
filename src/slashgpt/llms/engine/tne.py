from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, AsyncGenerator

from openai import OpenAI, AsyncOpenAI
import tiktoken  # for counting tokens

import base64
import binascii

from slashgpt.function.function_call import FunctionCall
from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_debug, print_error


if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


# Helper functions for determining image type
def is_base64_png(s):
    """Determines if a serialized string represents a base644 encoded PNG"""
    if type(s) is not str:
        return False

    try:
        s += "=" * ((4 - len(s) % 4) % 4)
        decoded_bytes = base64.b64decode(s, validate=True)
    except (ValueError, binascii.Error):
        return False

    # Check for PNG signature
    png_signature = b"\x89PNG\r\n\x1a\n"
    return decoded_bytes.startswith(png_signature)


def is_base64_jpg(s):
    """Determines if a serialized string represents a base64 encoded JPEG"""
    if type(s) is not str:
        return False

    try:
        s += "=" * ((4 - len(s) % 4) % 4)  # Ensure padding is correct
        decoded_bytes = base64.b64decode(s, validate=True)
    except (ValueError, binascii.Error):
        return False

    # Check for JPG signature (JPEG files start with FF D8)
    jpg_signature = b"\xFF\xD8"
    return decoded_bytes.startswith(jpg_signature)


class LLMEngineTNE(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("OPENAI_API_KEY environment variable is missing from .env")
            sys.exit()

        api_base = "http://chatapi.app.tne.ai/v1"
        self.client = OpenAI(api_key=key, base_url=api_base)
        self.async_client = AsyncOpenAI(api_key=key, base_url=api_base)

        return

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        model_name = self.llm_model.name()
        temperature = 0

        stream_response = await self.async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            max_tokens=4096,
            temperature=temperature,
        )

        async for chunk in stream_response:
            if chunk.choices[0].finish_reason is None:
                content = chunk.choices[0].delta.content
                yield content


    def __num_tokens(self, text: str):
        model_name = self.llm_model.name()
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def is_within_budget(self, text: str, verbose: bool = False):
        token_budget = self.llm_model.max_token() - 500
        return self.__num_tokens(text) <= token_budget
