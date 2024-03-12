from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, AsyncGenerator

import google.generativeai as genai

from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_error


if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


class LLMEngineGoogle(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("ANTHROPIC_API_KEY environment variable is missing from .env")
            sys.exit()

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel("gemini-pro")

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        # Parse the format to account for Anthropic API format
        message = {"role": "user"}
        content = ""
        for m in messages:
            msg_content = m["content"]
            content += msg_content
        message["parts"] = [content]

        config = genai.types.GenerationConfig(candidate_count=1, max_output_tokens=4096)
        params = {"generation_config": config, "stream": True}

        response = await self.model.generate_content_async(message, **params)
        async for chunk in response:
            yield chunk.text
