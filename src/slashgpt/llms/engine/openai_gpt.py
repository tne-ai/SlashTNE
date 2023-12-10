from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List, AsyncGenerator

import openai
from asyncer import asyncify  # For async OpenAI call
import tiktoken  # for counting tokens

from slashgpt.function.function_call import FunctionCall
from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_debug, print_error


if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


class LLMEngineOpenAIGPT(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("OPENAI_API_KEY environment variable is missing from .env")
            sys.exit()
        openai.api_key = key

        # Override default openai endpoint for custom-hosted models
        api_base = llm_model.get_api_base()
        if api_base:
            openai.api_base = api_base

        return

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        model_name = self.llm_model.name()
        temperature = manifest.temperature()
        functions = manifest.functions()
        stream = manifest.stream()
        num_completions = manifest.num_completions()
        max_tokens = manifest.max_tokens()

        params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            "n": num_completions,
            "max_tokens": max_tokens,
        }
        if functions:
            params["functions"] = functions

        if not stream:
            # Make a non-streaming API call
            response = await asyncify(openai.ChatCompletion.create)(**params)
            answer = response["choices"][0]["message"]
            res = answer["content"]
            role = answer["role"]

            function_call = None
            if functions is not None and answer.get("function_call") is not None:
                function_call = FunctionCall(answer.get("function_call"), manifest)

                if res and function_call is None:
                    function_call = self._extract_function_call(messages[-1], manifest, res, True)

            yield role, res, function_call
        else:
            async for response in openai.ChatCompletion.create(**params):
                for item in await self._process_response(response, functions, manifest, messages):
                    yield item

    def __num_tokens(self, text: str):
        model_name = self.llm_model.name()
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def is_within_budget(self, text: str, verbose: bool = False):
        token_budget = self.llm_model.max_token() - 500
        return self.__num_tokens(text) <= token_budget
