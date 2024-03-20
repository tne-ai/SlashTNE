from __future__ import annotations

import os
import aiohttp
import requests

from typing import TYPE_CHECKING, List, AsyncGenerator
from slashgpt.llms.engine.base import LLMEngineBase

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


API_BASE = "https://api-inference.huggingface.co/models"
RESP_503 = "Waiting for model to load into memory..."


def parse_hf_response(hf_response):
    try:
        response_str = hf_response[0].get("generated_text")
        return response_str
    except IndexError:
        return "Malformed response from HuggingFace inference endpoint. Please try again later."


class LLMEngineHF(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

        self.model_name = self.llm_model.name()
        self.url = os.path.join(API_BASE, self.model_name)
        self.headers = {"Authorization": f"Bearer {key}"}

    async def query(self, payload) -> AsyncGenerator:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json=payload, headers=self.headers) as response:
                # Success case
                if response.status == 200:
                    yield await response.json()
                # Waiting for model to load into memory on inference endpoint
                elif response.status == 503:
                    yield RESP_503
                else:
                    yield "HuggingFace inference failed. Please try again later."

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        payload_str = ""
        for i, message in enumerate(messages):
            message_content = message.get("content")
            if any(char.isalpha() for char in message_content):
                payload_str += f"{message_content.strip()}"
                if i < len(messages) - 1:
                    payload_str += "\n\n"

        payload = {"inputs": payload_str, "return_full_text": "false", "max_new_tokens": 250}
        async for chunk in self.query(payload):
            if type(chunk) is str:
                yield "\n"
                yield chunk
                yield "\n"
                if chunk == RESP_503:
                    payload.update({"wait_for_model": "true"})
                    async for c in self.query(payload):
                        yield parse_hf_response(c)
                break
            else:
                yield "\n"
                yield parse_hf_response(chunk)
                yield "\n"
