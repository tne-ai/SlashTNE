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


class LLMEngineOpenAIGPT(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)
        key = llm_model.get_api_key_value()
        if key == "":
            print_error("OPENAI_API_KEY environment variable is missing from .env")
            sys.exit()

        api_base = llm_model.get_api_base()
        if api_base:
            self.client = OpenAI(api_key=key, base_url=api_base)
            self.async_client = AsyncOpenAI(api_key=key, base_url=api_base)
        else:
            self.client = OpenAI(api_key=key)
            self.async_client = AsyncOpenAI(api_key=key)

        return

    def image_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> str:
        params = {
            "model": manifest.model().get("model_name"),
            # Prompt taken from the OpenAI guide
            "prompt": f"I NEED to test how the tool works with extremely simple prompts. DO NOT add any detail, just use it AS-IS\n{messages}",
            "size": "1024x1792",
            "quality": "standard",
            "n": 1,
        }
        response = self.client.images.generate(**params)
        if response:
            image_url = response.data[0].url
            return image_url

        return None

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        # For now, we are trying to force as much determinism as possible
        model_name = self.llm_model.name()
        functions = manifest.functions()
        stream = manifest.stream()
        num_completions = manifest.num_completions()
        images = manifest.images()
        seed = manifest.manifest().get("seed")
        top_p = 0.00000000000001
        # max_tokens = manifest.max_tokens()

        content = []
        # Check for images
        detected_img = False
        img_text_content = {"type": "text"}

        # First parse the messages to see if there are any image URLs
        for message in messages:
            # System prompt becomes text input to the model
            if message.get("role") == "system":
                img_text_content["text"] = message.get("content")
                content.append(img_text_content)
            # Now we extract the image url from the user content
            elif message.get("role") == "user":
                url_ind = message.get("content").find("https://")
                if url_ind >= 0:
                    detected_img = True
                    img_url = message.get("content")[url_ind:].split(" ")[0]
                    # TODO(lucas): Expose detail parameter in manifest
                    content.append({"type": "image_url", "image_url": {"url": img_url, "detail": "low"}})

        # Next, check if any base64 encoded images have been passed to us
        if images:
            detected_img = True
            for image in images:
                if is_base64_png(image):
                    img_url = f"data:image/png;base64, {image}"
                elif is_base64_jpg(image):
                    img_url = f"data:image/jpg;base64, {image}"
                else:
                    raise NotImplementedError
                content.append({"type": "image_url", "image_url": {"url": img_url, "detail": "auto"}})

        if detected_img:
            if model_name != "gpt-4o":
                raise ValueError(f"Image input not supported for model {model_name}")
            messages = [{"role": "user", "content": content}]
            params = {"model": model_name, "messages": messages, "stream": stream, "top_p": top_p}

        # GPT-o1* models do not support system role
        elif "o1-" in model_name:
            sanitized_messages = []
            for message in messages:
                sanitized_messages.append({"role": "user", "content": message.get("content")})
            messages = sanitized_messages
            # While in beta, some parameters are not supported
            top_p = 1
            stream = False
        else:
            params = {
                "model": model_name,
                "messages": messages,
                "top_p": top_p,
                "seed": seed,
                "stream": stream,
                "n": num_completions,
            }

        if not stream:
            # Make a non-streaming API call
            if model_name == "dall-e-3":
                response = self.async_client.images.generate(**params)
            else:
                if functions:
                    tools_list = []
                    for function in functions:
                        tools_list.append({"type": "function", "function": function})
                    params.update({"tools": tools_list, "tool_choice": "auto"})
                response = await self.async_client.chat.completions.create(**params)

            answer = response.choices[0].message

            res = answer.content

            if answer.tool_calls:
                function_call = FunctionCall(answer.tool_calls[0].function, manifest)
                yield function_call
            else:
                yield res

        else:
            # TODO(lucas): Support streaming and function calls (this only processes the text)
            stream_keys = ["model", "stream", "messages", "top_p", "seed"]
            stream_params = {k: params.get(k) for k in stream_keys}

            if model_name == "gpt-4-vision-preview":
                stream = self.async_client.chat.completions.create(max_tokens=4096, **stream_params)
            else:
                stream = self.async_client.chat.completions.create(**stream_params)

            collected_messages = []
            async for chunk in await stream:
                message = chunk.choices[0].delta.content
                function_call = chunk.choices[0].delta.tool_calls
                if function_call:
                    yield function_call
                if message:
                    collected_messages.append(message)
                    yield message

    def __num_tokens(self, text: str):
        model_name = self.llm_model.name()
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def is_within_budget(self, text: str, verbose: bool = False):
        token_budget = self.llm_model.max_token() - 500
        return self.__num_tokens(text) <= token_budget
