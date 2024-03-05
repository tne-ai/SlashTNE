from __future__ import annotations

from typing import TYPE_CHECKING, List, AsyncGenerator

try:
    import replicate
except ImportError:
    print("no replicate. pip install replicate")

from slashgpt.llms.engine.base import LLMEngineBase
from slashgpt.utils.print import print_debug

if TYPE_CHECKING:
    from slashgpt.llms.model import LlmModel
    from slashgpt.manifest import Manifest


default_model = "a16z-infra/llama7b-v2-chat:a845a72bb3fa3ae298143d13efa8873a2987dbf3d49c293513cd8abf4b845a83"


class LLMEngineReplicate(LLMEngineBase):
    def __init__(self, llm_model: LlmModel):
        super().__init__(llm_model)

    async def chat_completion(self, messages: List[dict], manifest: Manifest, verbose: bool) -> AsyncGenerator:
        temperature = manifest.temperature()
        replicate_model = self.llm_model.name()

        # Replicate doesn't allow a temperature of 0
        if temperature == 0:
            temperature = 0.01

        prompt, system_prompt = "", ""
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content")
            if message.get("role") == "user":
                prompt = message.get("content")

        replicate_input = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
        }
        async for event in await replicate.async_stream(replicate_model, input=replicate_input):
            yield str(event)

        """
        function_call = self._extract_function_call(messages[-1], manifest, res) if manifest.functions() is not None else None

        role = "assistant"
        if function_call:
            yield (role, None, function_call, None)
        else:
            yield (role, res, None, None)
        """
