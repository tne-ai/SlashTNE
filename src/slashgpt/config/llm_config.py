from slashgpt.llms.engine.openai_gpt import LLMEngineOpenAIGPT

llm_models = {
    "gpt3": {
        "engine_name": "openai-gpt",
        "model_name": "gpt-3.5-turbo-0613",
        "api_key": "OPENAI_API_KEY",
        "max_token": 4096,
    },
    "gpt31": {
        "engine_name": "openai-gpt",
        "model_name": "gpt-3.5-turbo-16k-0613",
        "api_key": "OPENAI_API_KEY",
        "max_token": 4096 * 4,
    },
    "gpt4": {
        "engine_name": "openai-gpt",
        "model_name": "gpt-4-0613",
        "api_key": "OPENAI_API_KEY",
        "max_token": 4096,
    },
    "llama2": {
        "engine_name": "replicate",
        "model_name": "llama2",
        "api_key": "REPLICATE_API_TOKEN",
        "replicate_model": "a16z-infra/llama7b-v2-chat:a845a72bb3fa3ae298143d13efa8873a2987dbf3d49c293513cd8abf4b845a83",
    },
    "local_llama2": {
        "engine_name": "hosted",
        "model_name": "local_llama2",
        "api_key": "KSERVE_API_KEY",
        "header_api_key": "x-api-key",
        "url": "https://llama2-7b-chat.staging.kubeflow.platform.nedra.app/v2/models/llama2-7b-chat/infer",
    },
    "local_embed": {
        "engine_name": "hosted",
        "model_name": "local_embed",
        "api_key": "KSERVE_API_KEY",
        "header_api_key": "x-api-key",
        "url": "https://bge-base-en.staging.kubeflow.platform.nedra.app/v2/models/bge-base-en/infer",
    },
    "llama270": {
        "engine_name": "replicate",
        "model_name": "llama270",
        "api_key": "REPLICATE_API_TOKEN",
        "replicate_model": "replicate/llama70b-v2-chat:2d19859030ff705a87c746f7e96eea03aefb71f166725aee39692f1476566d48",
    },
    "vicuna": {
        "engine_name": "replicate",
        "model_name": "vicuna",
        "api_key": "REPLICATE_API_TOKEN",
        "replicate_model": "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
    },
    "palm": {
        "engine_name": "palm",
        "model_name": "palm",
        "api_key": "GOOGLE_PALM_KEY",
    },
    "gpt2": {
        "engine_name": "from_pretrained",
        "model_name": "rinna/japanese-gpt2-xsmall",
        "max_token": 4096,
    },
    "rinna": {
        "engine_name": "from_pretrained-rinna",
        "model_name": "rinna/bilingual-gpt-neox-4b-instruction-sft",
        "max_token": 4096,
    },
}

llm_engine_configs = {
    "openai-gpt": LLMEngineOpenAIGPT,
    "replicate": {
        "module_name": "slashgpt.llms.engine.replicate",
        "class_name": "LLMEngineReplicate",
    },
    "palm": {
        "module_name": "slashgpt.llms.engine.palm",
        "class_name": "LLMEnginePaLM",
    },
    "hosted": {
        "module_name": "slashgpt.llms.engine.hosted",
        "class_name": "LLMEngineHosted",
    },
    "from_pretrained": {
        "module_name": "slashgpt.plugins.engine.from_pretrained",
        "class_name": "LLMEngineFromPretrained",
    },
    "from_pretrained-rinna": {
        "module_name": "slashgpt.plugins.engine.from_pretrained2",
        "class_name": "LLMEngineFromPretrained2",
    },
}
