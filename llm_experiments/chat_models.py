from langchain.chat_models import init_chat_model
from langchain_ollama.llms import OllamaLLM


def instantiate_chat_model(opt: str):
    match opt:
        case "gpt-4o-mini":
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        case "deepseek-r1:8b":
            return OllamaLLM(model="deepseek-r1:8b")
        case _:
            raise ValueError(f"Unknown model: {opt}")
