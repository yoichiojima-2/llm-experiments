from langchain_ollama import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.chat_models import init_chat_model


def instantiate_chat(opt):
    match opt:
        case "4o-mini":
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        case "deepseek":
            return init_chat_model("deepseek-r1:8b", model_provider="ollama")
        case "llama":
            return init_chat_model("llama3.2", model_provider="ollama")
        case _:
            raise ValueError(f"Unknown model: {opt}")


def instantiate_embedding(opt):
    match opt:
        case "mxbai-embed-large":
            return OllamaEmbeddings(model="mxbai-embed-large")
        case "nomic-embed-text":
            return OllamaEmbeddings(model="nomic-embed-text")
        case _:
            raise ValueError(f"Unknown embedding model: {opt}")


def instantiate_vision(opt):
    match opt:
        case "llava":
            return OllamaLLM(model="llava")
        case _:
            raise ValueError(f"Unknown embedding model: {opt}")
