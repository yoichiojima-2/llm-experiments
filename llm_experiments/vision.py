from langchain_ollama import OllamaLLM


def instantiate_vision(opt: str):
    match opt:
        case "llava":
            return OllamaLLM(model="llava")
        case _:
            raise ValueError(f"Unknown embedding model: {opt}")
