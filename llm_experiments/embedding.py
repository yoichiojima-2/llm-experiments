from langchain_ollama import OllamaEmbeddings


def instantiate_embedding(opt: str):
    match opt:
        case "mxbai-embed-large":
            return OllamaEmbeddings(model="mxbai-embed-large")
        case "nomic-embed-text":
            return OllamaEmbeddings(model="nomic-embed-text")
        case _:
            raise ValueError(f"Unknown embedding model: {opt}")
