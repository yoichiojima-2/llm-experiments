from langchain.chat_models import init_chat_model


def instantiate_chat(opt: str):
    match opt:
        case "4o-mini":
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        case "deepseek":
            return init_chat_model("deepseek-r1:8b", model_provider="ollama")
        case "llama":
            return init_chat_model("llama3.2", model_provider="ollama")
        case _:
            raise ValueError(f"Unknown model: {opt}")
