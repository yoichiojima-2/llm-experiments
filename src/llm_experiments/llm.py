from langchain.chat_models import init_chat_model


def create_model(model="gpt-4o-mini"):
    match model:
        case "gpt-4o-mini":
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        case "gpt-o3-mini":
            return init_chat_model("o3-mini-2025-01-31", model_provider="openai")
        case "deepseek-r1":
            return init_chat_model("deepseek-r1:latest", model_provider="ollama")
        case "llama3":
            return init_chat_model("llama3:latest", model_provider="ollama")
        case _:
            raise ValueError(f"unknown model: {model}")
