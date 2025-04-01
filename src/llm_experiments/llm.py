from langchain.chat_models import init_chat_model


def create_model(model="4o-mini", *a, **kw):
    match model:
        case "4o-mini":
            return init_chat_model("openai:gpt-4o-mini", *a, **kw)
        case "o3-mini":
            return init_chat_model("openai:o3-mini-2025-01-31", *a, **kw)
        case "gemini":
            return init_chat_model("google_genai:gemini-2.0-flash", *a, **kw)
        case "deepseek":
            return init_chat_model("ollama:deepseek-r1:latest", *a, **kw)
        case "llama":
            return init_chat_model("ollama:llama3:latest", *a, **kw)
        case _:
            raise ValueError(f"unknown model: {model}")
