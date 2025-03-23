from langchain.chat_models import init_chat_model


def create_model(model="gpt-4o-mini"):
    match model:
        case "gpt-4o-mini":
            return init_chat_model("gpt-4o-mini", model_provider="openai")
        case _:
            raise ValueError(f"unknown model: {model}")
