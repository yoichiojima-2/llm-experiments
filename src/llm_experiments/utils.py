import json


def get_last_message(messages):
    return messages["messages"][-1]


async def print_stream(stream):
    async for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


def parse_base_model(base_model):
    return json.dumps(base_model.model_json_schema()["properties"], indent=2)
