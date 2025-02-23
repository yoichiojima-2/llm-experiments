from llm_experiments import chat


def test_4o_mini():
    model = chat.instantiate_chat("gpt-4o-mini")
    assert model.invoke("test").content


def test_r1():
    model = chat.instantiate_chat("deepseek-r1:8b")
    assert model.invoke("test")


def test_llama():
    model = chat.instantiate_chat("llama3.2")
    assert model.invoke("test")
