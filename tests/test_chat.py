from llm_experiments import chat


def test_4o_mini():
    model = chat.instantiate_chat("gpt-4o-mini")
    assert model.invoke("test").content


def test_r1():
    deepseek_r1 = chat.instantiate_chat("deepseek-r1:8b")
    assert deepseek_r1.invoke("test")


