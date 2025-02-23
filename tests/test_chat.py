from llm_experiments import chat


def test_4o_mini():
    model = chat.instantiate_chat("gpt-4o-mini")
    res = model.invoke("test").content
    print(res)
    assert res


def test_r1():
    model = chat.instantiate_chat("deepseek-r1:8b")
    res = model.invoke("test")
    print(res)
    assert res


def test_llama():
    model = chat.instantiate_chat("llama3.2")
    res = model.invoke("test")
    print(res)
    assert res
