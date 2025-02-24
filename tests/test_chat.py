from llm_experiments import chat


def test_4o_mini():
    model = chat.instantiate_chat("4o-mini")
    res = model.invoke("test").content
    print(res)
    assert res


def test_r1():
    model = chat.instantiate_chat("deepseek")
    res = model.invoke("test")
    print(res)
    assert res


def test_llama():
    model = chat.instantiate_chat("llama")
    res = model.invoke("test")
    print(res)
    assert res
