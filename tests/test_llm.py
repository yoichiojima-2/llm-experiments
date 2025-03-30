from llm_experiments.llm import create_model


def test_default_model():
    model = create_model()
    assert model.invoke("test") is not None


def test_deepseek():
    model = create_model("deepseek-r1")
    assert model.invoke("test") is not None


def test_llama():
    model = create_model("llama3")
    assert model.invoke("test") is not None


def test_gpt_4o_mini():
    model = create_model("gpt-4o-mini")
    assert model.invoke("test") is not None


def test_gpt_o3_mini():
    model = create_model("gpt-o3-mini")
    assert model.invoke("test") is not None
