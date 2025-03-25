from llm_experiments.llm import create_model


def test_create_model():
    model = create_model()
    assert model is not None

    model = create_model("deepseek-r1")
    assert model is not None

    model = create_model("llama3")
    assert model is not None
