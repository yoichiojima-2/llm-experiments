from llm_experiments import embedding


def test_mxbai_embed_large():
    model = embedding.instantiate_embedding("mxbai-embed-large")
    assert model.embed_query("test")
