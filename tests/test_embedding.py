from llm_experiments import embedding


def test_mxbai_embed_large():
    model = embedding.instantiate_embedding("mxbai-embed-large")
    assert model.embed_query("test")

def test_nomic_embed_text():
    model = embedding.instantiate_embedding("nomic-embed-text")
    assert model.embed_query("test")
