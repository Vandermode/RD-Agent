"""Keep RD-Agent's chat backend and use local embeddings for its knowledge base."""

import os
from functools import lru_cache
from pathlib import Path

from fastembed import TextEmbedding

from rdagent.oai.backend.litellm import LiteLLMAPIBackend


EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def embedding_model() -> TextEmbedding:
    return TextEmbedding(
        model_name=EMBEDDING_MODEL,
        cache_dir=str(Path("git_ignore_folder/reproduction/embedding-models").resolve()),
        threads=int(os.environ.get("RDAGENT_EMBEDDING_THREADS", "2")),
    )


class LocalEmbeddingBackend(LiteLLMAPIBackend):
    def _create_embedding_inner_function(self, input_content_list: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in embedding_model().embed(input_content_list)]
