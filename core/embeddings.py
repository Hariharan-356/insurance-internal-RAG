
"""Qwen3-Embedding using sentence-transformers,
wrapped as a LangChain Embeddings class.
"""

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from config import settings


class Qwen3Embeddings(Embeddings):

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        # Select the configured device.
        target_device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Use FP16 only on CUDA; keep CPU inference stable with float32.
        model_kwargs = {"torch_dtype": torch.float32}

        if target_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        print(f"Loading embedding model: {model_name}")
        print(f"Embedding device: {target_device}")

        try:
            self.model = SentenceTransformer(
                model_name,
                device=target_device,
                model_kwargs=model_kwargs
            )
        except (OSError, ValueError, RuntimeError, MemoryError):
            if model_name != "sentence-transformers/all-MiniLM-L6-v2":
                print(
                    f"Falling back to lighter embedding model because '{model_name}' "
                    "could not be loaded on this machine."
                )
                self.model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    device=target_device,
                    model_kwargs={"torch_dtype": torch.float32}
                )
            else:
                raise

        self.batch_size = batch_size

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:

        try:
            vector = self.model.encode(
                [text],
                prompt_name="query",
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]

        except (ValueError, KeyError, TypeError):

            vector = self.model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False
            )[0]

        return vector.tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Qwen3Embeddings:

    return Qwen3Embeddings(
        settings.embedding_model,
        settings.embedding_device
    )
