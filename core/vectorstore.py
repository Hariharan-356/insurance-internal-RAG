from langchain_chroma import Chroma
from chromadb import PersistentClient

from config import settings
from core.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    client = PersistentClient(path=settings.chroma_dir)
    collection_name = settings.collection_name

    try:
        collection = client.get_collection(collection_name)
        if collection.count() > 0:
            sample = collection.peek(limit=1)
            embeddings = sample.get("embeddings") or []
            if embeddings and len(embeddings[0]) != len(get_embeddings().embed_query("probe")):
                client.delete_collection(collection_name)
                print(f"Rebuilt Chroma collection '{collection_name}' because the embedding dimension changed.")
    except Exception:
        pass

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine_sim
    )
