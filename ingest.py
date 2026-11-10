"""Load documents -> RecursiveCharacterTextSplitter -> embed -> Chroma.

Access level comes from the folder:  data/documents/<public|internal|confidential|restricted>/...
Files in unknown folders are SKIPPED (fail closed).

Usage:  python ingest.py
"""
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ACCESS_LEVELS, settings


def load_documents(root: Path):
    docs = []
    if not root.exists():
        return docs
    for level_dir in sorted(p for p in root.iterdir()):
        if not level_dir.is_dir():
            print(f"[skip] file '{level_dir.name}' is not in an access level folder")
            continue
        level = level_dir.name.lower()
        if level not in ACCESS_LEVELS:
            print(f"[skip] folder '{level_dir.name}' is not a valid access level")
            continue
        for f in sorted(level_dir.rglob("*")):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in (".txt", ".md"):
                loaded = TextLoader(str(f), encoding="utf-8").load()
            elif ext == ".pdf":
                loaded = PyPDFLoader(str(f)).load()
            else:
                continue
            for d in loaded:
                page = d.metadata.get("page", 0)
                d.metadata.update(
                    source=str(f.relative_to(root)).replace("\\", "/"),
                    page=page,
                    access_level=level,
                    access_rank=ACCESS_LEVELS[level],
                )
            docs.extend(loaded)
    return docs


def main():
    root = Path(settings.documents_dir)
    docs = load_documents(root)
    if not docs:
        raise SystemExit(f"No documents found under {root}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        key = f"{c.metadata['source']}|{c.metadata.get('page', 0)}|{i}|{c.page_content[:60]}"
        c.metadata["chunk_id"] = hashlib.sha1(key.encode()).hexdigest()
    print(f"{len(docs)} source docs/pages -> {len(chunks)} chunks")

    from core.vectorstore import get_vector_store

    store = get_vector_store()
    try:
        store.delete_collection()          # rebuild from scratch
    except Exception:
        pass
    store = get_vector_store()
    B = 32
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        print(f"  indexed {min(i + B, len(chunks))}/{len(chunks)}")
    print("Done.")


if __name__ == "__main__":
    main()
