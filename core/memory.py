from typing import Any

import chromadb

from core.config import CHROMA_DIR


_COLLECTION_NAME = "idamp_memory"


def get_chroma_client() -> chromadb.PersistentClient:
    """Return the persistent ChromaDB client for agent memory."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(name: str = _COLLECTION_NAME):
    """Get or create a collection for basic document storage."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        embedding_function=None,
    )


def store_document(
    doc_id: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Store or replace a document without interrupting the pipeline."""
    try:
        collection = get_collection()
        payload: dict[str, Any] = {"ids": [doc_id], "documents": [text]}
        if metadata is not None:
            payload["metadatas"] = [metadata]
        collection.upsert(**payload)
    except Exception:
        return


def query_memory(query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Return stored documents; query_text is kept for the public API."""
    del query_text
    try:
        result = get_collection().get(
            limit=n_results,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids", [])
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        return [
            {
                "id": doc_id,
                "document": document,
                "metadata": metadatas[index] if index < len(metadatas) else None,
            }
            for index, (doc_id, document) in enumerate(zip(ids, documents))
        ]
    except Exception:
        return []
