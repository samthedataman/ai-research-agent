from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.logging_config import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-based vector store for RAG capabilities."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "research_docs",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.encoder = SentenceTransformer(model_name)
        self.collection = self.client.get_or_create_collection(collection_name)
        logger.info("vector_store_initialized", collection=collection_name)

    def add_documents(self, docs: list[dict[str, Any]]) -> None:
        if not docs:
            return

        ids = [d["id"] for d in docs]
        texts = [d["text"] for d in docs]
        embeddings = self.encoder.encode(texts).tolist()
        metadatas = [d.get("metadata", {}) for d in docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("documents_added", count=len(docs))

    def query(self, question: str, n_results: int = 5) -> list[dict[str, Any]]:
        query_embedding = self.encoder.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )

        docs = []
        for i in range(len(results["ids"][0])):
            docs.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                }
            )
        return docs

    def get_context(self, question: str, n_results: int = 5) -> str:
        docs = self.query(question, n_results)
        return "\n\n".join(d["text"] for d in docs)

    @property
    def count(self) -> int:
        return self.collection.count()
