"""Jina Reranker for improving search result relevance."""

import logging
from typing import List, Dict, Any

from ..common.config import Settings

logger = logging.getLogger(__name__)


class JinaReranker:
    """Rerank search results using Jina Reranker v2."""

    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        top_k: int = None,
    ):
        """
        Initialize Jina Reranker.

        Args:
            model_name: HuggingFace model name (default: jinaai/jina-reranker-v2-base-multilingual)
            device: Device to use ("auto", "cpu", "cuda", "mps")
            top_k: Number of top results to return after reranking
        """
        settings = Settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device or settings.DEVICE
        self.top_k = top_k or settings.RERANKER_TOP_K

        # Lazy load the model
        self._model = None

    @property
    def model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        """Load the reranker model."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        # Determine device
        device = self._get_device()

        logger.info(f"Loading reranker model: {self.model_name} on {device}")

        model = CrossEncoder(
            self.model_name,
            device=device,
            trust_remote_code=True,
        )

        logger.info(f"Reranker model loaded successfully")
        return model

    def _get_device(self) -> str:
        """Determine the best device to use."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Rerank documents based on relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (default: self.top_k)

        Returns:
            Dict with:
                - scores: List of relevance scores for top_k documents
                - indices: List of original indices of top_k documents
                - documents: List of top_k document texts
        """
        if not documents:
            return {"scores": [], "indices": [], "documents": []}

        top_k = top_k or self.top_k
        top_k = min(top_k, len(documents))

        logger.info(f"Reranking {len(documents)} documents, returning top {top_k}")

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get relevance scores
        scores = self.model.predict(pairs)

        # Sort by score descending
        scored_indices = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )

        # Get top_k results
        top_indices = [idx for idx, _ in scored_indices[:top_k]]
        top_scores = [float(scores[idx]) for idx in top_indices]
        top_documents = [documents[idx] for idx in top_indices]

        return {
            "scores": top_scores,
            "indices": top_indices,
            "documents": top_documents,
        }

    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results that include metadata.

        Args:
            query: The search query
            results: List of result dicts containing text and metadata
            text_key: Key to access text in result dicts
            top_k: Number of top results to return

        Returns:
            List of top_k results with original metadata preserved
        """
        if not results:
            return []

        # Extract texts
        documents = [
            r.get("payload", {}).get(text_key, "") if "payload" in r else r.get(text_key, "")
            for r in results
        ]

        # Rerank
        reranked = self.rerank(query, documents, top_k)

        # Return results with preserved metadata
        return [results[idx] for idx in reranked["indices"]]
