"""BGE-M3 embedder — one encode() call produces dense (1024) + sparse lexical weights."""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Module-level singleton
_instance: "BGEM3Embedder | None" = None


def get_m3_embedder() -> "BGEM3Embedder":
    """Get or create the BGE-M3 embedder singleton."""
    global _instance
    if _instance is None:
        _instance = BGEM3Embedder()
    return _instance


class BGEM3Embedder:
    """BGE-M3 embedder producing dense (1024-dim) + sparse lexical weights."""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None

    @property
    def model(self):
        """Lazy load the BGE-M3 model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required. Install with: pip install FlagEmbedding"
            )

        logger.info(f"Loading BGE-M3 model: {self.model_name} (fp16={self.use_fp16})")
        model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        logger.info("BGE-M3 model loaded successfully")
        return model

    def encode(self, texts: List[str]) -> Dict:
        """Encode texts, returning dense vectors and sparse lexical weights.

        Returns:
            {"dense": list[ndarray(1024,)], "lexical": list[dict[int, float]]}
        """
        if not texts:
            return {"dense": [], "lexical": []}

        output = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": list(output["dense_vecs"]),
            "lexical": list(output["lexical_weights"]),
        }

    def encode_single(self, text: str) -> Dict:
        """Encode a single text.

        Returns:
            {"dense": ndarray(1024,), "lexical": dict[int, float]}
        """
        result = self.encode([text])
        return {
            "dense": result["dense"][0],
            "lexical": result["lexical"][0],
        }

    def dense_to_list(self, dense_vec) -> List[float]:
        """Convert numpy dense vector to plain list for Qdrant."""
        if isinstance(dense_vec, np.ndarray):
            return dense_vec.tolist()
        return list(dense_vec)

    @staticmethod
    def lexical_to_sparse(lexical_weights: dict) -> tuple[list[int], list[float]]:
        """Convert lexical weight dict {token_id: weight} to (indices, values) for Qdrant.

        Args:
            lexical_weights: dict mapping token_id (int or str) to weight (float)

        Returns:
            (indices, values) tuple for Qdrant SparseVector
        """
        if not lexical_weights:
            return [], []
        indices = [int(k) for k in lexical_weights.keys()]
        values = [float(v) for v in lexical_weights.values()]
        return indices, values
