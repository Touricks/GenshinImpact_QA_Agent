"""Chinese BM25 sparse vector generator using jieba tokenization."""

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_instance: "ChineseBM25Embedder | None" = None


def get_bm25_embedder(auto_load: bool = True) -> Optional["ChineseBM25Embedder"]:
    """Get or create the BM25 embedder singleton.

    Args:
        auto_load: If True, try to load from default IDF path.

    Returns:
        ChineseBM25Embedder or None if IDF not yet fitted/saved.
    """
    global _instance
    if _instance is not None:
        return _instance

    from ..config.settings import settings

    embedder = ChineseBM25Embedder(user_dict_path=settings.BM25_USER_DICT_PATH)

    if auto_load and settings.BM25_IDF_PATH.exists():
        embedder.load(settings.BM25_IDF_PATH)
        _instance = embedder
        return _instance

    # Not fitted yet — return unfitted instance (caller must fit or skip)
    return embedder


class ChineseBM25Embedder:
    """jieba + TF-IDF BM25 sparse vector generator for Chinese text."""

    def __init__(
        self,
        user_dict_path: Optional[Path] = None,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.user_dict_path = user_dict_path

        # Vocabulary: term -> index
        self.vocab: dict[str, int] = {}
        # IDF scores: term -> idf
        self.idf: dict[str, float] = {}
        # Average document length
        self.avg_dl: float = 0.0
        # Total documents used for fitting
        self.n_docs: int = 0
        # Whether the model has been fitted
        self._fitted = False

        self._jieba_initialized = False

    def _init_jieba(self):
        """Initialize jieba with custom dictionary."""
        if self._jieba_initialized:
            return
        import jieba

        if self.user_dict_path and Path(self.user_dict_path).exists():
            jieba.load_userdict(str(self.user_dict_path))
            logger.info(f"Loaded jieba user dict: {self.user_dict_path}")
        self._jieba_initialized = True

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using jieba, filtering stopwords and short tokens."""
        import jieba

        self._init_jieba()
        # Cut with HMM for better segmentation
        tokens = jieba.lcut(text, HMM=True)
        # Filter: keep tokens length >= 2 (skip single chars, punctuation)
        return [t.strip() for t in tokens if len(t.strip()) >= 2]

    def fit(self, corpus: List[str]):
        """Calculate IDF scores and build vocabulary from corpus.

        Args:
            corpus: List of document texts.
        """
        logger.info(f"Fitting BM25 on {len(corpus)} documents...")
        self._init_jieba()

        doc_freq: Counter = Counter()
        total_length = 0

        for text in corpus:
            tokens = self._tokenize(text)
            total_length += len(tokens)
            # Count unique terms per document
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freq[term] += 1

        self.n_docs = len(corpus)
        self.avg_dl = total_length / max(self.n_docs, 1)

        # Build vocabulary (sorted by frequency for deterministic indices)
        sorted_terms = sorted(doc_freq.keys(), key=lambda t: (-doc_freq[t], t))
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        # Calculate IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

        self._fitted = True
        logger.info(
            f"BM25 fitted: {len(self.vocab)} terms, avg_dl={self.avg_dl:.1f}"
        )

    def transform(self, text: str) -> tuple[list[int], list[float]]:
        """Transform text to BM25 sparse vector.

        Args:
            text: Input text.

        Returns:
            (indices, values) — Qdrant SparseVector format.
            Returns ([], []) if not fitted.
        """
        if not self._fitted:
            return [], []

        tokens = self._tokenize(text)
        if not tokens:
            return [], []

        tf_counts = Counter(tokens)
        doc_len = len(tokens)

        indices = []
        values = []

        for term, tf in tf_counts.items():
            if term not in self.vocab:
                continue
            idf = self.idf.get(term, 0.0)
            if idf <= 0:
                continue

            # BM25 score: IDF * (k1+1)*tf / (tf + k1*(1-b + b*dl/avg_dl))
            numerator = (self.k1 + 1) * tf
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            bm25_weight = idf * numerator / denominator

            indices.append(self.vocab[term])
            values.append(bm25_weight)

        return indices, values

    def save(self, path: Path):
        """Serialize vocabulary + IDF to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab": self.vocab,
            "idf": self.idf,
            "avg_dl": self.avg_dl,
            "n_docs": self.n_docs,
            "k1": self.k1,
            "b": self.b,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"BM25 model saved to {path} ({len(self.vocab)} terms)")

    def load(self, path: Path):
        """Load vocabulary + IDF from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.avg_dl = data["avg_dl"]
        self.n_docs = data["n_docs"]
        self.k1 = data.get("k1", 1.2)
        self.b = data.get("b", 0.75)
        self._fitted = True
        logger.info(f"BM25 model loaded from {path} ({len(self.vocab)} terms)")

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def generate_jieba_dict(output_path: Path) -> int:
    """Generate jieba user dictionary from alias config files.

    Reads all alias JSON files from src/common/config/aliases/ and writes
    entity names as high-frequency entries to ensure jieba doesn't split them.

    Args:
        output_path: Path to write the jieba user dict.

    Returns:
        Number of entries written.
    """
    aliases_dir = Path(__file__).parent.parent / "config" / "aliases"
    entries = set()

    for json_file in aliases_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            for key, info in data.items():
                # Add canonical name
                canonical = info.get("canonical_zh", key)
                if len(canonical) >= 2:
                    entries.add(canonical)
                # Add all aliases
                for alias in info.get("aliases", []):
                    if len(alias) >= 2:
                        entries.add(alias)
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse {json_file}: {e}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for entry in sorted(entries):
        # Format: word frequency part-of-speech
        # 999999 ensures jieba won't split the word
        lines.append(f"{entry} 999999 nr")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Generated jieba user dict: {output_path} ({len(lines)} entries)")
    return len(lines)
