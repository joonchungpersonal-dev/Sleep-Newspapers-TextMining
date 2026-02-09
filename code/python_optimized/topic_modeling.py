"""
Topic modelling engine: classical LDA (gensim) and neural BERTopic.

Mirrors the R STM pipeline -- estimating topic-word and document-topic
distributions, extracting top terms, computing prevalence over time,
and exporting representative documents.

Key differences from the R version
-----------------------------------
* LDA via gensim replaces R's ``stm::stm()``.  LDA does not natively
  support prevalence covariates, so temporal trends are computed
  post-hoc with LOESS / linear regression rather than b-spline
  ``estimateEffect``.
* BERTopic is offered as a modern, transformer-based alternative that
  can discover the number of topics automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from config import (
    RANDOM_SEED,
    TOPIC_MODEL as CFG,
    TOPICS_OF_INTEREST,
    TOP_DOCS_DIR,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class TopicModelResult:
    """Unified container holding results from either LDA or BERTopic."""

    model_type: str                      # "lda" or "bertopic"
    n_topics: int
    topic_term_matrix: np.ndarray        # (K x V) -- prob of word given topic
    doc_topic_matrix: np.ndarray         # (D x K) -- prob of topic given doc
    vocab: np.ndarray                    # (V,)
    model: Any = None                    # the underlying model object
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- convenience methods ------------------------------------------------

    def top_terms(self, topic_id: int, n: int = CFG.n_top_terms) -> pd.DataFrame:
        """Return the *n* highest-probability terms for *topic_id*.

        Parameters
        ----------
        topic_id : int
            Zero-based topic index.
        n : int
            Number of terms to return.

        Returns
        -------
        pd.DataFrame
            Columns: ``term``, ``probability``.
        """
        row = self.topic_term_matrix[topic_id]
        top_idx = row.argsort()[::-1][:n]
        return pd.DataFrame({
            "term": self.vocab[top_idx],
            "probability": row[top_idx],
        })

    def all_top_terms(self, n: int = CFG.n_top_terms) -> pd.DataFrame:
        """Top terms for every topic in long-form."""
        frames = []
        for k in range(self.n_topics):
            df = self.top_terms(k, n)
            df.insert(0, "topic", k)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def top_documents(
        self,
        topic_id: int,
        corpus_df: pd.DataFrame,
        n: int = CFG.n_top_docs,
    ) -> pd.DataFrame:
        """Return the *n* documents with highest loading on *topic_id*.

        Parameters
        ----------
        topic_id : int
            Zero-based topic index.
        corpus_df : pd.DataFrame
            Must contain at least a ``text`` column and ideally ``year``.
        n : int
            Number of documents.

        Returns
        -------
        pd.DataFrame
            Sorted by descending topic proportion.
        """
        proportions = self.doc_topic_matrix[:, topic_id]
        top_idx = proportions.argsort()[::-1][:n]
        result = corpus_df.iloc[top_idx].copy()
        result["topic_proportion"] = proportions[top_idx]
        result["rank"] = range(1, len(result) + 1)
        return result.reset_index(drop=True)

    def topic_prevalence_by_year(
        self,
        years: np.ndarray,
    ) -> pd.DataFrame:
        """Compute mean topic proportion per year for every topic.

        Parameters
        ----------
        years : array-like of int
            One entry per document.

        Returns
        -------
        pd.DataFrame
            Columns: ``year``, ``topic``, ``mean_proportion``.
        """
        years = np.asarray(years)
        records = []
        unique_years = np.sort(np.unique(years))
        for yr in unique_years:
            mask = years == yr
            means = self.doc_topic_matrix[mask].mean(axis=0)
            for k in range(self.n_topics):
                records.append({"year": int(yr), "topic": k,
                                "mean_proportion": float(means[k])})
        return pd.DataFrame(records)


# ============================================================================
# LDA via gensim
# ============================================================================

def train_lda(
    dtm: csr_matrix,
    vocab: np.ndarray,
    n_topics: int = CFG.n_topics_lda,
    seed: int = RANDOM_SEED,
) -> TopicModelResult:
    """Train an LDA model using gensim and return a :class:`TopicModelResult`.

    Parameters
    ----------
    dtm : sparse matrix (D x V)
    vocab : array of str
    n_topics : int
    seed : int

    Returns
    -------
    TopicModelResult
    """
    from gensim.corpora import Dictionary
    from gensim.matutils import Sparse2Corpus
    from gensim.models import LdaMulticore

    logger.info("Training LDA with K=%d ...", n_topics)

    # gensim expects a corpus in BoW format and a Dictionary
    corpus_bow = Sparse2Corpus(dtm, documents_columns=False)

    # Build a gensim Dictionary from the vocabulary
    id2word = dict(enumerate(vocab))

    model = LdaMulticore(
        corpus=corpus_bow,
        id2word=id2word,
        num_topics=n_topics,
        passes=CFG.lda_passes,
        iterations=CFG.lda_iterations,
        alpha=CFG.lda_alpha,
        eta=CFG.lda_eta,
        chunksize=CFG.lda_chunksize,
        eval_every=CFG.lda_eval_every,
        random_state=seed,
        per_word_topics=False,
    )

    # Extract matrices
    topic_term = model.get_topics()  # (K x V)

    # Document-topic matrix -- needs inference
    doc_topic = _infer_doc_topics_gensim(model, corpus_bow, n_topics)

    logger.info("LDA training complete.  Perplexity (held-out): computing...")

    return TopicModelResult(
        model_type="lda",
        n_topics=n_topics,
        topic_term_matrix=topic_term,
        doc_topic_matrix=doc_topic,
        vocab=vocab,
        model=model,
    )


def _infer_doc_topics_gensim(model, corpus_bow, n_topics: int) -> np.ndarray:
    """Batch-infer document-topic proportions and return a dense matrix."""
    from gensim.matutils import Sparse2Corpus

    rows = []
    for bow in corpus_bow:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        row = np.zeros(n_topics)
        for tid, prob in dist:
            row[tid] = prob
        rows.append(row)
    return np.vstack(rows)


# ============================================================================
# BERTopic (neural)
# ============================================================================

def train_bertopic(
    raw_texts: List[str],
    *,
    min_topic_size: int = CFG.bertopic_min_topic_size,
    embedding_model: str = CFG.bertopic_embedding_model,
    seed: int = RANDOM_SEED,
    nr_topics: int | str = CFG.bertopic_nr_topics,
) -> TopicModelResult:
    """Train a BERTopic model and return a :class:`TopicModelResult`.

    Parameters
    ----------
    raw_texts : list of str
        Original (uncleaned) documents -- BERTopic handles its own
        tokenization via sentence-transformers.
    min_topic_size : int
        Minimum cluster size.
    embedding_model : str
        Sentence-transformer model name.
    seed : int
    nr_topics : int or ``"auto"``
        If int, BERTopic will merge until this many topics remain.

    Returns
    -------
    TopicModelResult
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    logger.info("Training BERTopic (embedding=%s) ...", embedding_model)

    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                       metric="cosine", random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=min_topic_size,
                             prediction_data=True)

    bt = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics if isinstance(nr_topics, int) else None,
        top_n_words=CFG.bertopic_top_n_words,
        verbose=True,
    )

    topics, probs = bt.fit_transform(raw_texts)
    topic_info = bt.get_topic_info()
    # Remove outlier topic (-1)
    real_topics = topic_info[topic_info.Topic != -1]
    n_topics = len(real_topics)

    # Build topic-term matrix
    vocab_set: set = set()
    for tid in real_topics.Topic:
        for word, _ in bt.get_topic(tid):
            vocab_set.add(word)
    vocab_arr = np.array(sorted(vocab_set))
    word2idx = {w: i for i, w in enumerate(vocab_arr)}

    tt = np.zeros((n_topics, len(vocab_arr)))
    for i, tid in enumerate(real_topics.Topic):
        for word, score in bt.get_topic(tid):
            if word in word2idx:
                tt[i, word2idx[word]] = max(score, 0.0)
    # Normalise rows to sum to 1
    row_sums = tt.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    tt = tt / row_sums

    # Document-topic matrix (from probs)
    if probs is not None and probs.ndim == 2:
        dt = probs
    else:
        # Fallback: one-hot from hard assignments
        dt = np.zeros((len(raw_texts), n_topics))
        for i, t in enumerate(topics):
            if 0 <= t < n_topics:
                dt[i, t] = 1.0

    logger.info("BERTopic found %d topics", n_topics)

    return TopicModelResult(
        model_type="bertopic",
        n_topics=n_topics,
        topic_term_matrix=tt,
        doc_topic_matrix=dt,
        vocab=vocab_arr,
        model=bt,
        extra={"topics_assigned": topics, "topic_info": topic_info},
    )


# ============================================================================
# Export helpers
# ============================================================================

def export_top_documents(
    result: TopicModelResult,
    corpus_df: pd.DataFrame,
    topic_ids: Optional[Dict[int, str]] = None,
    n_docs: int = CFG.top_docs_for_export,
    output_dir: Path = TOP_DOCS_DIR,
) -> None:
    """Write per-topic CSV and plain-text files of representative documents.

    Mirrors the R ``extract_top_documents`` function and the batch loop
    that creates ``top_documents/topic_*_top100.csv`` etc.

    Parameters
    ----------
    result : TopicModelResult
    corpus_df : pd.DataFrame
    topic_ids : dict mapping topic index -> human label, or None for all
    n_docs : int
    output_dir : Path
    """
    if topic_ids is None:
        topic_ids = {k: f"Topic_{k}" for k in range(result.n_topics)}

    output_dir.mkdir(parents=True, exist_ok=True)

    for tid, label in topic_ids.items():
        if tid >= result.n_topics:
            logger.warning("Topic %d does not exist (K=%d), skipping.",
                           tid, result.n_topics)
            continue

        top = result.top_documents(tid, corpus_df, n=n_docs)
        safe_label = label.replace(" ", "_").replace("/", "-")

        # CSV
        csv_path = output_dir / f"topic_{tid:02d}_{safe_label}_top{n_docs}.csv"
        top.to_csv(csv_path, index=False)

        # Readable text
        txt_path = output_dir / f"topic_{tid:02d}_{safe_label}_top{n_docs}.txt"
        with open(txt_path, "w") as fh:
            fh.write(f"TOP {n_docs} DOCUMENTS FOR TOPIC {tid}: "
                     f"{label.upper()}\n")
            fh.write("=" * 80 + "\n\n")
            for _, row in top.iterrows():
                fh.write(f"RANK: {row['rank']}  |  "
                         f"PROPORTION: {row['topic_proportion']:.4f}\n")
                if "year" in row:
                    fh.write(f"YEAR: {row['year']}\n")
                fh.write("TEXT:\n")
                fh.write(str(row.get("text", ""))[:2000] + "\n")
                fh.write("-" * 80 + "\n\n")

    logger.info("Exported top documents to %s", output_dir)


def topic_term_summary(
    result: TopicModelResult,
    n_terms: int = 5,
) -> pd.DataFrame:
    """Wide-format table: one row per topic, columns for top-N terms.

    Analogous to R's ``sageLabels`` probability matrix export.
    """
    rows = []
    for k in range(result.n_topics):
        top = result.top_terms(k, n=n_terms)
        row = {"topic": k}
        for i, (_, r) in enumerate(top.iterrows(), 1):
            row[f"term_{i}"] = r["term"]
            row[f"prob_{i}"] = round(r["probability"], 6)
        rows.append(row)
    return pd.DataFrame(rows)
