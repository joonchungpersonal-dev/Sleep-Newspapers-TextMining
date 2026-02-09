"""
Text preprocessing pipeline for newspaper corpus analysis.

Converts raw newspaper text into clean, lemmatized tokens suitable for
topic modelling, frame analysis, and network construction.  Uses spaCy
for linguistic processing and scikit-learn for vocabulary pruning.

Design choices vs. the original R pipeline
------------------------------------------
* R used ``stm::textProcessor`` (stemming + stopword removal).
* Python version uses spaCy lemmatization -- produces more readable topics
  while achieving comparable dimensionality reduction.
* Vocabulary pruning mirrors R's ``prepDocuments(lower.thresh = 500)``
  via scikit-learn's ``CountVectorizer(min_df, max_df)``.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import spacy
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from config import PREPROCESSING as CFG, RANDOM_SEED, YEAR_RANGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus(path: str, text_col: str = "text", year_col: str = "Year",
                date_col: str = "datetime",
                id_col: str = "text_id") -> pd.DataFrame:
    """Load and minimally validate a newspaper corpus from CSV / Parquet / Rda-export.

    Parameters
    ----------
    path : str
        Path to the data file.  Accepts ``.csv``, ``.parquet``, ``.tsv``.
    text_col, year_col, date_col, id_col : str
        Column name mappings (case-insensitive lookup is attempted).

    Returns
    -------
    pd.DataFrame
        Columns: ``text``, ``year``, ``datetime``, ``text_id``
        (standardized names).  Rows with missing text are dropped.
    """
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext == "parquet":
        df = pd.read_parquet(path)
    elif ext in ("csv", "tsv"):
        sep = "\t" if ext == "tsv" else ","
        df = pd.read_csv(path, sep=sep, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    # --- case-insensitive column resolution ---
    col_map = _resolve_columns(df, text_col=text_col, year_col=year_col,
                               date_col=date_col, id_col=id_col)
    df = df.rename(columns=col_map)

    required = {"text", "year"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Could not find columns: {missing}.  Available: {list(df.columns)}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["text", "year"])
    df["year"] = df["year"].astype(int)

    # Restrict to study period
    lo, hi = YEAR_RANGE
    df = df.query("@lo <= year <= @hi").reset_index(drop=True)
    logger.info("Loaded corpus: %d documents, years %d--%d", len(df), lo, hi)
    return df


class SpacyPreprocessor:
    """Lemmatize + clean a corpus using spaCy.

    Produces one list of lemmas per document, ready for vectorization
    or direct ingestion by gensim / BERTopic.

    Parameters
    ----------
    model_name : str
        spaCy model to load (default from config).
    extra_stopwords : sequence of str
        Additional domain-specific stopwords.
    """

    def __init__(
        self,
        model_name: str = CFG.spacy_model,
        extra_stopwords: Sequence[str] = CFG.extra_stopwords,
    ) -> None:
        self.nlp = spacy.load(model_name, disable=["parser", "ner"])
        for w in extra_stopwords:
            self.nlp.vocab[w].is_stop = True

    # ---- public ----------------------------------------------------------

    def tokenize_corpus(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        n_process: int = 1,
    ) -> List[List[str]]:
        """Lemmatize every document and return lists of clean tokens.

        Parameters
        ----------
        texts : sequence of str
        batch_size, n_process : int
            Passed to ``nlp.pipe`` for efficient batched processing.

        Returns
        -------
        list of list of str
            One inner list per document.
        """
        corpus_tokens: List[List[str]] = []
        for doc in self.nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            tokens = [
                token.lemma_.lower()
                for token in doc
                if self._keep_token(token)
            ]
            corpus_tokens.append(tokens)
        logger.info("Tokenized %d documents", len(corpus_tokens))
        return corpus_tokens

    def texts_to_joined(self, texts: Sequence[str], **kw) -> List[str]:
        """Convenience: return space-joined token strings (for sklearn vectorizers)."""
        return [" ".join(toks) for toks in self.tokenize_corpus(texts, **kw)]

    # ---- private ---------------------------------------------------------

    def _keep_token(self, token: spacy.tokens.Token) -> bool:
        """Return True if token passes all quality filters."""
        if token.is_stop or token.is_punct or token.is_space:
            return False
        if token.like_num or token.like_url or token.like_email:
            return False
        lemma = token.lemma_.lower()
        if len(lemma) < CFG.min_token_len or len(lemma) > CFG.max_token_len:
            return False
        if not re.match(r"^[a-z]+$", lemma):
            return False
        return True


def build_dtm(
    clean_texts: Sequence[str],
    min_df: int = CFG.min_doc_freq,
    max_df: float = CFG.max_doc_freq_ratio,
) -> Tuple[csr_matrix, np.ndarray, CountVectorizer]:
    """Build a document-term matrix with vocabulary pruning.

    Parameters
    ----------
    clean_texts : list of str
        Pre-processed, space-joined token strings.
    min_df : int
        Minimum document frequency for a term to be retained.
    max_df : float
        Maximum document-frequency ratio (0-1).

    Returns
    -------
    dtm : scipy.sparse.csr_matrix  (n_docs x n_terms)
    vocab : np.ndarray of str
    vectorizer : fitted CountVectorizer
    """
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-z]{3,}\b",
    )
    dtm = vectorizer.fit_transform(clean_texts)
    vocab = np.array(vectorizer.get_feature_names_out())
    logger.info("DTM shape: %s  |  vocab size: %d", dtm.shape, len(vocab))
    return dtm, vocab, vectorizer


def word_frequency_table(dtm: csr_matrix, vocab: np.ndarray) -> pd.DataFrame:
    """Compute corpus-wide word frequencies from a DTM.

    Returns
    -------
    pd.DataFrame
        Columns ``word`` and ``freq``, sorted descending.
    """
    freqs = np.asarray(dtm.sum(axis=0)).ravel()
    df = pd.DataFrame({"word": vocab, "freq": freqs})
    return df.sort_values("freq", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_columns(df: pd.DataFrame, **target_map) -> dict:
    """Case-insensitive column name resolution.

    ``target_map`` maps canonical name -> user-supplied hint.
    Returns a dict suitable for ``df.rename(columns=...)``.
    """
    lower_to_actual = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, hint in target_map.items():
        # Try exact match first, then case-insensitive
        if hint in df.columns:
            rename[hint] = canonical
        elif hint.lower() in lower_to_actual:
            rename[lower_to_actual[hint.lower()]] = canonical
    return rename
