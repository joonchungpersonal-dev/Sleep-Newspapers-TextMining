"""
Centralized configuration for the Sleep-Newspapers text mining pipeline.

All tunable parameters live here so the analysis is reproducible
and easy to reconfigure without touching logic code.

Author: Joon Chung (Python port)
Original R analysis: Joon Chung, University of Miami Miller School of Medicine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "python"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
NETWORKS_DIR = OUTPUT_DIR / "networks"
TOP_DOCS_DIR = OUTPUT_DIR / "top_documents"

# Create all output directories on import
for _d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, NETWORKS_DIR, TOP_DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Corpus metadata
# ---------------------------------------------------------------------------
YEAR_RANGE: Tuple[int, int] = (1983, 2017)
RANDOM_SEED: int = 8675309  # same seed as the original R analysis


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters controlling tokenization and vocabulary pruning."""

    spacy_model: str = "en_core_web_sm"
    min_token_len: int = 3
    max_token_len: int = 50
    # Vocabulary pruning (analogous to R's lower.thresh = 500)
    min_doc_freq: int = 10
    max_doc_freq_ratio: float = 0.85   # drop terms in > 85 % of docs
    extra_stopwords: Tuple[str, ...] = (
        "said", "say", "says", "would", "could", "also", "one", "two",
        "new", "like", "get", "got", "go", "went", "make", "made",
        "time", "year", "people", "may", "even", "much",
    )


PREPROCESSING = PreprocessingConfig()


# ---------------------------------------------------------------------------
# Topic modelling
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TopicModelConfig:
    """Parameters for both classical (LDA) and neural (BERTopic) models."""

    # --- LDA via gensim ---
    n_topics_lda: int = 70           # match the original R STM K = 70
    lda_passes: int = 15
    lda_iterations: int = 400
    lda_alpha: str = "auto"
    lda_eta: str = "auto"
    lda_chunksize: int = 2000
    lda_eval_every: int = 10

    # --- BERTopic (neural alternative) ---
    # BERTopic determines n_topics automatically; we set limits.
    bertopic_min_topic_size: int = 30
    bertopic_nr_topics: int | str = "auto"  # or an int to force merging
    bertopic_embedding_model: str = "all-MiniLM-L6-v2"
    bertopic_top_n_words: int = 15

    # Shared
    n_top_terms: int = 10            # terms shown per topic in summaries
    n_top_docs: int = 100            # representative docs per topic
    top_docs_for_export: int = 25    # documents written to per-topic files


TOPIC_MODEL = TopicModelConfig()


# ---------------------------------------------------------------------------
# Topics of interest  (R analysis focal topics, 1-indexed like the original)
# ---------------------------------------------------------------------------
TOPICS_OF_INTEREST: Dict[int, str] = {
    3:  "Sleep and Work",
    7:  "Sleep Medicine / Drugs",
    15: "Circadian Science",
    32: "Sleep Apnea / Hospitals",
    47: "Health Research",
    51: "Academic Sleep Research",
    59: "School Start Times",
    14: "Dear Ann (validation)",
    23: "Iraq Wars",
}


# ---------------------------------------------------------------------------
# Frame analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameConfig:
    """Dictionary-based media frame definitions.

    Each frame is a list of seed words.  Frame scores are computed as
    simple keyword-hit counts on lowercased text.  The dictionaries mirror
    the R ``frames_detailed`` object.
    """

    frames: Dict[str, List[str]] = field(default_factory=lambda: {
        "health": [
            "health", "wellness", "medical", "doctor", "disease", "risk",
            "benefits", "harmful", "illness", "sick", "recovery", "healing",
            "treatment", "therapy", "medication", "symptoms", "diagnosis",
            "patient", "clinic", "hospital",
        ],
        "productivity": [
            "efficiency", "performance", "output", "profit", "competitive",
            "productive", "effective", "success", "achievement", "goals",
            "results", "optimization", "maximize", "improve", "enhance",
            "boost", "increase", "economic", "business", "revenue", "growth",
        ],
        "moral": [
            "lazy", "disciplined", "responsible", "work ethic", "character",
            "virtue", "vice", "dedication", "commitment", "diligent",
            "hardworking", "slacker", "irresponsible", "duty", "obligation",
            "values", "integrity", "shame", "guilt", "pride", "honor",
            "respect",
        ],
        "scientific": [
            "research", "study", "evidence", "data", "findings", "scientist",
            "experiment", "analysis", "investigation", "hypothesis", "theory",
            "methodology", "journal", "publication", "statistics",
            "correlation", "causation", "sample", "survey", "clinical",
            "trial",
        ],
        "economic": [
            "cost", "expensive", "cheap", "money", "financial", "budget",
            "savings", "investment", "return", "economics", "market",
            "competition", "industry", "corporate", "commerce", "trade",
        ],
        "lifestyle": [
            "balance", "wellbeing", "quality of life", "happiness",
            "satisfaction", "fulfillment", "personal", "individual",
            "choice", "preference", "comfort", "convenience", "leisure",
            "recreation", "hobby", "family", "relationships",
        ],
    })


FRAME = FrameConfig()


# ---------------------------------------------------------------------------
# Network analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NetworkConfig:
    """Parameters for term co-occurrence network construction."""

    word_counts: Tuple[int, ...] = (15, 30, 50)
    n_top_docs_for_network: int = 100
    min_cooccurrence: int = 2
    layout_algorithm: str = "spring"   # networkx spring layout (FR-like)
    node_color: str = "steelblue"
    edge_color: str = "#999999"


NETWORK = NetworkConfig()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VizConfig:
    """Shared visual-styling constants."""

    figure_dpi: int = 300
    figure_format: str = "png"
    matplotlib_style: str = "seaborn-v0_8-whitegrid"
    color_palette: str = "viridis"
    font_sizes: Tuple[int, int] = (16, 18)
    plotly_template: str = "plotly_white"


VIZ = VizConfig()
