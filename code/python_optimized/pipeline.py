#!/usr/bin/env python3
"""
Sleep-Newspapers Text Mining Pipeline
======================================

Main orchestration script that reproduces the full R analysis
(5+ separate R scripts) in a single, configurable Python pipeline.

Usage
-----
    # Full pipeline (LDA only -- BERTopic requires GPU or patience)
    python pipeline.py --data ../data/clean_full_text_20k.csv --model lda

    # BERTopic alternative
    python pipeline.py --data ../data/clean_full_text_20k.csv --model bertopic

    # Both models for comparison
    python pipeline.py --data ../data/clean_full_text_20k.csv --model both

    # Dry run: preprocessing + word frequency only (no topic model)
    python pipeline.py --data ../data/clean_full_text_20k.csv --steps preprocess,wordfreq

Equivalent R scripts consolidated
-----------------------------------
1. ``01 STM models 6_12_2025.R``         -> preprocessing + topic_modeling
2. ``03 Top 25 documents to file.R``     -> topic_modeling.export_top_documents
3. ``TOPIC 3 Frame analysis.R``          -> frame_analysis
4. ``Chung. Word Frequency Analysis.R``  -> preprocessing + visualization
5. ``Master R script 8_29_2025.R``        -> this pipeline (all of the above)

Author
------
Joon Chung, University of Miami Miller School of Medicine
Python port and redesign.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# --- project imports ---
from config import (
    DATA_DIR,
    FIGURES_DIR,
    NETWORKS_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
    TABLES_DIR,
    TOPICS_OF_INTEREST,
    TOPIC_MODEL as TM_CFG,
)

logger = logging.getLogger("pipeline")


# ============================================================================
# Pipeline steps
# ============================================================================

def step_preprocess(corpus_df: pd.DataFrame) -> Dict:
    """Tokenize and vectorize the corpus.

    Returns a dict with ``clean_texts``, ``dtm``, ``vocab``, ``vectorizer``,
    ``word_freq``.
    """
    from preprocessing import SpacyPreprocessor, build_dtm, word_frequency_table

    logger.info("=== STEP 1: Preprocessing ===")
    t0 = time.time()

    preprocessor = SpacyPreprocessor()
    clean_texts = preprocessor.texts_to_joined(corpus_df["text"].tolist())

    dtm, vocab, vectorizer = build_dtm(clean_texts)
    word_freq = word_frequency_table(dtm, vocab)

    elapsed = time.time() - t0
    logger.info("Preprocessing complete in %.1f s  |  vocab = %d terms",
                elapsed, len(vocab))

    return {
        "clean_texts": clean_texts,
        "dtm": dtm,
        "vocab": vocab,
        "vectorizer": vectorizer,
        "word_freq": word_freq,
    }


def step_wordfreq_viz(word_freq: pd.DataFrame) -> None:
    """Generate corpus-wide word frequency visualizations."""
    from visualization import (plot_word_frequency,
                                plot_word_frequency_distribution)

    logger.info("=== Word Frequency Visualization ===")
    plot_word_frequency(word_freq)
    plot_word_frequency_distribution(word_freq)
    word_freq.head(100).to_csv(TABLES_DIR / "word_frequency_top100.csv",
                               index=False)
    logger.info("Word frequency summary saved.")


def step_topic_model(
    corpus_df: pd.DataFrame,
    preprocess_out: Dict,
    model_type: str = "lda",
) -> Dict:
    """Train topic model(s) and produce all topic-level outputs.

    Returns a dict with ``result`` (TopicModelResult), ``prevalence``,
    ``all_terms``.
    """
    from topic_modeling import (
        TopicModelResult,
        export_top_documents,
        topic_term_summary,
        train_bertopic,
        train_lda,
    )

    logger.info("=== STEP 2: Topic Modelling (%s) ===", model_type)
    t0 = time.time()

    if model_type == "lda":
        result = train_lda(
            preprocess_out["dtm"],
            preprocess_out["vocab"],
        )
    elif model_type == "bertopic":
        result = train_bertopic(corpus_df["text"].tolist())
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    elapsed = time.time() - t0
    logger.info("Model trained in %.1f s  |  K = %d topics", elapsed,
                result.n_topics)

    # --- Topic-term summary table ---
    all_terms = result.all_top_terms(n=TM_CFG.n_top_terms)
    all_terms.to_csv(TABLES_DIR / f"topic_terms_{model_type}.csv", index=False)

    summary = topic_term_summary(result, n_terms=5)
    summary.to_csv(TABLES_DIR / f"topic_summary_wide_{model_type}.csv",
                   index=False)

    # --- Prevalence over time ---
    prevalence = result.topic_prevalence_by_year(corpus_df["year"].values)
    prevalence.to_csv(TABLES_DIR / f"topic_prevalence_{model_type}.csv",
                      index=False)

    # --- Export top documents for topics of interest ---
    # Remap to 0-based if config uses 1-based (R convention)
    topic_map = _to_zero_based(TOPICS_OF_INTEREST, result.n_topics)
    export_top_documents(result, corpus_df, topic_ids=topic_map)

    return {
        "result": result,
        "prevalence": prevalence,
        "all_terms": all_terms,
        "topic_map": topic_map,
    }


def step_topic_viz(tm_out: Dict, model_type: str = "lda") -> None:
    """Generate all topic model visualizations."""
    from visualization import (
        plot_all_topics_summary,
        plot_doc_topic_distributions,
        plot_prevalence_grid,
        plot_wordcloud,
        plotly_topic_prevalence,
    )

    logger.info("=== STEP 3: Topic Visualization ===")
    result = tm_out["result"]
    topic_map = tm_out["topic_map"]

    # Term bar charts
    plot_all_topics_summary(tm_out["all_terms"], topic_map)

    # Prevalence over time
    plot_prevalence_grid(tm_out["prevalence"], topic_map)

    # Word clouds for focal topics
    for tid, label in topic_map.items():
        if tid < result.n_topics:
            term_df = result.top_terms(tid, n=30)
            plot_wordcloud(term_df, topic_label=label, topic_id=tid)

    # Document-topic histograms
    plot_doc_topic_distributions(result.doc_topic_matrix, topic_map)

    # Interactive (Plotly) prevalence -- save as HTML
    try:
        pfig = plotly_topic_prevalence(tm_out["prevalence"], topic_map)
        html_path = FIGURES_DIR / f"prevalence_interactive_{model_type}.html"
        pfig.write_html(str(html_path))
        logger.info("Interactive prevalence chart saved: %s", html_path)
    except Exception as e:
        logger.warning("Plotly export skipped: %s", e)


def step_frame_analysis(corpus_df: pd.DataFrame) -> Dict:
    """Run dictionary-based frame analysis on the corpus.

    Returns dict of DataFrames from ``frame_analysis.run_frame_analysis``.
    """
    from frame_analysis import run_frame_analysis

    logger.info("=== STEP 4: Frame Analysis ===")
    return run_frame_analysis(corpus_df["text"], corpus_df["year"])


def step_frame_viz(frame_out: Dict) -> None:
    """Generate frame analysis visualizations."""
    from visualization import (
        plot_frame_by_decade,
        plot_frame_correlation_heatmap,
        plot_frame_distribution,
        plot_frame_temporal_trends,
        plotly_frame_heatmap,
    )

    logger.info("=== STEP 5: Frame Visualization ===")
    plot_frame_distribution(frame_out["summary"])
    plot_frame_correlation_heatmap(frame_out["correlation"])
    plot_frame_temporal_trends(frame_out["by_year"])
    plot_frame_by_decade(frame_out["by_decade"])

    # Interactive heatmap
    try:
        pfig = plotly_frame_heatmap(frame_out["correlation"])
        pfig.write_html(str(FIGURES_DIR / "frame_correlation_interactive.html"))
    except Exception as e:
        logger.warning("Plotly frame heatmap skipped: %s", e)


def step_network_analysis(
    tm_out: Dict,
    corpus_df: pd.DataFrame,
    clean_texts,
) -> Dict:
    """Build co-occurrence networks for focal topics."""
    from network_analysis import (
        all_network_summaries,
        build_topic_networks,
    )

    logger.info("=== STEP 6: Network Analysis ===")
    topic_map = tm_out["topic_map"]
    result = tm_out["result"]

    networks = build_topic_networks(
        result, corpus_df, clean_texts, topic_ids=topic_map,
    )

    # Summary table
    if networks:
        summaries = all_network_summaries(networks)
        summaries.to_csv(TABLES_DIR / "network_summaries.csv", index=False)
        logger.info("Network summaries:\n%s", summaries.to_string(index=False))
    else:
        logger.warning("No networks were created.")

    return networks


def step_network_viz(networks: Dict) -> None:
    """Render all network plots."""
    from visualization import plot_all_topic_networks

    logger.info("=== STEP 7: Network Visualization ===")
    plot_all_topic_networks(networks)


# ============================================================================
# Helpers
# ============================================================================

def _to_zero_based(topic_map: Dict[int, str], n_topics: int) -> Dict[int, str]:
    """Convert 1-based R topic indices to 0-based Python indices.

    If all keys are >= 1 and the largest key exceeds n_topics when
    0-indexed, assume 1-based and subtract 1.  Otherwise keep as-is.
    """
    if all(k >= 1 for k in topic_map) and max(topic_map) >= n_topics:
        return {k - 1: v for k, v in topic_map.items()}
    return dict(topic_map)


# ============================================================================
# CLI
# ============================================================================

ALL_STEPS = [
    "preprocess", "wordfreq", "topic_model", "topic_viz",
    "frame", "frame_viz", "network", "network_viz",
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sleep-Newspapers Text Mining Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", type=str, required=True,
                   help="Path to corpus CSV / Parquet file")
    p.add_argument("--model", type=str, default="lda",
                   choices=["lda", "bertopic", "both"],
                   help="Topic model to train (default: lda)")
    p.add_argument("--steps", type=str, default=",".join(ALL_STEPS),
                   help="Comma-separated steps to run (default: all)")
    p.add_argument("--text-col", type=str, default="text",
                   help="Column name for article text")
    p.add_argument("--year-col", type=str, default="Year",
                   help="Column name for publication year")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    np.random.seed(RANDOM_SEED)

    steps = set(args.steps.split(","))
    logger.info("Pipeline starting  |  steps: %s  |  model: %s",
                args.steps, args.model)

    # --- Load data ---
    from preprocessing import load_corpus

    corpus_df = load_corpus(args.data, text_col=args.text_col,
                            year_col=args.year_col)
    logger.info("Corpus: %d documents", len(corpus_df))

    # --- Preprocessing ---
    preprocess_out = None
    if "preprocess" in steps:
        preprocess_out = step_preprocess(corpus_df)

    if "wordfreq" in steps and preprocess_out:
        step_wordfreq_viz(preprocess_out["word_freq"])

    # --- Topic Modelling ---
    models_to_run = (["lda", "bertopic"] if args.model == "both"
                     else [args.model])

    tm_outputs: Dict[str, Dict] = {}
    for mt in models_to_run:
        if "topic_model" in steps and preprocess_out:
            tm_out = step_topic_model(corpus_df, preprocess_out, model_type=mt)
            tm_outputs[mt] = tm_out

            if "topic_viz" in steps:
                step_topic_viz(tm_out, model_type=mt)

    # Use the first model's output for downstream analyses
    primary_tm = tm_outputs.get(models_to_run[0])

    # --- Frame Analysis ---
    frame_out = None
    if "frame" in steps:
        frame_out = step_frame_analysis(corpus_df)
    if "frame_viz" in steps and frame_out:
        step_frame_viz(frame_out)

    # --- Network Analysis ---
    if "network" in steps and primary_tm and preprocess_out:
        networks = step_network_analysis(
            primary_tm, corpus_df, preprocess_out["clean_texts"],
        )
        if "network_viz" in steps and networks:
            step_network_viz(networks)

    logger.info("Pipeline complete.  Outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
