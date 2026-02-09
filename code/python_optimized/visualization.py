"""
Unified visualization module for the Sleep-Newspapers text mining pipeline.

Provides both static (matplotlib) and interactive (plotly) figures covering:
  - Topic-term bar charts and word clouds
  - Topic prevalence over time (with confidence bands)
  - Document-topic distribution histograms
  - Frame analysis charts (distribution, prevalence, correlation heatmap,
    temporal evolution)
  - Co-occurrence network layouts
  - Corpus-wide word frequency plots

All functions return their figure objects so callers can further customise
or save them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

from config import VIZ as VCFG, FIGURES_DIR, YEAR_RANGE

logger = logging.getLogger(__name__)


# ============================================================================
# Styling helpers
# ============================================================================

def _apply_style() -> None:
    """Set the global matplotlib style once."""
    try:
        plt.style.use(VCFG.matplotlib_style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")


_apply_style()


def _save(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Save a figure and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{VCFG.figure_format}"
    fig.savefig(path, dpi=VCFG.figure_dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


# ============================================================================
# 1. Topic-term bar charts
# ============================================================================

def plot_topic_top_terms(
    topic_terms_df: pd.DataFrame,
    topic_id: int,
    topic_label: str = "",
    n_terms: int = 10,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of the top terms for a single topic.

    Parameters
    ----------
    topic_terms_df : pd.DataFrame
        Long-form with columns ``topic``, ``term``, ``probability``.
    topic_id : int
    topic_label : str
    n_terms : int
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    subset = (topic_terms_df
              .query("topic == @topic_id")
              .nlargest(n_terms, "probability"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(subset["term"], subset["probability"], color="steelblue", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("P(word | topic)")
    ax.set_title(topic_label or f"Topic {topic_id}: Top {n_terms} terms")
    fig.tight_layout()
    if save:
        _save(fig, f"topic_{topic_id:02d}_terms")
    return fig


def plot_all_topics_summary(
    topic_terms_df: pd.DataFrame,
    topic_labels: Dict[int, str],
    n_terms: int = 8,
    save: bool = True,
) -> plt.Figure:
    """Grid of bar charts for the focal topics.

    Parameters
    ----------
    topic_terms_df : pd.DataFrame
    topic_labels : dict mapping topic_id -> human label
    n_terms : int
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    n = len(topic_labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (tid, label) in enumerate(topic_labels.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        subset = (topic_terms_df
                  .query("topic == @tid")
                  .nlargest(n_terms, "probability"))
        ax.barh(subset["term"], subset["probability"],
                color="steelblue", alpha=0.85)
        ax.invert_yaxis()
        ax.set_title(label, fontsize=10)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Top Terms per Topic", fontsize=14, y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, "all_topics_terms_summary")
    return fig


# ============================================================================
# 2. Topic prevalence over time
# ============================================================================

def plot_topic_prevalence(
    prevalence_df: pd.DataFrame,
    topic_id: int,
    topic_label: str = "",
    save: bool = True,
) -> plt.Figure:
    """Line plot of mean topic proportion by year with LOESS-style smoothing.

    Parameters
    ----------
    prevalence_df : pd.DataFrame
        Columns ``year``, ``topic``, ``mean_proportion``.
    topic_id : int
    topic_label : str
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    sub = prevalence_df.query("topic == @topic_id").sort_values("year")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub["year"], sub["mean_proportion"], "-o",
            color="steelblue", markersize=3, linewidth=1.5)
    ax.fill_between(sub["year"], sub["mean_proportion"] * 0.85,
                    sub["mean_proportion"] * 1.15,
                    alpha=0.15, color="steelblue")
    ax.set_xlabel("Year")
    ax.set_ylabel("Expected topic proportion")
    ax.set_title(topic_label or f"Topic {topic_id}: Prevalence over time")
    ax.set_xlim(*YEAR_RANGE)
    fig.tight_layout()
    if save:
        _save(fig, f"topic_{topic_id:02d}_prevalence")
    return fig


def plot_prevalence_grid(
    prevalence_df: pd.DataFrame,
    topic_labels: Dict[int, str],
    save: bool = True,
) -> plt.Figure:
    """Small-multiples of topic prevalence over time.

    Parameters
    ----------
    prevalence_df : pd.DataFrame
    topic_labels : dict
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    n = len(topic_labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                             sharex=True)
    axes = np.atleast_2d(axes)

    for idx, (tid, label) in enumerate(topic_labels.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        sub = prevalence_df.query("topic == @tid").sort_values("year")
        ax.plot(sub["year"], sub["mean_proportion"],
                color="steelblue", linewidth=1.2)
        ax.fill_between(sub["year"], sub["mean_proportion"] * 0.85,
                        sub["mean_proportion"] * 1.15,
                        alpha=0.12, color="steelblue")
        ax.set_title(label, fontsize=9)
        ax.set_xlim(*YEAR_RANGE)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Topic Prevalence Over Time", fontsize=13, y=1.01)
    fig.tight_layout()
    if save:
        _save(fig, "prevalence_grid")
    return fig


# ============================================================================
# 3. Word clouds
# ============================================================================

def plot_wordcloud(
    term_probs: pd.DataFrame,
    topic_label: str = "",
    topic_id: int = 0,
    save: bool = True,
) -> plt.Figure:
    """Generate a word cloud from term probabilities.

    Parameters
    ----------
    term_probs : pd.DataFrame
        Columns ``term``, ``probability``.
    topic_label : str
    topic_id : int
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("wordcloud not installed -- skipping word cloud")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "wordcloud not installed", ha="center")
        return fig

    freq_dict = dict(zip(term_probs["term"], term_probs["probability"]))
    wc = WordCloud(
        width=800, height=500, background_color="white",
        colormap=VCFG.color_palette, max_words=100,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(topic_label or f"Topic {topic_id}", fontsize=14)
    fig.tight_layout()
    if save:
        _save(fig, f"wordcloud_topic_{topic_id:02d}")
    return fig


# ============================================================================
# 4. Frame analysis visualizations
# ============================================================================

def plot_frame_distribution(
    frame_summary_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Horizontal bar chart of total frame matches across the corpus.

    Parameters
    ----------
    frame_summary_df : pd.DataFrame
        Output of ``frame_analysis.frame_summary()``.
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    df = frame_summary_df.sort_values("total_matches")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.get_cmap(VCFG.color_palette)(
        np.linspace(0.25, 0.85, len(df))
    )
    ax.barh(df["frame"], df["total_matches"], color=colors, alpha=0.85)
    ax.set_xlabel("Total keyword matches")
    ax.set_title("Frame Prevalence Across Corpus")
    fig.tight_layout()
    if save:
        _save(fig, "frame_distribution")
    return fig


def plot_frame_correlation_heatmap(
    corr_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Heatmap of frame score correlations.

    Parameters
    ----------
    corr_df : pd.DataFrame  (square, from ``frame_analysis.frame_correlation``)
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_df.index, fontsize=9)
    # Annotate cells
    for i in range(len(corr_df)):
        for j in range(len(corr_df)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(corr_df.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Frame Correlation Matrix")
    fig.tight_layout()
    if save:
        _save(fig, "frame_correlation_heatmap")
    return fig


def plot_frame_temporal_trends(
    by_year_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Line plot of mean frame score per year for all frames.

    Parameters
    ----------
    by_year_df : pd.DataFrame
        Output of ``frame_analysis.frame_prevalence_by_period(..., 'year')``.
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap(VCFG.color_palette)
    frames = by_year_df["frame"].unique()
    for i, frame in enumerate(frames):
        sub = by_year_df.query("frame == @frame").sort_values("period")
        color = cmap(i / max(len(frames) - 1, 1))
        ax.plot(sub["period"], sub["mean_score"], label=frame,
                linewidth=1.4, color=color)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean frame score")
    ax.set_title("Frame Usage Evolution Over Time")
    fig.tight_layout()
    if save:
        _save(fig, "frame_temporal_trends")
    return fig


def plot_frame_by_decade(
    by_decade_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Grouped bar chart of mean frame score per decade.

    Parameters
    ----------
    by_decade_df : pd.DataFrame
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    pivot = by_decade_df.pivot(index="period", columns="frame",
                               values="mean_score").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, alpha=0.85, width=0.8,
               colormap=VCFG.color_palette)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Mean frame score")
    ax.set_title("Frame Usage by Decade")
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    if save:
        _save(fig, "frame_by_decade")
    return fig


# ============================================================================
# 5. Co-occurrence network plots
# ============================================================================

def plot_network(
    G: nx.Graph,
    title: str = "",
    layout: str = "spring",
    save: bool = True,
    filename: str = "network",
    font_size: int = 16,
) -> plt.Figure:
    """Matplotlib rendering of a co-occurrence network.

    Parameters
    ----------
    G : nx.Graph
    title : str
    layout : str  (``"spring"``, ``"kamada_kawai"``, ``"circular"``)
    save : bool
    filename : str
    font_size : int

    Returns
    -------
    matplotlib.Figure
    """
    if G.number_of_nodes() == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Empty network", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    if layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=1.5 / np.sqrt(G.number_of_nodes()),
                                seed=42, iterations=50)

    # Node sizes from frequency
    freqs = np.array([G.nodes[n].get("frequency", 1) for n in G.nodes])
    node_sizes = 100 + (freqs / max(freqs.max(), 1)) * 1500

    # Edge widths from weight
    weights = np.array([G[u][v]["weight"] for u, v in G.edges])
    edge_widths = 0.5 + (weights / max(weights.max(), 1)) * 4
    edge_alphas = 0.3 + (weights / max(weights.max(), 1)) * 0.5

    # Draw edges
    for (u, v), w, a in zip(G.edges, edge_widths, edge_alphas):
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#999999", linewidth=w, alpha=a, zorder=1)

    # Draw nodes
    xs = [pos[n][0] for n in G.nodes]
    ys = [pos[n][1] for n in G.nodes]
    ax.scatter(xs, ys, s=node_sizes, c="steelblue", alpha=0.8,
               edgecolors="white", linewidth=0.5, zorder=2)

    # Labels
    for n in G.nodes:
        ax.annotate(n, pos[n], fontsize=font_size * 0.55,
                    ha="center", va="center", zorder=3,
                    fontweight="bold", color="black")

    ax.set_title(title, fontsize=font_size)
    ax.axis("off")
    fig.tight_layout()
    if save:
        _save(fig, filename, FIGURES_DIR / "networks")
    return fig


def plot_all_topic_networks(
    networks: Dict[str, Dict],
    save: bool = True,
) -> None:
    """Batch-generate network plots for all topics in *networks*.

    Parameters
    ----------
    networks : dict returned by ``network_analysis.build_topic_networks``
    save : bool
    """
    for key, info in networks.items():
        G = info["graph"]
        label = info.get("topic_label", "")
        n_words = info.get("n_words", "")
        title = f"{label} ({n_words} terms)"
        plot_network(G, title=title, save=save, filename=key)


# ============================================================================
# 6. Word frequency plots
# ============================================================================

def plot_word_frequency(
    word_freq_df: pd.DataFrame,
    n_top: int = 25,
    save: bool = True,
) -> plt.Figure:
    """Bar chart of the most frequent words in the corpus.

    Parameters
    ----------
    word_freq_df : pd.DataFrame
        Columns ``word``, ``freq`` (from ``preprocessing.word_frequency_table``).
    n_top : int
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    top = word_freq_df.head(n_top)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["word"], top["freq"], color="steelblue", alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top {n_top} Most Frequent Words")
    fig.tight_layout()
    if save:
        _save(fig, "word_frequency_top")
    return fig


def plot_word_frequency_distribution(
    word_freq_df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Histogram of the word frequency distribution (log scale).

    Parameters
    ----------
    word_freq_df : pd.DataFrame
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(word_freq_df["freq"], bins=50, color="steelblue", alpha=0.8,
            edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("Word frequency")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Word Frequency Distribution")
    fig.tight_layout()
    if save:
        _save(fig, "word_frequency_distribution")
    return fig


# ============================================================================
# 7. Document-topic distribution
# ============================================================================

def plot_doc_topic_distributions(
    doc_topic_matrix: np.ndarray,
    topic_labels: Dict[int, str],
    save: bool = True,
) -> plt.Figure:
    """Histograms of document-topic proportions for focal topics.

    Parameters
    ----------
    doc_topic_matrix : np.ndarray  (D x K)
    topic_labels : dict
    save : bool

    Returns
    -------
    matplotlib.Figure
    """
    n = len(topic_labels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (tid, label) in enumerate(topic_labels.items()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        vals = doc_topic_matrix[:, tid]
        ax.hist(vals, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Topic proportion", fontsize=8)
        ax.set_ylabel("Documents", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Distribution of Document-Topic Proportions", fontsize=13, y=1.01)
    fig.tight_layout()
    if save:
        _save(fig, "doc_topic_distributions")
    return fig


# ============================================================================
# 8. Interactive (Plotly) helpers
# ============================================================================

def plotly_topic_prevalence(
    prevalence_df: pd.DataFrame,
    topic_labels: Dict[int, str],
) -> "plotly.graph_objects.Figure":
    """Interactive line chart of topic prevalence over time.

    Parameters
    ----------
    prevalence_df : pd.DataFrame
    topic_labels : dict

    Returns
    -------
    plotly Figure (can be shown in Jupyter or exported to HTML)
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    for tid, label in topic_labels.items():
        sub = prevalence_df.query("topic == @tid").sort_values("year")
        fig.add_trace(go.Scatter(
            x=sub["year"], y=sub["mean_proportion"],
            mode="lines+markers", name=label,
            line=dict(width=2), marker=dict(size=4),
        ))
    fig.update_layout(
        title="Topic Prevalence Over Time (interactive)",
        xaxis_title="Year",
        yaxis_title="Mean topic proportion",
        template=VCFG.plotly_template,
        legend=dict(font=dict(size=10)),
        height=500,
    )
    return fig


def plotly_frame_heatmap(corr_df: pd.DataFrame) -> "plotly.graph_objects.Figure":
    """Interactive frame correlation heatmap.

    Parameters
    ----------
    corr_df : pd.DataFrame

    Returns
    -------
    plotly Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=list(corr_df.columns),
        y=list(corr_df.index),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_df.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title="Frame Correlation Matrix (interactive)",
        template=VCFG.plotly_template,
        height=500, width=600,
    )
    return fig
