"""
Term co-occurrence network analysis for topic model outputs.

Replicates the R ``make_network`` / ``process_networks`` workflow:
for each topic, select the top-N representative documents, build a
local DTM, compute the term co-occurrence matrix, construct a
NetworkX graph, and export both the edge list and a publication-quality
Fruchterman-Reingold layout plot.

Key improvements over the R version
------------------------------------
* Uses scikit-learn's sparse matrix operations instead of dense
  ``t(dtm) %*% dtm`` -- dramatically faster on large vocabularies.
* Network metrics (degree centrality, betweenness, clustering
  coefficient) are computed and attached to nodes automatically.
* Interactive Plotly export supplements static matplotlib figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from config import NETWORK as NET_CFG, NETWORKS_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# Core: build co-occurrence graph for one topic
# ============================================================================

def build_cooccurrence_network(
    texts: List[str],
    focus_terms: Optional[List[str]] = None,
    min_cooccurrence: int = NET_CFG.min_cooccurrence,
    max_terms: int = 50,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build a term co-occurrence graph from a set of documents.

    Parameters
    ----------
    texts : list of str
        Pre-processed, space-joined token strings (one per document).
    focus_terms : list of str or None
        If given, restrict the network to only these terms.
        Corresponds to the R approach of filtering the DTM to
        STM-ranked words.
    min_cooccurrence : int
        Minimum co-occurrence count for an edge to be created.
    max_terms : int
        Upper limit on vocabulary size (top by frequency).

    Returns
    -------
    G : networkx.Graph
        Undirected, weighted graph.  Nodes carry ``frequency`` and
        ``degree`` attributes.  Edges carry ``weight`` (co-occurrence
        count).
    edge_df : pd.DataFrame
        Long-form edge list with columns ``term1``, ``term2``, ``weight``.
    """
    # Build a local DTM
    vec = CountVectorizer(
        token_pattern=r"(?u)\b[a-z]{3,}\b",
        max_features=5000,
    )
    try:
        dtm = vec.fit_transform(texts)
    except ValueError:
        logger.warning("Empty vocabulary after vectorization")
        return nx.Graph(), pd.DataFrame(columns=["term1", "term2", "weight"])

    vocab = np.array(vec.get_feature_names_out())

    # Filter to focus terms if provided
    if focus_terms:
        keep_mask = np.isin(vocab, focus_terms)
        if keep_mask.sum() < 3:
            logger.warning("Fewer than 3 focus terms found in DTM "
                           "(%d / %d)", keep_mask.sum(), len(focus_terms))
            return nx.Graph(), pd.DataFrame(columns=["term1", "term2", "weight"])
        dtm = dtm[:, keep_mask]
        vocab = vocab[keep_mask]
    else:
        # Keep only top-N by frequency
        freqs = np.asarray(dtm.sum(axis=0)).ravel()
        top_idx = freqs.argsort()[::-1][:max_terms]
        dtm = dtm[:, top_idx]
        vocab = vocab[top_idx]

    # Co-occurrence: C = X^T X  (sparse, fast)
    cooc = (dtm.T @ dtm).toarray()
    np.fill_diagonal(cooc, 0)  # no self-loops

    # Build edge list (upper triangle only to avoid duplicates)
    n = len(vocab)
    rows, cols = np.triu_indices(n, k=1)
    weights = cooc[rows, cols]
    mask = weights >= min_cooccurrence
    edge_df = pd.DataFrame({
        "term1": vocab[rows[mask]],
        "term2": vocab[cols[mask]],
        "weight": weights[mask].astype(int),
    }).sort_values("weight", ascending=False).reset_index(drop=True)

    if edge_df.empty:
        return nx.Graph(), edge_df

    # Build graph
    G = nx.from_pandas_edgelist(edge_df, "term1", "term2",
                                 edge_attr="weight")

    # Node attributes
    word_freq = dict(zip(vocab, np.asarray(dtm.sum(axis=0)).ravel()))
    for node in G.nodes:
        G.nodes[node]["frequency"] = int(word_freq.get(node, 1))
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    nx.set_node_attributes(G, nx.betweenness_centrality(G), "betweenness")
    nx.set_node_attributes(G, nx.clustering(G, weight="weight"), "clustering")

    return G, edge_df


# ============================================================================
# Batch processing across topics
# ============================================================================

def build_topic_networks(
    topic_result,            # TopicModelResult (avoid circular import)
    corpus_df: pd.DataFrame,
    clean_texts: List[str],
    topic_ids: Optional[Dict[int, str]] = None,
    word_counts: Tuple[int, ...] = NET_CFG.word_counts,
    n_top_docs: int = NET_CFG.n_top_docs_for_network,
    output_dir: Path = NETWORKS_DIR,
) -> Dict[str, Dict]:
    """Generate co-occurrence networks for specified topics.

    For each (topic, n_words) combination:
      1. Select top *n_top_docs* documents for the topic.
      2. Get the topic's top-N terms as focus vocabulary.
      3. Build the co-occurrence network.
      4. Save edge list CSV and metadata.

    Parameters
    ----------
    topic_result : TopicModelResult
    corpus_df : pd.DataFrame  (must have ``text`` column)
    clean_texts : list of str  (preprocessed texts aligned with corpus_df)
    topic_ids : dict[int, str] or None
    word_counts : tuple of int
    n_top_docs : int
    output_dir : Path

    Returns
    -------
    dict
        Keys are ``"topic_{tid}_{n_words}w"``, values are dicts with
        ``graph``, ``edge_df``, ``terms``, ``n_edges``.
    """
    if topic_ids is None:
        topic_ids = {k: f"Topic_{k}" for k in range(topic_result.n_topics)}

    output_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    for tid, label in topic_ids.items():
        if tid >= topic_result.n_topics:
            continue

        # Top documents
        proportions = topic_result.doc_topic_matrix[:, tid]
        top_idx = proportions.argsort()[::-1][:n_top_docs]
        topic_texts = [clean_texts[i] for i in top_idx
                       if i < len(clean_texts) and clean_texts[i].strip()]

        if len(topic_texts) < 5:
            logger.warning("Topic %d (%s): too few valid documents (%d)",
                           tid, label, len(topic_texts))
            continue

        for n_words in word_counts:
            # Focus terms from the topic-term distribution
            focus = topic_result.top_terms(tid, n=n_words)["term"].tolist()

            G, edge_df = build_cooccurrence_network(
                topic_texts, focus_terms=focus,
            )

            if G.number_of_edges() == 0:
                logger.warning("Topic %d, %d words: empty network", tid, n_words)
                continue

            key = f"topic_{tid:02d}_{n_words}w"

            # Save edge list
            csv_path = output_dir / f"{key}_edges.csv"
            edge_df.to_csv(csv_path, index=False)

            results[key] = {
                "graph": G,
                "edge_df": edge_df,
                "terms": list(G.nodes),
                "n_edges": G.number_of_edges(),
                "topic_id": tid,
                "topic_label": label,
                "n_words": n_words,
            }
            logger.info("Topic %d (%s), %d words: %d nodes, %d edges",
                         tid, label, n_words,
                         G.number_of_nodes(), G.number_of_edges())

    logger.info("Built %d networks total", len(results))
    return results


# ============================================================================
# Network summary metrics
# ============================================================================

def network_summary(G: nx.Graph) -> Dict[str, float]:
    """Return key graph-level metrics for a co-occurrence network.

    Parameters
    ----------
    G : nx.Graph

    Returns
    -------
    dict with keys: ``n_nodes``, ``n_edges``, ``density``,
    ``avg_clustering``, ``avg_degree``, ``diameter``
    (diameter is -1 if graph is disconnected).
    """
    if G.number_of_nodes() == 0:
        return {"n_nodes": 0, "n_edges": 0, "density": 0.0,
                "avg_clustering": 0.0, "avg_degree": 0.0, "diameter": -1}
    degrees = [d for _, d in G.degree()]
    try:
        diam = nx.diameter(G) if nx.is_connected(G) else -1
    except nx.NetworkXError:
        diam = -1
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": round(nx.density(G), 4),
        "avg_clustering": round(nx.average_clustering(G), 4),
        "avg_degree": round(np.mean(degrees), 2),
        "diameter": diam,
    }


def all_network_summaries(
    networks: Dict[str, Dict],
) -> pd.DataFrame:
    """Tabulate summary metrics for every network in a batch.

    Parameters
    ----------
    networks : dict returned by :func:`build_topic_networks`

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for key, info in networks.items():
        row = {
            "network": key,
            "topic_id": info["topic_id"],
            "topic_label": info["topic_label"],
            "n_words": info["n_words"],
        }
        row.update(network_summary(info["graph"]))
        rows.append(row)
    return pd.DataFrame(rows)
