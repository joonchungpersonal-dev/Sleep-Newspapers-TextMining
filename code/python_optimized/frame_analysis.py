"""
Dictionary-based media frame analysis with temporal decomposition.

Replicates and extends the R ``TOPIC 3 Frame analysis.R`` script.
Frame scores are simple keyword-hit counts on lowercased article text;
the module then computes frame prevalence, dominance, co-occurrence,
temporal trends, and statistical tests for change over time.

Design notes
------------
* The R version used ``stringr::str_count`` with regex alternation.
  We do the same with compiled ``re`` patterns for speed.
* All frame definitions live in ``config.FRAME.frames`` so they can
  be edited without touching analysis code.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import FRAME as FRAME_CFG, TABLES_DIR

logger = logging.getLogger(__name__)


# ============================================================================
# Core scoring
# ============================================================================

class FrameScorer:
    """Score documents against a set of keyword-based media frames.

    Parameters
    ----------
    frame_dict : dict mapping frame name -> list of seed words.
        Defaults to ``config.FRAME.frames``.
    """

    def __init__(
        self,
        frame_dict: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.frame_dict = frame_dict or FRAME_CFG.frames
        # Pre-compile one regex per frame for performance
        self._patterns: Dict[str, re.Pattern] = {
            name: re.compile("|".join(re.escape(w) for w in words), re.IGNORECASE)
            for name, words in self.frame_dict.items()
        }

    def score(self, texts: pd.Series) -> pd.DataFrame:
        """Return a DataFrame with one column per frame, values = hit counts.

        Parameters
        ----------
        texts : pd.Series of str

        Returns
        -------
        pd.DataFrame
            Index aligned with *texts*, columns named ``{frame}_frame``.
        """
        result = pd.DataFrame(index=texts.index)
        for name, pat in self._patterns.items():
            result[f"{name}_frame"] = texts.str.lower().apply(
                lambda t: len(pat.findall(t)) if isinstance(t, str) else 0
            )
        return result

    @property
    def frame_names(self) -> List[str]:
        return list(self.frame_dict.keys())


# ============================================================================
# Analytical helpers
# ============================================================================

def frame_summary(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each frame column.

    Parameters
    ----------
    scores : pd.DataFrame
        Output of :meth:`FrameScorer.score`.

    Returns
    -------
    pd.DataFrame
        Columns: ``frame``, ``total_matches``, ``articles_with_frame``,
        ``pct_articles``, ``mean_score``, ``max_score``.
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    records = []
    for col in frame_cols:
        name = col.replace("_frame", "")
        s = scores[col]
        records.append({
            "frame": name,
            "total_matches": int(s.sum()),
            "articles_with_frame": int((s > 0).sum()),
            "pct_articles": round((s > 0).mean() * 100, 1),
            "mean_score": round(s.mean(), 3),
            "max_score": int(s.max()),
        })
    return pd.DataFrame(records).sort_values("total_matches", ascending=False)


def frame_prevalence_by_period(
    scores: pd.DataFrame,
    years: pd.Series,
    period: str = "decade",
) -> pd.DataFrame:
    """Compute frame prevalence grouped by decade or year.

    Parameters
    ----------
    scores : pd.DataFrame
    years : pd.Series of int
    period : ``"decade"`` or ``"year"``

    Returns
    -------
    pd.DataFrame  (long format)
        Columns: ``period``, ``frame``, ``mean_score``, ``prevalence_pct``.
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    combined = scores.copy()
    combined["year"] = years.values

    if period == "decade":
        combined["period"] = (combined["year"] // 10) * 10
    else:
        combined["period"] = combined["year"]

    records = []
    for p, grp in combined.groupby("period"):
        for col in frame_cols:
            name = col.replace("_frame", "")
            records.append({
                "period": int(p),
                "frame": name,
                "mean_score": float(grp[col].mean()),
                "prevalence_pct": float((grp[col] > 0).mean() * 100),
                "total_score": int(grp[col].sum()),
            })
    return pd.DataFrame(records)


def frame_temporal_trends(
    scores: pd.DataFrame,
    years: pd.Series,
) -> pd.DataFrame:
    """OLS regression of each frame score on year to detect trends.

    Replicates the R ``lm(frame_score ~ year)`` tests.

    Parameters
    ----------
    scores : pd.DataFrame
    years : pd.Series of int

    Returns
    -------
    pd.DataFrame
        Columns: ``frame``, ``slope``, ``p_value``, ``r_squared``.
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    y_arr = years.values.astype(float)
    results = []
    for col in frame_cols:
        name = col.replace("_frame", "")
        x = y_arr
        y = scores[col].values.astype(float)
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 10:
            continue
        slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
        results.append({
            "frame": name,
            "slope": round(slope, 6),
            "p_value": round(p, 4),
            "r_squared": round(r ** 2, 4),
        })
    return pd.DataFrame(results).sort_values("p_value")


def frame_cooccurrence(scores: pd.DataFrame) -> pd.DataFrame:
    """Binary co-occurrence matrix: how often frames appear together.

    Parameters
    ----------
    scores : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Square matrix with frame names as both index and columns.
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    binary = (scores[frame_cols] > 0).astype(int)
    cooc = binary.T.dot(binary)
    np.fill_diagonal(cooc.values, 0)
    cooc.index = [c.replace("_frame", "") for c in cooc.index]
    cooc.columns = [c.replace("_frame", "") for c in cooc.columns]
    return cooc


def frame_correlation(scores: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix among frame scores.

    Parameters
    ----------
    scores : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    corr = scores[frame_cols].corr()
    corr.index = [c.replace("_frame", "") for c in corr.index]
    corr.columns = [c.replace("_frame", "") for c in corr.columns]
    return corr


def dominant_frame(scores: pd.DataFrame) -> pd.Series:
    """Identify the dominant (highest-scoring) frame for each document.

    Parameters
    ----------
    scores : pd.DataFrame

    Returns
    -------
    pd.Series of str
        Frame name, or ``"none"`` when all scores are zero.
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    totals = scores[frame_cols].sum(axis=1)
    dominant = scores[frame_cols].idxmax(axis=1).str.replace("_frame", "", regex=False)
    dominant[totals == 0] = "none"
    return dominant


def frame_complexity(scores: pd.DataFrame) -> pd.Series:
    """Count how many distinct frames each document activates.

    Parameters
    ----------
    scores : pd.DataFrame

    Returns
    -------
    pd.Series of int
    """
    frame_cols = [c for c in scores.columns if c.endswith("_frame")]
    return (scores[frame_cols] > 0).sum(axis=1)


# ============================================================================
# Convenience: run full analysis
# ============================================================================

def run_frame_analysis(
    texts: pd.Series,
    years: pd.Series,
    output_dir=TABLES_DIR,
) -> Dict[str, pd.DataFrame]:
    """Execute the complete frame analysis pipeline and save CSVs.

    Parameters
    ----------
    texts : pd.Series of str
    years : pd.Series of int
    output_dir : Path

    Returns
    -------
    dict of DataFrames keyed by analysis name
    """
    scorer = FrameScorer()
    scores = scorer.score(texts)

    results = {
        "frame_scores": scores,
        "summary": frame_summary(scores),
        "by_decade": frame_prevalence_by_period(scores, years, "decade"),
        "by_year": frame_prevalence_by_period(scores, years, "year"),
        "temporal_trends": frame_temporal_trends(scores, years),
        "cooccurrence": frame_cooccurrence(scores),
        "correlation": frame_correlation(scores),
    }

    # Persist
    for name, df in results.items():
        path = output_dir / f"frame_{name}.csv"
        df.to_csv(path, index=(name in ("cooccurrence", "correlation")))
        logger.info("Saved %s -> %s", name, path)

    logger.info("Frame analysis complete: %d frames, %d documents",
                len(scorer.frame_names), len(texts))
    return results
