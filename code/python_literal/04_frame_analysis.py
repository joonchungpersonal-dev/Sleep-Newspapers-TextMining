"""
04_frame_analysis.py

Literal Python translation of:
    "TOPIC 3 Frame analysis.R"

Author: Joon Chung (translation from R)

Purpose:
    Deep dive frame analysis with temporal trends for Topic 3
    (Sleep and Work). Analyzes how different discursive frames
    (health, productivity, moral, scientific, economic, lifestyle)
    are used in newspaper articles about sleep.

IMPORTANT NOTES ON R-TO-PYTHON DIFFERENCES:
    - R's tidyverse (dplyr, tidyr, stringr) -> pandas, re
    - R's ggplot2 -> matplotlib + seaborn
    - R's viridis -> matplotlib viridis colormap
    - R's patchwork -> matplotlib subplots
    - R's corrplot -> seaborn heatmap
    - R's str_count() -> re-based counting or pandas str.count()
    - R's pivot_longer() -> pandas melt()
    - R's pivot_wider() -> pandas pivot_table()
    - R's group_by() %>% summarise() -> pandas groupby().agg()
    - R's rowwise() %>% mutate() -> pandas apply()
    - R's c_across() -> row operations with .apply()
    - R's lm() -> statsmodels OLS or scipy linregress
    - R's cor() -> pandas .corr() or np.corrcoef()
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm


# =============================================================================
# R equivalent: library(tidyverse); library(ggplot2); library(viridis);
#               library(patchwork); library(corrplot)
# =============================================================================

print("=== DEEP DIVE FRAME ANALYSIS ===")


# =============================================================================
# 1. ENHANCED FRAME DEFINITIONS AND CALCULATION
# =============================================================================

print("\n1. ENHANCED FRAME DEFINITIONS")
print("=" * 40)

# R: frames_detailed <- list(
#   health = c("health", "wellness", ...),
#   productivity = c("efficiency", "performance", ...),
#   ...
# )

frames_detailed = {
    'health': [
        "health", "wellness", "medical", "doctor", "disease", "risk",
        "benefits", "harmful", "illness", "sick", "recovery", "healing",
        "treatment", "therapy", "medication", "symptoms", "diagnosis",
        "patient", "clinic", "hospital"
    ],

    'productivity': [
        "efficiency", "performance", "output", "profit", "competitive",
        "productive", "effective", "success", "achievement", "goals",
        "results", "optimization", "maximize", "improve", "enhance",
        "boost", "increase", "economic", "business", "profit", "revenue",
        "growth"
    ],

    'moral': [
        "lazy", "disciplined", "responsible", "work ethic", "character",
        "virtue", "vice", "dedication", "commitment", "diligent",
        "hardworking", "slacker", "irresponsible", "duty", "obligation",
        "values", "integrity", "shame", "guilt", "pride", "honor", "respect"
    ],

    'scientific': [
        "research", "study", "evidence", "data", "findings", "scientist",
        "experiment", "analysis", "investigation", "hypothesis", "theory",
        "methodology", "peer-reviewed", "journal", "publication",
        "statistics", "correlation", "causation", "sample", "survey",
        "clinical", "trial"
    ],

    'economic': [
        "cost", "expensive", "cheap", "money", "financial", "budget",
        "savings", "investment", "return", "roi", "economics", "market",
        "competition", "industry", "corporate", "business", "commerce",
        "trade", "capitalism"
    ],

    'lifestyle': [
        "balance", "wellbeing", "quality of life", "happiness",
        "satisfaction", "fulfillment", "personal", "individual", "choice",
        "preference", "comfort", "convenience", "leisure", "recreation",
        "hobby", "family", "relationships"
    ]
}


def calculate_frame_scores(df, text_column='text'):
    """
    Calculate frame scores for each document.

    R equivalent:
        for (frame_name in names(frames_detailed)) {
          frame_words <- frames_detailed[[frame_name]]
          pattern <- paste(frame_words, collapse = "|")
          scores <- str_count(tolower(top_100_topic3$text), pattern)
          top_100_topic3[[paste0(frame_name, "_frame")]] <- scores
          ...
        }

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a text column.
    text_column : str
        Name of the text column.

    Returns
    -------
    pd.DataFrame : input df with added frame score columns
    pd.DataFrame : frame_results summary
    """
    # R: Remove any existing frame columns to start fresh
    existing_frame_cols = [c for c in df.columns
                           if c.endswith('_frame') or
                           (c.endswith('_score') and c != 'sentiment_score')]
    if existing_frame_cols:
        df = df.drop(columns=existing_frame_cols)
        print(f"Removed existing frame columns: {', '.join(existing_frame_cols)}")

    frame_results_list = []

    for frame_name, frame_words in frames_detailed.items():
        # R: pattern <- paste(frame_words, collapse = "|")
        pattern = '|'.join(re.escape(w) for w in frame_words)

        # R: scores <- str_count(tolower(top_100_topic3$text), pattern)
        scores = df[text_column].fillna('').str.lower().str.count(
            pattern, flags=re.IGNORECASE
        )

        # R: top_100_topic3[[paste0(frame_name, "_frame")]] <- scores
        df[f'{frame_name}_frame'] = scores

        # R: Record statistics
        frame_results_list.append({
            'frame': frame_name,
            'total_matches': scores.sum(),
            'articles_with_frame': (scores > 0).sum(),
            'mean_score': round(scores.mean(), 2),
            'max_score': scores.max()
        })

        print(f"Frame: {frame_name} - Total matches: {scores.sum()} "
              f"- Articles with frame: {(scores > 0).sum()}")

    frame_results = pd.DataFrame(frame_results_list)
    print()
    print(frame_results.to_string(index=False))

    return df, frame_results


def frame_comparison_analysis(df):
    """
    Frame comparison analysis.

    R equivalent (Section 2):
        frame_cols <- grep("_frame$", names(top_100_topic3), value = TRUE)
        frame_data <- top_100_topic3 %>%
          select(rank, doc_index, any_of("year"), ...) %>%
          pivot_longer(cols = all_of(frame_cols), ...)

    Parameters
    ----------
    df : pd.DataFrame with frame score columns

    Returns
    -------
    pd.DataFrame : frame_data (long format)
    pd.DataFrame : frame_summary
    """
    print("\n2. FRAME COMPARISON ANALYSIS")
    print("=" * 40)

    # R: frame_cols <- grep("_frame$", names(top_100_topic3), value = TRUE)
    frame_cols = [c for c in df.columns if c.endswith('_frame')]
    print(f"Frame columns identified: {', '.join(frame_cols)}")

    # R: pivot_longer(cols = all_of(frame_cols), ...)
    id_cols = []
    for col in ['rank', 'doc_index', 'year', 'source']:
        if col in df.columns:
            id_cols.append(col)

    frame_data = df[id_cols + frame_cols].melt(
        id_vars=id_cols,
        value_vars=frame_cols,
        var_name='frame_type',
        value_name='frame_score'
    )

    # R: mutate(frame_type = str_remove(frame_type, "_frame"))
    frame_data['frame_type'] = frame_data['frame_type'].str.replace(
        '_frame$', '', regex=True
    )

    # R: frame_summary <- frame_data %>% group_by(frame_type) %>% summarise(...)
    frame_summary = frame_data.groupby('frame_type').agg(
        total_score=('frame_score', 'sum'),
        mean_score=('frame_score', lambda x: round(x.mean(), 3)),
        median_score=('frame_score', 'median'),
        articles_with_frame=('frame_score', lambda x: (x > 0).sum()),
        percentage_articles=('frame_score',
                             lambda x: round((x > 0).sum() / len(x) * 100, 1)),
        max_score=('frame_score', 'max')
    ).reset_index().sort_values('total_score', ascending=False)

    print("\nFrame Summary Statistics:")
    print(frame_summary.to_string(index=False))

    return frame_data, frame_summary


def temporal_frame_analysis(df, frame_data):
    """
    Comprehensive temporal frame analysis.

    R equivalent (Section 3):
        if ("year" %in% names(top_100_topic3)) {
          frame_temporal <- frame_data %>%
            filter(!is.na(year)) %>%
            group_by(year, frame_type) %>% summarise(...)
          ...
        }

    Parameters
    ----------
    df : pd.DataFrame with frame scores and 'year' column
    frame_data : pd.DataFrame (long format from frame_comparison_analysis)

    Returns
    -------
    dict with temporal analysis results
    """
    results = {}

    if 'year' not in df.columns:
        print("No 'year' column found. Skipping temporal analysis.")
        return results

    print("\n3. TEMPORAL FRAME ANALYSIS")
    print("=" * 40)

    frame_cols = [c for c in df.columns if c.endswith('_frame')]

    # R: 3.1 Frame evolution by year
    frame_temporal = frame_data[frame_data['year'].notna()].groupby(
        ['year', 'frame_type']
    ).agg(
        mean_frame_score=('frame_score', 'mean'),
        total_frame_score=('frame_score', 'sum'),
        articles_with_frame=('frame_score', lambda x: (x > 0).sum())
    ).reset_index()

    results['frame_temporal'] = frame_temporal

    # R: 3.2 Frame evolution by decade
    # R: mutate(decade = floor(year/10)*10)
    df_with_decade = df[df['year'].notna()].copy()
    df_with_decade['decade'] = (df_with_decade['year'] // 10 * 10).astype(int)

    frame_decade_data = df_with_decade[['decade'] + frame_cols].melt(
        id_vars=['decade'],
        value_vars=frame_cols,
        var_name='frame_type',
        value_name='frame_score'
    )
    frame_decade_data['frame_type'] = frame_decade_data['frame_type'].str.replace(
        '_frame$', '', regex=True
    )

    frame_decade = frame_decade_data.groupby(['decade', 'frame_type']).agg(
        mean_score=('frame_score', 'mean'),
        total_score=('frame_score', 'sum'),
        prevalence=('frame_score', lambda x: (x > 0).sum() / len(x) * 100)
    ).reset_index()

    results['frame_decade'] = frame_decade

    # R: decade_summary <- frame_decade %>%
    #      select(decade, frame_type, mean_score) %>%
    #      pivot_wider(names_from = frame_type, values_from = mean_score, ...)
    print("Frame evolution by decade:")
    decade_summary = frame_decade.pivot_table(
        index='decade', columns='frame_type',
        values='mean_score', fill_value=0
    ).reset_index()
    print(decade_summary.to_string(index=False))

    results['decade_summary'] = decade_summary

    # Prevalence by decade
    print("\nFrame prevalence by decade (% of articles):")
    decade_prevalence = frame_decade.pivot_table(
        index='decade', columns='frame_type',
        values='prevalence', fill_value=0
    ).reset_index()
    print(decade_prevalence.to_string(index=False))

    results['decade_prevalence'] = decade_prevalence

    # R: 3.3 Frame dominance over time
    frame_dominance = frame_temporal.copy()
    total_by_year = frame_dominance.groupby('year')['total_frame_score'].transform('sum')
    frame_dominance['total_frames_year'] = total_by_year
    frame_dominance['frame_proportion'] = np.where(
        total_by_year > 0,
        frame_dominance['total_frame_score'] / total_by_year * 100,
        0
    )

    results['frame_dominance'] = frame_dominance

    # R: 3.4 Statistical tests for temporal trends
    print("\nTesting for temporal trends in frame usage:")

    frame_trend_tests_list = []

    for frame in frame_data['frame_type'].unique():
        frame_subset = frame_data[
            (frame_data['frame_type'] == frame) &
            (frame_data['year'].notna())
        ]

        if len(frame_subset) > 5:
            try:
                # R: model <- lm(frame_score ~ year, data = frame_subset)
                X = sm.add_constant(frame_subset['year'].values)
                y = frame_subset['frame_score'].values
                ols_model = sm.OLS(y, X).fit()

                frame_trend_tests_list.append({
                    'frame_type': frame,
                    'slope': round(ols_model.params[1], 6),
                    'p_value': round(ols_model.pvalues[1], 4),
                    'r_squared': round(ols_model.rsquared, 4)
                })
            except Exception as e:
                print(f"Error analyzing frame: {frame} - {e}")

    frame_trend_tests = pd.DataFrame(frame_trend_tests_list)
    if not frame_trend_tests.empty:
        frame_trend_tests = frame_trend_tests.sort_values('p_value')
    print(frame_trend_tests.to_string(index=False))

    results['frame_trend_tests'] = frame_trend_tests

    # R: 3.5 Frame correlation over time periods
    # R: early_period <- top_100_topic3 %>% filter(year <= 2000)
    early_period = df[df['year'] <= 2000]
    late_period = df[df['year'] > 2000]

    if len(early_period) > 10 and len(late_period) > 10:
        # R: early_corr <- early_period %>% select(all_of(frame_cols)) %>% cor(...)
        early_corr = early_period[frame_cols].corr()
        late_corr = late_period[frame_cols].corr()

        print("\nFrame correlations - Early period (<=2000):")
        print(early_corr.round(3).to_string())

        print("\nFrame correlations - Late period (>2000):")
        print(late_corr.round(3).to_string())

        results['early_corr'] = early_corr
        results['late_corr'] = late_corr

    return results


def create_frame_visualizations(df, frame_data, frame_summary,
                                temporal_results=None):
    """
    Comprehensive frame visualizations.

    R equivalent (Section 4):
        frame_plots <- list()
        frame_plots$distribution <- frame_data %>% ggplot(...) + geom_boxplot(...)
        ...

    Parameters
    ----------
    df : pd.DataFrame with frame scores
    frame_data : pd.DataFrame (long format)
    frame_summary : pd.DataFrame
    temporal_results : dict or None

    Returns
    -------
    dict of matplotlib figures
    """
    print("\n4. CREATING FRAME VISUALIZATIONS")
    print("=" * 40)

    frame_plots = {}
    frame_cols = [c for c in df.columns if c.endswith('_frame')]

    # R: 4.1 Frame distribution comparison (boxplot)
    # R: frame_data %>% ggplot(aes(x = reorder(frame_type, frame_score, median),
    #      y = frame_score, fill = frame_type)) + geom_boxplot(alpha = 0.7, ...)
    fig, ax = plt.subplots(figsize=(10, 6))
    order = frame_data.groupby('frame_type')['frame_score'].median().sort_values().index
    palette = sns.color_palette("viridis", len(order))
    sns.boxplot(
        data=frame_data, x='frame_score', y='frame_type',
        order=order, palette=palette, ax=ax
    )
    ax.set_title('Frame Usage Distribution')
    ax.set_xlabel('Frame Score')
    ax.set_ylabel('Frame Type')
    plt.tight_layout()
    frame_plots['distribution'] = fig

    # R: 4.2 Frame prevalence bar chart
    # R: frame_prevalence %>% ggplot(aes(x = reorder(frame_type, prevalence), ...))
    fig, ax = plt.subplots(figsize=(10, 6))
    frame_prevalence = frame_data.groupby('frame_type').apply(
        lambda x: (x['frame_score'] > 0).sum() / len(x) * 100
    ).reset_index()
    frame_prevalence.columns = ['frame_type', 'prevalence']
    frame_prevalence = frame_prevalence.sort_values('prevalence')

    ax.barh(
        frame_prevalence['frame_type'],
        frame_prevalence['prevalence'],
        color=sns.color_palette("viridis", len(frame_prevalence))
    )
    ax.set_title('Frame Prevalence')
    ax.set_xlabel('Prevalence (%)')
    ax.set_ylabel('Frame Type')
    plt.tight_layout()
    frame_plots['prevalence'] = fig

    # R: 4.3 Temporal frame evolution (if year available)
    if temporal_results and 'frame_temporal' in temporal_results:
        frame_temporal = temporal_results['frame_temporal']

        # R: geom_smooth(method = "loess", se = FALSE) + geom_point()
        fig, ax = plt.subplots(figsize=(12, 6))
        for frame_type, color in zip(
            frame_temporal['frame_type'].unique(),
            sns.color_palette("viridis", frame_temporal['frame_type'].nunique())
        ):
            subset = frame_temporal[frame_temporal['frame_type'] == frame_type]
            ax.plot(subset['year'], subset['mean_frame_score'],
                    'o-', label=frame_type, color=color, alpha=0.7, markersize=4)

        ax.set_title('Frame Usage Evolution Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Mean Frame Score')
        ax.legend(loc='best', title='Frame')
        plt.tight_layout()
        frame_plots['temporal_trends'] = fig

        # R: Frame dominance over time
        frame_dominance = temporal_results.get('frame_dominance')
        if frame_dominance is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            for frame_type, color in zip(
                frame_dominance['frame_type'].unique(),
                sns.color_palette("viridis",
                                  frame_dominance['frame_type'].nunique())
            ):
                subset = frame_dominance[
                    frame_dominance['frame_type'] == frame_type
                ]
                ax.plot(subset['year'], subset['frame_proportion'],
                        '-', label=frame_type, color=color, linewidth=1.5)

            ax.set_title('Frame Dominance Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Frame Proportion (%)')
            ax.legend(loc='best', title='Frame')
            plt.tight_layout()
            frame_plots['dominance_trends'] = fig

        # R: Decade comparison bar chart
        frame_decade = temporal_results.get('frame_decade')
        if frame_decade is not None:
            fig, ax = plt.subplots(figsize=(12, 6))
            decades = sorted(frame_decade['decade'].unique())
            frame_types = frame_decade['frame_type'].unique()
            x = np.arange(len(decades))
            width = 0.8 / len(frame_types)

            for i, ft in enumerate(frame_types):
                subset = frame_decade[frame_decade['frame_type'] == ft]
                vals = [subset[subset['decade'] == d]['mean_score'].values[0]
                        if len(subset[subset['decade'] == d]) > 0 else 0
                        for d in decades]
                ax.bar(x + i * width, vals, width, label=ft, alpha=0.8)

            ax.set_xticks(x + width * len(frame_types) / 2)
            ax.set_xticklabels([str(d) for d in decades])
            ax.set_title('Frame Usage by Decade')
            ax.set_xlabel('Decade')
            ax.set_ylabel('Mean Frame Score')
            ax.legend(loc='best', title='Frame')
            plt.tight_layout()
            frame_plots['decade_comparison'] = fig

    # R: 4.4 Frame correlation heatmap
    # R: corrplot -> seaborn heatmap
    frame_correlation_data = df[frame_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    # R: Clean labels by removing _frame suffix
    labels = [c.replace('_frame', '') for c in frame_correlation_data.columns]
    sns.heatmap(
        frame_correlation_data, ax=ax,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        annot=True, fmt='.2f',
        xticklabels=labels, yticklabels=labels
    )
    ax.set_title('Frame Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    frame_plots['correlation_heatmap'] = fig

    # R: 4.5 Frame intensity distribution
    # R: frame_data %>% filter(frame_score > 0) %>%
    #      ggplot(aes(x = frame_score, fill = frame_type)) +
    #      geom_histogram(bins = 15) + facet_wrap(~frame_type, scales = "free")
    nonzero = frame_data[frame_data['frame_score'] > 0]
    frame_types = nonzero['frame_type'].unique()
    n_frames = len(frame_types)
    ncols = min(3, n_frames)
    nrows = (n_frames + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, ft in enumerate(frame_types):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        subset = nonzero[nonzero['frame_type'] == ft]
        ax.hist(subset['frame_score'], bins=15, alpha=0.7, color='steelblue')
        ax.set_title(ft)
        ax.set_xlabel('Frame Score')
        ax.set_ylabel('Count')

    # Hide empty subplots
    for idx in range(n_frames, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle('Frame Intensity Distribution', fontsize=14)
    plt.tight_layout()
    frame_plots['intensity'] = fig

    return frame_plots


def advanced_frame_analysis(df):
    """
    Advanced frame analysis.

    R equivalent (Section 5):
        5.1 Frame co-occurrence analysis
        5.2 Frame dominance patterns
        5.3 Frame complexity

    Parameters
    ----------
    df : pd.DataFrame with frame score columns

    Returns
    -------
    dict with analysis results
    """
    print("\n5. ADVANCED FRAME ANALYSIS")
    print("=" * 40)

    frame_cols = [c for c in df.columns if c.endswith('_frame')]
    results = {}

    # R: 5.1 Frame co-occurrence analysis
    # R: frame_cooccurrence <- top_100_topic3 %>% select(...) %>%
    #      mutate_all(~ ifelse(. > 0, 1, 0)) %>% as.matrix()
    frame_binary = (df[frame_cols] > 0).astype(int).values

    # R: cooccur_matrix <- t(frame_cooccurrence) %*% frame_cooccurrence
    cooccur_matrix = frame_binary.T @ frame_binary
    np.fill_diagonal(cooccur_matrix, 0)  # R: diag(cooccur_matrix) <- 0

    cooccur_df = pd.DataFrame(
        cooccur_matrix,
        index=[c.replace('_frame', '') for c in frame_cols],
        columns=[c.replace('_frame', '') for c in frame_cols]
    )
    print("Frame co-occurrence counts (how often frames appear together):")
    print(cooccur_df.to_string())

    results['cooccurrence'] = cooccur_df

    # R: 5.2 Frame dominance patterns
    # R: frame_dominance_analysis <- top_100_topic3 %>% rowwise() %>% mutate(
    #      total_frame_score = sum(c_across(all_of(frame_cols))),
    #      dominant_frame = ...)
    df_analysis = df.copy()
    df_analysis['total_frame_score'] = df[frame_cols].sum(axis=1)

    def get_dominant_frame(row):
        scores = row[frame_cols]
        if scores.sum() > 0:
            return scores.idxmax().replace('_frame', '')
        return 'none'

    df_analysis['dominant_frame'] = df.apply(get_dominant_frame, axis=1)

    print("\nDominant frame distribution:")
    dominant_frame_summary = df_analysis['dominant_frame'].value_counts()
    print(dominant_frame_summary.to_string())

    results['dominant_frames'] = dominant_frame_summary

    # R: 5.3 Frame complexity
    # R: frames_used = rowSums(select(., all_of(frame_cols)) > 0)
    df_analysis['frames_used'] = (df[frame_cols] > 0).sum(axis=1)

    def categorize_complexity(n):
        if n == 0:
            return "No frames"
        elif n == 1:
            return "Single frame"
        elif n == 2:
            return "Two frames"
        else:
            return "Multiple frames"

    df_analysis['frame_diversity'] = df_analysis['frames_used'].apply(
        categorize_complexity
    )

    print("\nFrame complexity distribution:")
    complexity_summary = df_analysis['frame_diversity'].value_counts()
    print(complexity_summary.to_string())

    results['complexity'] = complexity_summary
    results['analysis_df'] = df_analysis

    return results


def save_frame_results(frame_plots, frame_results, frame_summary,
                       temporal_results=None, advanced_results=None,
                       output_dir="frame_analysis_results"):
    """
    Save frame analysis results.

    R equivalent (Section 7):
        if (!dir.exists("frame_analysis_results"))
          dir.create("frame_analysis_results")
        for (plot_name in names(frame_plots)) {
          ggsave(filename, frame_plots[[plot_name]], width=12, height=8, dpi=300)
        }
    """
    print("\n7. SAVING FRAME ANALYSIS RESULTS")
    print("=" * 40)

    os.makedirs(output_dir, exist_ok=True)

    # Save plots
    for plot_name, fig in frame_plots.items():
        filename = os.path.join(output_dir, f"{plot_name}.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filename}")

    # Save data summaries
    frame_results.to_csv(
        os.path.join(output_dir, "frame_summary_statistics.csv"), index=False
    )
    frame_summary.to_csv(
        os.path.join(output_dir, "frame_detailed_summary.csv"), index=False
    )

    if temporal_results:
        if 'decade_summary' in temporal_results:
            temporal_results['decade_summary'].to_csv(
                os.path.join(output_dir, "frame_by_decade.csv"), index=False
            )
        if 'frame_trend_tests' in temporal_results:
            temporal_results['frame_trend_tests'].to_csv(
                os.path.join(output_dir, "frame_temporal_trends.csv"), index=False
            )

    if advanced_results and 'analysis_df' in advanced_results:
        advanced_results['analysis_df'].to_csv(
            os.path.join(output_dir, "frame_dominance_by_article.csv"),
            index=False
        )

    print(f"\nFrame analysis complete! Results saved to '{output_dir}/' directory")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block.

    R equivalent flow:
        1. Load top_100_topic3 data
        2. Calculate frame scores
        3. Frame comparison analysis
        4. Temporal frame analysis
        5. Create visualizations
        6. Advanced frame analysis
        7. Save results

    NOTE: This script expects a DataFrame named 'top_100_topic3' with
    columns including 'text', 'rank', 'doc_index', 'year', and possibly 'source'.
    This would typically be the top 100 documents from Topic 3 (Sleep and Work).
    """

    print("=" * 80)
    print("FRAME ANALYSIS - Python Translation")
    print("=" * 80)
    print()
    print("Purpose: Deep dive frame analysis with temporal trends for Topic 3")
    print()
    print("NOTE: This script requires a DataFrame 'top_100_topic3' with")
    print("      columns: text, rank, doc_index, year (optional: source)")
    print()

    # ------------------------------------------------------------------
    # PLACEHOLDER: Load data
    # ------------------------------------------------------------------
    # In practice, load your top 100 documents for Topic 3:
    # top_100_topic3 = pd.read_csv("top_documents/topic_3_Sleep_and_Work_top100.csv")

    print("  [DATA LOADING PLACEHOLDER - exiting]")
    print("  To run, load top_100_topic3 DataFrame above.")
    import sys
    sys.exit(0)

    # ------------------------------------------------------------------
    # Step 1: Calculate frame scores
    # ------------------------------------------------------------------
    top_100_topic3, frame_results = calculate_frame_scores(
        top_100_topic3, text_column='text'
    )

    # ------------------------------------------------------------------
    # Step 2: Frame comparison analysis
    # ------------------------------------------------------------------
    frame_data, frame_summary = frame_comparison_analysis(top_100_topic3)

    # ------------------------------------------------------------------
    # Step 3: Temporal frame analysis
    # ------------------------------------------------------------------
    temporal_results = temporal_frame_analysis(top_100_topic3, frame_data)

    # ------------------------------------------------------------------
    # Step 4: Create visualizations
    # ------------------------------------------------------------------
    frame_plots = create_frame_visualizations(
        top_100_topic3, frame_data, frame_summary, temporal_results
    )

    # ------------------------------------------------------------------
    # Step 5: Advanced frame analysis
    # ------------------------------------------------------------------
    advanced_results = advanced_frame_analysis(top_100_topic3)

    # ------------------------------------------------------------------
    # Step 6: Display visualizations
    # ------------------------------------------------------------------
    print("\n6. DISPLAYING FRAME VISUALIZATIONS")
    print("=" * 40)
    for plot_name in frame_plots:
        print(f"Created: {plot_name}")

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    save_frame_results(
        frame_plots, frame_results, frame_summary,
        temporal_results, advanced_results
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    frame_cols = [c for c in top_100_topic3.columns if c.endswith('_frame')]
    print("\n=== FRAME ANALYSIS SUMMARY ===")
    print(f"Frames analyzed: {len(frame_cols)}")
    print(f"Articles analyzed: {len(top_100_topic3)}")
    if 'year' in top_100_topic3.columns:
        print(f"Time span: {top_100_topic3['year'].min()} - "
              f"{top_100_topic3['year'].max()}")
    print(f"Most prevalent frame: {frame_summary['frame_type'].iloc[0]} "
          f"({frame_summary['percentage_articles'].iloc[0]}% of articles)")
    print(f"Least prevalent frame: {frame_summary['frame_type'].iloc[-1]} "
          f"({frame_summary['percentage_articles'].iloc[-1]}% of articles)")
