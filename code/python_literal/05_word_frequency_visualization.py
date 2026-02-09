"""
05_word_frequency_visualization.py

Literal Python translation of:
    "Chung. Word Frequency Analysis.R"

Author: Joon Chung (translation from R)

Purpose:
    Word Frequency Analysis and Word Clouds for Structural Topic Model.
    Includes:
      1. Basic word frequency analysis
      2. Basic word cloud
      3. Interactive word cloud (approximated with static in Python)
      4. Topic-specific word clouds
      5. Temporal word frequency analysis
      6. Comparative word clouds
      7. Styled word frequency plots
      8. Frequency distribution analysis
      9. Custom styling functions
     10. Summary statistics

IMPORTANT NOTES ON R-TO-PYTHON DIFFERENCES:
    - R's wordcloud package -> Python wordcloud package
    - R's wordcloud2 (interactive) -> static WordCloud in Python
      (for interactive, consider plotly or bokeh)
    - R's RColorBrewer -> matplotlib colormaps or palettable
    - R's colSums() -> np.sum() or .sum()
    - R's out$documents (STM sparse format) -> gensim corpus or sklearn DTM
    - R's comparison.cloud() -> side-by-side word clouds in Python
    - R's ggplot2 -> matplotlib
    - R's viridis() -> matplotlib 'viridis' colormap
    - R's case_when() -> np.select() or pd.cut()
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from wordcloud import WordCloud
from gensim.models import LdaModel
from gensim import corpora

import nltk
from nltk.corpus import stopwords

# Ensure NLTK data
for resource in ['stopwords']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


# =============================================================================
# R equivalent: library(stm); library(wordcloud); library(wordcloud2);
#               library(RColorBrewer); library(dplyr); library(ggplot2);
#               library(tidytext); library(viridis)
# =============================================================================


# =============================================================================
# 1. BASIC WORD FREQUENCY ANALYSIS
# =============================================================================

def basic_word_frequency_analysis(corpus, dictionary):
    """
    Python equivalent of R's Section 1: BASIC WORD FREQUENCY ANALYSIS.

    R equivalent:
        vocab <- out$vocab
        # Handle different formats of out$documents
        if (is.list(out$documents)) {
          word_counts <- rep(0, length(vocab))
          for (i in seq_along(out$documents)) {
            doc <- out$documents[[i]]
            word_indices <- doc[1, ]
            word_frequencies <- doc[2, ]
            for (j in seq_along(word_indices)) {
              word_counts[word_indices[j]] <- word_counts[word_indices[j]] + word_frequencies[j]
            }
          }
        }
        word_freq_df <- data.frame(word = names(word_counts),
                                    freq = as.numeric(word_counts)) %>%
          arrange(desc(freq))

    Parameters
    ----------
    corpus : list of list of (word_id, count) -- gensim corpus format
        R equivalent: out$documents
    dictionary : gensim.corpora.Dictionary
        R equivalent: out$vocab

    Returns
    -------
    pd.DataFrame with columns: word, freq (sorted descending)
    """
    # R: vocab <- out$vocab
    vocab = list(dictionary.values())

    # R: Check structure of out$documents and compute word counts
    print("Computing word frequencies from corpus...")

    # R: word_counts <- rep(0, length(vocab))
    word_counts = {}

    # R: Sum word counts across all documents
    # R: for (i in seq_along(out$documents)) { ... }
    for doc in corpus:
        for word_id, count in doc:
            word = dictionary[word_id]
            word_counts[word] = word_counts.get(word, 0) + count

    # R: word_freq_df <- data.frame(word = ..., freq = ...) %>% arrange(desc(freq))
    word_freq_df = pd.DataFrame([
        {'word': word, 'freq': freq}
        for word, freq in word_counts.items()
    ]).sort_values('freq', ascending=False).reset_index(drop=True)

    # R: print("Top 20 most frequent words:")
    # R: print(head(word_freq_df, 20))
    print("Top 20 most frequent words:")
    print(word_freq_df.head(20).to_string(index=False))

    return word_freq_df


# =============================================================================
# 2. BASIC WORD CLOUD
# =============================================================================

def basic_wordcloud(word_freq_df, output_file="basic_wordcloud.png",
                    min_freq=5, max_words=100):
    """
    Python equivalent of R's Section 2: BASIC WORD CLOUD.

    R equivalent:
        png("basic_wordcloud.png", width = 800, height = 600)
        wordcloud(words = word_freq_df$word,
                  freq = word_freq_df$freq,
                  min.freq = 5, max.words = 100,
                  random.order = FALSE, rot.per = 0.35,
                  colors = brewer.pal(8, "Dark2"))
        dev.off()

    Parameters
    ----------
    word_freq_df : pd.DataFrame with columns: word, freq
    output_file : str
    min_freq : int
    max_words : int
    """
    # Filter by min_freq
    filtered = word_freq_df[word_freq_df['freq'] >= min_freq]

    # Create word frequency dict
    word_freq_dict = dict(zip(filtered['word'], filtered['freq']))

    # R: colors = brewer.pal(8, "Dark2")
    wc = WordCloud(
        width=800, height=600,
        background_color='white',
        max_words=max_words,
        colormap='Dark2',  # Equivalent to brewer.pal "Dark2"
        random_state=42,
        prefer_horizontal=0.65  # R: rot.per = 0.35 means 35% rotated
    )
    wc.generate_from_frequencies(word_freq_dict)

    # R: png(...) ... dev.off()
    wc.to_file(output_file)
    print(f"Saved basic word cloud: {output_file}")


# =============================================================================
# 3. INTERACTIVE WORD CLOUD (wordcloud2 approximation)
# =============================================================================

def interactive_wordcloud(word_freq_df, max_words=100):
    """
    Python approximation of R's Section 3: INTERACTIVE WORD CLOUD (wordcloud2).

    R equivalent:
        wordcloud2(word_freq_df[1:100, ],
                   size = 0.8,
                   color = 'random-light',
                   backgroundColor = "black")

    NOTE: R's wordcloud2 is interactive (HTML widget). This Python version
    creates a static word cloud. For interactive, use plotly or bokeh.
    """
    top_words = word_freq_df.head(max_words)
    word_freq_dict = dict(zip(top_words['word'], top_words['freq']))

    # R: backgroundColor = "black", color = 'random-light'
    wc = WordCloud(
        width=800, height=600,
        background_color='black',
        max_words=max_words,
        colormap='Set3',  # Light colors on dark background
        random_state=42
    )
    wc.generate_from_frequencies(word_freq_dict)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud (interactive equivalent)', color='white')
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    plt.savefig("interactive_wordcloud.png", dpi=150,
                facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print("Saved interactive_wordcloud.png (static approximation)")


# =============================================================================
# 4. TOPIC-SPECIFIC WORD CLOUDS
# =============================================================================

def create_topic_wordcloud(model, topic_num, n_words=50, output_file=None):
    """
    Python equivalent of R's create_topic_wordcloud() function.

    R equivalent:
        create_topic_wordcloud <- function(stm_model, topic_num, n_words = 50) {
          top_words <- labelTopics(stm_model, n = n_words)$prob[topic_num, ]
          topic_df <- data.frame(word = top_words, prob = word_probs)
          wordcloud(words = topic_df$word, freq = topic_df$prob * 1000, ...)
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    topic_num : int (0-based for gensim, 1-based in R)
    n_words : int
    output_file : str or None
    """
    # R: top_words <- labelTopics(stm_model, n = n_words)
    topic_terms = model.show_topic(topic_num, topn=n_words)
    word_freq = {word: prob for word, prob in topic_terms}

    # R: wordcloud(words = ..., freq = ..., colors = brewer.pal(8, "Set2"))
    wc = WordCloud(
        width=600, height=600,
        background_color='white',
        max_words=n_words,
        colormap='Set2',
        random_state=42,
        prefer_horizontal=0.65
    )
    wc.generate_from_frequencies(word_freq)

    if output_file:
        wc.to_file(output_file)
        print(f"Saved topic word cloud: {output_file}")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {topic_num + 1}')
        plt.tight_layout()
        plt.show()


# =============================================================================
# 5. TEMPORAL WORD FREQUENCY ANALYSIS
# =============================================================================

def temporal_word_analysis(corpus, dictionary, metadata, time_var="Year"):
    """
    Python equivalent of R's temporal_word_analysis() function.

    R equivalent:
        temporal_word_analysis <- function(out_data, meta_data, time_var = "Year") {
          doc_topics <- data.frame(doc_id = 1:length(out_data$documents),
                                   time_period = meta_data[[time_var]])
          for (period in unique(doc_topics$time_period)) {
            period_docs <- which(doc_topics$time_period == period)
            ...
          }
        }

    Parameters
    ----------
    corpus : list of gensim corpus documents
    dictionary : gensim.corpora.Dictionary
    metadata : pd.DataFrame
    time_var : str

    Returns
    -------
    pd.DataFrame with columns: word, freq, period
    """
    # R: doc_topics <- data.frame(doc_id = ..., time_period = ...)
    if time_var not in metadata.columns:
        print(f"Warning: '{time_var}' not found in metadata.")
        return pd.DataFrame()

    periods = metadata[time_var].dropna().unique()
    temporal_freq_list = []

    for period in sorted(periods):
        # R: period_docs <- which(doc_topics$time_period == period)
        period_indices = metadata[metadata[time_var] == period].index.tolist()

        # Compute word counts for this period
        period_word_counts = {}
        for idx in period_indices:
            if idx < len(corpus):
                doc = corpus[idx]
                for word_id, count in doc:
                    word = dictionary[word_id]
                    period_word_counts[word] = \
                        period_word_counts.get(word, 0) + count

        # R: temporal_freq[[as.character(period)]] <- data.frame(...)
        for word, freq in period_word_counts.items():
            temporal_freq_list.append({
                'word': word,
                'freq': freq,
                'period': period
            })

    # R: return(do.call(rbind, temporal_freq))
    return pd.DataFrame(temporal_freq_list)


# =============================================================================
# 5b. YEARLY WORD CLOUDS
# =============================================================================

def create_yearly_wordclouds(temporal_data, min_freq=5, output_dir="."):
    """
    Python equivalent of R's create_yearly_wordclouds() function.

    R equivalent:
        create_yearly_wordclouds <- function(temporal_data, min_freq = 5) {
          for (year in years) {
            year_data <- temporal_data[temporal_data$period == year, ]
            year_data <- year_data[year_data$freq >= min_freq, ]
            png(paste0("wordcloud_", year, ".png"), ...)
            wordcloud(words = year_data$word, freq = year_data$freq, ...)
            dev.off()
          }
        }
    """
    years = sorted(temporal_data['period'].unique())

    for year in years:
        year_data = temporal_data[
            (temporal_data['period'] == year) &
            (temporal_data['freq'] >= min_freq)
        ].sort_values('freq', ascending=False)

        if len(year_data) > 0:
            word_freq = dict(zip(year_data['word'], year_data['freq']))
            wc = WordCloud(
                width=600, height=600,
                background_color='white',
                max_words=100,
                colormap='Spectral',
                random_state=42
            )
            wc.generate_from_frequencies(word_freq)

            filename = os.path.join(output_dir, f"wordcloud_{year}.png")
            wc.to_file(filename)
            print(f"Saved: {filename}")


# =============================================================================
# 6. COMPARATIVE WORD CLOUDS
# =============================================================================

def comparison_wordcloud(temporal_data, period1, period2,
                         output_file="comparison_wordcloud.png"):
    """
    Python approximation of R's comparison_wordcloud() function.

    R equivalent:
        comparison_wordcloud <- function(temporal_data, period1, period2) {
          data1 <- temporal_data[temporal_data$period == period1, ]
          data2 <- temporal_data[temporal_data$period == period2, ]
          comparison_df <- merge(data1, data2, by = "word", all = TRUE)
          comparison_df$diff <- comparison_df$freq_2 - comparison_df$freq_1
          comparison.cloud(...)
        }

    NOTE: R's comparison.cloud() has no direct Python equivalent.
    This creates two side-by-side word clouds instead.

    Parameters
    ----------
    temporal_data : pd.DataFrame
    period1, period2 : values for 'period' column
    output_file : str
    """
    # R: data1 <- temporal_data[temporal_data$period == period1, ]
    data1 = temporal_data[temporal_data['period'] == period1][['word', 'freq']]
    data2 = temporal_data[temporal_data['period'] == period2][['word', 'freq']]

    # R: comparison_df <- merge(data1, data2, by="word", suffixes=c("_1","_2"), all=TRUE)
    comparison_df = pd.merge(
        data1, data2, on='word', how='outer',
        suffixes=('_1', '_2')
    )
    comparison_df = comparison_df.fillna(0)

    # R: comparison_df$diff <- comparison_df$freq_2 - comparison_df$freq_1
    comparison_df['diff'] = comparison_df['freq_2'] - comparison_df['freq_1']

    # Create side-by-side word clouds (Python approximation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Period 1
    freq1 = dict(zip(
        comparison_df['word'],
        comparison_df['freq_1'].clip(lower=0)
    ))
    freq1 = {k: v for k, v in freq1.items() if v > 0}
    if freq1:
        wc1 = WordCloud(
            width=600, height=400, background_color='white',
            max_words=100, colormap='Reds', random_state=42
        )
        wc1.generate_from_frequencies(freq1)
        ax1.imshow(wc1, interpolation='bilinear')
    ax1.set_title(f'Period: {period1}', fontsize=14, color='red')
    ax1.axis('off')

    # Period 2
    freq2 = dict(zip(
        comparison_df['word'],
        comparison_df['freq_2'].clip(lower=0)
    ))
    freq2 = {k: v for k, v in freq2.items() if v > 0}
    if freq2:
        wc2 = WordCloud(
            width=600, height=400, background_color='white',
            max_words=100, colormap='Blues', random_state=42
        )
        wc2.generate_from_frequencies(freq2)
        ax2.imshow(wc2, interpolation='bilinear')
    ax2.set_title(f'Period: {period2}', fontsize=14, color='blue')
    ax2.axis('off')

    plt.suptitle('Comparison Word Cloud', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved comparison word cloud: {output_file}")


# =============================================================================
# 7. STYLED WORD FREQUENCY PLOTS
# =============================================================================

def styled_word_frequency_plot(word_freq_df, n_top=20,
                                output_file="top_words_plot.png"):
    """
    Python equivalent of R's styled word frequency bar plot.

    R equivalent:
        top_words_plot <- word_freq_df %>%
          head(20) %>%
          ggplot(aes(x = reorder(word, freq), y = freq)) +
          geom_col(fill = "steelblue", alpha = 0.8) +
          coord_flip() +
          labs(title = "Top 20 Most Frequent Words", ...) +
          theme_minimal() +
          theme(plot.title = element_text(size = 16, face = "bold"))

    Parameters
    ----------
    word_freq_df : pd.DataFrame
    n_top : int
    output_file : str
    """
    # R: word_freq_df %>% head(20) %>% ggplot(...)
    top_words = word_freq_df.head(n_top).sort_values('freq', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_words['word'], top_words['freq'],
            color='steelblue', alpha=0.8)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Words')
    ax.set_title(f'Top {n_top} Most Frequent Words',
                 fontsize=16, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_file}")

    return fig


# =============================================================================
# 8. FREQUENCY DISTRIBUTION ANALYSIS
# =============================================================================

def frequency_distribution_analysis(word_freq_df):
    """
    Python equivalent of R's frequency distribution analysis.

    R equivalent:
        freq_distribution <- word_freq_df %>%
          mutate(freq_category = case_when(
            freq >= 100 ~ "Very High (100+)",
            freq >= 50 ~ "High (50-99)",
            freq >= 20 ~ "Medium (20-49)",
            freq >= 10 ~ "Low (10-19)",
            TRUE ~ "Very Low (<10)"
          )) %>%
          count(freq_category) %>%
          mutate(percentage = round(n/sum(n) * 100, 1))

    Parameters
    ----------
    word_freq_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # R: case_when(...) -> np.select()
    conditions = [
        word_freq_df['freq'] >= 100,
        word_freq_df['freq'] >= 50,
        word_freq_df['freq'] >= 20,
        word_freq_df['freq'] >= 10,
    ]
    choices = [
        "Very High (100+)",
        "High (50-99)",
        "Medium (20-49)",
        "Low (10-19)",
    ]
    word_freq_df_copy = word_freq_df.copy()
    word_freq_df_copy['freq_category'] = np.select(
        conditions, choices, default="Very Low (<10)"
    )

    # R: count(freq_category) %>% mutate(percentage = ...)
    freq_distribution = word_freq_df_copy.groupby(
        'freq_category'
    ).size().reset_index(name='n')
    freq_distribution['percentage'] = round(
        freq_distribution['n'] / freq_distribution['n'].sum() * 100, 1
    )

    print("Word Frequency Distribution:")
    print(freq_distribution.to_string(index=False))

    return freq_distribution


# =============================================================================
# 9. CUSTOM STYLING FUNCTIONS
# =============================================================================

def custom_wordcloud(words_df, title="Word Cloud",
                     color_palette="viridis",
                     background_color="white",
                     output_file=None):
    """
    Python equivalent of R's custom_wordcloud() function.

    R equivalent:
        custom_wordcloud <- function(words_df, title = "Word Cloud",
                                     color_palette = "viridis",
                                     background_color = "white") {
          if (color_palette == "viridis") {
            colors <- viridis(n = nrow(words_df), alpha = 0.8)
          } else {
            colors <- brewer.pal(min(nrow(words_df), 11), color_palette)
          }
          wordcloud(words = words_df$word, freq = words_df$freq,
                    min.freq = 2, max.words = 150, ...)
          title(main = title, cex.main = 1.5)
        }

    Parameters
    ----------
    words_df : pd.DataFrame with columns: word, freq
    title : str
    color_palette : str (matplotlib colormap name)
    background_color : str
    output_file : str or None
    """
    word_freq = dict(zip(words_df['word'], words_df['freq']))
    # Filter min_freq >= 2
    word_freq = {k: v for k, v in word_freq.items() if v >= 2}

    wc = WordCloud(
        width=800, height=600,
        background_color=background_color,
        max_words=150,
        colormap=color_palette,
        random_state=42,
        prefer_horizontal=0.65
    )
    wc.generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=18)

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
    else:
        plt.tight_layout()
        plt.show()

    return fig


# =============================================================================
# 10. SUMMARY STATISTICS
# =============================================================================

def print_summary_statistics(word_freq_df):
    """
    Python equivalent of R's summary statistics output.

    R equivalent:
        cat("=== WORD FREQUENCY SUMMARY ===\n")
        cat("Total unique words:", nrow(word_freq_df), "\n")
        cat("Total word occurrences:", sum(word_freq_df$freq), "\n")
        cat("Most frequent word:", word_freq_df$word[1], "(", word_freq_df$freq[1], "times)\n")
        cat("Median frequency:", median(word_freq_df$freq), "\n")
        cat("Words appearing only once:", sum(word_freq_df$freq == 1), "\n")
    """
    print("=== WORD FREQUENCY SUMMARY ===")
    print(f"Total unique words: {len(word_freq_df)}")
    print(f"Total word occurrences: {word_freq_df['freq'].sum()}")
    print(f"Most frequent word: {word_freq_df['word'].iloc[0]} "
          f"({word_freq_df['freq'].iloc[0]} times)")
    print(f"Median frequency: {word_freq_df['freq'].median()}")
    print(f"Words appearing only once: {(word_freq_df['freq'] == 1).sum()}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block.

    R script flow:
        1. Get vocab and word counts from STM processed data
        2. Create basic word cloud
        3. Create interactive word cloud
        4. Topic-specific word clouds
        5. Temporal word frequency analysis
        6. Comparative word clouds
        7. Styled word frequency plots
        8. Frequency distribution analysis
        9. Custom styling
       10. Summary statistics
    """

    print("=" * 80)
    print("WORD FREQUENCY ANALYSIS - Python Translation")
    print("=" * 80)
    print()
    print("This script performs word frequency analysis and generates word clouds.")
    print()
    print("NOTE: This script requires:")
    print("  1. A trained topic model (gensim LdaModel)")
    print("  2. The corpus and dictionary used for training")
    print("  3. Metadata with 'Year' column for temporal analysis")
    print()

    # ------------------------------------------------------------------
    # PLACEHOLDER: Load model and data
    # ------------------------------------------------------------------
    # topic_model_prev = LdaModel.load("topic_model_prev.gensim")
    # dictionary = corpora.Dictionary.load("dictionary.gensim")
    # corpus = list(corpora.MmCorpus("corpus.mm"))
    # metadata = pd.read_csv("metadata.csv")  # with 'Year' column

    print("  [DATA LOADING PLACEHOLDER - exiting]")
    print("  To run, uncomment and configure data loading above.")
    import sys
    sys.exit(0)

    # ------------------------------------------------------------------
    # 1. Basic word frequency analysis
    # ------------------------------------------------------------------
    # R: vocab <- out$vocab; word_counts <- ...
    word_freq_df = basic_word_frequency_analysis(corpus, dictionary)

    # ------------------------------------------------------------------
    # 2. Basic word cloud
    # ------------------------------------------------------------------
    # R: png("basic_wordcloud.png", ...); wordcloud(...); dev.off()
    basic_wordcloud(word_freq_df, output_file="basic_wordcloud.png")

    # ------------------------------------------------------------------
    # 3. Interactive word cloud (static approximation)
    # ------------------------------------------------------------------
    # R: wordcloud2(word_freq_df[1:100, ], ...)
    interactive_wordcloud(word_freq_df, max_words=100)

    # ------------------------------------------------------------------
    # 4. Topic-specific word clouds
    # ------------------------------------------------------------------
    # R: for(i in 1:topic_model_prev$settings$dim$K) {
    #      create_topic_wordcloud(topic_model_prev, i)
    #    }
    # Uncomment to create topic-specific word clouds:
    # for topic_id in range(topic_model_prev.num_topics):
    #     create_topic_wordcloud(
    #         topic_model_prev, topic_id,
    #         output_file=f"topic_{topic_id+1}_wordcloud.png"
    #     )

    # ------------------------------------------------------------------
    # 5. Temporal word frequency analysis
    # ------------------------------------------------------------------
    # R: temporal_data <- temporal_word_analysis(out, out$meta, "Year")
    temporal_data = temporal_word_analysis(
        corpus, dictionary, metadata, time_var="Year"
    )

    # R: create_yearly_wordclouds(temporal_data)
    # Uncomment to create yearly word clouds:
    # create_yearly_wordclouds(temporal_data, min_freq=5)

    # ------------------------------------------------------------------
    # 6. Comparative word clouds
    # ------------------------------------------------------------------
    # R: comparison_wordcloud(temporal_data, min(period), max(period))
    if not temporal_data.empty:
        min_period = temporal_data['period'].min()
        max_period = temporal_data['period'].max()
        comparison_wordcloud(temporal_data, min_period, max_period)

    # ------------------------------------------------------------------
    # 7. Styled word frequency plot
    # ------------------------------------------------------------------
    # R: top_words_plot <- word_freq_df %>% head(20) %>% ggplot(...)
    styled_word_frequency_plot(word_freq_df, n_top=20)

    # ------------------------------------------------------------------
    # 8. Frequency distribution analysis
    # ------------------------------------------------------------------
    freq_distribution = frequency_distribution_analysis(word_freq_df)

    # ------------------------------------------------------------------
    # 9. Custom word cloud examples
    # ------------------------------------------------------------------
    # R: custom_wordcloud(word_freq_df[1:100, ], "Top 100 Words", "Set3")
    custom_wordcloud(
        word_freq_df.head(100),
        title="Top 100 Words",
        color_palette="Set3",
        output_file="custom_wordcloud_top100.png"
    )

    # ------------------------------------------------------------------
    # 10. Summary statistics
    # ------------------------------------------------------------------
    print_summary_statistics(word_freq_df)

    print()
    print("=" * 80)
    print("WORD FREQUENCY ANALYSIS COMPLETE")
    print("=" * 80)
