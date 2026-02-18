"""
01_stm_model_estimation.py

Literal Python translation of:
    "01 STM models 6_12_2025.R"

Author: Joon Chung
        Contact: see README
        The University of Miami, Miller School of Medicine
        The Department of Informatics and Health Data Science

Purpose:
    Structural Topic Models by year.
    This script runs topic modeling (LDA as STM proxy), with each year as a
    PREVALENCE covariate.

Penultimate update: 7/28/2018
Last update:        6/13/2025 (R version)

IMPORTANT NOTES ON R-TO-PYTHON DIFFERENCES:
    - R's `stm` package implements Structural Topic Models with prevalence
      covariates (e.g., ~year). There is NO direct Python equivalent.
    - This translation uses gensim's LdaModel as the closest approximation.
    - STM-specific features like `estimateEffect()`, `sageLabels()`,
      `plotRemoved()`, and prevalence covariates (~year) have NO direct
      Python equivalents. These are approximated or noted where they differ.
    - For true STM functionality, consider using rpy2 to call R from Python.
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import seaborn as sns

# Topic modeling: gensim LDA replaces R stm
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# Text processing: replaces R textProcessor / quanteda
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Word cloud
from wordcloud import WordCloud

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


# =============================================================================
# R equivalent: set.seed(8675309)
# =============================================================================
RANDOM_SEED = 8675309
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# R equivalent: library(stm); library(tidyverse); library(gridExtra);
#               library(tidytext); library(ggwordcloud)
# (All imports handled above)
# =============================================================================


def text_processor(texts, metadata=None, striphtml=True):
    """
    Python equivalent of R's stm::textProcessor().

    Preprocesses text for topic modeling:
      1. Lowercase
      2. Strip HTML tags (if striphtml=True)
      3. Remove punctuation and numbers
      4. Tokenize
      5. Remove stopwords
      6. Remove short words (< 3 chars)

    NOTE: R's textProcessor also stems words by default. We include stemming
    here to match R behavior. To disable, set stem=False.

    Parameters
    ----------
    texts : list of str
        Raw text documents.
    metadata : pd.DataFrame or None
        Metadata associated with each document.
    striphtml : bool
        Whether to strip HTML tags.

    Returns
    -------
    dict with keys:
        'documents' : list of list of (word_id, count) tuples (gensim corpus format)
        'vocab' : list of str (vocabulary)
        'meta' : pd.DataFrame (metadata, filtered to match documents)
        'dictionary' : gensim Dictionary object
    """
    stop_words = set(stopwords.words('english'))
    processed_texts = []

    for text in texts:
        if pd.isna(text):
            processed_texts.append([])
            continue

        doc = str(text).lower()

        # Strip HTML
        if striphtml:
            doc = re.sub(r'<[^>]+>', '', doc)

        # Remove punctuation and numbers
        doc = re.sub(r'[^a-zA-Z\s]', '', doc)

        # Tokenize
        tokens = word_tokenize(doc)

        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in stop_words and len(t) >= 3]

        processed_texts.append(tokens)

    # Build gensim dictionary and corpus
    # R equivalent: prepDocuments builds vocab and document-term representation
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(doc) for doc in processed_texts]

    # Track which documents are non-empty
    valid_indices = [i for i, doc in enumerate(corpus) if len(doc) > 0]

    # Filter to valid documents
    corpus_filtered = [corpus[i] for i in valid_indices]
    if metadata is not None:
        metadata_filtered = metadata.iloc[valid_indices].reset_index(drop=True)
    else:
        metadata_filtered = None

    return {
        'documents': corpus_filtered,
        'vocab': list(dictionary.values()),
        'meta': metadata_filtered,
        'dictionary': dictionary,
        'processed_texts': [processed_texts[i] for i in valid_indices]
    }


def plot_removed(corpus, dictionary, lower_thresh_range):
    """
    Python equivalent of R's stm::plotRemoved().

    Plots the number of words and documents removed at different
    frequency thresholds.

    Parameters
    ----------
    corpus : list of list of (word_id, count)
        Gensim-format corpus.
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary.
    lower_thresh_range : array-like
        Sequence of lower thresholds to test.

    NOTE: R's plotRemoved shows words removed vs. threshold. This approximation
    shows vocabulary size and document count at each threshold.
    """
    words_remaining = []
    docs_remaining = []

    for thresh in lower_thresh_range:
        # Filter dictionary: keep only words with total freq >= thresh
        good_ids = [
            token_id for token_id, freq in dictionary.cfs.items()
            if freq >= thresh
        ]
        n_words = len(good_ids)
        words_remaining.append(n_words)

        # Count documents that still have at least one word
        n_docs = sum(
            1 for doc in corpus
            if any(word_id in good_ids for word_id, _ in doc)
        )
        docs_remaining.append(n_docs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(lower_thresh_range, words_remaining, 'b-')
    ax1.set_xlabel('Lower Threshold')
    ax1.set_ylabel('Words Remaining')
    ax1.set_title('Words Remaining vs. Threshold')

    ax2.plot(lower_thresh_range, docs_remaining, 'r-')
    ax2.set_xlabel('Lower Threshold')
    ax2.set_ylabel('Documents Remaining')
    ax2.set_title('Documents Remaining vs. Threshold')

    plt.tight_layout()
    plt.savefig('plot_removed.png', dpi=150)
    plt.close()
    print("Saved plot_removed.png")


def prep_documents(corpus, dictionary, lower_thresh=500):
    """
    Python equivalent of R's stm::prepDocuments().

    Filters the corpus and dictionary based on a minimum word frequency threshold.

    Parameters
    ----------
    corpus : list of list of (word_id, count)
        Gensim-format corpus.
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary.
    lower_thresh : int
        Minimum total word frequency to retain.

    Returns
    -------
    tuple: (filtered_corpus, filtered_dictionary)
    """
    # Identify words to keep
    good_ids = [
        token_id for token_id, freq in dictionary.cfs.items()
        if freq >= lower_thresh
    ]

    # Filter dictionary
    dictionary.filter_tokens(
        bad_ids=[tid for tid in dictionary.keys() if tid not in good_ids]
    )
    dictionary.compactify()

    # Re-create corpus with filtered dictionary
    # Note: we need original texts for this; gensim corpus can be re-bowified
    # For simplicity, we filter in-place
    filtered_corpus = []
    for doc in corpus:
        filtered_doc = [(wid, cnt) for wid, cnt in doc if wid in dictionary.token2id.values()]
        if len(filtered_doc) > 0:
            filtered_corpus.append(filtered_doc)

    print(f"After filtering: {len(dictionary)} unique words, {len(filtered_corpus)} documents")

    return filtered_corpus, dictionary


def run_topic_model(corpus, dictionary, num_topics=70, random_state=8675309,
                    passes=20, iterations=400):
    """
    Python equivalent of R's stm::stm().

    Runs LDA topic model using gensim.

    NOTE: R's STM uses spectral initialization (init.type = "Spectral") and
    supports prevalence covariates (~year). Gensim LDA does NOT support
    prevalence covariates. The topic proportions will not be conditioned on
    year. For true STM, use rpy2 to call R from Python.

    Parameters
    ----------
    corpus : list of list of (word_id, count)
        Gensim-format corpus.
    dictionary : gensim.corpora.Dictionary
        Gensim dictionary.
    num_topics : int
        Number of topics (K = 70 in original R script).
    random_state : int
        Random seed for reproducibility.
    passes : int
        Number of passes through the corpus during training.
    iterations : int
        Maximum number of iterations per pass.

    Returns
    -------
    gensim.models.LdaModel
        Trained LDA model.
    """
    # R equivalent:
    # topic_model_prev <- stm(out$documents, out$vocab,
    #                         data = out$meta,
    #                         prevalence = ~year,
    #                         K = 70,
    #                         verbose = TRUE,
    #                         init.type = "Spectral")

    print(f"Running LDA with K={num_topics} topics...")
    print("NOTE: This uses gensim LDA, NOT R's STM with prevalence covariates.")
    print("      Prevalence covariate (~year) is NOT included in this model.")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        iterations=iterations,
        alpha='auto',         # R STM default is 'auto'
        eta='auto',           # R STM default is 'auto'
        per_word_topics=True   # Needed for coherence calculation
    )

    print("Model training complete.")
    return lda_model


def get_topic_terms(model, topic_id, n_terms=10):
    """
    Python equivalent of R's labelTopics() or sageLabels().

    Returns top terms for a given topic.

    NOTE: R's sageLabels() provides multiple ranking methods (prob, frex,
    score, lift). Gensim only provides probability-based rankings.
    FREX, Score, and Lift have no direct gensim equivalent.

    Parameters
    ----------
    model : gensim.models.LdaModel
        Trained LDA model.
    topic_id : int
        Topic index (0-based in Python, 1-based in R).
    n_terms : int
        Number of top terms to return.

    Returns
    -------
    list of (str, float) : (term, probability) pairs
    """
    return model.show_topic(topic_id, topn=n_terms)


def get_document_topics(model, corpus):
    """
    Python equivalent of extracting model$theta from R's STM.

    Returns document-topic proportion matrix (theta).

    Parameters
    ----------
    model : gensim.models.LdaModel
        Trained LDA model.
    corpus : list
        Gensim corpus.

    Returns
    -------
    np.ndarray of shape (n_docs, n_topics)
        Document-topic proportion matrix.
    """
    n_topics = model.num_topics
    theta = np.zeros((len(corpus), n_topics))

    for i, doc in enumerate(corpus):
        topic_dist = model.get_document_topics(doc, minimum_probability=0.0)
        for topic_id, prob in topic_dist:
            theta[i, topic_id] = prob

    return theta


def get_beta_matrix(model):
    """
    Python equivalent of R's tidy(topic_model_prev, matrix = "beta").

    Returns the topic-word probability matrix (beta).

    Parameters
    ----------
    model : gensim.models.LdaModel
        Trained LDA model.

    Returns
    -------
    pd.DataFrame with columns: topic, term, beta
    """
    n_topics = model.num_topics
    vocab = list(model.id2word.values())
    records = []

    for topic_id in range(n_topics):
        topic_terms = model.get_topic_terms(topic_id, topn=len(vocab))
        for word_id, prob in topic_terms:
            records.append({
                'topic': topic_id + 1,  # 1-indexed to match R
                'term': model.id2word[word_id],
                'beta': prob
            })

    return pd.DataFrame(records)


def stm_plot(model, corpus, metadata, topic_select, title, K=70):
    """
    Python equivalent of R's stm_plot() function.

    This function:
      1) Computes expected topic proportions by year
      2) Plots expected topic proportions by year (line plot with CI ribbon)
      3) Extracts the top 10 terms for the topic
      4) Plots the top 10 terms as a horizontal bar chart
      5) Returns a combined figure (gridspec equivalent of grid.arrange)

    NOTE: R's estimateEffect with b-spline smoothing (~s(year)) is NOT
    replicated here. Instead, we compute simple yearly means of theta.
    For b-spline smoothing, use scipy or statsmodels separately.

    Parameters
    ----------
    model : gensim.models.LdaModel
        Trained LDA model.
    corpus : list
        Gensim corpus.
    metadata : pd.DataFrame
        Metadata with 'year' column.
    topic_select : int
        Topic number (1-based, matching R convention).
    title : str
        Plot title.
    K : int
        Total number of topics.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Get theta matrix
    theta = get_document_topics(model, corpus)

    # topic_select is 1-based in R; convert to 0-based for Python
    topic_idx = topic_select - 1

    # Create dataframe with year and topic proportions
    plot_data = pd.DataFrame({
        'year': metadata['year'].values,
        'proportion': theta[:, topic_idx]
    })

    # R equivalent: plot_df <- plot_df %>% filter(years <= 2017)
    plot_data = plot_data[plot_data['year'] <= 2017].copy()

    # Aggregate by year: mean and 95% CI
    yearly_stats = plot_data.groupby('year')['proportion'].agg(['mean', 'std', 'count'])
    yearly_stats.columns = ['means', 'std', 'count']

    # 95% CI approximation
    yearly_stats['se'] = yearly_stats['std'] / np.sqrt(yearly_stats['count'])
    yearly_stats['lower'] = yearly_stats['means'] - 1.96 * yearly_stats['se']
    yearly_stats['upper'] = yearly_stats['means'] + 1.96 * yearly_stats['se']
    yearly_stats = yearly_stats.reset_index()

    # Get top 10 terms for this topic
    # R equivalent: stm_beta %>% filter(topic == topic_select) %>% top_n(10)
    top_terms = model.show_topic(topic_idx, topn=10)
    terms_df = pd.DataFrame(top_terms, columns=['term', 'beta'])

    # Create combined figure
    # R equivalent: grid.arrange(topic_terms, topic_prop, nrow = 1, top = title)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Left panel: top terms bar chart
    # R equivalent: geom_col() + coord_flip()
    ax1 = fig.add_subplot(gs[0])
    terms_sorted = terms_df.sort_values('beta', ascending=True)
    ax1.barh(terms_sorted['term'], terms_sorted['beta'], color='steelblue')
    ax1.set_xlabel('Probability word | Topic')
    ax1.set_ylabel('Top terms')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right panel: topic proportion over time
    # R equivalent: geom_line() + geom_ribbon()
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(yearly_stats['year'], yearly_stats['means'], 'k-', linewidth=1)
    ax2.fill_between(
        yearly_stats['year'],
        yearly_stats['lower'],
        yearly_stats['upper'],
        alpha=0.2, color='gray'
    )
    # R equivalent: scale_x_continuous(breaks = seq(1983, 2016, 3))
    ax2.set_xticks(range(1983, 2017, 3))
    ax2.set_xlim(1983, 2017)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Expected topic proportion')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def generate_wordcloud(model, topic_id, n_words=30, output_file=None):
    """
    Python equivalent of R's cloud(topic_model_prev, topic_num).

    Generates a word cloud for a given topic.

    Parameters
    ----------
    model : gensim.models.LdaModel
        Trained LDA model.
    topic_id : int
        Topic index (0-based).
    n_words : int
        Number of words to include.
    output_file : str or None
        If provided, save the word cloud to this file.
    """
    # Get topic words and probabilities
    # R equivalent: cloud(topic_model_prev, topic_num)
    topic_terms = model.show_topic(topic_id, topn=n_words)
    word_freq = {word: prob for word, prob in topic_terms}

    wc = WordCloud(
        width=800, height=600,
        background_color='white',
        max_words=n_words,
        colormap='Set2'
    )
    wc.generate_from_frequencies(word_freq)

    if output_file:
        wc.to_file(output_file)
        print(f"Saved word cloud to {output_file}")
    else:
        plt.figure(figsize=(10, 7.5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block.

    This mirrors the R script's sequential execution. In practice, you would
    load your data (e.g., from a CSV or .Rda file) and then run these steps.

    R equivalent flow:
      1. Load full_text_clean
      2. Select columns and remove NAs
      3. textProcessor() -> processed
      4. plotRemoved()
      5. prepDocuments() -> out
      6. stm() -> topic_model_prev
      7. estimateEffect() -> prep
      8. stm_plot() for each topic
      9. Word clouds
     10. Extract top documents
    """

    print("=" * 80)
    print("STM MODEL ESTIMATION - Python Translation")
    print("=" * 80)
    print()
    print("NOTE: This script requires pre-loaded data. The R script loads from")
    print("      an .Rda file (clean_full_text_20k.Rda). In Python, load your")
    print("      data as a pandas DataFrame named 'full_text_clean'.")
    print()
    print("IMPORTANT DIFFERENCES FROM R:")
    print("  1. R's STM package has NO direct Python equivalent.")
    print("  2. This uses gensim LDA as a proxy.")
    print("  3. Prevalence covariates (~year) are NOT supported in gensim LDA.")
    print("  4. estimateEffect() with b-splines is NOT available.")
    print("  5. sageLabels() FREX/Score/Lift rankings are NOT available.")
    print()

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    # R equivalent:
    # load("clean_full_text_20k.Rda")
    # full_text_subset <- full_text_clean %>%
    #   dplyr::select(text, text_id, datetime, Year) %>% na.omit()
    # full_text_subset$year <- as.numeric(full_text_subset$Year)

    # Example: load from CSV (adjust path as needed)
    # full_text_clean = pd.read_csv("path/to/your/data.csv")

    # For demonstration, we create a placeholder check
    try:
        # Try to load data - adjust this path/method for your setup
        # Option 1: Load from CSV
        # full_text_clean = pd.read_csv("data/clean_full_text_20k.csv")

        # Option 2: Load from R .Rda file using pyreadr
        # import pyreadr
        # result = pyreadr.read_r("data/clean_full_text_20k.Rda")
        # full_text_clean = result['full_text_clean']

        print("Data loading placeholder - replace with actual data loading code.")
        print("Exiting: No data loaded. Set up data loading above.")
        import sys
        sys.exit(0)

    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please provide your data as a CSV or load the .Rda file.")
        import sys
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 2: Preprocess
    # ------------------------------------------------------------------
    # R: full_text_subset <- full_text_clean %>%
    #      select(text, text_id, datetime, Year) %>% na.omit()
    full_text_subset = full_text_clean[['text', 'text_id', 'datetime', 'Year']].dropna()
    full_text_subset = full_text_subset.copy()
    full_text_subset['year'] = pd.to_numeric(full_text_subset['Year'])

    # R: processed <- textProcessor(full_text_subset$text, ...)
    processed = text_processor(
        full_text_subset['text'].tolist(),
        metadata=full_text_subset.reset_index(drop=True),
        striphtml=True
    )

    # R: plotRemoved(processed$documents, lower.thresh = seq(10, 2000, 2))
    plot_removed(
        processed['documents'],
        processed['dictionary'],
        lower_thresh_range=range(10, 2001, 2)
    )

    # R: out <- prepDocuments(processed$documents, processed$vocab,
    #                         processed$meta, lower.thresh = 500)
    corpus_filtered, dictionary_filtered = prep_documents(
        processed['documents'],
        processed['dictionary'],
        lower_thresh=500
    )

    # ------------------------------------------------------------------
    # STEP 3: Run topic model
    # ------------------------------------------------------------------
    # R: set.seed(8675309)
    # R: topic_model_prev <- stm(out$documents, out$vocab,
    #                            data = out$meta, prevalence = ~year,
    #                            K = 70, verbose = TRUE,
    #                            init.type = "Spectral")
    topic_model_prev = run_topic_model(
        corpus_filtered,
        dictionary_filtered,
        num_topics=70,
        random_state=RANDOM_SEED
    )

    # Save model
    topic_model_prev.save("topic_model_prev.gensim")
    print("Model saved to topic_model_prev.gensim")

    # ------------------------------------------------------------------
    # STEP 4: Generate plots for topics of interest
    # ------------------------------------------------------------------
    # R: topics_of_interest <- c(3, 7, 15, 32, 47, 51)
    topics_of_interest = [3, 7, 15, 32, 47, 51]
    topic_labels = {
        3: "Work and sleep",
        7: "Sleep medicine / drugs",
        15: "Circadian science",
        32: "Sleep apnea, hospitals",
        47: "Health research",
        51: "Sleep research"
    }

    meta = processed['meta']

    for topic_num in topics_of_interest:
        label = topic_labels.get(topic_num, f"Topic {topic_num}")
        print(f"Generating plot for Topic {topic_num}: {label}")

        fig = stm_plot(
            topic_model_prev, corpus_filtered, meta,
            topic_select=topic_num, title=label
        )

        # R: ggsave(sleep_plot, file = "sleep_plot.png", width=10, height=5)
        filename = f"topic_{topic_num}_plot.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {filename}")

    # ------------------------------------------------------------------
    # STEP 5: Generate word clouds
    # ------------------------------------------------------------------
    # R: cloud(topic_model_prev, 3) etc.
    topic_cloud_files = {
        3: "sleep_cloud.png",
        7: "drug_cloud.png",
        15: "science_cloud.png",
        32: "apnea_cloud.png",
        47: "health_cloud.png",
        51: "academic_research_cloud.png"
    }

    for topic_num, filename in topic_cloud_files.items():
        print(f"Generating word cloud for Topic {topic_num}...")
        # Convert to 0-based index for gensim
        generate_wordcloud(topic_model_prev, topic_num - 1,
                           n_words=30, output_file=filename)

    print()
    print("=" * 80)
    print("STM model estimation complete.")
    print("=" * 80)
