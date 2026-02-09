"""
02_master_stm_analysis.py

THE MAIN ANALYSIS SCRIPT.

Literal Python translation of:
    "Master R script 8_29_2025.R" (primary, 87KB)
    "Master R script.R" (original reference, 91KB)

Author: Joon Chung
        jxc3388@miami.edu
        The University of Miami, Miller School of Medicine
        The Department of Informatics and Health Data Science

Purpose:
    Master analysis script for structural topic modeling of US newspaper
    discourse about sleep (1983-2017). Includes:
      - STM model estimation (via LDA proxy)
      - Topic proportion plotting with confidence intervals
      - Word cloud generation
      - Top document extraction
      - Topic term extraction (probability, score, FREX)
      - Co-occurrence network analysis
      - sageLabels-style formatted output

Penultimate update: 7/28/2018
Last update:        10/16/2025 (R version)

IMPORTANT NOTES ON R-TO-PYTHON DIFFERENCES:
    - R's `stm` package implements Structural Topic Models with prevalence
      covariates. This translation uses gensim LDA as a proxy.
    - STM-specific: estimateEffect(), sageLabels(), plotRemoved(),
      prevalence covariates (~year) have no direct Python equivalents.
    - R's ggplot2 -> matplotlib/seaborn
    - R's gridExtra -> matplotlib gridspec / subplots
    - R's tidyverse/dplyr -> pandas
    - R's quanteda -> nltk/spacy for text processing
    - R's igraph -> networkx
    - R's ggraph -> networkx + matplotlib
    - R's corrplot -> seaborn heatmap
    - R's ggwordcloud -> wordcloud (Python package)
"""

import os
import gc
import re
import sys
import random
import textwrap
import warnings
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
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

# Word cloud
from wordcloud import WordCloud

# Network analysis: replaces R igraph / ggraph
import networkx as nx

# Statistical modeling: replaces R estimateEffect / b-spline / lm
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline
import statsmodels.api as sm

# Progress bars for batch processing
from tqdm import tqdm

# Ensure NLTK data is available
for resource in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource != 'stopwords'
                       else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)


# =============================================================================
# R equivalent: set.seed(8675309)
# =============================================================================
RANDOM_SEED = 8675309
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =============================================================================
# R equivalent: library(stm); library(tidyverse); library(gridExtra);
#               library(tidytext); library(ggwordcloud);
#               library(quanteda); library(igraph); library(ggraph)
# =============================================================================


###############################################################################
#                                                                             #
# SECTION 1: TEXT PROCESSING AND MODEL ESTIMATION                             #
#                                                                             #
###############################################################################


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

    NOTE: R's textProcessor also stems words by default.

    Parameters
    ----------
    texts : list of str
        Raw text documents.
    metadata : pd.DataFrame or None
        Metadata DataFrame.
    striphtml : bool
        Whether to strip HTML tags.

    Returns
    -------
    dict with keys: 'documents', 'vocab', 'meta', 'dictionary', 'processed_texts'
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
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(doc) for doc in processed_texts]

    # Track non-empty documents
    valid_indices = [i for i, doc in enumerate(corpus) if len(doc) > 0]
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


def prep_documents(corpus, dictionary, lower_thresh=500):
    """
    Python equivalent of R's stm::prepDocuments().

    Filters corpus and dictionary based on minimum word frequency threshold.

    Parameters
    ----------
    corpus : list of list of (word_id, count)
    dictionary : gensim.corpora.Dictionary
    lower_thresh : int

    Returns
    -------
    tuple: (filtered_corpus, filtered_dictionary)
    """
    good_ids = [
        token_id for token_id, freq in dictionary.cfs.items()
        if freq >= lower_thresh
    ]

    bad_ids = [tid for tid in dictionary.keys() if tid not in good_ids]
    dictionary.filter_tokens(bad_ids=bad_ids)
    dictionary.compactify()

    filtered_corpus = []
    for doc in corpus:
        filtered_doc = [
            (wid, cnt) for wid, cnt in doc
            if wid in dictionary.token2id.values()
        ]
        if len(filtered_doc) > 0:
            filtered_corpus.append(filtered_doc)

    print(f"After filtering: {len(dictionary)} unique words, "
          f"{len(filtered_corpus)} documents")

    return filtered_corpus, dictionary


def run_topic_model(corpus, dictionary, num_topics=70, random_state=8675309,
                    passes=20, iterations=400):
    """
    Python equivalent of R's stm::stm().

    Runs LDA topic model. NOTE: R's STM with prevalence covariates (~year)
    and spectral initialization are NOT replicated here.

    Parameters
    ----------
    corpus, dictionary, num_topics, random_state, passes, iterations

    Returns
    -------
    gensim.models.LdaModel
    """
    # R equivalent:
    # topic_model_prev <- stm(out$documents, out$vocab,
    #                         data = out$meta, prevalence = ~year,
    #                         K = 70, verbose = TRUE, init.type = "Spectral")
    print(f"Running LDA with K={num_topics} topics...")
    print("NOTE: gensim LDA does NOT support prevalence covariates (~year).")

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=passes,
        iterations=iterations,
        alpha='auto',
        eta='auto',
        per_word_topics=True
    )

    print("Model training complete.")
    return lda_model


###############################################################################
#                                                                             #
# SECTION 2: TOPIC ANALYSIS HELPER FUNCTIONS                                  #
#                                                                             #
###############################################################################


def get_document_topics(model, corpus):
    """
    Python equivalent of extracting model$theta from R STM.

    Returns document-topic proportion matrix (theta).

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list

    Returns
    -------
    np.ndarray of shape (n_docs, n_topics)
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

    Returns topic-word probability matrix as a long-format DataFrame.

    Parameters
    ----------
    model : gensim.models.LdaModel

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


def sage_labels(model, n_terms=5):
    """
    Python approximation of R's sageLabels().

    R's sageLabels() returns multiple ranking methods:
      - marginal$prob: highest probability terms
      - marginal$frex: FREX (frequency + exclusivity) terms
      - marginal$score: score-based terms
      - marginal$lift: lift-based terms

    NOTE: Only probability-based ranking is directly available in gensim.
    FREX, Score, and Lift are approximated below.

    Parameters
    ----------
    model : gensim.models.LdaModel
    n_terms : int

    Returns
    -------
    dict with keys: 'prob', 'frex', 'score', 'lift'
        Each is a list of lists (topics x terms).
    """
    n_topics = model.num_topics
    # Get full topic-word matrix
    topic_word_matrix = model.get_topics()  # shape: (n_topics, n_vocab)
    vocab = [model.id2word[i] for i in range(len(model.id2word))]

    # Probability terms (straightforward)
    prob_terms = []
    for topic_id in range(n_topics):
        top_indices = np.argsort(topic_word_matrix[topic_id])[::-1][:n_terms]
        prob_terms.append([vocab[i] for i in top_indices])

    # FREX approximation: balance frequency (probability) and exclusivity
    # R's FREX = harmonic mean of word rank by probability and exclusivity
    # Exclusivity ~ P(word|topic) / sum_k P(word|topic_k)
    word_sums = topic_word_matrix.sum(axis=0)
    word_sums[word_sums == 0] = 1e-10
    exclusivity = topic_word_matrix / word_sums[np.newaxis, :]

    frex_terms = []
    for topic_id in range(n_topics):
        # FREX = harmonic mean of probability rank and exclusivity rank
        prob_rank = np.argsort(np.argsort(-topic_word_matrix[topic_id])) + 1
        excl_rank = np.argsort(np.argsort(-exclusivity[topic_id])) + 1
        # Harmonic mean of ranks (lower is better)
        frex_score = 2 * (prob_rank * excl_rank) / (prob_rank + excl_rank)
        top_indices = np.argsort(frex_score)[:n_terms]
        frex_terms.append([vocab[i] for i in top_indices])

    # Score approximation: log(P(word|topic)) - log(P(word))
    word_probs = word_sums / word_sums.sum()
    word_probs[word_probs == 0] = 1e-10

    score_terms = []
    for topic_id in range(n_topics):
        topic_probs = topic_word_matrix[topic_id]
        topic_probs_safe = np.where(topic_probs > 0, topic_probs, 1e-10)
        score = np.log(topic_probs_safe) - np.log(word_probs)
        top_indices = np.argsort(score)[::-1][:n_terms]
        score_terms.append([vocab[i] for i in top_indices])

    # Lift approximation: P(word|topic) / P(word)
    lift_terms = []
    for topic_id in range(n_topics):
        lift = topic_word_matrix[topic_id] / word_probs
        top_indices = np.argsort(lift)[::-1][:n_terms]
        lift_terms.append([vocab[i] for i in top_indices])

    return {
        'prob': prob_terms,
        'frex': frex_terms,
        'score': score_terms,
        'lift': lift_terms
    }


###############################################################################
#                                                                             #
# SECTION 3: PLOTTING FUNCTIONS                                               #
#                                                                             #
###############################################################################


def stm_plot(model, corpus, metadata, topic_select, title, K=70):
    """
    Python equivalent of R's stm_plot() function.

    This function:
      1) Computes expected topic proportions by year
      2) Plots expected topic proportions by year with CI ribbon
      3) Plots top 10 terms as horizontal bar chart
      4) Returns combined figure (grid.arrange equivalent)

    NOTE: R uses estimateEffect with b-spline smoothing (~s(year)).
    This Python version uses simple yearly aggregation.
    For b-spline smoothing, consider statsmodels or scipy.

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    metadata : pd.DataFrame with 'year' column
    topic_select : int (1-based)
    title : str
    K : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    theta = get_document_topics(model, corpus)
    topic_idx = topic_select - 1  # Convert to 0-based

    # R: plot_df <- data.frame(years=years, means=means, lower=lower, upper=upper)
    plot_data = pd.DataFrame({
        'year': metadata['year'].values[:len(theta)],
        'proportion': theta[:, topic_idx]
    })

    # R: plot_df <- plot_df %>% filter(years <= 2017)
    plot_data = plot_data[plot_data['year'] <= 2017].copy()

    # Aggregate by year
    yearly_stats = plot_data.groupby('year')['proportion'].agg(
        ['mean', 'std', 'count']
    )
    yearly_stats.columns = ['means', 'std', 'count']
    yearly_stats['se'] = yearly_stats['std'] / np.sqrt(yearly_stats['count'])
    yearly_stats['lower'] = yearly_stats['means'] - 1.96 * yearly_stats['se']
    yearly_stats['upper'] = yearly_stats['means'] + 1.96 * yearly_stats['se']
    yearly_stats = yearly_stats.reset_index()

    # Get top 10 terms
    # R: stm_beta %>% filter(topic == topic_select) %>% arrange(desc(beta)) %>% top_n(10)
    top_terms = model.show_topic(topic_idx, topn=10)
    terms_df = pd.DataFrame(top_terms, columns=['term', 'beta'])

    # R: grid.arrange(topic_terms, topic_prop, nrow = 1, top = title)
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # Left: top terms (R: geom_col() + coord_flip())
    ax1 = fig.add_subplot(gs[0])
    terms_sorted = terms_df.sort_values('beta', ascending=True)
    ax1.barh(terms_sorted['term'], terms_sorted['beta'], color='steelblue')
    ax1.set_xlabel('Probability word | Topic')
    ax1.set_ylabel('Top terms')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: topic proportion over time (R: geom_line() + geom_ribbon())
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(yearly_stats['year'], yearly_stats['means'], 'k-', linewidth=1)
    ax2.fill_between(
        yearly_stats['year'],
        yearly_stats['lower'], yearly_stats['upper'],
        alpha=0.2, color='gray'
    )
    # R: scale_x_continuous(breaks = seq(1983, 2016, 3), limits = c(1983, 2017))
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

    Parameters
    ----------
    model : gensim.models.LdaModel
    topic_id : int (0-based)
    n_words : int
    output_file : str or None
    """
    # R: cloud(topic_model_prev, topic_num)
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
        print(f"Saved word cloud: {output_file}")
    else:
        plt.figure(figsize=(10, 7.5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def stm_to_ggwordcloud(model, topic_id, topic_name="", n_words=30):
    """
    Python equivalent of R's stm_to_ggwordcloud() function.

    Creates a matplotlib-based word cloud figure.

    R equivalent:
        stm_to_ggwordcloud <- function(model, topic, topic_name) {
          beta <- exp(model$beta$logbeta[[1]])
          topic_words <- beta[topic, ]
          top_indices <- order(topic_words, decreasing = TRUE)[1:30]
          word_probs <- data.frame(word = model$vocab[top_indices],
                                   freq = topic_words[top_indices])
          ggplot(word_probs, aes(label = word, size = freq)) +
            geom_text_wordcloud() + ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    topic_id : int (0-based)
    topic_name : str
    n_words : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    topic_terms = model.show_topic(topic_id, topn=n_words)
    word_freq = {word: prob for word, prob in topic_terms}

    wc = WordCloud(
        width=800, height=600,
        background_color='white',
        max_words=n_words,
        colormap='viridis'
    )
    wc.generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    if topic_name:
        ax.set_title(topic_name, fontsize=14)

    return fig


def plot_topic_summary(model, corpus, topics_of_interest, topic_labels=None):
    """
    Python equivalent of R's plot.STM(type="summary", labeltype="prob").

    Creates a summary bar plot showing expected topic proportions
    for selected topics.

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    topics_of_interest : list of int (1-based)
    topic_labels : dict or None
    """
    theta = get_document_topics(model, corpus)
    mean_proportions = theta.mean(axis=0)

    topic_data = []
    for t in topics_of_interest:
        idx = t - 1  # Convert to 0-based
        top_terms = model.show_topic(idx, topn=5)
        label = ', '.join([w for w, _ in top_terms])
        if topic_labels and t in topic_labels:
            label = f"T{t}: {topic_labels[t]}"
        else:
            label = f"T{t}: {label}"
        topic_data.append({
            'topic': t,
            'proportion': mean_proportions[idx],
            'label': label
        })

    df = pd.DataFrame(topic_data)
    df = df.sort_values('proportion', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(topics_of_interest) * 0.5)))
    ax.barh(df['label'], df['proportion'], color='steelblue')
    ax.set_xlabel('Expected Topic Proportion')
    ax.set_title('Topic Summary')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def plot_topic_correlations(model, corpus):
    """
    Python equivalent of R's plot(topicCorr(topic_model_prev)).

    Computes and plots the correlation matrix of topic proportions.

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list

    Returns
    -------
    matplotlib.figure.Figure
    """
    # R: plot(topicCorr(topic_model_prev))
    theta = get_document_topics(model, corpus)
    corr_matrix = np.corrcoef(theta.T)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, ax=ax,
        cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        xticklabels=range(1, model.num_topics + 1),
        yticklabels=range(1, model.num_topics + 1)
    )
    ax.set_title('Topic Correlations')
    ax.set_xlabel('Topic')
    ax.set_ylabel('Topic')

    plt.tight_layout()
    return fig


def plot_document_probability_distribution(model, corpus, topics_of_interest):
    """
    Python equivalent of R's ggplot for document probability distributions.

    R equivalent:
        ggplot(stm_theta, aes(gamma, fill = as.factor(topic))) +
          geom_histogram(alpha = 0.8, show.legend = FALSE) +
          facet_wrap(~ topic, ncol = 3) + ...

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    topics_of_interest : list of int (1-based)

    Returns
    -------
    matplotlib.figure.Figure
    """
    theta = get_document_topics(model, corpus)

    n_topics = len(topics_of_interest)
    ncols = 3
    nrows = (n_topics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    fig.suptitle('Distribution of document probabilities for each topic',
                 fontsize=14)

    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, topic_num in enumerate(topics_of_interest):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        topic_idx = topic_num - 1
        ax.hist(theta[:, topic_idx], bins=30, alpha=0.8, color='steelblue')
        ax.set_title(f'Topic {topic_num}')
        ax.set_xlabel('gamma')
        ax.set_ylabel('Number of stories')

    # Hide empty subplots
    for idx in range(n_topics, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


###############################################################################
#                                                                             #
# SECTION 4: TOP DOCUMENT EXTRACTION                                          #
#                                                                             #
###############################################################################


def extract_top_documents(model, corpus, topic_num, meta_data, n_docs=100):
    """
    Python equivalent of R's extract_top_documents() function.

    Extracts the top n_docs documents most associated with a given topic.

    R equivalent:
        extract_top_documents <- function(model, topic_num, topic_name,
                                          meta_data, n_docs = 100) {
          topic_proportions <- model$theta[, topic_num]
          top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
          ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    topic_num : int (1-based, matching R convention)
    meta_data : pd.DataFrame
    n_docs : int

    Returns
    -------
    pd.DataFrame
    """
    theta = get_document_topics(model, corpus)
    topic_idx = topic_num - 1  # Convert to 0-based

    # R: topic_proportions <- model$theta[, topic_num]
    topic_proportions = theta[:, topic_idx]

    # R: top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
    top_indices = np.argsort(topic_proportions)[::-1][:n_docs]
    top_proportions = topic_proportions[top_indices]

    # R: top_docs_data <- data.frame(Rank = 1:n_docs, ...)
    top_docs_data = pd.DataFrame({
        'Rank': range(1, n_docs + 1),
        'Document_Index': top_indices + 1,  # 1-based to match R
        'Topic_Proportion': np.round(top_proportions, 4)
    })

    # R: Add metadata columns by checking column names
    if meta_data is not None:
        # Text content
        for col_name in ['text', 'Text', 'content']:
            if col_name in meta_data.columns:
                top_docs_data['Text'] = meta_data[col_name].iloc[top_indices].values
                break

        # Publication date
        for col_name in ['datetime', 'date', 'Date', 'publication_date', 'pub_date']:
            if col_name in meta_data.columns:
                top_docs_data['Date'] = meta_data[col_name].iloc[top_indices].values
                break

        # Year
        for col_name in ['year', 'Year']:
            if col_name in meta_data.columns:
                top_docs_data['Year'] = meta_data[col_name].iloc[top_indices].values
                break

        # Source/Newspaper
        for col_name in ['source', 'Source', 'newspaper', 'Newspaper', 'publication']:
            if col_name in meta_data.columns:
                top_docs_data['Source'] = meta_data[col_name].iloc[top_indices].values
                break

        # Title
        for col_name in ['title', 'Title', 'headline']:
            if col_name in meta_data.columns:
                top_docs_data['Title'] = meta_data[col_name].iloc[top_indices].values
                break

        # Document ID
        for col_name in ['id', 'ID', 'doc_id', 'text_id']:
            if col_name in meta_data.columns:
                top_docs_data['Document_ID'] = meta_data[col_name].iloc[top_indices].values
                break

    return top_docs_data


def write_top_documents_to_files(model, corpus, topics_of_interest,
                                 topic_names, meta_data, output_dir="top_documents"):
    """
    Python equivalent of the R loop that writes top 100 articles to file.

    Writes CSV, formatted TXT, and individual TXT files for each topic.

    R equivalent: The for loop starting at
        for (i in 1:length(topics_of_interest)) { ... }
    """
    # R: if (!dir.exists("top_documents")) dir.create("top_documents")
    os.makedirs(output_dir, exist_ok=True)

    for i, (topic_num, topic_name) in enumerate(zip(topics_of_interest, topic_names)):
        print(f"Processing topic {topic_num}: {topic_name}")

        # Extract top documents
        top_docs = extract_top_documents(model, corpus, topic_num, meta_data)

        # Option 1: Save as CSV
        # R: write.csv(top_docs, csv_filename, row.names = FALSE)
        csv_filename = os.path.join(
            output_dir,
            f"topic_{topic_num}_{topic_name}_top100.csv"
        )
        top_docs.to_csv(csv_filename, index=False)

        # Option 2: Save as readable text file
        # R: sink(txt_filename) ... sink()
        txt_filename = os.path.join(
            output_dir,
            f"topic_{topic_num}_{topic_name}_top100.txt"
        )
        with open(txt_filename, 'w') as f:
            f.write(f"TOP 100 DOCUMENTS FOR TOPIC {topic_num}: "
                    f"{topic_name.replace('_', ' ').upper()}\n")
            f.write("=" * 80 + "\n\n")

            for j in range(len(top_docs)):
                row = top_docs.iloc[j]
                f.write(f"RANK: {j+1} | PROPORTION: {row['Topic_Proportion']} "
                        f"| DOC INDEX: {row['Document_Index']}\n")

                if 'Date' in top_docs.columns:
                    f.write(f"DATE: {row['Date']}\n")
                if 'Year' in top_docs.columns:
                    f.write(f"YEAR: {row['Year']}\n")
                if 'Source' in top_docs.columns:
                    f.write(f"SOURCE/NEWSPAPER: {row['Source']}\n")
                if 'Title' in top_docs.columns:
                    f.write(f"TITLE: {row['Title']}\n")
                if 'Document_ID' in top_docs.columns:
                    f.write(f"DOCUMENT ID: {row['Document_ID']}\n")

                f.write("TEXT:\n")
                if 'Text' in top_docs.columns:
                    # R: wrapped_text <- strwrap(text, width = 80)
                    wrapped = textwrap.fill(str(row['Text']), width=80)
                    f.write(wrapped + "\n")

                f.write("\n" + "-" * 80 + "\n\n")

        # Option 3: Individual files
        # R: doc_dir <- paste0("top_documents/topic_", topic_num, "_", ...)
        doc_dir = os.path.join(
            output_dir,
            f"topic_{topic_num}_{topic_name}_individual"
        )
        os.makedirs(doc_dir, exist_ok=True)

        for j in range(len(top_docs)):
            row = top_docs.iloc[j]
            individual_filename = os.path.join(
                doc_dir,
                f"rank_{j+1:03d}_doc_{int(row['Document_Index'])}.txt"
            )
            with open(individual_filename, 'w') as f:
                f.write(f"TOPIC: {topic_name} | RANK: {j+1} | "
                        f"PROPORTION: {row['Topic_Proportion']}\n")
                f.write(f"DOCUMENT INDEX: {int(row['Document_Index'])}\n")

                if 'Date' in top_docs.columns:
                    f.write(f"PUBLICATION DATE: {row['Date']}\n")
                if 'Year' in top_docs.columns:
                    f.write(f"YEAR: {row['Year']}\n")
                if 'Source' in top_docs.columns:
                    f.write(f"SOURCE/NEWSPAPER: {row['Source']}\n")
                if 'Title' in top_docs.columns:
                    f.write(f"TITLE/HEADLINE: {row['Title']}\n")
                if 'Document_ID' in top_docs.columns:
                    f.write(f"DOCUMENT ID: {row['Document_ID']}\n")

                f.write("\nTEXT:\n")
                f.write("=" * 50 + "\n")

                if 'Text' in top_docs.columns:
                    f.write(str(row['Text']) + "\n")

    # Summary file
    # R: summary_filename <- "top_documents/summary_all_topics.txt"
    summary_filename = os.path.join(output_dir, "summary_all_topics.txt")
    with open(summary_filename, 'w') as f:
        f.write("SUMMARY: TOP 100 DOCUMENTS FOR ALL TOPICS OF INTEREST\n")
        f.write("=" * 60 + "\n\n")

        for topic_num, topic_name in zip(topics_of_interest, topic_names):
            top_docs = extract_top_documents(model, corpus, topic_num, meta_data)

            f.write(f"TOPIC {topic_num}: {topic_name.replace('_', ' ').upper()}\n")
            f.write("Top 10 documents (proportion):\n")

            for j in range(min(10, len(top_docs))):
                row = top_docs.iloc[j]
                f.write(f"  {j+1}. Doc {int(row['Document_Index'])} "
                        f"({row['Topic_Proportion']})\n")
            f.write("\n")


###############################################################################
#                                                                             #
# SECTION 5: STM TOPIC TERMS EXTRACTION - SAGELABELS STYLE                   #
#                                                                             #
###############################################################################


def extract_all_topic_terms(model, n_terms=5):
    """
    Python equivalent of R's extract_all_topic_terms() function.

    Uses sage_labels() to extract top terms for all topics using both
    probability and score (approximated) ranking methods.

    R equivalent:
        extract_all_topic_terms <- function(topic_model, n_terms = 5) {
          sage_result <- sageLabels(topic_model, n = n_terms)
          prob_terms <- sage_result$marginal$prob
          score_terms <- sage_result$marginal$score
          ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    n_terms : int

    Returns
    -------
    pd.DataFrame with columns: topic, method, rank, term
    """
    print(f"Extracting {n_terms} terms per method for {model.num_topics} topics...")

    sage_result = sage_labels(model, n_terms=n_terms)

    results_list = []

    # Process probability terms
    for topic_num, terms in enumerate(sage_result['prob'], start=1):
        for rank, term in enumerate(terms, start=1):
            results_list.append({
                'topic': topic_num,
                'method': 'probability',
                'rank': rank,
                'term': term
            })

    # Process score terms
    for topic_num, terms in enumerate(sage_result['score'], start=1):
        for rank, term in enumerate(terms, start=1):
            results_list.append({
                'topic': topic_num,
                'method': 'score',
                'rank': rank,
                'term': term
            })

    final_results = pd.DataFrame(results_list)

    print(f"Extracted {len(final_results)} term entries")
    print(f"Topics: {model.num_topics}")
    print(f"Methods: probability, score")
    print(f"Terms per method per topic: {n_terms}")

    return final_results


def extract_terms_wide_format(model, n_terms=5):
    """
    Python equivalent of R's extract_terms_wide_format() function.

    Creates a wide-format table (Excel-friendly) with one row per topic.

    R equivalent:
        extract_terms_wide_format <- function(topic_model, n_terms = 5) { ... }

    Parameters
    ----------
    model : gensim.models.LdaModel
    n_terms : int

    Returns
    -------
    pd.DataFrame
    """
    print("Extracting terms in wide format...")

    sage_result = sage_labels(model, n_terms=n_terms)

    wide_results = pd.DataFrame({'topic': range(1, model.num_topics + 1)})

    for i in range(n_terms):
        wide_results[f'prob_{i+1}'] = [
            sage_result['prob'][t][i] for t in range(model.num_topics)
        ]
        wide_results[f'score_{i+1}'] = [
            sage_result['score'][t][i] for t in range(model.num_topics)
        ]

    print(f"Created wide format with {len(wide_results.columns)} columns")
    return wide_results


def create_sagelabels_format(model, n_terms=5, topics=None):
    """
    Python equivalent of R's create_sagelabels_format() function.

    Generates output exactly like R's sageLabels console display.

    R equivalent:
        create_sagelabels_format <- function(topic_model, n_terms = 5, topics = NULL) {
          sage_result <- sageLabels(topic_model, n = n_terms)
          for (topic_num in topics) {
            prob_terms <- sage_result$marginal$prob[topic_num, ]
            score_terms <- sage_result$marginal$score[topic_num, ]
            ...
          }
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    n_terms : int
    topics : list of int or None (1-based)

    Returns
    -------
    list of str (output lines)
    """
    if topics is None:
        topics = list(range(1, model.num_topics + 1))

    sage_result = sage_labels(model, n_terms=n_terms)

    output_lines = []

    for topic_num in topics:
        topic_idx = topic_num - 1  # Convert to 0-based
        if topic_idx < model.num_topics:
            prob_terms = sage_result['prob'][topic_idx]
            score_terms = sage_result['score'][topic_idx]

            # R format:
            # Topic X:
            #   Marginal Highest Prob: term1, term2, ...
            #   Marginal Score: term1, term2, ...
            output_lines.append(f"Topic {topic_num}:")
            output_lines.append(
                f" \t Marginal Highest Prob: {', '.join(prob_terms)}"
            )
            output_lines.append(
                f" \t Marginal Score: {', '.join(score_terms)}"
            )
            output_lines.append("")

    return output_lines


def print_sagelabels_format(model, n_terms=5, topics=None):
    """
    Python equivalent of R's print_sagelabels_format().

    Prints sageLabels-format output to console.
    """
    output_lines = create_sagelabels_format(model, n_terms, topics)
    for line in output_lines:
        print(line)


def save_sagelabels_format(model, filename="stm_sagelabels_format.txt",
                            n_terms=5, topics=None):
    """
    Python equivalent of R's save_sagelabels_format().

    Saves sageLabels-format output to a text file.
    """
    output_lines = create_sagelabels_format(model, n_terms, topics)

    with open(filename, 'w') as f:
        f.write('\n'.join(output_lines))

    n_topics = len(topics) if topics else model.num_topics
    print(f"Saved sageLabels format to: {filename}")
    print(f"Topics included: {n_topics}")
    print(f"Terms per method: {n_terms}")


def format_specific_topics(model, topics, n_terms=5):
    """
    Python equivalent of R's format_specific_topics().
    """
    print("=" * 77)
    print("STM TOPIC TERMS - sageLabels Format")
    print("=" * 77)
    print_sagelabels_format(model, n_terms, topics)
    print("=" * 77)


def process_all_sagelabels_format(model, n_terms=5, save_file=True,
                                   filename="stm_all_topics_sagelabels.txt"):
    """
    Python equivalent of R's process_all_sagelabels_format().
    """
    total_topics = model.num_topics
    print(f"Creating sageLabels format for {total_topics} topics "
          f"with {n_terms} terms each...")

    if save_file:
        save_sagelabels_format(model, filename, n_terms, topics=None)
        print()

    print("PREVIEW (First 5 topics):")
    print("=" * 77)
    print_sagelabels_format(model, n_terms, topics=list(range(1, 6)))
    print("=" * 77)

    if save_file:
        print(f"Full output saved to: {filename}")


def save_sagelabels_matrices(model, n_terms=5, prefix="stm_terms"):
    """
    Python equivalent of R's save_sagelabels_matrices().

    Creates separate CSV files for probability and score matrices.
    """
    print("Extracting matrices in sageLabels format...")

    sage_result = sage_labels(model, n_terms=n_terms)

    # Probability matrix
    prob_data = {'topic': range(1, model.num_topics + 1)}
    for i in range(n_terms):
        prob_data[f'term_{i+1}'] = [
            sage_result['prob'][t][i] for t in range(model.num_topics)
        ]
    prob_df = pd.DataFrame(prob_data)

    # Score matrix
    score_data = {'topic': range(1, model.num_topics + 1)}
    for i in range(n_terms):
        score_data[f'term_{i+1}'] = [
            sage_result['score'][t][i] for t in range(model.num_topics)
        ]
    score_df = pd.DataFrame(score_data)

    # Save files
    prob_file = f"{prefix}_probability_matrix.csv"
    score_file = f"{prefix}_score_matrix.csv"

    prob_df.to_csv(prob_file, index=False)
    score_df.to_csv(score_file, index=False)

    print(f"Saved probability matrix: {prob_file}")
    print(f"Saved score matrix: {score_file}")

    return {'probability': prob_df, 'score': score_df}


def analyze_term_overlap(model, n_terms=5):
    """
    Python equivalent of R's analyze_term_overlap() function.

    Shows overlap between probability and score methods.

    R equivalent:
        analyze_term_overlap <- function(topic_model, n_terms = 5) {
          sage_result <- sageLabels(topic_model, n = n_terms)
          ...
          overlap <- intersect(prob_terms, score_terms)
          ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    n_terms : int

    Returns
    -------
    pd.DataFrame
    """
    sage_result = sage_labels(model, n_terms=n_terms)

    records = []
    for topic_idx in range(model.num_topics):
        prob_terms = sage_result['prob'][topic_idx]
        score_terms = sage_result['score'][topic_idx]

        overlap = set(prob_terms) & set(score_terms)
        unique_prob = set(prob_terms) - set(score_terms)
        unique_score = set(score_terms) - set(prob_terms)

        records.append({
            'topic': topic_idx + 1,
            'prob_terms': ', '.join(prob_terms),
            'score_terms': ', '.join(score_terms),
            'overlap_count': len(overlap),
            'overlap_terms': ', '.join(overlap),
            'unique_prob': ', '.join(unique_prob),
            'unique_score': ', '.join(unique_score)
        })

    return pd.DataFrame(records)


###############################################################################
#                                                                             #
# SECTION 6: CO-OCCURRENCE NETWORK ANALYSIS                                   #
#                                                                             #
###############################################################################


def make_dtm(texts):
    """
    Python equivalent of R's make_dtm() function using quanteda.

    Creates a document-term matrix from raw text.

    R equivalent:
        make_dtm <- function(texts) {
          corpus(texts) %>%
            tokens(remove_punct=TRUE, remove_numbers=TRUE) %>%
            tokens_tolower() %>%
            tokens_remove(stopwords("en")) %>%
            tokens_keep(min_nchar = 3) %>%
            dfm() %>%
            dfm_trim(min_termfreq = 2, min_docfreq = 1)
        }

    Parameters
    ----------
    texts : list of str

    Returns
    -------
    tuple: (scipy sparse matrix or np array, list of str vocab)
    """
    from sklearn.feature_extraction.text import CountVectorizer

    stop_words_list = list(stopwords.words('english'))

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=stop_words_list,
        token_pattern=r'[a-zA-Z]{3,}',  # min 3 chars, letters only
        min_df=1,
        # R: dfm_trim(min_termfreq = 2)
    )

    # Filter out None/NaN
    clean_texts = [str(t) if t and not pd.isna(t) else "" for t in texts]

    try:
        dtm = vectorizer.fit_transform(clean_texts)
    except ValueError:
        return None, []

    vocab = vectorizer.get_feature_names_out().tolist()

    # Apply min_termfreq filter (total frequency >= 2)
    term_freqs = np.asarray(dtm.sum(axis=0)).flatten()
    keep_mask = term_freqs >= 2
    dtm = dtm[:, keep_mask]
    vocab = [v for v, keep in zip(vocab, keep_mask) if keep]

    return dtm, vocab


def get_stm_terms(model, topic_num, n_words, method="frex"):
    """
    Python equivalent of R's get_stm_terms() function.

    Extracts top terms for a topic using the specified ranking method.

    R equivalent:
        get_stm_terms <- function(topic_model, topic_num, n_words, method = "frex") {
          sage_result <- sageLabels(topic_model, n = n_words)
          terms <- switch(method,
            "frex" = sage_result$marginal$frex[topic_num, ],
            ...
          )
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    topic_num : int (1-based)
    n_words : int
    method : str, one of "frex", "score", "prob", "lift"

    Returns
    -------
    list of str
    """
    sage_result = sage_labels(model, n_terms=n_words)
    topic_idx = topic_num - 1  # Convert to 0-based

    if method not in sage_result:
        raise ValueError(f"Method must be one of: {list(sage_result.keys())}")

    terms = sage_result[method][topic_idx]
    # Remove any None/empty
    terms = [t for t in terms if t]

    return terms


def make_network(model, corpus, text_data, topic_num, n_words, method="frex"):
    """
    Python equivalent of R's make_network() function.

    Creates a co-occurrence network for a single topic:
      1. Extracts top documents for the topic
      2. Creates DTM from those documents
      3. Gets STM-ranked terms and filters DTM
      4. Calculates word co-occurrence matrix
      5. Creates network graph and visualization
      6. Saves plot and data files

    R equivalent:
        make_network <- function(topic_model, text_data, topic_num, n_words, method) {
          topic_weights <- topic_model$theta[, topic_num]
          top_doc_indices <- order(topic_weights, decreasing = TRUE)[1:100]
          ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    text_data : pd.DataFrame with 'text' column
    topic_num : int (1-based)
    n_words : int
    method : str

    Returns
    -------
    dict or None
    """
    # STEP 1: Get top documents
    # R: topic_weights <- topic_model$theta[, topic_num]
    theta = get_document_topics(model, corpus)
    topic_idx = topic_num - 1
    topic_weights = theta[:, topic_idx]

    # R: top_doc_indices <- order(topic_weights, decreasing = TRUE)[1:100]
    top_doc_indices = np.argsort(topic_weights)[::-1][:100]

    # Extract text
    topic_texts = []
    for idx in top_doc_indices:
        if idx < len(text_data):
            text = text_data.iloc[idx].get('text', '')
            if pd.notna(text) and len(str(text)) > 10:
                topic_texts.append(str(text))

    if len(topic_texts) < 5:
        warnings.warn(f"Topic {topic_num} has insufficient documents ({len(topic_texts)})")
        return None

    # STEP 2: Create DTM
    dtm, vocab = make_dtm(topic_texts)
    if dtm is None or len(vocab) < 5:
        warnings.warn(f"Topic {topic_num} has insufficient terms after preprocessing")
        return None

    # STEP 3: Get STM-ranked terms and filter
    stm_terms = get_stm_terms(model, topic_num, n_words, method)

    # R: available_terms <- intersect(stm_terms, colnames(dtm))
    available_terms = [t for t in stm_terms if t in vocab]

    if len(available_terms) < 3:
        warnings.warn(f"Topic {topic_num} has too few available terms ({len(available_terms)})")
        return None

    # Filter DTM to available terms
    term_indices = [vocab.index(t) for t in available_terms]
    dtm_filtered = dtm[:, term_indices]

    # STEP 4: Calculate co-occurrence matrix
    # R: cooccur_matrix <- t(dtm_filtered) %*% dtm_filtered
    cooccur_matrix = (dtm_filtered.T @ dtm_filtered).toarray()

    # STEP 5: Convert to edge list
    edge_list = []
    for i in range(len(available_terms)):
        for j in range(i + 1, len(available_terms)):
            weight = cooccur_matrix[i, j]
            if weight >= 1:
                edge_list.append({
                    'term1': available_terms[i],
                    'term2': available_terms[j],
                    'weight': weight
                })

    edge_df = pd.DataFrame(edge_list)
    if len(edge_df) == 0:
        warnings.warn(f"Topic {topic_num} has no word co-occurrences")
        return None

    edge_df = edge_df.sort_values('weight', ascending=False)

    # STEP 6: Create network graph
    # R: network_graph <- graph_from_data_frame(edge_list, directed = FALSE)
    G = nx.Graph()
    for _, row in edge_df.iterrows():
        G.add_edge(row['term1'], row['term2'], weight=row['weight'])

    # Add node attributes
    # R: V(network_graph)$frequency <- word_frequencies[V(network_graph)$name]
    word_frequencies = np.asarray(dtm_filtered.sum(axis=0)).flatten()
    freq_dict = dict(zip(available_terms, word_frequencies))
    for node in G.nodes():
        G.nodes[node]['frequency'] = freq_dict.get(node, 1)
        G.nodes[node]['degree'] = G.degree(node)

    # STEP 7: Create visualization
    # R: ggraph(network_graph, layout = "fr") + geom_edge_link() + ...
    node_color = 'steelblue'
    edge_color = 'gray'
    text_color = 'black'

    def create_network_plot(font_size):
        fig, ax = plt.subplots(figsize=(12, 10))

        # Fruchterman-Reingold layout
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=2.0)

        # Draw edges
        edges = G.edges(data=True)
        weights = [d['weight'] for _, _, d in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [0.3 + 2.7 * (w / max_weight) for w in weights]
        edge_alphas = [0.4 + 0.5 * (w / max_weight) for w in weights]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=edge_widths,
            alpha=0.6,
            edge_color=edge_color
        )

        # Draw nodes
        node_sizes = [G.nodes[n]['frequency'] * 50 for n in G.nodes()]
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_color,
            alpha=0.8
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=font_size * 0.6,
            font_color=text_color
        )

        ax.set_axis_off()
        plt.tight_layout()
        return fig

    # STEP 8: Save files
    # R: base_directory <- paste0("stm_networks_", method)
    base_directory = f"stm_networks_{method}"
    os.makedirs(base_directory, exist_ok=True)

    font_sizes = [24, 36]
    plots_created = {}

    for font_size in font_sizes:
        plot_dir = os.path.join(base_directory, f"plots_font{font_size}")
        os.makedirs(plot_dir, exist_ok=True)

        fig = create_network_plot(font_size)

        # Save PNG
        plot_filename = os.path.join(
            plot_dir,
            f"topic_{topic_num:02d}_{n_words}words_font{font_size}.png"
        )
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight',
                    facecolor='white')

        # Save EPS
        eps_dir = os.path.join(base_directory, f"eps_font{font_size}")
        os.makedirs(eps_dir, exist_ok=True)
        eps_filename = os.path.join(
            eps_dir,
            f"topic_{topic_num:02d}_{n_words}words_font{font_size}.eps"
        )
        try:
            fig.savefig(eps_filename, format='eps',
                        bbox_inches='tight', facecolor='white')
        except Exception as e:
            warnings.warn(f"Failed to save EPS for topic {topic_num}: {e}")

        plots_created[f'font{font_size}'] = fig
        plt.close(fig)

    # Save edge list data
    data_dir = os.path.join(base_directory, "data")
    os.makedirs(data_dir, exist_ok=True)
    data_filename = os.path.join(
        data_dir,
        f"topic_{topic_num:02d}_{n_words}words_edges.csv"
    )

    # R: edge_list_annotated <- edge_list %>% mutate(topic_number = topic_num, ...)
    edge_df_annotated = edge_df.copy()
    edge_df_annotated['topic_number'] = topic_num
    edge_df_annotated['ranking_method'] = method
    edge_df_annotated['word_count'] = n_words
    edge_df_annotated['extraction_date'] = date.today().isoformat()

    edge_df_annotated.to_csv(data_filename, index=False)

    print(f"  Topic {topic_num}: {len(available_terms)} terms, "
          f"{G.number_of_edges()} connections")

    return {
        'graph': G,
        'plots': plots_created,
        'terms': available_terms,
        'edges': len(edge_df),
        'method': method,
        'font_sizes_created': font_sizes
    }


def process_networks(model, corpus, text_data, method="frex",
                     topics=None, word_counts=None):
    """
    Python equivalent of R's process_networks() function.

    Batch processing function that creates networks for multiple topics.

    R equivalent:
        process_networks <- function(topic_model, text_data, method = "frex",
                                     topics = 1:70, word_counts = c(15, 30, 50)) { ... }

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list
    text_data : pd.DataFrame
    method : str
    topics : list of int or None
    word_counts : list of int or None

    Returns
    -------
    dict with processing statistics
    """
    if topics is None:
        topics = list(range(1, model.num_topics + 1))
    if word_counts is None:
        word_counts = [15, 30, 50]

    if method not in ['frex', 'score', 'prob', 'lift']:
        raise ValueError("method must be one of: frex, score, prob, lift")

    total_combinations = len(topics) * len(word_counts)
    success_count = 0
    failed_topics = []

    print("=" * 77)
    print("STM CO-OCCURRENCE NETWORK BATCH PROCESSING")
    print("=" * 77)
    print(f"Settings:")
    print(f"  Method: {method.upper()}")
    print(f"  Topics: {len(topics)} ({min(topics)} to {max(topics)})")
    print(f"  Word counts: {', '.join(map(str, word_counts))}")
    print(f"  Font sizes: 24 and 36 (both generated automatically)")
    print(f"  Total networks: {total_combinations * 4} "
          f"({total_combinations} x 2 font sizes x 2 formats)")
    print(f"  Output directory: stm_networks_{method}/")
    print("=" * 77)

    start_time = datetime.now()

    for topic_num in tqdm(topics, desc="Processing topics"):
        topic_success = False

        for n_words in word_counts:
            try:
                result = make_network(model, corpus, text_data,
                                      topic_num, n_words, method)
                if result is not None:
                    success_count += 1
                    topic_success = True
            except Exception as e:
                warnings.warn(f"Error processing Topic {topic_num} "
                              f"({n_words} words): {e}")

        if not topic_success:
            failed_topics.append(topic_num)

        # Memory cleanup every 20 topics
        if topic_num % 20 == 0:
            gc.collect()

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds() / 60.0
    success_rate = round((success_count / total_combinations) * 100, 1)
    total_files_created = success_count * 4

    summary_stats = {
        'method': method,
        'total_attempted': total_combinations,
        'successful_networks': success_count,
        'total_files_created': total_files_created,
        'failed_topics': failed_topics,
        'success_rate_percent': success_rate,
        'processing_time_minutes': round(processing_time, 2),
        'networks_per_minute': round(success_count / max(processing_time, 0.01), 1),
        'font_sizes': [24, 36],
        'formats': ['PNG', 'EPS'],
        'timestamp': datetime.now().isoformat()
    }

    print()
    print("=" * 77)
    print("PROCESSING COMPLETE!")
    print("=" * 77)
    print(f"Results:")
    print(f"  Successful networks: {success_count} of {total_combinations}")
    print(f"  Total files created: {total_files_created} "
          f"({success_count} networks x 2 fonts x 2 formats)")
    print(f"  Success rate: {success_rate}%")
    print(f"  Processing time: {round(processing_time, 2)} minutes")
    print(f"  Speed: {summary_stats['networks_per_minute']} networks/minute")

    if failed_topics:
        print(f"  Failed topics: {', '.join(map(str, failed_topics))}")

    print(f"  Output location: stm_networks_{method}/")
    print("=" * 77)

    return summary_stats


def test_term_extraction(model, test_topics=None, method="frex", n_words=8):
    """
    Python equivalent of R's test_term_extraction() function.

    Quick validation to verify STM term extraction works correctly.

    R equivalent:
        test_term_extraction <- function(topic_model,
            test_topics = c(1, 5, 10, 15, 20), method = "frex", n_words = 8) { ... }
    """
    if test_topics is None:
        test_topics = [1, 5, 10, 15, 20]

    print("=" * 77)
    print(f"TESTING STM TERM EXTRACTION - {method.upper()} METHOD")
    print("=" * 77)
    print("This test verifies that different topics return different terms.")
    print("If all topics show the same terms, there is an indexing problem.")
    print()

    for topic_num in test_topics:
        terms = get_stm_terms(model, topic_num, n_words, method)
        terms_display = ', '.join(terms[:6])
        suffix = ', ...' if len(terms) > 6 else ''
        print(f"Topic {topic_num:2d}: {terms_display}{suffix}")

    print()
    print("If you see different terms for each topic, extraction is working correctly.")
    print("=" * 77)


def test_single_network(model, corpus, text_data, topic_num=1,
                         method="frex", n_words=30):
    """
    Python equivalent of R's test_single_network() function.

    Quick function to test network creation on a single topic.
    """
    print(f"Testing network creation for Topic {topic_num} "
          f"with {method} rankings...")

    result = make_network(model, corpus, text_data, topic_num, n_words, method)

    if result is not None:
        print("Success! Network created with:")
        print(f"  Terms: {len(result['terms'])}")
        print(f"  Edges: {result['edges']}")
        print(f"  Font sizes: {result['font_sizes_created']}")
        print(f"  Files saved in: stm_networks_{method}/")
        print(f"  Top terms: {', '.join(result['terms'][:6])}")
    else:
        print(f"Failed to create network for Topic {topic_num}")

    return result


###############################################################################
#                                                                             #
# SECTION 7: MAIN EXECUTION                                                   #
#                                                                             #
###############################################################################

if __name__ == "__main__":
    """
    Main execution block mirroring the R Master script flow.

    R script flow:
      1. Load data (full_text_clean)
      2. Preprocess (textProcessor, prepDocuments)
      3. Run STM model
      4. estimateEffect with b-spline
      5. Generate topic plots (stm_plot for each topic)
      6. Generate word clouds
      7. Topic correlations
      8. Beta/theta extraction
      9. Document probability distributions
     10. Extract top documents to file
     11. sageLabels extraction
     12. Co-occurrence network analysis
    """

    print("=" * 80)
    print("MASTER STM ANALYSIS - Python Translation")
    print("=" * 80)
    print()
    print("Author: Joon Chung, jxc3388@miami.edu")
    print("University of Miami, Miller School of Medicine")
    print()
    print("IMPORTANT: This script requires pre-loaded data.")
    print("Load your newspaper corpus as a pandas DataFrame.")
    print()

    # ------------------------------------------------------------------
    # STEP 1: Load data
    # ------------------------------------------------------------------
    # R equivalent:
    # load("clean_full_text_20k.Rda")
    # full_text_subset <- full_text_clean %>%
    #   select(text, text_id, datetime, Year) %>% na.omit()
    # full_text_subset$year <- as.numeric(full_text_subset$Year)

    print("STEP 1: Load data")
    print("  Replace the placeholder below with your actual data loading code.")
    print("  Expected columns: text, text_id, datetime, Year")
    print()

    # --- PLACEHOLDER: Replace with actual data loading ---
    # Option A: CSV
    # full_text_clean = pd.read_csv("path/to/clean_full_text_20k.csv")
    #
    # Option B: R .Rda file
    # import pyreadr
    # result = pyreadr.read_r("path/to/clean_full_text_20k.Rda")
    # full_text_clean = result['full_text_clean']
    #
    # Option C: Already loaded in memory
    # (no action needed)

    print("  [DATA LOADING PLACEHOLDER - exiting]")
    print("  To run the full analysis, uncomment and configure data loading above.")
    sys.exit(0)

    # ------------------------------------------------------------------
    # STEP 2: Preprocess
    # ------------------------------------------------------------------
    # R: full_text_subset <- full_text_clean %>%
    #      select(text, text_id, datetime, Year) %>% na.omit()
    full_text_subset = full_text_clean[
        ['text', 'text_id', 'datetime', 'Year']
    ].dropna().copy()
    full_text_subset['year'] = pd.to_numeric(full_text_subset['Year'])

    # R: processed <- textProcessor(full_text_subset$text, ...)
    processed = text_processor(
        full_text_subset['text'].tolist(),
        metadata=full_text_subset.reset_index(drop=True),
        striphtml=True
    )

    # R: out <- prepDocuments(processed$documents, processed$vocab,
    #                         processed$meta, lower.thresh = 500)
    corpus_out, dict_out = prep_documents(
        processed['documents'], processed['dictionary'], lower_thresh=500
    )
    meta_out = processed['meta']

    # ------------------------------------------------------------------
    # STEP 3: Run topic model
    # ------------------------------------------------------------------
    # R: set.seed(8675309)
    # R: topic_model_prev <- stm(out$documents, out$vocab, ...)
    topic_model_prev = run_topic_model(
        corpus_out, dict_out, num_topics=70, random_state=RANDOM_SEED
    )

    # R: sageLabels(topic_model_prev)
    print("\n--- sageLabels output ---")
    print_sagelabels_format(topic_model_prev, n_terms=5, topics=list(range(1, 11)))

    # ------------------------------------------------------------------
    # STEP 4: estimateEffect (b-spline) - APPROXIMATED
    # ------------------------------------------------------------------
    # R: prep <- estimateEffect(1:70 ~ s(year), topic_model_prev, meta = out$meta)
    # NOTE: R's estimateEffect with b-spline smoothing is NOT available in Python.
    # The stm_plot function uses simple yearly aggregation as an approximation.
    print("\nNOTE: estimateEffect with b-spline is NOT available in Python.")
    print("      Using simple yearly aggregation as an approximation.")

    # ------------------------------------------------------------------
    # STEP 5: Generate topic plots
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

    for topic_num in topics_of_interest:
        label = topic_labels.get(topic_num, f"Topic {topic_num}")
        print(f"\nGenerating plot for Topic {topic_num}: {label}")

        fig = stm_plot(
            topic_model_prev, corpus_out, meta_out,
            topic_select=topic_num, title=label
        )
        filename = f"topic_{topic_num}_plot.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {filename}")

    # ------------------------------------------------------------------
    # STEP 6: Word clouds
    # ------------------------------------------------------------------
    print("\n--- Generating word clouds ---")
    cloud_map = {
        3: "sleep_cloud.png",
        7: "drug_cloud.png",
        15: "science_cloud.png",
        32: "apnea_cloud.png",
        47: "health_cloud.png",
        51: "academic_research_cloud.png"
    }

    for topic_num, filename in cloud_map.items():
        generate_wordcloud(topic_model_prev, topic_num - 1,
                           n_words=30, output_file=filename)

    # ------------------------------------------------------------------
    # STEP 7: Topic correlations
    # ------------------------------------------------------------------
    # R: plot(topicCorr(topic_model_prev))
    print("\n--- Topic correlations ---")
    fig = plot_topic_correlations(topic_model_prev, corpus_out)
    fig.savefig("topic_correlations.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # STEP 8: Beta/theta matrices
    # ------------------------------------------------------------------
    # R: stm_beta <- tidy(topic_model_prev, matrix = "beta")
    print("\n--- Extracting beta matrix ---")
    stm_beta = get_beta_matrix(topic_model_prev)
    print(f"  Beta matrix: {len(stm_beta)} rows")

    # R: stm_theta <- tidy(topic_model_prev, matrix = "theta") %>%
    #      filter(topic == topics_of_interest)
    theta_matrix = get_document_topics(topic_model_prev, corpus_out)
    print(f"  Theta matrix: {theta_matrix.shape}")

    # ------------------------------------------------------------------
    # STEP 9: Document probability distributions
    # ------------------------------------------------------------------
    # R: ggplot(stm_theta, aes(gamma, ...)) + geom_histogram() + facet_wrap()
    print("\n--- Document probability distributions ---")
    fig = plot_document_probability_distribution(
        topic_model_prev, corpus_out, topics_of_interest
    )
    fig.savefig("doc_prob_distributions.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # STEP 10: Topic summary
    # ------------------------------------------------------------------
    # R: plot.STM(topic_model_prev, type="summary", labeltype="prob",
    #             topics=topics_of_interest)
    print("\n--- Topic summary ---")
    fig = plot_topic_summary(
        topic_model_prev, corpus_out, topics_of_interest, topic_labels
    )
    fig.savefig("topic_summary.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ------------------------------------------------------------------
    # STEP 11: Write top 100 articles to file
    # ------------------------------------------------------------------
    topic_names = [
        "Sleep_and_Work", "Sleep_medicine_drugs", "Circadian_science",
        "Sleep_apnea_hospitals", "Sleep_and_health", "Academic_sleep_research"
    ]

    print("\n--- Writing top documents to file ---")
    write_top_documents_to_files(
        topic_model_prev, corpus_out,
        topics_of_interest, topic_names,
        meta_out, output_dir="top_documents"
    )

    # ------------------------------------------------------------------
    # STEP 12: sageLabels extraction
    # ------------------------------------------------------------------
    print("\n--- sageLabels extraction ---")
    process_all_sagelabels_format(
        topic_model_prev, n_terms=5,
        filename="stm_all_topics_sagelabels.txt"
    )

    save_sagelabels_matrices(topic_model_prev, n_terms=5, prefix="stm_terms")

    overlap_stats = analyze_term_overlap(topic_model_prev, n_terms=5)
    overlap_stats.to_csv("term_overlap_analysis.csv", index=False)

    # ------------------------------------------------------------------
    # STEP 13: Co-occurrence network analysis
    # ------------------------------------------------------------------
    print("\n--- Co-occurrence network analysis ---")
    print("Testing term extraction...")
    test_term_extraction(topic_model_prev, method='frex')
    test_term_extraction(topic_model_prev, method='score')

    print("\nTesting single network...")
    test_single_network(topic_model_prev, corpus_out, meta_out,
                         topic_num=1, method='frex')

    print("\nProcessing all networks (FREX)...")
    results_frex = process_networks(
        topic_model_prev, corpus_out, meta_out,
        method='frex', topics=list(range(1, 71)),
        word_counts=[15, 30, 50]
    )

    print("\nProcessing all networks (Score)...")
    results_score = process_networks(
        topic_model_prev, corpus_out, meta_out,
        method='score', topics=list(range(1, 71)),
        word_counts=[15, 30, 50]
    )

    print()
    print("=" * 80)
    print("MASTER STM ANALYSIS COMPLETE")
    print("=" * 80)
