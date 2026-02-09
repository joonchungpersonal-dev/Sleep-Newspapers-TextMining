"""
03_top_documents_extraction.py

Literal Python translation of:
    "03 Top 25 documents to file.R"

Author: Joon Chung
        jxc3388@miami.edu
        The University of Miami, Miller School of Medicine
        The Department of Informatics and Health Data Science

Date: 6/13/2025 (R version)

Purpose:
    Grab top 25 documents per topic (70 topics).
    Extracts the highest-probability documents for each of the 70 topics
    in the STM model and saves them as CSV and formatted text files.

IMPORTANT NOTES ON R-TO-PYTHON DIFFERENCES:
    - R's model$theta is accessed via get_document_topics() in Python
    - R's sink() / cat() for writing to file -> Python open() / write()
    - R's order() with decreasing=TRUE -> np.argsort()[::-1]
    - R's sprintf("%02d", x) -> f"{x:02d}" in Python
    - R's paste0() -> f-strings or os.path.join()
    - R's exists("full_text_subset") -> variable in scope
    - R's pipe operator %>% -> chained method calls or intermediate variables
"""

import os
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd

# Topic modeling
from gensim.models import LdaModel
from gensim import corpora

# Re-use helper from the master script
# If running standalone, these functions are defined below


# =============================================================================
# R equivalent: library(stm); library(dplyr)
# =============================================================================


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


def extract_top_documents(model, corpus, topic_num, full_text_subset=None,
                          n_docs=25):
    """
    Python equivalent of R's extract_top_documents() function.

    Extracts the top n_docs documents most associated with a given topic.

    R equivalent:
        extract_top_documents <- function(model, topic_num, n_docs = 25) {
          topic_proportions <- model$theta[, topic_num]
          top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
          top_proportions <- topic_proportions[top_indices]
          top_docs_data <- data.frame(
            Rank = 1:n_docs,
            Document_Index = top_indices,
            Topic_Proportion = round(top_proportions, 4),
            stringsAsFactors = FALSE
          )
          ...
        }

    Parameters
    ----------
    model : gensim.models.LdaModel
    corpus : list of gensim corpus documents
    topic_num : int
        Topic number (1-based, matching R convention).
    full_text_subset : pd.DataFrame or None
        Metadata DataFrame. R equivalent: full_text_subset global variable.
    n_docs : int
        Number of top documents to extract.

    Returns
    -------
    pd.DataFrame
    """
    # R: topic_proportions <- model$theta[, topic_num]
    theta = get_document_topics(model, corpus)
    topic_idx = topic_num - 1  # Convert to 0-based (R is 1-based)

    topic_proportions = theta[:, topic_idx]

    # R: top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
    top_indices = np.argsort(topic_proportions)[::-1][:n_docs]

    # R: top_proportions <- topic_proportions[top_indices]
    top_proportions = topic_proportions[top_indices]

    # R: top_docs_data <- data.frame(Rank = 1:n_docs, ...)
    top_docs_data = pd.DataFrame({
        'Rank': range(1, n_docs + 1),
        'Document_Index': top_indices + 1,  # 1-based to match R
        'Topic_Proportion': np.round(top_proportions, 4)
    })

    # R: if (exists("full_text_subset")) { ... }
    if full_text_subset is not None:
        # Text content
        # R: if ("text" %in% names(full_text_subset)) { ... }
        for col_name in ['text', 'Text', 'content']:
            if col_name in full_text_subset.columns:
                top_docs_data['Text'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

        # Publication date
        for col_name in ['datetime', 'date', 'Date']:
            if col_name in full_text_subset.columns:
                top_docs_data['Date'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

        # Year
        for col_name in ['year', 'Year']:
            if col_name in full_text_subset.columns:
                top_docs_data['Year'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

        # Source/Newspaper
        for col_name in ['source', 'Source', 'newspaper', 'Newspaper', 'publication']:
            if col_name in full_text_subset.columns:
                top_docs_data['Source'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

        # Title
        for col_name in ['title', 'Title', 'headline']:
            if col_name in full_text_subset.columns:
                top_docs_data['Title'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

        # Document ID
        for col_name in ['id', 'ID', 'doc_id', 'text_id']:
            if col_name in full_text_subset.columns:
                top_docs_data['Document_ID'] = \
                    full_text_subset[col_name].iloc[top_indices].values
                break

    return top_docs_data


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block.

    R equivalent flow:
        1. Define all_topics <- 1:70
        2. Create output directories
        3. Loop through all 70 topics
        4. Extract top 25 documents for each
        5. Save as CSV and formatted TXT
        6. Create master summary file
        7. Create topic index file
    """

    print("=" * 80)
    print("TOP DOCUMENTS EXTRACTION - Python Translation")
    print("=" * 80)
    print()
    print("Purpose: Grab top 25 documents per topic (70 topics)")
    print()
    print("NOTE: This script requires:")
    print("  1. A trained topic model (gensim LdaModel)")
    print("  2. The corpus used for training")
    print("  3. The metadata DataFrame (full_text_subset)")
    print()

    # ------------------------------------------------------------------
    # PLACEHOLDER: Load model and data
    # ------------------------------------------------------------------
    # In practice, load your saved model and data:
    #
    # topic_model_prev = LdaModel.load("topic_model_prev.gensim")
    # dictionary = corpora.Dictionary.load("dictionary.gensim")
    # corpus = corpora.MmCorpus("corpus.mm")
    # full_text_subset = pd.read_csv("full_text_subset.csv")

    print("  [DATA LOADING PLACEHOLDER - exiting]")
    print("  To run, uncomment and configure data loading above.")
    import sys
    sys.exit(0)

    # ------------------------------------------------------------------
    # R: all_topics <- 1:70
    # ------------------------------------------------------------------
    all_topics = list(range(1, 71))

    # ------------------------------------------------------------------
    # R: Create output directories
    # ------------------------------------------------------------------
    # R: if (!dir.exists("all_topics_documents")) dir.create(...)
    output_base = "all_topics_documents"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(os.path.join(output_base, "csv_files"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "txt_files"), exist_ok=True)

    # ------------------------------------------------------------------
    # R: Extract and save top documents for all 70 topics
    # ------------------------------------------------------------------
    # R: cat("Processing all 70 topics...\n")
    print("Processing all 70 topics...")
    print("This may take a few minutes.\n")

    # R: for (topic_num in all_topics) { ... }
    for topic_num in all_topics:
        # R: cat("Processing topic", topic_num, "of 70\n")
        print(f"Processing topic {topic_num} of 70")

        # R: top_docs <- extract_top_documents(topic_model_prev, topic_num)
        top_docs = extract_top_documents(
            topic_model_prev, corpus, topic_num,
            full_text_subset=full_text_subset, n_docs=25
        )

        # R: Save as CSV
        # R: csv_filename <- paste0("all_topics_documents/csv_files/topic_",
        #                           sprintf("%02d", topic_num), "_top25.csv")
        csv_filename = os.path.join(
            output_base, "csv_files",
            f"topic_{topic_num:02d}_top25.csv"
        )
        # R: write.csv(top_docs, csv_filename, row.names = FALSE)
        top_docs.to_csv(csv_filename, index=False)

        # R: Save as formatted text file
        # R: txt_filename <- paste0("all_topics_documents/txt_files/topic_",
        #                           sprintf("%02d", topic_num), "_top25.txt")
        txt_filename = os.path.join(
            output_base, "txt_files",
            f"topic_{topic_num:02d}_top25.txt"
        )

        # R: sink(txt_filename) ... sink()
        with open(txt_filename, 'w') as f:
            # R: cat("TOP 25 DOCUMENTS FOR TOPIC", topic_num, "\n")
            f.write(f"TOP 25 DOCUMENTS FOR TOPIC {topic_num}\n")
            # R: cat("=" %>% rep(50) %>% paste(collapse = ""), "\n\n")
            f.write("=" * 50 + "\n\n")

            # R: for (j in 1:nrow(top_docs)) { ... }
            for j in range(len(top_docs)):
                row = top_docs.iloc[j]

                # R: cat("RANK:", j, "| PROPORTION:", ..., "| DOC INDEX:", ..., "\n")
                f.write(f"RANK: {j+1} | PROPORTION: {row['Topic_Proportion']} "
                        f"| DOC INDEX: {int(row['Document_Index'])}\n")

                # R: if ("Date" %in% names(top_docs)) cat("DATE:", ..., "\n")
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

                # R: cat("\n", "-" %>% rep(80) %>% paste(collapse = ""), "\n\n")
                f.write("\n" + "-" * 80 + "\n\n")

    # ------------------------------------------------------------------
    # R: Create master summary file
    # ------------------------------------------------------------------
    # R: summary_filename <- "all_topics_documents/master_summary_all_70_topics.txt"
    summary_filename = os.path.join(
        output_base, "master_summary_all_70_topics.txt"
    )

    # R: sink(summary_filename) ... sink()
    with open(summary_filename, 'w') as f:
        # R: cat("MASTER SUMMARY: TOP 5 DOCUMENTS FOR ALL 70 TOPICS\n")
        f.write("MASTER SUMMARY: TOP 5 DOCUMENTS FOR ALL 70 TOPICS\n")
        # R: cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")
        f.write("=" * 70 + "\n\n")

        for topic_num in all_topics:
            # R: top_docs <- extract_top_documents(topic_model_prev, topic_num, n_docs = 5)
            top_docs = extract_top_documents(
                topic_model_prev, corpus, topic_num,
                full_text_subset=full_text_subset, n_docs=5
            )

            # R: cat("TOPIC", sprintf("%02d", topic_num), "\n")
            f.write(f"TOPIC {topic_num:02d}\n")
            f.write("Top 5 documents (proportion):\n")

            # R: for (j in 1:min(5, nrow(top_docs))) { ... }
            for j in range(min(5, len(top_docs))):
                row = top_docs.iloc[j]
                line = f"  {j+1}. Doc {int(row['Document_Index'])} " \
                       f"({row['Topic_Proportion']})"

                if 'Year' in top_docs.columns:
                    line += f" [{row['Year']}]"
                if 'Source' in top_docs.columns:
                    line += f" - {row['Source']}"

                f.write(line + "\n")

            f.write("\n")

    # ------------------------------------------------------------------
    # R: Create topic index file
    # ------------------------------------------------------------------
    # R: index_filename <- "all_topics_documents/topic_index.txt"
    index_filename = os.path.join(output_base, "topic_index.txt")

    # R: sink(index_filename) ... sink()
    with open(index_filename, 'w') as f:
        f.write("TOPIC INDEX - ALL 70 TOPICS\n")
        # R: cat("Generated:", Sys.time(), "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 50 + "\n\n")

        f.write("File Structure:\n")
        f.write("- csv_files/: Contains CSV data for each topic\n")
        f.write("- txt_files/: Contains formatted text files for each topic\n")
        f.write("- master_summary_all_70_topics.txt: "
                "Overview of top 5 docs per topic\n\n")

        f.write("Topics Processed:\n")
        for topic_num in all_topics:
            f.write(f"Topic {topic_num:02d} - Files: "
                    f"topic_{topic_num:02d}_top25.csv/.txt\n")

        f.write(f"\nTotal Topics: {len(all_topics)}\n")
        f.write("Documents per Topic: 25\n")
        f.write(f"Total Document Extractions: {len(all_topics) * 25}\n")

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------
    # R: cat("\n=== PROCESSING COMPLETE ===\n")
    print()
    print("=== PROCESSING COMPLETE ===")
    print("All 70 topics processed successfully!")
    print(f"Files saved in '{output_base}' directory:")
    print("- CSV files in csv_files/ subdirectory")
    print("- TXT files in txt_files/ subdirectory")
    print("- Master summary file: master_summary_all_70_topics.txt")
    print("- Topic index file: topic_index.txt")
    print(f"\nTotal files created: {(len(all_topics) * 2) + 2}")
