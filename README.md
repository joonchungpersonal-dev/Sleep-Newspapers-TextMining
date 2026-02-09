# The Social Construction of Sleep in U.S. Media

Text mining analysis of U.S. newspaper discourse on sleep (1987–2018).

## Overview

This project examines how major U.S. newspapers have framed sleep over three decades using structural topic modeling (STM), latent Dirichlet allocation (LDA), and word embeddings (Word2Vec).

## Data

- **Source:** ProQuest newspaper database
- **Search:** Articles with "sleep" in the title
- **Date range:** 1987–2018
- **Publications:** 15+ major U.S. newspapers (NYT, Washington Post, USA Today, WSJ, Chicago Tribune, Boston Globe, LA Times, and others)

Note: Raw and processed data files are not included in this repository due to size and licensing. See `data/metadata/` for search parameters and codebooks.

## Repository Structure

```
code/
  01_data_collection/    # ProQuest download and scraping scripts
  02_preprocessing/      # Cleaning, deduplication, corpus construction
  03_topic_modeling/     # LDA and STM analyses
  04_word_embeddings/    # Word2Vec training and temporal alignment
  05_visualization/      # Figure and table generation
data/
  metadata/              # Codebooks, search parameters, newspaper list
manuscript/
  main/                  # Manuscript files
  supplementary/         # Supplementary materials
output/
  figures/               # Publication-ready figures
  tables/                # Results tables
references/              # Key methodology papers
```

## Requirements

- R (with packages: `stm`, `tidytext`, `quanteda`, `topicmodels`)
- Python 3 (with `gensim`, `numpy`)

## Citation

TODO: Add citation once published.
