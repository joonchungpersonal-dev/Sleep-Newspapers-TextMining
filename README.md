# Discourse around Sleep, Performance, and Work in U.S. Print Media (1983–2017)

Code and supplementary materials for a computational social science study of sleep discourse in major U.S. newspapers.

## Overview

This project uses **Structural Topic Models (STM)** to analyze how sleep has been discussed in U.S. print media over 35 years. The study examines the tension between sleep and work in American culture, the growing medicalization of sleep, and the incomplete "customization" of sleep as a performance tool.

**Manuscript:** Under review at *Humanities and Social Science Communications* (Nature Portfolio).

## Data

| Parameter | Value |
|-----------|-------|
| Source | ProQuest newspaper database |
| Search | "sleep" in title or abstract |
| Date range | January 1, 1983 – December 31, 2017 |
| Final corpus | 25,766 articles (after deduplication) |
| Newspapers | NYT, Washington Post, Chicago Tribune, USA Today, WSJ, OC Register, LA Times |

Raw and processed data files are not included due to size and licensing restrictions. See `data/metadata/` for topic term matrices and model outputs.

## Methods

- **70-topic STM** with publication year as prevalence covariate (B-spline smoothed)
- **Frame analysis** across 8 media frames (health, moral, economic, technology, lifestyle, productivity, scientific, social)
- **Close reading** of top-50 representative documents per topic
- Preprocessing: stop word removal, stemming, min document frequency = 500
- Software: R 4.4.1 (`stm`, `tidyverse`, `quanteda`, `igraph`)

## Repository Structure

```
code/
  original_R/            # Original R scripts as authored
  R_verbose/             # Same code with detailed comments for non-R users
  python_literal/        # Line-by-line Python translation (gensim LDA)
  python_optimized/      # Modern Python pipeline (BERTopic + CLI)
  03_topic_modeling/     # Working analysis scripts
  05_visualization/      # Word frequency and visualization scripts
data/
  metadata/              # Topic terms, STM label matrices, codebooks
output/
  figures/               # Topic trend plots, word clouds, publication figures
  tables/                # Frame analysis results, per-topic document extracts
```

### Code Variants

The analysis was originally written in R. Two Python translations are provided for accessibility:

| Variant | Path | Description |
|---------|------|-------------|
| **R (original)** | `code/original_R/` | The scripts used in the study |
| **R (verbose)** | `code/R_verbose/` | Same code + section headers, concept explanations, Python cross-references |
| **Python (literal)** | `code/python_literal/` | Mirrors R structure; uses gensim LDA as STM proxy |
| **Python (modern)** | `code/python_optimized/` | Reimagined pipeline with BERTopic, spaCy, plotly; single CLI entry point |

> **Note:** R's `stm` package has no exact Python equivalent. The Python versions approximate STM using gensim LDA with custom FREX/Score calculations. All approximations are documented inline.

## Key Findings

- **Sleep & work discourse** (Topic 33) increased 2–3x from 1983 to 2017, reflecting growing cultural tension between productivity norms and sleep advocacy
- **Health outcomes of poor sleep** (Topic 32) quadrupled in media attention, driven by epidemiological research linking sleep to chronic disease
- **Sleep & performance** (Topic 47) increased 3–4x as scientific research on cognition, memory, and workplace safety grew
- Validation topics (Iraq Wars, school start times) confirmed the model captures real-world events

## Requirements

**R:**
```
stm, tidyverse, quanteda, igraph, ggplot2, wordcloud, corrplot
```

**Python (literal):**
```
pip install -r code/python_literal/requirements.txt
```

**Python (modern):**
```
pip install -r code/python_optimized/requirements.txt
python code/python_optimized/pipeline.py --data <corpus.csv> --steps all
```

## Citation

> Chung, J. (2025). Discourse around sleep, performance, and work in the U.S. print media: 1983–2017. *Humanities and Social Science Communications*. (Under review)

## License

TBD
