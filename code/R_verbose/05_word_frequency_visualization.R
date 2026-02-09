# =============================================================================
# SCRIPT:   05_word_frequency_visualization.R
# PURPOSE:  Analyze word frequencies across the preprocessed newspaper corpus,
#           generate word clouds (static, interactive, and topic-specific),
#           perform temporal word frequency analysis, create comparison clouds,
#           and produce publication-quality frequency distribution plots.
#
# AUTHOR:   Joon Chung
# EMAIL:    jxc3388@miami.edu
# AFFIL:    The University of Miami, Miller School of Medicine
#           The Department of Informatics and Health Data Science
#
# INPUTS:
#   - out: The preprocessed STM data object (from prepDocuments()) containing:
#       out$documents: List of document-term count matrices (sparse format)
#       out$vocab: Character vector of vocabulary terms
#       out$meta: Metadata data frame with document-level covariates
#   - topic_model_prev: (Optional) Fitted STM model for topic-specific clouds
#
# OUTPUTS:
#   - basic_wordcloud.png: Overall corpus word cloud
#   - wordcloud_YEAR.png: Year-specific word clouds (when uncommented)
#   - Top-words bar plots displayed in the R graphics device
#   - Console output of frequency distribution statistics
#
# REQUIRED PACKAGES:
#   - stm         : STM preprocessing objects and topic model access
#   - wordcloud   : Base R word cloud generation (static images)
#   - wordcloud2  : Interactive HTML word clouds (requires browser/viewer)
#   - RColorBrewer: Color palettes for visualizations
#   - dplyr       : Data manipulation (filtering, arranging, mutating)
#   - ggplot2     : Publication-quality statistical graphics
#   - tidytext    : Tidy text mining utilities
#   - viridis     : Color-blind-friendly palettes
#
# NOTES FOR PYTHON USERS:
#   - The R 'wordcloud' package is equivalent to Python's 'wordcloud' library
#     (from wordcloud import WordCloud). The API differs but the concept is
#     identical: size words proportional to frequency.
#   - STM stores documents in a sparse list format where each document is a
#     2-row matrix: row 1 = word indices, row 2 = word frequencies. This is
#     similar to scipy.sparse matrices or gensim's bag-of-words format:
#       [(word_id, count), (word_id, count), ...]
#   - RColorBrewer palettes correspond to matplotlib colormaps:
#       brewer.pal(8, "Dark2") ~ plt.cm.Dark2
#   - The comparison.cloud() function creates a plot showing which words are
#     distinctive to each group. In Python, you would compute TF-IDF
#     differences and plot manually.
# =============================================================================


# =============================================================================
#### SECTION: Load Required Libraries ####
# =============================================================================

# Word Frequency Analysis and Word Clouds for Structural Topic Model
# Required libraries
library(stm)
# wordcloud: creates static word clouds from word-frequency pairs.
# The main function wordcloud() accepts vectors of words and frequencies.
# In Python: from wordcloud import WordCloud; wc = WordCloud(); wc.generate_from_frequencies(freq_dict)
library(wordcloud)
# wordcloud2: creates interactive HTML/JavaScript word clouds that can be
# viewed in RStudio's Viewer pane or a web browser. Not available in Python
# wordcloud, but JavaScript libraries like d3-cloud provide similar interactivity.
library(wordcloud2)
# RColorBrewer: provides color palettes designed by Cynthia Brewer for
# cartography. brewer.pal(n, "palette_name") returns n colors.
# In Python: matplotlib.cm provides similar palettes.
library(RColorBrewer)
library(dplyr)
library(ggplot2)
library(tidytext)
# viridis: perceptually uniform, color-blind-friendly palettes.
# In Python: matplotlib.cm.viridis or seaborn's "viridis" palette.
library(viridis)


# =============================================================================
#### SECTION 1: Basic Word Frequency Analysis ####
# =============================================================================

# =============================================================================
# 1. BASIC WORD FREQUENCY ANALYSIS
# =============================================================================

# The 'out' object was created by stm::prepDocuments() and contains:
#   out$documents: a list where each element is a 2-row matrix for one document
#     Row 1: vocabulary indices (1-based) for words present in that document
#     Row 2: frequency counts for those words
#   out$vocab: character vector mapping indices to actual words
#
# This sparse representation is memory-efficient for large corpora. In Python
# with gensim, documents are stored similarly as lists of (word_id, count) tuples.
# With scikit-learn, you would use a scipy.sparse CSR matrix instead.

# Get vocabulary and word counts from STM processed data
vocab <- out$vocab

# First, let's check the structure of out$documents.
# str() displays the internal structure of an R object (like type, dimensions).
# class() returns the object's class name. These are diagnostic/debugging steps.
# In Python: type(out_documents), out_documents.shape (for numpy arrays).
cat("Structure of out$documents:\n")
str(out$documents)
cat("\nClass of out$documents:", class(out$documents), "\n")

# Handle different possible formats of out$documents.
# STM typically stores documents as a list (sparse format), but we check
# for alternative formats (matrix) for robustness.
if (is.list(out$documents)) {
  # If documents is a list (sparse format), convert to word counts
  cat("Documents are in list format (sparse). Converting to word counts...\n")

  # Initialize a word count vector with zeros, one entry per vocabulary term.
  # names() assigns the vocabulary words as names for easy lookup.
  # In Python: word_counts = np.zeros(len(vocab))
  word_counts <- rep(0, length(vocab))
  names(word_counts) <- vocab

  # Iterate over each document's sparse representation.
  # seq_along() generates indices 1:length(out$documents).
  # In Python: for i, doc in enumerate(out_documents):
  for (i in seq_along(out$documents)) {
    doc <- out$documents[[i]]
    # Each doc is a 2-row matrix:
    #   doc[1, ] = word indices (into vocab)
    #   doc[2, ] = frequencies (how many times each word appears)
    if (length(doc) > 0 && is.matrix(doc)) {
      word_indices <- doc[1, ]  # First row contains word indices
      word_frequencies <- doc[2, ]  # Second row contains frequencies

      # Accumulate word frequencies across all documents.
      for (j in seq_along(word_indices)) {
        word_counts[word_indices[j]] <- word_counts[word_indices[j]] + word_frequencies[j]
      }
    }
  }

} else if (is.matrix(out$documents)) {
  # If documents is already a dense matrix (documents x words)
  cat("Documents are in matrix format. Using colSums...\n")
  # colSums() sums each column (word) across all rows (documents).
  # In Python: word_counts = doc_term_matrix.sum(axis=0)
  word_counts <- colSums(out$documents)
  names(word_counts) <- vocab

} else {
  # Fallback: try using STM's make.dt() helper function
  cat("Attempting to convert documents to matrix format...\n")

  if (requireNamespace("stm", quietly = TRUE)) {
    # stm::make.dt() converts the sparse document list to a dense document-term
    # matrix (data.table format). This is memory-intensive for large corpora.
    dtm <- stm::make.dt(out$documents, vocab = vocab)
    word_counts <- colSums(dtm)
    names(word_counts) <- vocab
  } else {
    stop("Cannot determine format of out$documents. Please check the structure.")
  }
}

# Create a sorted word frequency data frame.
# arrange(desc(freq)) sorts by frequency in descending order.
# In Python: word_freq_df = pd.DataFrame({'word': vocab, 'freq': counts}).sort_values('freq', ascending=False)
word_freq_df <- data.frame(
  word = names(word_counts),
  freq = as.numeric(word_counts),
  stringsAsFactors = FALSE
) %>%
  arrange(desc(freq))

# Display top words
print("Top 20 most frequent words:")
# head() returns the first N rows (like Python's df.head(20)).
print(head(word_freq_df, 20))


# =============================================================================
#### SECTION 2: Basic Word Cloud ####
# =============================================================================

# =============================================================================
# 2. BASIC WORD CLOUD
# =============================================================================

# Create a static word cloud of the most frequent words in the corpus.
# png() opens a PNG graphics device; all subsequent plotting goes to the file.
# dev.off() closes the device and writes the file.
# In Python: wc = WordCloud(width=800, height=600, max_words=100)
#             wc.generate_from_frequencies(word_freq_dict)
#             wc.to_file("basic_wordcloud.png")

# Simple word cloud with most frequent words
png("basic_wordcloud.png", width = 800, height = 600)
wordcloud(words = word_freq_df$word,
          freq = word_freq_df$freq,
          min.freq = 5,           # Only include words appearing >= 5 times
          max.words = 100,        # Display at most 100 words
          random.order = FALSE,   # Place most frequent words in the center
          rot.per = 0.35,         # 35% of words are rotated 90 degrees
          colors = brewer.pal(8, "Dark2"))  # Use Dark2 color palette (8 colors)
dev.off()


# =============================================================================
#### SECTION 3: Interactive Word Cloud (wordcloud2) ####
# =============================================================================

# =============================================================================
# 3. INTERACTIVE WORD CLOUD (wordcloud2)
# =============================================================================

# wordcloud2() creates an interactive HTML word cloud. In RStudio, this opens
# in the Viewer pane. Users can hover over words to see exact frequencies.
# word_freq_df[1:100, ] selects the top 100 rows (most frequent words).
# In Python, you would use a JavaScript library like d3-cloud for interactivity,
# or Plotly's word cloud capabilities.
wordcloud2(word_freq_df[1:100, ],
           size = 0.8,
           color = 'random-light',
           backgroundColor = "black")


# =============================================================================
#### SECTION 4: Topic-Specific Word Clouds ####
# =============================================================================

# =============================================================================
# 4. TOPIC-SPECIFIC WORD CLOUDS
# =============================================================================

# This function creates a word cloud for a specific topic from the STM model.
# It extracts the top words using labelTopics() and sizes them by probability.
#
# WHAT IS labelTopics()?
# labelTopics() returns the most informative words for each topic using several
# ranking methods: prob (probability), frex (frequency-exclusivity), lift, and
# score. Here we use the probability ranking.
#
# In Python/gensim: model.show_topic(topic_id, topn=50)

# Function to create word clouds for specific topics
create_topic_wordcloud <- function(stm_model, topic_num, n_words = 50) {
  # Get top words for the topic
  top_words <- labelTopics(stm_model, n = n_words)$prob[topic_num, ]
  word_probs <- labelTopics(stm_model, n = n_words)$prob[topic_num, ]

  # Create dataframe
  topic_df <- data.frame(
    word = top_words,
    prob = word_probs,
    stringsAsFactors = FALSE
  )

  # Create word cloud. Multiply probabilities by 1000 to create visible size
  # differences (wordcloud expects integer-like frequency values).
  wordcloud(words = topic_df$word,
            freq = topic_df$prob * 1000,  # Scale probabilities
            min.freq = 1,
            max.words = n_words,
            random.order = FALSE,
            rot.per = 0.35,
            colors = brewer.pal(8, "Set2"),
            main = paste("Topic", topic_num))
}

# Uncomment and modify to create word clouds for all topics:
# for(i in 1:topic_model_prev$settings$dim$K) {
#   png(paste0("topic_", i, "_wordcloud.png"), width = 600, height = 600)
#   create_topic_wordcloud(topic_model_prev, i)
#   dev.off()
# }


# =============================================================================
#### SECTION 5: Temporal Word Frequency Analysis ####
# =============================================================================

# =============================================================================
# 5. TEMPORAL WORD FREQUENCY ANALYSIS
# =============================================================================

# This function computes word frequencies separately for each time period
# (e.g., year), enabling analysis of how vocabulary usage changes over time.
#
# The function handles the sparse document format used by STM, where each
# document is a 2-row matrix of [word_indices; frequencies].
#
# In Python, you would group documents by time period and compute
# TF (term frequency) or TF-IDF for each group:
#   from sklearn.feature_extraction.text import CountVectorizer
#   for year, docs in grouped_docs.items():
#       vectorizer.fit_transform(docs)

# Function to analyze word frequency over time
temporal_word_analysis <- function(out_data, meta_data, time_var = "Year") {
  # Create a mapping from document index to time period
  doc_topics <- data.frame(
    doc_id = 1:length(out_data$documents),
    time_period = meta_data[[time_var]]
  )

  # Get word frequencies by time period
  temporal_freq <- list()

  # unique() returns distinct values (like Python's set() or .unique()).
  for(period in unique(doc_topics$time_period)) {
    # which() returns indices where the condition is TRUE.
    # In Python: period_docs = np.where(doc_topics['time_period'] == period)[0]
    period_docs <- which(doc_topics$time_period == period)

    # Handle different document formats
    if (is.list(out_data$documents)) {
      # For list format (sparse): aggregate word counts for this period
      period_word_counts <- rep(0, length(out_data$vocab))
      names(period_word_counts) <- out_data$vocab

      for (doc_idx in period_docs) {
        if (doc_idx <= length(out_data$documents)) {
          doc <- out_data$documents[[doc_idx]]
          if (length(doc) > 0 && is.matrix(doc)) {
            word_indices <- doc[1, ]
            word_frequencies <- doc[2, ]
            for (j in seq_along(word_indices)) {
              period_word_counts[word_indices[j]] <- period_word_counts[word_indices[j]] + word_frequencies[j]
            }
          }
        }
      }
    } else if (is.matrix(out_data$documents)) {
      # For matrix format: subset rows and sum columns
      # drop = FALSE prevents R from simplifying a single-row matrix to a vector.
      period_word_counts <- colSums(out_data$documents[period_docs, , drop = FALSE])
      names(period_word_counts) <- out_data$vocab
    } else {
      # Fallback using STM's make.dt function
      dtm <- stm::make.dt(out_data$documents[period_docs], vocab = out_data$vocab)
      period_word_counts <- colSums(dtm)
      names(period_word_counts) <- out_data$vocab
    }

    # Store results for this period as a data frame
    temporal_freq[[as.character(period)]] <- data.frame(
      word = names(period_word_counts),
      freq = as.numeric(period_word_counts),
      period = period,
      stringsAsFactors = FALSE
    )
  }

  # do.call(rbind, list_of_dataframes) row-binds all data frames in the list.
  # This is equivalent to pd.concat(list_of_dfs) in Python.
  return(do.call(rbind, temporal_freq))
}

# Generate temporal analysis using the 'Year' column from metadata.
temporal_data <- temporal_word_analysis(out, out$meta, "Year")


# =============================================================================
#### SECTION: Create Yearly Word Clouds (Function Definition) ####
# =============================================================================

# This function generates a separate word cloud for each year in the corpus.
# It filters to words with minimum frequency and saves each as a PNG file.

# Create word clouds by year
create_yearly_wordclouds <- function(temporal_data, min_freq = 5) {
  years <- unique(temporal_data$period)

  for(year in years) {
    year_data <- temporal_data[temporal_data$period == year, ]
    year_data <- year_data[year_data$freq >= min_freq, ]
    year_data <- year_data[order(-year_data$freq), ]

    if(nrow(year_data) > 0) {
      png(paste0("wordcloud_", year, ".png"), width = 600, height = 600)
      wordcloud(words = year_data$word,
                freq = year_data$freq,
                max.words = 100,
                random.order = FALSE,
                rot.per = 0.35,
                colors = brewer.pal(8, "Spectral"),
                main = paste("Word Cloud -", year))
      dev.off()
    }
  }
}

# Uncomment to create yearly word clouds
# create_yearly_wordclouds(temporal_data)


# =============================================================================
#### SECTION 6: Comparative Word Clouds ####
# =============================================================================

# =============================================================================
# 6. COMPARATIVE WORD CLOUDS
# =============================================================================

# This function creates a "comparison cloud" showing words that are distinctive
# to each of two time periods. Words appearing more in period 2 are shown in
# one color, words more common in period 1 in another.
#
# merge() with all=TRUE performs a full outer join (like pd.merge(how='outer')).
# comparison.cloud() is from the wordcloud package.
#
# In Python, you would compute the difference in TF-IDF or raw frequencies
# between two groups and visualize the most different words.

# Compare word frequencies between two time periods or conditions
comparison_wordcloud <- function(temporal_data, period1, period2) {
  data1 <- temporal_data[temporal_data$period == period1, ]
  data2 <- temporal_data[temporal_data$period == period2, ]

  # Merge and calculate differences
  # suffixes = c("_1", "_2") adds these suffixes to disambiguate freq columns.
  comparison_df <- merge(data1[, c("word", "freq")],
                         data2[, c("word", "freq")],
                         by = "word",
                         suffixes = c("_1", "_2"),
                         all = TRUE)

  # Replace NA with 0 for words that appear in only one period.
  comparison_df[is.na(comparison_df)] <- 0
  comparison_df$diff <- comparison_df$freq_2 - comparison_df$freq_1

  # comparison.cloud() from the wordcloud package visualizes word differences
  # between two groups. The term.matrix has one column per group.
  comparison.cloud(term.matrix = as.matrix(comparison_df[, c("freq_1", "freq_2")]),
                   colors = c("red", "blue"),
                   max.words = 100)
}


# =============================================================================
#### SECTION 7: Styled Word Frequency Plots ####
# =============================================================================

# =============================================================================
# 7. STYLED WORD FREQUENCY PLOTS
# =============================================================================

# Create a publication-quality bar chart of the top 20 most frequent words.
# reorder(word, freq) sorts the bars by frequency for visual clarity.
# coord_flip() makes the bars horizontal (easier to read long words).
# In Python: sns.barplot(x='freq', y='word', data=top20_df, orient='h')

# Create bar plot of top words
top_words_plot <- word_freq_df %>%
  head(20) %>%
  ggplot(aes(x = reorder(word, freq), y = freq)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Top 20 Most Frequent Words",
       x = "Words",
       y = "Frequency") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"))

print(top_words_plot)


# =============================================================================
#### SECTION 8: Frequency Distribution Analysis ####
# =============================================================================

# =============================================================================
# 8. FREQUENCY DISTRIBUTION ANALYSIS
# =============================================================================

# Categorize words by frequency range to understand the distribution shape.
# case_when() is a vectorized conditional that evaluates conditions in order
# (like a series of if/elif statements in Python).
# count() is shorthand for group_by() + summarise(n = n()).
# In Python: pd.cut(df['freq'], bins=[0, 10, 20, 50, 100, np.inf])

# Analyze word frequency distribution
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

print("Word Frequency Distribution:")
print(freq_distribution)


# =============================================================================
#### SECTION 9: Custom Styling Functions ####
# =============================================================================

# =============================================================================
# 9. CUSTOM STYLING FUNCTIONS
# =============================================================================

# A reusable function for creating styled word clouds with custom color palettes.
# This provides more flexibility than the basic wordcloud() call above.

# Function for custom colored word cloud
custom_wordcloud <- function(words_df, title = "Word Cloud",
                             color_palette = "viridis",
                             background_color = "white") {

  # Generate colors based on frequency.
  # viridis() generates n colors from the viridis palette.
  # brewer.pal() generates colors from RColorBrewer palettes.
  if(color_palette == "viridis") {
    colors <- viridis(n = nrow(words_df), alpha = 0.8)
  } else {
    colors <- brewer.pal(min(nrow(words_df), 11), color_palette)
  }

  wordcloud(words = words_df$word,
            freq = words_df$freq,
            min.freq = 2,
            max.words = 150,
            random.order = FALSE,
            rot.per = 0.35,
            colors = colors,
            use.r.layout = FALSE,
            fixed.asp = TRUE)

  # title() adds a main title to the current base R plot.
  # cex.main controls the title font size multiplier.
  title(main = title, cex.main = 1.5)
}


# =============================================================================
#### SECTION 10: Summary Statistics ####
# =============================================================================

# =============================================================================
# 10. SUMMARY STATISTICS
# =============================================================================

# Print corpus-level vocabulary statistics to the console.
# nrow() returns the number of rows (unique words).
# sum() totals all frequencies.
# median() returns the middle value.
cat("=== WORD FREQUENCY SUMMARY ===\n")
cat("Total unique words:", nrow(word_freq_df), "\n")
cat("Total word occurrences:", sum(word_freq_df$freq), "\n")
cat("Most frequent word:", word_freq_df$word[1], "(", word_freq_df$freq[1], "times)\n")
cat("Median frequency:", median(word_freq_df$freq), "\n")
# Hapax legomena: words appearing exactly once (important in linguistics).
cat("Words appearing only once:", sum(word_freq_df$freq == 1), "\n")


# =============================================================================
#### SECTION: Usage Examples ####
# =============================================================================

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

# Example 1: Basic word cloud
# custom_wordcloud(word_freq_df[1:100, ], "Top 100 Words", "Set3")

# Example 2: Topic-specific analysis (if you have the topic model)
# create_topic_wordcloud(topic_model_prev, topic_num = 1, n_words = 30)

# Example 3: Temporal comparison
# comparison_wordcloud(temporal_data, min(temporal_data$period), max(temporal_data$period))

# Note: Uncomment the relevant sections based on your specific needs
# and ensure your topic model object is properly loaded
