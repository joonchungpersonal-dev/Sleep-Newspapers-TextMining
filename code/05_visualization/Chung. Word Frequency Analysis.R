# Word Frequency Analysis and Word Clouds for Structural Topic Model
# Required libraries
library(stm)
library(wordcloud)
library(wordcloud2)
library(RColorBrewer)
library(dplyr)
library(ggplot2)
library(tidytext)
library(viridis)

# =============================================================================
# 1. BASIC WORD FREQUENCY ANALYSIS
# =============================================================================

# Get vocabulary and word counts from STM processed data
vocab <- out$vocab

# First, let's check the structure of out$documents
cat("Structure of out$documents:\n")
str(out$documents)
cat("\nClass of out$documents:", class(out$documents), "\n")

# Handle different possible formats of out$documents
if (is.list(out$documents)) {
  # If documents is a list (sparse format), convert to word counts
  cat("Documents are in list format (sparse). Converting to word counts...\n")
  
  # Initialize word count vector
  word_counts <- rep(0, length(vocab))
  names(word_counts) <- vocab
  
  # Sum word counts across all documents
  for (i in seq_along(out$documents)) {
    doc <- out$documents[[i]]
    if (length(doc) > 0 && is.matrix(doc)) {
      # doc is a matrix with word indices and counts
      word_indices <- doc[1, ]  # First row contains word indices
      word_frequencies <- doc[2, ]  # Second row contains frequencies
      
      for (j in seq_along(word_indices)) {
        word_counts[word_indices[j]] <- word_counts[word_indices[j]] + word_frequencies[j]
      }
    }
  }
  
} else if (is.matrix(out$documents)) {
  # If documents is already a matrix
  cat("Documents are in matrix format. Using colSums...\n")
  word_counts <- colSums(out$documents)
  names(word_counts) <- vocab
  
} else {
  # Try to convert to matrix format
  cat("Attempting to convert documents to matrix format...\n")
  
  # Alternative approach using STM's make.dt function if available
  if (requireNamespace("stm", quietly = TRUE)) {
    # Create document-term matrix
    dtm <- stm::make.dt(out$documents, vocab = vocab)
    word_counts <- colSums(dtm)
    names(word_counts) <- vocab
  } else {
    stop("Cannot determine format of out$documents. Please check the structure.")
  }
}

# Create word frequency dataframe
word_freq_df <- data.frame(
  word = names(word_counts),
  freq = as.numeric(word_counts),
  stringsAsFactors = FALSE
) %>%
  arrange(desc(freq))

# Display top words
print("Top 20 most frequent words:")
print(head(word_freq_df, 20))

# =============================================================================
# 2. BASIC WORD CLOUD
# =============================================================================

# Simple word cloud with most frequent words
png("basic_wordcloud.png", width = 800, height = 600)
wordcloud(words = word_freq_df$word, 
          freq = word_freq_df$freq,
          min.freq = 5,
          max.words = 100,
          random.order = FALSE,
          rot.per = 0.35,
          colors = brewer.pal(8, "Dark2"))
dev.off()

# =============================================================================
# 3. INTERACTIVE WORD CLOUD (wordcloud2)
# =============================================================================

# Create interactive word cloud
wordcloud2(word_freq_df[1:100, ], 
           size = 0.8, 
           color = 'random-light',
           backgroundColor = "black")

# =============================================================================
# 4. TOPIC-SPECIFIC WORD CLOUDS
# =============================================================================

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
  
  # Create word cloud
  wordcloud(words = topic_df$word,
            freq = topic_df$prob * 1000,  # Scale probabilities
            min.freq = 1,
            max.words = n_words,
            random.order = FALSE,
            rot.per = 0.35,
            colors = brewer.pal(8, "Set2"),
            main = paste("Topic", topic_num))
}

# Create word clouds for each topic (assuming you have your topic model)
# Uncomment and modify based on your number of topics
# for(i in 1:topic_model_prev$settings$dim$K) {
#   png(paste0("topic_", i, "_wordcloud.png"), width = 600, height = 600)
#   create_topic_wordcloud(topic_model_prev, i)
#   dev.off()
# }

# =============================================================================
# 5. TEMPORAL WORD FREQUENCY ANALYSIS
# =============================================================================

# Function to analyze word frequency over time
temporal_word_analysis <- function(out_data, meta_data, time_var = "Year") {
  # Create document-term matrix with time information
  doc_topics <- data.frame(
    doc_id = 1:length(out_data$documents),
    time_period = meta_data[[time_var]]
  )
  
  # Get word frequencies by time period
  temporal_freq <- list()
  
  for(period in unique(doc_topics$time_period)) {
    period_docs <- which(doc_topics$time_period == period)
    
    # Handle different document formats
    if (is.list(out_data$documents)) {
      # For list format (sparse)
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
      # For matrix format
      period_word_counts <- colSums(out_data$documents[period_docs, , drop = FALSE])
      names(period_word_counts) <- out_data$vocab
    } else {
      # Try alternative approach
      dtm <- stm::make.dt(out_data$documents[period_docs], vocab = out_data$vocab)
      period_word_counts <- colSums(dtm)
      names(period_word_counts) <- out_data$vocab
    }
    
    temporal_freq[[as.character(period)]] <- data.frame(
      word = names(period_word_counts),
      freq = as.numeric(period_word_counts),
      period = period,
      stringsAsFactors = FALSE
    )
  }
  
  return(do.call(rbind, temporal_freq))
}

# Generate temporal analysis
temporal_data <- temporal_word_analysis(out, out$meta, "Year")

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
# 6. COMPARATIVE WORD CLOUDS
# =============================================================================

# Compare word frequencies between two time periods or conditions
comparison_wordcloud <- function(temporal_data, period1, period2) {
  data1 <- temporal_data[temporal_data$period == period1, ]
  data2 <- temporal_data[temporal_data$period == period2, ]
  
  # Merge and calculate differences
  comparison_df <- merge(data1[, c("word", "freq")], 
                         data2[, c("word", "freq")], 
                         by = "word", 
                         suffixes = c("_1", "_2"),
                         all = TRUE)
  
  comparison_df[is.na(comparison_df)] <- 0
  comparison_df$diff <- comparison_df$freq_2 - comparison_df$freq_1
  
  # Create comparison cloud
  comparison.cloud(term.matrix = as.matrix(comparison_df[, c("freq_1", "freq_2")]),
                   colors = c("red", "blue"),
                   max.words = 100)
}

# =============================================================================
# 7. STYLED WORD FREQUENCY PLOTS
# =============================================================================

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
# 8. FREQUENCY DISTRIBUTION ANALYSIS
# =============================================================================

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
# 9. CUSTOM STYLING FUNCTIONS
# =============================================================================

# Function for custom colored word cloud
custom_wordcloud <- function(words_df, title = "Word Cloud", 
                             color_palette = "viridis", 
                             background_color = "white") {
  
  # Generate colors based on frequency
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
  
  title(main = title, cex.main = 1.5)
}

# =============================================================================
# 10. SUMMARY STATISTICS
# =============================================================================

# Generate summary statistics
cat("=== WORD FREQUENCY SUMMARY ===\n")
cat("Total unique words:", nrow(word_freq_df), "\n")
cat("Total word occurrences:", sum(word_freq_df$freq), "\n")
cat("Most frequent word:", word_freq_df$word[1], "(", word_freq_df$freq[1], "times)\n")
cat("Median frequency:", median(word_freq_df$freq), "\n")
cat("Words appearing only once:", sum(word_freq_df$freq == 1), "\n")

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
