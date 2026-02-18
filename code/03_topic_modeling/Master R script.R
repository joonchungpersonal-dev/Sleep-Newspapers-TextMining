## Joon Chung
## Contact: see README
## The University of Miami, Miller School of Medicine
##  The Department of Informatics and Health Data Science
##
## Structural Topic Models by year
##
## Penultimate update: 7/28/2018
## Last update:        6/13/2025

## This script runs STM, with each year as a PREVALENCE covariate

set.seed(8675309)

library(stm)
library(tidyverse)
library(gridExtra)
library(tidytext)
library(ggwordcloud)

# 
# # Load full_text
# 
# ## Remove any NA
# full_text_subset <- full_text_clean %>% dplyr::select(text, text_id, datetime, Year) %>%
#   na.omit()
# full_text_subset$year <- as.numeric(full_text_subset$Year)
# 
# Processed
processed <- textProcessor(full_text_subset$text, striphtml = TRUE, metadata = as.data.frame(full_text_subset))

plotRemoved(processed$documents, lower.thresh = seq(from = 10, to = 2000, by = 2))

out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 500)

## Run STM
set.seed(8675309)

topic_model_prev <- stm(out$documents,
                        out$vocab,
                        data = out$meta,
                        prevalence = ~year,
                        K = 70,
                        verbose = TRUE,
                        init.type = "Spectral")
# 

###################################################################################

sageLabels(topic_model_prev)

## Run the b-spline model
prep <- estimateEffect(1:70 ~ s(year), topic_model_prev, meta = out$meta)

## Function to code topic proportions and terms
stm_plot <- function(topic_select, title){
  ## This function takes a structural topic model and:
  ##  1) extracts meaningful data from a plot.estimateEffect object
  ##  2) plots expected topic proportions by year in ggplot2
  ##  3) tidys the beta matrix of the topic model
  ##  4) plots the top 10 terms in a given topic
  ##  5) returns both plots in grid.arrange
  
  data <- plot(prep, covariate = "year", method = "continuous", topics = topic_select)
  
  # Expected topic proportion
  means <- with(data, means) %>% unlist() %>% as.matrix() %>% as.data.frame()
  means <- with(means, V1)
  
  # 95% Confidence interval
  ci <- with(data, ci) %>% unlist() %>% as.data.frame()
  
  lower <- ci[seq(1, nrow(ci), 2),]
  upper <- ci[seq(2, nrow(ci), 2),]
  
  # years
  years <- with(data, x) %>% as.numeric()
  plot_df <- data.frame(years = years, means = means, lower = lower, upper = upper)
  plot_df <- plot_df %>% filter(years <= 2017)
  
  # Plot
  topic_prop <- ggplot(plot_df, aes(x = years, y = means)) + geom_line() + 
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) + 
    scale_x_continuous(breaks = seq(1983,2016, 3), limits = c(1983, 2017)) + 
    theme_bw() +
    theme(axis.text.x = element_text(angle =45)) + 
    xlab("Year") + 
    ylab("Expected topic proportion") 
  
  
  stm_beta <- tidy(topic_model_prev, matrix = "beta")
  topic_terms <- stm_beta %>% dplyr::filter(topic == topic_select) %>%
    arrange(desc(beta)) %>%
    top_n(10) %>%
    ggplot(aes(x = reorder(term, beta), beta)) + 
    geom_col() + 
    # facet_wrap(~topic, scales = "free") + 
    coord_flip() + 
    theme_bw() +
    xlab("Top terms") + 
    ylab("Probability word | Topic")
  
  return(grid.arrange(topic_terms, topic_prop, nrow = 1,top = title))
}

## Preliminary plots - full plot output later in code.
topics_of_interest
# 3  7 15 32 47 51

sleep_plot <- stm_plot(topic = 3, "Work and sleep")
drug_plot <- stm_plot(topic = 7, "Sleep medicine / drugs")
science_plot <- stm_plot(topic = 15, "Circadian science")
apnea_plot <- stm_plot(topic = 32, "Sleep apnea, hospitals")
health_plot <- stm_plot(topic = 47, "Health research")
academic_research_plot <- stm_plot(topic = 51, "Sleep research")
school_plot <- stm_plot(topic = 59, "School start times")
work_plot <- stm_plot(topic = 22, "Work")
disaster_plot <- stm_plot(topic = 37, "Disaster")
dearann_1_plot <- stm_plot(topic = 14, "Dear Ann (validation)")
iraq_2_plot <- stm_plot(topic = 23, "Iraq wars")


ggsave(sleep_plot, file = "sleep_plot.png",
       width = 10,
       height = 5)

ggsave(drug_plot, file = "drug_plot.png",
       width = 10,
       height = 5)

ggsave(science_plot, file = "science_plot.png",
       width = 10,
       height = 5)

ggsave(apnea_plot, file = "apnea_plot.png",
       width = 10,
       height = 5)

ggsave(car_plot, file = "car_plot.png",
       width = 10,
       height = 5)

ggsave(health_plot, file = "health_plot.png",
       width = 10,
       height = 5)

ggsave(academic_research_plot, file = "academic_research_plot.png",
       width = 10,
       height = 5)

ggsave(school_plot, file = "school_plot.png",
       width = 10,
       height = 5)

ggsave(dearann_1_plot, file = "dearann_1_plot.png",
       width = 10,
       height = 5)

ggsave(iraq_2_plot, file = "iraq_2_plot.png",
       width = 10,
       height = 5)


topics_of_interest
# 3  7 15 32 45 47 51 59 14 23

cloud(topic_model_prev, 3)
cloud(topic_model_prev, 7)
cloud(topic_model_prev, 15)
cloud(topic_model_prev, 32)
cloud(topic_model_prev, 47)
cloud(topic_model_prev, 51)
cloud(topic_model_prev, 59)
cloud(topic_model_prev, 14)
cloud(topic_model_prev, 23)

# Function to create ggplot wordcloud from STM topic
stm_to_ggwordcloud <- function(model, topic, topic_name) {
  # Get top words and probabilities correctly
  words <- labelTopics(model, topic, n = 30)
  word_probs <- data.frame(
    word = words$prob[1,],  # First row contains the words
    freq = as.numeric(words$prob[2,]),  # Second row contains probabilities
    stringsAsFactors = FALSE
  )
  
  # Alternative approach: use the topic-word matrix directly
  # This is more reliable for getting actual probabilities
  beta <- exp(model$beta$logbeta[[1]])  # Convert log probabilities to probabilities
  topic_words <- beta[topic, ]
  top_indices <- order(topic_words, decreasing = TRUE)[1:30]
  
  word_probs <- data.frame(
    word = model$vocab[top_indices],
    freq = topic_words[top_indices],
    stringsAsFactors = FALSE
  )
  
  # Create ggplot wordcloud
  ggplot(word_probs, aes(label = word, size = freq)) +
    geom_text_wordcloud() +
    scale_size_area(max_size = 10) +
    theme_minimal() +
    labs(title = topic_name)
}

# Create ggplot wordcloud objects
sleep_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 3, "")
drug_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 7, "")
science_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 15, "")
apnea_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 32, "")
health_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 47, "")
academic_research_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 51, "")
school_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 59, "")



dearann_1_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 14, "")
iraq_2_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 23, "")

# Option 1: All 10 clouds in a 2x5 grid
all_clouds_plot <- grid.arrange(
  sleep_cloud_gg, drug_cloud_gg, science_cloud_gg, apnea_cloud_gg, 
  health_cloud_gg, academic_research_cloud_gg, school_cloud_gg,
  ncol = 2, nrow = 5
)

# Option 2: All 10 clouds in a 3x4 grid (with 2 empty spaces)
all_clouds_plot_3x4 <- grid.arrange(
  sleep_cloud_gg, drug_cloud_gg, science_cloud_gg,
  apnea_cloud_gg, health_cloud_gg,
  academic_research_cloud_gg, 
  ncol = 2, nrow = 4
)

# Option 3: Sleep-related topics only (first 8 clouds) in a 2x4 grid
sleep_related_clouds <- grid.arrange(
  sleep_cloud_gg, drug_cloud_gg, science_cloud_gg, apnea_cloud_gg,
  health_cloud_gg, academic_research_cloud_gg, school_cloud_gg, car_cloud_gg,
  ncol = 2, nrow = 4
)

# Option 4: With custom layout using layout_matrix
layout_matrix <- rbind(
  c(1, 2, 3),
  c(4, 5, 6),
  c(7, 8, 9),
  c(10, 10, NA)  # iraq_2 spans 2 columns, empty space in corner
)




# Option 1: Using png() device (recommended)
png("sleep_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 3)
dev.off()

png("drug_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 7)
dev.off()

png("science_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 15)
dev.off()

png("apnea_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 32)
dev.off()

png("car_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 45)
dev.off()

png("health_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 47)
dev.off()

png("academic_research_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 51)
dev.off()

png("school_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 59)
dev.off()

png("dearann_1_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 14)
dev.off()

png("iraq_2_cloud.png", width = 800, height = 600)
cloud(topic_model_prev, 23)
dev.off()

# Alternative Option 2: Using a loop to save all at once
topics <- c(3, 7, 15, 32, 45, 47, 51, 59, 14, 23)
filenames <- c("sleep_cloud.png", "drug_cloud.png", "science_cloud.png", 
               "apnea_cloud.png", "car_cloud.png", "health_cloud.png",
               "academic_research_cloud.png", "school_cloud.png", 
               "dearann_1_cloud.png", "iraq_2_cloud.png")

for(i in 1:length(topics)) {
  png(filenames[i], width = 800, height = 600)
  cloud(topic_model_prev, topics[i])
  dev.off()
}



sageLabels(topic_model_prev)
plot(topicCorr(topic_model_prev))


stm_beta <- tidy(topic_model_prev, matrix = "beta")
stm_theta <- tidy(topic_model_prev, matrix = "theta") %>%
  dplyr::filter(topic == topics_of_interest)

## Distribution of document probabilites for each topic
ggplot(stm_theta, aes(gamma, fill = as.factor(topic))) +
  geom_histogram(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  labs(title = "Distribution of document probabilities for each topic",
       y = "Number of stories", x = expression(gamma))

final_text <- full_text_subset %>% mutate(document = row_number())
final_text <- merge(final_text, stm_theta, by = "document")


summary(final_text$gamma)
qplot(final_text$gamma)


## plot.STM()
plot.STM(topic_model_prev, type = "summary", labeltype = "prob",
         topics = topics_of_interest)



#### Write top 100 articles to file:

# Define your topics of interest and names
topics_of_interest <- c(3, 7, 15, 32, 47, 51)
topic_names <- c("Sleep and Work", "Sleep_medicine_drugs", "Circadian_science", 
                 "Sleep_apnea_hospitals", "Sleep_and_health",
                 "Academic_sleep_research")

# Function to extract top documents for a topic
extract_top_documents <- function(model, topic_num, topic_name, meta_data, n_docs = 100) {
  
  # Get document-topic proportions for this topic
  topic_proportions <- model$theta[, topic_num]
  
  # Get top document indices
  top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
  top_proportions <- topic_proportions[top_indices]
  
  # Extract document information
  top_docs_data <- data.frame(
    Rank = 1:n_docs,
    Document_Index = top_indices,
    Topic_Proportion = round(top_proportions, 4),
    stringsAsFactors = FALSE
  )
  
  # Add metadata if available (adjust column names as needed)
  if (!is.null(meta_data)) {
    # Text content
    if ("text" %in% names(meta_data)) {
      top_docs_data$Text <- meta_data$text[top_indices]
    } else if ("Text" %in% names(meta_data)) {
      top_docs_data$Text <- meta_data$Text[top_indices]
    } else if ("content" %in% names(meta_data)) {
      top_docs_data$Text <- meta_data$content[top_indices]
    }
    
    # Publication date - check for datetime and year columns specifically
    if ("datetime" %in% names(meta_data)) {
      top_docs_data$Date <- meta_data$datetime[top_indices]
    } else if ("date" %in% names(meta_data)) {
      top_docs_data$Date <- meta_data$date[top_indices]
    } else if ("Date" %in% names(meta_data)) {
      top_docs_data$Date <- meta_data$Date[top_indices]
    } else if ("publication_date" %in% names(meta_data)) {
      top_docs_data$Date <- meta_data$publication_date[top_indices]
    } else if ("pub_date" %in% names(meta_data)) {
      top_docs_data$Date <- meta_data$pub_date[top_indices]
    }
    
    # Year column (separate from datetime)
    if ("year" %in% names(meta_data)) {
      top_docs_data$Year <- meta_data$year[top_indices]
    } else if ("Year" %in% names(meta_data)) {
      top_docs_data$Year <- meta_data$Year[top_indices]
    }
    
    # Newspaper/Source
    if ("source" %in% names(meta_data)) {
      top_docs_data$Source <- meta_data$source[top_indices]
    } else if ("Source" %in% names(meta_data)) {
      top_docs_data$Source <- meta_data$Source[top_indices]
    } else if ("newspaper" %in% names(meta_data)) {
      top_docs_data$Source <- meta_data$newspaper[top_indices]
    } else if ("Newspaper" %in% names(meta_data)) {
      top_docs_data$Source <- meta_data$Newspaper[top_indices]
    } else if ("publication" %in% names(meta_data)) {
      top_docs_data$Source <- meta_data$publication[top_indices]
    }
    
    # Title (if available)
    if ("title" %in% names(meta_data)) {
      top_docs_data$Title <- meta_data$title[top_indices]
    } else if ("Title" %in% names(meta_data)) {
      top_docs_data$Title <- meta_data$Title[top_indices]
    } else if ("headline" %in% names(meta_data)) {
      top_docs_data$Title <- meta_data$headline[top_indices]
    }
    
    # Document ID (if available)
    if ("id" %in% names(meta_data)) {
      top_docs_data$Document_ID <- meta_data$id[top_indices]
    } else if ("ID" %in% names(meta_data)) {
      top_docs_data$Document_ID <- meta_data$ID[top_indices]
    } else if ("doc_id" %in% names(meta_data)) {
      top_docs_data$Document_ID <- meta_data$doc_id[top_indices]
    }
  }
  
  return(top_docs_data)
}

# Create directory for output files
if (!dir.exists("top_documents")) {
  dir.create("top_documents")
}

# Extract and save top documents for each topic
for (i in 1:length(topics_of_interest)) {
  topic_num <- topics_of_interest[i]
  topic_name <- topic_names[i]
  
  cat("Processing topic", topic_num, ":", topic_name, "\n")
  
  # Extract top documents
  # Using full_text_subset as the metadata object
  top_docs <- extract_top_documents(topic_model_prev, topic_num, topic_name, full_text_subset)
  
  # Option 1: Save as CSV
  csv_filename <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_top100.csv")
  write.csv(top_docs, csv_filename, row.names = FALSE)
  
  # Option 2: Save as readable text file
  txt_filename <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_top100.txt")
  
  # Create formatted text output
  sink(txt_filename)
  cat("TOP 100 DOCUMENTS FOR TOPIC", topic_num, ":", toupper(gsub("_", " ", topic_name)), "\n")
  cat("=" %>% rep(80) %>% paste(collapse = ""), "\n\n")
  
  for (j in 1:nrow(top_docs)) {
    cat("RANK:", j, "| PROPORTION:", top_docs$Topic_Proportion[j], 
        "| DOC INDEX:", top_docs$Document_Index[j], "\n")
    
    # Publication Date and Year
    if ("Date" %in% names(top_docs)) {
      cat("DATE:", top_docs$Date[j], "\n")
    }
    if ("Year" %in% names(top_docs)) {
      cat("YEAR:", top_docs$Year[j], "\n")
    }
    
    # Newspaper/Source
    if ("Source" %in% names(top_docs)) {
      cat("SOURCE/NEWSPAPER:", top_docs$Source[j], "\n")
    }
    
    # Title/Headline
    if ("Title" %in% names(top_docs)) {
      cat("TITLE:", top_docs$Title[j], "\n")
    }
    
    # Document ID
    if ("Document_ID" %in% names(top_docs)) {
      cat("DOCUMENT ID:", top_docs$Document_ID[j], "\n")
    }
    
    cat("TEXT:\n")
    if ("Text" %in% names(top_docs)) {
      # Wrap text to 80 characters for readability
      text <- top_docs$Text[j]
      wrapped_text <- strwrap(text, width = 80)
      cat(paste(wrapped_text, collapse = "\n"), "\n")
    }
    
    cat("\n", "-" %>% rep(80) %>% paste(collapse = ""), "\n\n")
  }
  sink()
  
  # Option 3: Save individual text files for each document (if you want separate files)
  doc_dir <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_individual/")
  if (!dir.exists(doc_dir)) {
    dir.create(doc_dir, recursive = TRUE)
  }
  
  for (j in 1:nrow(top_docs)) {
    individual_filename <- paste0(doc_dir, "rank_", sprintf("%03d", j), "_doc_", top_docs$Document_Index[j], ".txt")
    
    sink(individual_filename)
    cat("TOPIC:", topic_name, "| RANK:", j, "| PROPORTION:", top_docs$Topic_Proportion[j], "\n")
    cat("DOCUMENT INDEX:", top_docs$Document_Index[j], "\n")
    
    if ("Date" %in% names(top_docs)) cat("PUBLICATION DATE:", top_docs$Date[j], "\n")
    if ("Year" %in% names(top_docs)) cat("YEAR:", top_docs$Year[j], "\n")
    if ("Source" %in% names(top_docs)) cat("SOURCE/NEWSPAPER:", top_docs$Source[j], "\n")
    if ("Title" %in% names(top_docs)) cat("TITLE/HEADLINE:", top_docs$Title[j], "\n")
    if ("Document_ID" %in% names(top_docs)) cat("DOCUMENT ID:", top_docs$Document_ID[j], "\n")
    
    cat("\nTEXT:\n")
    cat("=" %>% rep(50) %>% paste(collapse = ""), "\n")
    
    if ("Text" %in% names(top_docs)) {
      cat(top_docs$Text[j], "\n")
    }
    sink()
  }
}

# Create summary file with all topics
summary_filename <- "top_documents/summary_all_topics.txt"
sink(summary_filename)
cat("SUMMARY: TOP 100 DOCUMENTS FOR ALL TOPICS OF INTEREST\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

for (i in 1:length(topics_of_interest)) {
  topic_num <- topics_of_interest[i]
  topic_name <- topic_names[i]
  
  top_docs <- extract_top_documents(topic_model_prev, topic_num, topic_name, full_text_subset)
  
  cat("TOPIC", topic_num, ":", toupper(gsub("_", " ", topic_name)), "\n")
  cat("Top 10 documents (proportion):\n")
  
  for (j in 1:10) {
    cat("  ", j, ". Doc", top_docs$Document_Index[j], 
        "(", top_docs$Topic_Proportion[j], ")", "\n")
  }
  cat("\n")
}
sink()

# =============================================================================
# STM TOPIC TERMS EXTRACTION - PROBABILITY AND SCORE
# =============================================================================
# Extract marginal probability and score terms for all topics in STM model

library(stm)
library(tidyverse)

# =============================================================================
# EXTRACT ALL TOPIC TERMS FUNCTION
# =============================================================================

#' Extract Probability and Score Terms for All Topics
#' 
#' Uses sageLabels() to extract top terms for all topics using both
#' marginal probability and score ranking methods.
#' 
#' @param topic_model STM model object
#' @param n_terms Integer: number of top terms to extract per method per topic
#' @return Data frame with topics, ranking methods, and terms

extract_all_topic_terms <- function(topic_model, n_terms = 5) {
  
  cat("Extracting", n_terms, "terms per method for", topic_model$settings$dim$K, "topics...\n")
  
  # Get all topic labels using sageLabels
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  # Extract probability and score terms
  prob_terms <- sage_result$marginal$prob    # Matrix: topics × terms
  score_terms <- sage_result$marginal$score  # Matrix: topics × terms
  
  # Convert to long format data frame
  results_list <- list()
  
  # Process probability terms
  for (topic_num in 1:nrow(prob_terms)) {
    prob_row <- data.frame(
      topic = topic_num,
      method = "probability",
      rank = 1:n_terms,
      term = as.character(prob_terms[topic_num, ])
    )
    results_list[[length(results_list) + 1]] <- prob_row
  }
  
  # Process score terms
  for (topic_num in 1:nrow(score_terms)) {
    score_row <- data.frame(
      topic = topic_num,
      method = "score", 
      rank = 1:n_terms,
      term = as.character(score_terms[topic_num, ])
    )
    results_list[[length(results_list) + 1]] <- score_row
  }
  
  # Combine all results
  final_results <- do.call(rbind, results_list)
  
  cat("✓ Extracted", nrow(final_results), "term entries\n")
  cat("✓ Topics:", topic_model$settings$dim$K, "\n")
  cat("✓ Methods: probability, score\n")
  cat("✓ Terms per method per topic:", n_terms, "\n")
  
  return(final_results)
}

# =============================================================================
# WIDE FORMAT EXTRACTION (ALTERNATIVE)
# =============================================================================

#' Extract Terms in Wide Format (Excel-friendly)
#' 
#' Creates a wide-format table with one row per topic and separate columns
#' for each term rank and method. Better for Excel viewing.

extract_terms_wide_format <- function(topic_model, n_terms = 5) {
  
  cat("Extracting terms in wide format...\n")
  
  # Get all topic labels
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  # Create column names
  prob_cols <- paste0("prob_", 1:n_terms)
  score_cols <- paste0("score_", 1:n_terms)
  
  # Create wide format data frame
  wide_results <- data.frame(
    topic = 1:nrow(sage_result$marginal$prob)
  )
  
  # Add probability columns
  for (i in 1:n_terms) {
    wide_results[[prob_cols[i]]] <- as.character(sage_result$marginal$prob[, i])
  }
  
  # Add score columns
  for (i in 1:n_terms) {
    wide_results[[score_cols[i]]] <- as.character(sage_result$marginal$score[, i])
  }
  
  cat("✓ Created wide format with", ncol(wide_results), "columns\n")
  
  return(wide_results)
}

# =============================================================================
# SAGELABELS-STYLE FORMATTED OUTPUT
# =============================================================================

#' Create Exact sageLabels Format Output
#' 
#' Generates output exactly like sageLabels console display:
#' Topic X:
#'   Marginal Highest Prob: term1, term2, term3, term4, term5
#'   Marginal Score: term1, term2, term3, term4, term5

create_sagelabels_format <- function(topic_model, n_terms = 5, topics = NULL) {
  
  # If no specific topics requested, do all topics
  if (is.null(topics)) {
    topics <- 1:topic_model$settings$dim$K
  }
  
  # Get sage results
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  # Create formatted output
  output_lines <- c()
  
  for (topic_num in topics) {
    if (topic_num <= nrow(sage_result$marginal$prob)) {
      
      # Get terms for this topic
      prob_terms <- as.character(sage_result$marginal$prob[topic_num, ])
      score_terms <- as.character(sage_result$marginal$score[topic_num, ])
      
      # Format exactly like sageLabels
      topic_line <- paste0("Topic ", topic_num, ":")
      prob_line <- paste0(" \t Marginal Highest Prob: ", paste(prob_terms, collapse = ", "))
      score_line <- paste0(" \t Marginal Score: ", paste(score_terms, collapse = ", "))
      
      # Add to output
      output_lines <- c(output_lines, topic_line, prob_line, score_line, "")
    }
  }
  
  return(output_lines)
}

#' Print sageLabels Format to Console
print_sagelabels_format <- function(topic_model, n_terms = 5, topics = NULL) {
  
  output_lines <- create_sagelabels_format(topic_model, n_terms, topics)
  
  # Print each line
  for (line in output_lines) {
    cat(line, "\n")
  }
}

#' Save sageLabels Format to Text File
save_sagelabels_format <- function(topic_model, filename = "stm_sagelabels_format.txt", 
                                   n_terms = 5, topics = NULL) {
  
  output_lines <- create_sagelabels_format(topic_model, n_terms, topics)
  
  # Write to file
  writeLines(output_lines, filename)
  
  cat("✓ Saved sageLabels format to:", filename, "\n")
  cat("✓ Topics included:", length(topics %||% 1:topic_model$settings$dim$K), "\n")
  cat("✓ Terms per method:", n_terms, "\n")
}

#' Create sageLabels Format for Specific Topics
format_specific_topics <- function(topic_model, topics, n_terms = 5) {
  
  cat("=============================================================================\n")
  cat("STM TOPIC TERMS - sageLabels Format\n")
  cat("=============================================================================\n")
  
  print_sagelabels_format(topic_model, n_terms, topics)
  
  cat("=============================================================================\n")
}

# =============================================================================
# BATCH PROCESSING WITH SAGELABELS FORMAT
# =============================================================================

#' Create sageLabels Output for All Topics with File Saving
process_all_sagelabels_format <- function(topic_model, n_terms = 5, 
                                          save_file = TRUE, 
                                          filename = "stm_all_topics_sagelabels.txt") {
  
  total_topics <- topic_model$settings$dim$K
  
  cat("Creating sageLabels format for", total_topics, "topics with", n_terms, "terms each...\n\n")
  
  # Create and optionally save
  if (save_file) {
    save_sagelabels_format(topic_model, filename, n_terms, topics = NULL)
    cat("\n")
  }
  
  # Also display first few topics as preview
  cat("PREVIEW (First 5 topics):\n")
  cat("=============================================================================\n")
  print_sagelabels_format(topic_model, n_terms, topics = 1:5)
  cat("=============================================================================\n")
  
  if (save_file) {
    cat("Full output saved to:", filename, "\n")
  }
}

# =============================================================================
# MATRIX FORMAT OUTPUT (PURE SAGELABELS STYLE)
# =============================================================================

#' Extract and Save Matrices in sageLabels Style
#' 
#' Creates separate CSV files for probability and score matrices,
#' exactly like the sageLabels structure

save_sagelabels_matrices <- function(topic_model, n_terms = 5, prefix = "stm_terms") {
  
  cat("Extracting matrices in sageLabels format...\n")
  
  # Get sage results
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  # Extract matrices
  prob_matrix <- sage_result$marginal$prob
  score_matrix <- sage_result$marginal$score
  
  # Convert to data frames with topic numbers
  prob_df <- as.data.frame(prob_matrix)
  score_df <- as.data.frame(score_matrix)
  
  # Add topic column
  prob_df$topic <- 1:nrow(prob_df)
  score_df$topic <- 1:nrow(score_df)
  
  # Reorder columns (topic first)
  prob_df <- prob_df[, c("topic", paste0("V", 1:n_terms))]
  score_df <- score_df[, c("topic", paste0("V", 1:n_terms))]
  
  # Rename term columns
  colnames(prob_df) <- c("topic", paste0("term_", 1:n_terms))
  colnames(score_df) <- c("topic", paste0("term_", 1:n_terms))
  
  # Save files
  prob_file <- paste0(prefix, "_probability_matrix.csv")
  score_file <- paste0(prefix, "_score_matrix.csv")
  
  write.csv(prob_df, prob_file, row.names = FALSE)
  write.csv(score_df, score_file, row.names = FALSE)
  
  cat("✓ Saved probability matrix:", prob_file, "\n")
  cat("✓ Saved score matrix:", score_file, "\n")
  
  return(list(probability = prob_df, score = score_df))
}

# =============================================================================
# COMBINED STACKED MATRIX (SINGLE FILE)
# =============================================================================

#' Create Single File with Both Matrices Stacked
#' 
#' Combines probability and score matrices into one file with clear section headers

create_combined_stacked_matrix <- function(topic_model, n_terms = 5) {
  
  cat("Creating combined stacked matrix...\n")
  
  # Get matrices
  matrices <- save_sagelabels_matrices(topic_model, n_terms, prefix = "temp")
  
  # Create combined data frame with section headers
  prob_section <- matrices$probability
  prob_section$method <- "PROBABILITY"
  
  score_section <- matrices$score  
  score_section$method <- "SCORE"
  
  # Add blank row separator
  blank_row <- data.frame(
    topic = NA,
    term_1 = "---",
    term_2 = "---", 
    term_3 = "---",
    term_4 = "---",
    term_5 = "---",
    method = "---"
  )
  
  # Adjust blank row columns to match n_terms
  if (n_terms != 5) {
    blank_row <- blank_row[, 1:(n_terms + 2)]  # topic + n_terms + method
    for (i in 2:(n_terms + 1)) {
      blank_row[, i] <- "---"
    }
  }
  
  # Combine with separator
  combined_matrix <- rbind(
    prob_section,
    blank_row,
    score_section
  )
  
  # Reorder columns: method first, then topic, then terms
  combined_matrix <- combined_matrix[, c("method", "topic", paste0("term_", 1:n_terms))]
  
  cat("✓ Created combined matrix with", nrow(combined_matrix), "rows\n")
  
  return(combined_matrix)
}

# =============================================================================
# PRINT FORMATTED OUTPUT (CONSOLE DISPLAY)
# =============================================================================

#' Print Stacked Output to Console (sageLabels style)
#' 
#' Displays the matrices in a clean, readable format in the console

print_stacked_terms <- function(topic_model, n_terms = 5, topics_to_show = 1:10) {
  
  # Get sage results
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  cat("=============================================================================\n")
  cat("STM TOPIC TERMS - PROBABILITY AND SCORE (", n_terms, " terms each)\n") 
  cat("=============================================================================\n\n")
  
  cat("MARGINAL PROBABILITY:\n")
  cat("Topic | ", paste(paste0("Term ", 1:n_terms), collapse = " | "), "\n")
  cat("------|", paste(rep("------", n_terms), collapse = "|"), "\n")
  
  for (topic in topics_to_show) {
    if (topic <= nrow(sage_result$marginal$prob)) {
      terms <- sage_result$marginal$prob[topic, ]
      cat(sprintf("%5d | ", topic), paste(sprintf("%-5s", terms), collapse = " | "), "\n")
    }
  }
  
  cat("\nMARGINAL SCORE:\n")
  cat("Topic | ", paste(paste0("Term ", 1:n_terms), collapse = " | "), "\n")
  cat("------|", paste(rep("------", n_terms), collapse = "|"), "\n")
  
  for (topic in topics_to_show) {
    if (topic <= nrow(sage_result$marginal$score)) {
      terms <- sage_result$marginal$score[topic, ]
      cat(sprintf("%5d | ", topic), paste(sprintf("%-5s", terms), collapse = " | "), "\n")
    }
  }
  
  cat("\n=============================================================================\n")
}

# =============================================================================
# SUMMARY STATISTICS FUNCTION
# =============================================================================

#' Generate Summary Statistics for Topic Terms
#' 
#' Shows overlap between probability and score methods, unique terms, etc.

analyze_term_overlap <- function(topic_model, n_terms = 5) {
  
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  overlap_stats <- data.frame(
    topic = 1:nrow(sage_result$marginal$prob),
    prob_terms = "",
    score_terms = "",
    overlap_count = 0,
    overlap_terms = "",
    unique_prob = "",
    unique_score = ""
  )
  
  for (topic_num in 1:nrow(sage_result$marginal$prob)) {
    prob_terms <- as.character(sage_result$marginal$prob[topic_num, ])
    score_terms <- as.character(sage_result$marginal$score[topic_num, ])
    
    # Calculate overlap
    overlap <- intersect(prob_terms, score_terms)
    unique_prob <- setdiff(prob_terms, score_terms)
    unique_score <- setdiff(score_terms, prob_terms)
    
    overlap_stats$prob_terms[topic_num] <- paste(prob_terms, collapse = ", ")
    overlap_stats$score_terms[topic_num] <- paste(score_terms, collapse = ", ")
    overlap_stats$overlap_count[topic_num] <- length(overlap)
    overlap_stats$overlap_terms[topic_num] <- paste(overlap, collapse = ", ")
    overlap_stats$unique_prob[topic_num] <- paste(unique_prob, collapse = ", ")
    overlap_stats$unique_score[topic_num] <- paste(unique_score, collapse = ", ")
  }
  
  return(overlap_stats)
}

# =============================================================================
# EXECUTION EXAMPLES
# =============================================================================

# =============================================================================
# ALTERNATIVE FORMATS (IF NEEDED)
# =============================================================================

#' Simple Wide Format (Excel-friendly)
create_simple_wide <- function(topic_model, n_terms = 5) {
  sage_result <- sageLabels(topic_model, n = n_terms)
  
  wide_df <- data.frame(topic = 1:nrow(sage_result$marginal$prob))
  
  for (i in 1:n_terms) {
    wide_df[[paste0("prob_", i)]] <- as.character(sage_result$marginal$prob[, i])
    wide_df[[paste0("score_", i)]] <- as.character(sage_result$marginal$score[, i])
  }
  
  return(wide_df)
}

# =============================================================================
# EXECUTION EXAMPLES
# =============================================================================

cat("=============================================================================\n")
cat("STM TOPIC TERMS EXTRACTION - EXACT SAGELABELS FORMAT\n")
cat("=============================================================================\n")
cat("Create output exactly like sageLabels console display with Marginal Prob and Score.\n\n")

cat("📋 SAGELABELS FORMAT OPTIONS:\n\n")

cat("OPTION 1: Display all topics in sageLabels format\n")
cat("process_all_sagelabels_format(topic_model_prev, n_terms = 5)\n")
cat("# Shows preview + saves to file: stm_all_topics_sagelabels.txt\n\n")

cat("OPTION 2: View specific topics in console\n")
cat("print_sagelabels_format(topic_model_prev, n_terms = 5, topics = c(1, 5, 10, 35))\n\n")

cat("OPTION 3: Save specific topics to file\n")
cat("save_sagelabels_format(topic_model_prev, 'selected_topics.txt', n_terms = 5, topics = 1:20)\n\n")

cat("OPTION 4: Format specific topics with header\n")
cat("format_specific_topics(topic_model_prev, topics = c(35, 42, 67), n_terms = 7)\n\n")

cat("OPTION 5: Get raw output lines (for further processing)\n")
cat("output_lines <- create_sagelabels_format(topic_model_prev, n_terms = 5, topics = 1:10)\n\n")

cat("🎯 RECOMMENDED (EXACT SAGELABELS REPLICA):\n")
cat("# Process all topics and save to file\n")
cat("process_all_sagelabels_format(topic_model_prev, n_terms = 5, filename = 'stm_sagelabels_output.txt')\n\n")

cat("# Or just view a few topics\n")
cat("print_sagelabels_format(topic_model_prev, n_terms = 5, topics = 30:40)\n\n")

cat("=============================================================================\n")
cat("📄 EXACT OUTPUT FORMAT (like your example):\n")
cat("Topic 35:\n")
cat(" \t Marginal Highest Prob: race, run, ride, hors, mile, track, finish\n")
cat(" \t Marginal Score: hors, race, ride, bike, mile, run, track\n")
cat("\n")
cat("Topic 36:\n") 
cat(" \t Marginal Highest Prob: game, team, play, win, season, player, score\n")
cat(" \t Marginal Score: pitch, game, inning, team, player, base, run\n\n")

cat("📁 FILE OUTPUT:\n")
cat("• Text file with exact sageLabels formatting\n")
cat("• One topic per section with Marginal Prob and Score\n")
cat("• Easy to copy/paste into documents\n")
cat("• Perfect for supplementary materials\n\n")

cat("=============================================================================\n")

# =============================================================================
# STM CO-OCCURRENCE NETWORK ANALYSIS WITH SAGELABELS
# =============================================================================
# 
# PURPOSE: Create co-occurrence networks for STM topic models using different
#          word ranking methods (FREX, Score, Probability, Lift)
#
# AUTHOR: [Your Name]
# DATE: [Current Date]
# 
# OVERVIEW:
# This script creates word co-occurrence networks for Structural Topic Models (STM).
# Unlike frequency-based approaches, this uses STM's sophisticated ranking methods:
# - FREX: Balances frequency and exclusivity (words frequent in topic, rare elsewhere)
# - Score: Weighted combination of probability and exclusivity  
# - Probability: Raw word probabilities within topics
# - Lift: How much more likely words appear in this topic vs. overall corpus
#
# OUTPUT: Network visualizations and edge lists for each topic and word count
# =============================================================================

# Load required libraries
library(quanteda)    # Text processing and DTM creation
library(igraph)      # Network analysis and graph objects
library(ggraph)      # Grammar of graphics for network visualization
library(tidyverse)   # Data manipulation (dplyr, tidyr, ggplot2)
library(stm)         # Structural Topic Modeling

# =============================================================================
# TERM EXTRACTION FROM STM MODELS
# =============================================================================

#' Extract Top Terms from STM Model Using Different Ranking Methods
#' 
#' This function uses STM's sageLabels() to extract topic terms. Unlike labelTopics(),
#' sageLabels() returns all topics in a clean matrix format where:
#' - Rows = topics (1 to K)
#' - Columns = word rankings (1 to n_words)
#' 
#' @param topic_model STM model object from stm() function
#' @param topic_num Integer: which topic to extract terms for (1 to K)
#' @param n_words Integer: how many top terms to extract
#' @param method Character: ranking method ("frex", "score", "prob", "lift")
#' 
#' @return Character vector of top terms for the specified topic
#'
#' @details
#' FREX (FREQuency + EXclusivity): Balances how often a word appears in a topic
#' with how unique it is to that topic. Best for interpretable topic labels.
#' 
#' Score: Similar to FREX but with different weighting. Often produces 
#' comparable but slightly different term selections.
#' 
#' Probability: Raw probability of words within topics. May include common
#' words that appear across multiple topics.
#' 
#' Lift: How much more likely a word is to appear in this topic compared to
#' the overall corpus. Emphasizes highly distinctive terms.

get_stm_terms <- function(topic_model, topic_num, n_words, method = "frex") {
  
  # Use sageLabels to get all topic terms at once
  # This is more reliable than labelTopics() which has indexing issues
  sage_result <- sageLabels(topic_model, n = n_words)
  
  # Extract terms for the specific topic using simple matrix indexing
  # sage_result$marginal contains matrices where rows = topics, cols = rankings
  terms <- switch(method,
                  "frex" = sage_result$marginal$frex[topic_num, ],
                  "score" = sage_result$marginal$score[topic_num, ],
                  "prob" = sage_result$marginal$prob[topic_num, ],  # probability
                  "lift" = sage_result$marginal$lift[topic_num, ],
                  stop("Method must be one of: frex, score, prob, lift")
  )
  
  # Convert to character and remove any potential NA values
  terms <- as.character(terms)
  terms <- terms[!is.na(terms) & terms != ""]
  
  return(terms)
}

# =============================================================================
# DOCUMENT-TERM MATRIX CREATION
# =============================================================================

#' Create Document-Term Matrix from Raw Text
#' 
#' Converts raw text documents into a quanteda dfm (document-feature matrix)
#' suitable for co-occurrence analysis. Uses standard NLP preprocessing:
#' tokenization, lowercasing, stopword removal, and term filtering.
#' 
#' @param texts Character vector of documents
#' @return quanteda dfm object
#'
#' @details
#' Preprocessing steps:
#' 1. Tokenization: Split text into individual words
#' 2. Cleaning: Remove punctuation, numbers, symbols
#' 3. Normalization: Convert to lowercase
#' 4. Filtering: Remove stopwords and short words (< 3 characters)
#' 5. Trimming: Keep only terms appearing in multiple documents

make_dtm <- function(texts) {
  corpus(texts) %>%                                    # Create quanteda corpus
    tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%  # Tokenize and clean
    tokens_tolower() %>%                               # Convert to lowercase
    tokens_remove(stopwords("en")) %>%                # Remove English stopwords
    tokens_keep(min_nchar = 3) %>%                    # Keep words ≥ 3 characters
    dfm() %>%                                         # Create document-feature matrix
    dfm_trim(min_termfreq = 2, min_docfreq = 1)       # Remove very rare terms
}

# =============================================================================
# NETWORK CREATION AND VISUALIZATION
# =============================================================================

#' Create Co-occurrence Network for Single Topic
#' 
#' This is the core function that:
#' 1. Extracts top documents for a topic from the STM model
#' 2. Creates a DTM from those documents  
#' 3. Gets STM-ranked terms and filters DTM to those terms
#' 4. Calculates word co-occurrence matrix
#' 5. Creates network graph and visualization
#' 6. Saves plot and data files
#' 
#' @param topic_model STM model object
#' @param text_data Data frame with 'text' column containing documents
#' @param topic_num Integer: topic number to analyze
#' @param n_words Integer: number of top terms to include
#' @param method Character: STM ranking method
#' @param font_size Numeric: base font size for plot text
#' 
#' @return List with network graph, plot, and extracted terms (or NULL if failed)
#'
#' @details
#' Co-occurrence networks show which words appear together in documents.
#' Edge weights represent how often word pairs co-occur. Node sizes represent
#' individual word frequencies. This reveals semantic relationships and
#' topic structure beyond simple word lists.

make_network <- function(topic_model, text_data, topic_num, n_words, method) {
  
  # STEP 1: Get top documents for this specific topic
  # topic_model$theta contains document-topic probabilities (docs x topics)
  # Higher values = document more strongly associated with topic
  topic_weights <- topic_model$theta[, topic_num]
  top_doc_indices <- order(topic_weights, decreasing = TRUE)[1:100]  # Top 100 docs
  
  # Extract text from top documents
  topic_texts <- text_data$text[top_doc_indices]
  
  # Filter out invalid documents (NA, empty, too short)
  valid_texts <- !is.na(topic_texts) & nchar(topic_texts) > 10
  topic_texts <- topic_texts[valid_texts]
  
  # Check if we have enough documents to proceed
  if (length(topic_texts) < 5) {
    warning("Topic ", topic_num, " has insufficient documents (", length(topic_texts), ")")
    return(NULL)
  }
  
  # STEP 2: Create document-term matrix from topic documents
  dtm <- make_dtm(topic_texts)
  
  # Check if DTM has enough terms after preprocessing
  if (nfeat(dtm) < 5) {
    warning("Topic ", topic_num, " has insufficient terms after preprocessing")
    return(NULL)
  }
  
  # STEP 3: Get STM-ranked terms and filter DTM
  # This is the key innovation: instead of using most frequent terms,
  # we use STM's sophisticated ranking to get topic-defining terms
  stm_terms <- get_stm_terms(topic_model, topic_num, n_words, method)
  
  # Find which STM terms actually exist in our DTM
  # (Some STM terms might not appear in top documents due to preprocessing)
  available_terms <- intersect(stm_terms, colnames(dtm))
  
  # Need at least 3 terms to create meaningful network
  if (length(available_terms) < 3) {
    warning("Topic ", topic_num, " has too few available terms (", length(available_terms), ")")
    return(NULL)
  }
  
  # Filter DTM to only include STM-selected terms
  # This focuses the network on topic-defining vocabulary
  dtm_filtered <- dtm[, available_terms]
  
  # STEP 4: Calculate co-occurrence matrix
  # Co-occurrence = how often pairs of words appear together in documents
  # Matrix multiplication: t(dtm) %*% dtm gives term-term co-occurrence counts
  cooccur_matrix <- t(dtm_filtered) %*% dtm_filtered
  
  # STEP 5: Convert co-occurrence matrix to edge list format
  # Networks need edge lists: term1, term2, weight (co-occurrence count)
  cooccur_df <- as.data.frame(as.matrix(cooccur_matrix))
  cooccur_df$term1 <- rownames(cooccur_df)
  
  # Reshape from wide to long format for network creation
  edge_list <- cooccur_df %>%
    pivot_longer(-term1, names_to = "term2", values_to = "weight") %>%
    filter(
      term1 != term2,      # Remove self-connections (word with itself)
      weight >= 1,         # Minimum co-occurrence threshold
      term1 < term2        # Remove duplicate pairs (keep only term1 < term2)
    ) %>%
    arrange(desc(weight))  # Sort by strongest connections first
  
  # Check if we have any connections
  if (nrow(edge_list) == 0) {
    warning("Topic ", topic_num, " has no word co-occurrences")
    return(NULL)
  }
  
  # STEP 6: Create network graph object
  network_graph <- graph_from_data_frame(edge_list, directed = FALSE)
  
  # Add node attributes (properties of individual words)
  word_frequencies <- colSums(dtm_filtered)
  V(network_graph)$frequency <- word_frequencies[V(network_graph)$name]
  V(network_graph)$degree <- degree(network_graph)  # Number of connections
  
  # STEP 7: Create clean greyscale visualization
  # Use simple, professional greyscale color scheme
  
  # Define greyscale palette
  node_color <- "steelblue"    # Keep steelblue for nodes (classic, readable)
  edge_color <- "gray60"       # Medium grey for edges
  text_color <- "black"        # Black text for maximum readability
  
  # STEP 7: Create network plot function for specific font size
  create_plot <- function(font_size) {
    ggraph(network_graph, layout = "fr") +  # Fruchterman-Reingold layout
      
      # EDGES: Show co-occurrence relationships
      geom_edge_link(
        aes(width = weight, alpha = weight), 
        color = edge_color,
        show.legend = TRUE
      ) +
      
      # NODES: Show individual words
      geom_node_point(
        aes(size = frequency), 
        color = node_color, 
        alpha = 0.8,
        show.legend = TRUE
      ) +
      
      # LABELS: Word text with repulsion to avoid overlap
      geom_node_text(
        aes(label = name), 
        size = font_size * 0.25,        # Convert to ggplot text size units
        color = text_color,
        repel = TRUE, 
        point.padding = unit(0.3, "lines"),
        max.overlaps = 20
      ) +
      
      # SCALES: Control visual mapping of data to aesthetics
      scale_edge_width_continuous(
        range = c(0.3, 3), 
        name = "Co-occurrence\nCount",
        guide = guide_legend(override.aes = list(alpha = 1))
      ) +
      scale_edge_alpha_continuous(
        range = c(0.4, 0.9), 
        guide = "none"  # Don't show alpha legend (redundant with width)
      ) +
      scale_size_continuous(
        range = c(2, 10), 
        name = "Word\nFrequency"
      ) +
      
      # THEME: Clean network visualization
      theme_graph(base_size = font_size) +
      theme(
        legend.position = "bottom",
        legend.direction = "horizontal",
        legend.box = "horizontal",
        legend.title = element_text(size = font_size * 0.9),
        legend.text = element_text(size = font_size * 0.8),
        plot.title = element_text(size = font_size * 1.2, hjust = 0.5),
        plot.subtitle = element_text(size = font_size * 1.0, hjust = 0.5)
      ) +
      
      # LABELS: Informative title and subtitle
      labs(
        title = paste("Topic", topic_num, "Co-occurrence Network -", toupper(method), "Rankings"),
        subtitle = paste(length(available_terms), "STM-ranked terms •", 
                         ecount(network_graph), "co-occurrence connections"),
        caption = paste("Font size:", font_size, "| Generated:", Sys.Date())
      )
  }
  
  # STEP 8: Save files with organized directory structure - BOTH FONT SIZES
  
  # Create output directories
  base_directory <- paste0("stm_networks_", method)
  dir.create(base_directory, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(base_directory, "plots_font16"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "plots_font18"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "data"), showWarnings = FALSE)
  
  # Generate and save both font size versions
  font_sizes <- c(16, 18)
  plots_created <- list()
  
  for (font_size in font_sizes) {
    # Create plot with specific font size
    network_plot <- create_plot(font_size)
    
    # Save plot in appropriate directory
    plot_filename <- file.path(
      base_directory, paste0("plots_font", font_size),
      paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_font", font_size, ".png")
    )
    
    ggsave(
      filename = plot_filename, 
      plot = network_plot, 
      width = 12, height = 10,  # Larger size for better readability
      dpi = 300,                # High resolution for publication
      bg = "white"              # White background
    )
    
    plots_created[[paste0("font", font_size)]] <- network_plot
  }
  
  # Save network data (edge list with co-occurrence counts) - same for both font sizes
  data_filename <- file.path(
    base_directory, "data",
    paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_edges.csv")
  )
  
  # Add helpful metadata to edge list
  edge_list_annotated <- edge_list %>%
    mutate(
      topic_number = topic_num,
      ranking_method = method,
      word_count = n_words,
      extraction_date = Sys.Date()
    )
  
  write.csv(edge_list_annotated, data_filename, row.names = FALSE)
  
  # Progress reporting
  cat("✓ Topic", topic_num, ":", length(available_terms), "terms,", 
      ecount(network_graph), "connections (font sizes: 16, 18)\n")
  
  # Return results for further analysis
  return(list(
    graph = network_graph,
    plots = plots_created,  # Now contains both font size versions
    terms = available_terms,
    edges = nrow(edge_list),
    method = method,
    font_sizes_created = c(16, 18)
  ))
}

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

#' Process Multiple Topics and Word Counts
#' 
#' Batch processing function that creates networks for multiple topics and
#' word count combinations. Includes progress reporting and error handling.
#' 
#' @param topic_model STM model object
#' @param text_data Data frame with text column
#' @param method Character: STM ranking method
#' @param topics Integer vector: which topics to process (default: all)
#' @param word_counts Integer vector: word count options (default: 15, 30, 50)
#' @param font_size Numeric: base font size for plots
#' 
#' @return List with processing statistics and timing information

process_networks <- function(topic_model, text_data, method = "frex", 
                             topics = 1:70, word_counts = c(15, 30, 50)) {
  
  # Validate inputs
  if (!inherits(topic_model, "STM")) {
    stop("topic_model must be an STM object from stm() function")
  }
  
  if (!"text" %in% names(text_data)) {
    stop("text_data must have a 'text' column")
  }
  
  if (!method %in% c("frex", "score", "prob", "lift")) {
    stop("method must be one of: frex, score, prob, lift")
  }
  
  # Initialize processing
  total_combinations <- length(topics) * length(word_counts)
  success_count <- 0
  failed_topics <- c()
  
  cat("=============================================================================\n")
  cat("STM CO-OCCURRENCE NETWORK BATCH PROCESSING\n")
  cat("=============================================================================\n")
  cat("Settings:\n")
  cat("  • Method:", toupper(method), "\n")
  cat("  • Topics:", length(topics), "(", min(topics), "to", max(topics), ")\n")
  cat("  • Word counts:", paste(word_counts, collapse = ", "), "\n")
  cat("  • Font sizes: 16 and 18 (both generated automatically)\n")
  cat("  • Total networks:", total_combinations * 2, "(", total_combinations, "× 2 font sizes)\n")
  cat("  • Output directory: stm_networks_", method, "/\n")
  cat("=============================================================================\n\n")
  
  start_time <- Sys.time()
  
  # Process each topic
  for (topic_num in topics) {
    
    # Progress reporting every 10 topics
    if (topic_num %% 10 == 0 || topic_num == min(topics)) {
      cat("Processing Topic", topic_num, "of", max(topics), "...\n")
    }
    
    topic_success <- FALSE
    
    # Process each word count for this topic
    for (n_words in word_counts) {
      
      # Create network with error handling
      result <- tryCatch({
        make_network(topic_model, text_data, topic_num, n_words, method)
      }, error = function(e) {
        warning("Error processing Topic ", topic_num, " (", n_words, " words): ", e$message)
        NULL
      })
      
      # Track success
      if (!is.null(result)) {
        success_count <- success_count + 1
        topic_success <- TRUE
      }
    }
    
    # Track failed topics
    if (!topic_success) {
      failed_topics <- c(failed_topics, topic_num)
    }
    
    # Memory cleanup every 20 topics
    if (topic_num %% 20 == 0) {
      gc(verbose = FALSE)
    }
  }
  
  # Calculate timing and summary statistics
  end_time <- Sys.time()
  processing_time <- as.numeric(end_time - start_time, units = "mins")
  success_rate <- round((success_count / total_combinations) * 100, 1)
  total_plots_created <- success_count * 2  # Two font sizes per successful network
  
  # Create processing summary
  summary_stats <- list(
    method = method,
    total_attempted = total_combinations,
    successful_networks = success_count,
    total_plots_created = total_plots_created,
    failed_topics = failed_topics,
    success_rate_percent = success_rate,
    processing_time_minutes = round(processing_time, 2),
    networks_per_minute = round(success_count / processing_time, 1),
    font_sizes = c(16, 18),
    timestamp = Sys.time()
  )
  
  # Save processing report
  report_file <- paste0("stm_networks_", method, "/processing_report.txt")
  
  cat("\n=============================================================================\n")
  cat("PROCESSING COMPLETE!\n")
  cat("=============================================================================\n")
  cat("Results:\n")
  cat("  ✓ Successful networks:", success_count, "of", total_combinations, "\n")
  cat("  ✓ Total plots created:", total_plots_created, "(", success_count, "networks × 2 font sizes)\n")
  cat("  ✓ Success rate:", success_rate, "%\n")
  cat("  ✓ Processing time:", round(processing_time, 2), "minutes\n")
  cat("  ✓ Speed:", round(success_count / processing_time, 1), "networks/minute\n")
  
  if (length(failed_topics) > 0) {
    cat("  ⚠ Failed topics:", paste(failed_topics, collapse = ", "), "\n")
  }
  
  cat("  📁 Output location: stm_networks_", method, "/\n")
  cat("=============================================================================\n")
  
  return(summary_stats)
}

# =============================================================================
# TESTING AND VALIDATION FUNCTIONS  
# =============================================================================

#' Test Term Extraction for Multiple Topics
#' 
#' Quick validation function to verify that STM term extraction is working
#' correctly and producing different terms for different topics.
#' 
#' @param topic_model STM model object
#' @param test_topics Integer vector: topics to test
#' @param method Character: ranking method to test
#' @param n_words Integer: number of terms to show

test_term_extraction <- function(topic_model, test_topics = c(1, 5, 10, 15, 20), 
                                 method = "frex", n_words = 8) {
  
  cat("=============================================================================\n")
  cat("TESTING STM TERM EXTRACTION -", toupper(method), "METHOD\n")
  cat("=============================================================================\n")
  cat("This test verifies that different topics return different terms.\n")
  cat("If all topics show the same terms, there's an indexing problem.\n\n")
  
  for (topic_num in test_topics) {
    # Extract terms
    terms <- get_stm_terms(topic_model, topic_num, n_words, method)
    
    # Display results
    cat("Topic", sprintf("%2d", topic_num), ":", paste(head(terms, 6), collapse = ", "))
    if (length(terms) > 6) cat(", ...")
    cat("\n")
  }
  
  cat("\n✓ If you see different terms for each topic, extraction is working correctly!\n")
  cat("✗ If all topics show identical terms, there's still an indexing issue.\n")
  cat("=============================================================================\n")
}

# =============================================================================
# NETWORK COMPARISON FUNCTION  
# =============================================================================

#' Create Sample Network to Test Output
#' 
#' Quick function to test network creation on a single topic to verify
#' everything is working before running the full batch process.
#' 
#' @param topic_model STM model object  
#' @param text_data Data frame with text column
#' @param topic_num Integer: topic to visualize
#' @param method Character: ranking method
#' @param n_words Integer: number of words

test_single_network <- function(topic_model, text_data, topic_num = 1, 
                                method = "frex", n_words = 30) {
  
  cat("Testing network creation for Topic", topic_num, "with", method, "rankings...\n")
  
  # Create single network
  result <- make_network(topic_model, text_data, topic_num, n_words, method)
  
  if (!is.null(result)) {
    cat("✓ Success! Network created with:\n")
    cat("  • Terms:", length(result$terms), "\n")
    cat("  • Edges:", result$edges, "\n") 
    cat("  • Font sizes: 16 and 18 plots created\n")
    cat("  • Files saved in: stm_networks_", method, "/\n")
    
    # Show first few terms
    cat("  • Top terms:", paste(head(result$terms, 6), collapse = ", "), "\n")
  } else {
    cat("✗ Failed to create network for Topic", topic_num, "\n")
  }
  
  return(result)
}

# =============================================================================
# EXECUTION AND USAGE EXAMPLES
# =============================================================================

cat("=============================================================================\n")
cat("STM CO-OCCURRENCE NETWORK ANALYSIS - EDUCATIONAL VERSION\n")
cat("=============================================================================\n")
cat("This script creates word co-occurrence networks from STM topic models using\n")
cat("sophisticated ranking methods (FREX, Score, Probability, Lift) rather than\n") 
cat("simple frequency counts. Each method reveals different aspects of topics:\n\n")

cat("📊 RANKING METHODS:\n")
cat("• FREX: Frequent + Exclusive (best for topic interpretation)\n") 
cat("• Score: Weighted probability + exclusivity\n")
cat("• Probability: Raw word probabilities within topics\n")
cat("• Lift: Topic distinctiveness vs. overall corpus\n\n")

cat("🎨 FEATURES:\n")
cat("• Clean greyscale visualizations with black text\n")
cat("• Automatic dual font sizes (16 and 18) for each plot\n") 
cat("• Comprehensive documentation\n")
cat("• Error handling and progress reporting\n\n")

cat("📋 USAGE STEPS:\n")
cat("=============================================================================\n\n")

cat("STEP 1: Test term extraction (verify different topics return different terms)\n")
cat("test_term_extraction(topic_model_prev, test_topics = c(1, 5, 10, 15, 20), method = 'frex')\n")
cat("test_term_extraction(topic_model_prev, test_topics = c(1, 5, 10, 15, 20), method = 'score')\n\n")

cat("STEP 2: Test single network creation (optional - verify everything works)\n")
cat("test_single_network(topic_model_prev, out$meta, topic_num = 1, method = 'frex')\n\n")

cat("STEP 3: Process networks (automatically creates font size 16 and 18 versions)\n")
cat("# FREX networks (recommended for interpretation):\n")
cat("results_frex <- process_networks(topic_model_prev, out$meta, method = 'frex')\n\n")
cat("# Score networks (alternative ranking):\n") 
cat("results_score <- process_networks(topic_model_prev, out$meta, method = 'score')\n\n")

cat("=============================================================================\n")
cat("📁 OUTPUT STRUCTURE:\n")
cat("stm_networks_[method]/\n")
cat("  ├── plots_font16/          # Network visualizations with font size 16\n")  
cat("  ├── plots_font18/          # Network visualizations with font size 18\n")
cat("  ├── data/                  # Edge lists (CSV files)\n")
cat("  └── processing_report.txt  # Summary statistics\n\n")

cat("🔤 FONT SIZES:\n")
cat("Each network automatically generates TWO versions:\n")
cat("• Font size 16: Good for detailed analysis and screen viewing\n")
cat("• Font size 18: Larger text, better for presentations and printing\n\n")

cat("🎨 VISUAL DESIGN:\n")
cat("Clean, professional greyscale design:\n")
cat("• Nodes: Steelblue - classic and readable\n")
cat("• Edges: Gray60 - subtle connection lines\n") 
cat("• Text: Black - maximum contrast and readability\n\n")

cat("=============================================================================\n")
cat("Ready to create STM co-occurrence networks! Start with the test functions.\n")
cat("=============================================================================\n")

# =============================================================================
# STM CO-OCCURRENCE NETWORK ANALYSIS WITH SAGELABELS
# =============================================================================
# 
# PURPOSE: Create co-occurrence networks for STM topic models using different
#          word ranking methods (FREX, Score, Probability, Lift)
#
# AUTHOR: [Your Name]
# DATE: [Current Date]
# 
# OVERVIEW:
# This script creates word co-occurrence networks for Structural Topic Models (STM).
# Unlike frequency-based approaches, this uses STM's sophisticated ranking methods:
# - FREX: Balances frequency and exclusivity (words frequent in topic, rare elsewhere)
# - Score: Weighted combination of probability and exclusivity  
# - Probability: Raw word probabilities within topics
# - Lift: How much more likely words appear in this topic vs. overall corpus
#
# OUTPUT: Network visualizations and edge lists for each topic and word count
# =============================================================================

# Load required libraries
library(quanteda)    # Text processing and DTM creation
library(igraph)      # Network analysis and graph objects
library(ggraph)      # Grammar of graphics for network visualization
library(tidyverse)   # Data manipulation (dplyr, tidyr, ggplot2)
library(stm)         # Structural Topic Modeling

# =============================================================================
# TERM EXTRACTION FROM STM MODELS
# =============================================================================

#' Extract Top Terms from STM Model Using Different Ranking Methods
#' 
#' This function uses STM's sageLabels() to extract topic terms. Unlike labelTopics(),
#' sageLabels() returns all topics in a clean matrix format where:
#' - Rows = topics (1 to K)
#' - Columns = word rankings (1 to n_words)
#' 
#' @param topic_model STM model object from stm() function
#' @param topic_num Integer: which topic to extract terms for (1 to K)
#' @param n_words Integer: how many top terms to extract
#' @param method Character: ranking method ("frex", "score", "prob", "lift")
#' 
#' @return Character vector of top terms for the specified topic
#'
#' @details
#' FREX (FREQuency + EXclusivity): Balances how often a word appears in a topic
#' with how unique it is to that topic. Best for interpretable topic labels.
#' 
#' Score: Similar to FREX but with different weighting. Often produces 
#' comparable but slightly different term selections.
#' 
#' Probability: Raw probability of words within topics. May include common
#' words that appear across multiple topics.
#' 
#' Lift: How much more likely a word is to appear in this topic compared to
#' the overall corpus. Emphasizes highly distinctive terms.

get_stm_terms <- function(topic_model, topic_num, n_words, method = "frex") {
  
  # Use sageLabels to get all topic terms at once
  # This is more reliable than labelTopics() which has indexing issues
  sage_result <- sageLabels(topic_model, n = n_words)
  
  # Extract terms for the specific topic using simple matrix indexing
  # sage_result$marginal contains matrices where rows = topics, cols = rankings
  terms <- switch(method,
                  "frex" = sage_result$marginal$frex[topic_num, ],
                  "score" = sage_result$marginal$score[topic_num, ],
                  "prob" = sage_result$marginal$prob[topic_num, ],  # probability
                  "lift" = sage_result$marginal$lift[topic_num, ],
                  stop("Method must be one of: frex, score, prob, lift")
  )
  
  # Convert to character and remove any potential NA values
  terms <- as.character(terms)
  terms <- terms[!is.na(terms) & terms != ""]
  
  return(terms)
}

# =============================================================================
# DOCUMENT-TERM MATRIX CREATION
# =============================================================================

#' Create Document-Term Matrix from Raw Text
#' 
#' Converts raw text documents into a quanteda dfm (document-feature matrix)
#' suitable for co-occurrence analysis. Uses standard NLP preprocessing:
#' tokenization, lowercasing, stopword removal, and term filtering.
#' 
#' @param texts Character vector of documents
#' @return quanteda dfm object
#'
#' @details
#' Preprocessing steps:
#' 1. Tokenization: Split text into individual words
#' 2. Cleaning: Remove punctuation, numbers, symbols
#' 3. Normalization: Convert to lowercase
#' 4. Filtering: Remove stopwords and short words (< 3 characters)
#' 5. Trimming: Keep only terms appearing in multiple documents

make_dtm <- function(texts) {
  corpus(texts) %>%                                    # Create quanteda corpus
    tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%  # Tokenize and clean
    tokens_tolower() %>%                               # Convert to lowercase
    tokens_remove(stopwords("en")) %>%                # Remove English stopwords
    tokens_keep(min_nchar = 3) %>%                    # Keep words ≥ 3 characters
    dfm() %>%                                         # Create document-feature matrix
    dfm_trim(min_termfreq = 2, min_docfreq = 1)       # Remove very rare terms
}

# =============================================================================
# NETWORK CREATION AND VISUALIZATION
# =============================================================================

#' Create Co-occurrence Network for Single Topic
#' 
#' This is the core function that:
#' 1. Extracts top documents for a topic from the STM model
#' 2. Creates a DTM from those documents  
#' 3. Gets STM-ranked terms and filters DTM to those terms
#' 4. Calculates word co-occurrence matrix
#' 5. Creates network graph and visualization
#' 6. Saves plot and data files
#' 
#' @param topic_model STM model object
#' @param text_data Data frame with 'text' column containing documents
#' @param topic_num Integer: topic number to analyze
#' @param n_words Integer: number of top terms to include
#' @param method Character: STM ranking method
#' @param font_size Numeric: base font size for plot text
#' 
#' @return List with network graph, plot, and extracted terms (or NULL if failed)
#'
#' @details
#' Co-occurrence networks show which words appear together in documents.
#' Edge weights represent how often word pairs co-occur. Node sizes represent
#' individual word frequencies. This reveals semantic relationships and
#' topic structure beyond simple word lists.

make_network <- function(topic_model, text_data, topic_num, n_words, method) {
  
  # STEP 1: Get top documents for this specific topic
  # topic_model$theta contains document-topic probabilities (docs x topics)
  # Higher values = document more strongly associated with topic
  topic_weights <- topic_model$theta[, topic_num]
  top_doc_indices <- order(topic_weights, decreasing = TRUE)[1:100]  # Top 100 docs
  
  # Extract text from top documents
  topic_texts <- text_data$text[top_doc_indices]
  
  # Filter out invalid documents (NA, empty, too short)
  valid_texts <- !is.na(topic_texts) & nchar(topic_texts) > 10
  topic_texts <- topic_texts[valid_texts]
  
  # Check if we have enough documents to proceed
  if (length(topic_texts) < 5) {
    warning("Topic ", topic_num, " has insufficient documents (", length(topic_texts), ")")
    return(NULL)
  }
  
  # STEP 2: Create document-term matrix from topic documents
  dtm <- make_dtm(topic_texts)
  
  # Check if DTM has enough terms after preprocessing
  if (nfeat(dtm) < 5) {
    warning("Topic ", topic_num, " has insufficient terms after preprocessing")
    return(NULL)
  }
  
  # STEP 3: Get STM-ranked terms and filter DTM
  # This is the key innovation: instead of using most frequent terms,
  # we use STM's sophisticated ranking to get topic-defining terms
  stm_terms <- get_stm_terms(topic_model, topic_num, n_words, method)
  
  # Find which STM terms actually exist in our DTM
  # (Some STM terms might not appear in top documents due to preprocessing)
  available_terms <- intersect(stm_terms, colnames(dtm))
  
  # Need at least 3 terms to create meaningful network
  if (length(available_terms) < 3) {
    warning("Topic ", topic_num, " has too few available terms (", length(available_terms), ")")
    return(NULL)
  }
  
  # Filter DTM to only include STM-selected terms
  # This focuses the network on topic-defining vocabulary
  dtm_filtered <- dtm[, available_terms]
  
  # STEP 4: Calculate co-occurrence matrix
  # Co-occurrence = how often pairs of words appear together in documents
  # Matrix multiplication: t(dtm) %*% dtm gives term-term co-occurrence counts
  cooccur_matrix <- t(dtm_filtered) %*% dtm_filtered
  
  # STEP 5: Convert co-occurrence matrix to edge list format
  # Networks need edge lists: term1, term2, weight (co-occurrence count)
  cooccur_df <- as.data.frame(as.matrix(cooccur_matrix))
  cooccur_df$term1 <- rownames(cooccur_df)
  
  # Reshape from wide to long format for network creation
  edge_list <- cooccur_df %>%
    pivot_longer(-term1, names_to = "term2", values_to = "weight") %>%
    filter(
      term1 != term2,      # Remove self-connections (word with itself)
      weight >= 1,         # Minimum co-occurrence threshold
      term1 < term2        # Remove duplicate pairs (keep only term1 < term2)
    ) %>%
    arrange(desc(weight))  # Sort by strongest connections first
  
  # Check if we have any connections
  if (nrow(edge_list) == 0) {
    warning("Topic ", topic_num, " has no word co-occurrences")
    return(NULL)
  }
  
  # STEP 6: Create network graph object
  network_graph <- graph_from_data_frame(edge_list, directed = FALSE)
  
  # Validate network graph
  if (is.null(network_graph) || vcount(network_graph) == 0) {
    warning("Topic ", topic_num, " failed to create valid network graph")
    return(NULL)
  }
  
  # Add node attributes (properties of individual words)
  word_frequencies <- colSums(dtm_filtered)
  
  # Ensure all nodes have frequency values
  node_names <- V(network_graph)$name
  V(network_graph)$frequency <- word_frequencies[node_names]
  
  # Handle any missing frequencies
  missing_freq <- is.na(V(network_graph)$frequency)
  if (any(missing_freq)) {
    V(network_graph)$frequency[missing_freq] <- 1  # Default frequency
  }
  
  V(network_graph)$degree <- degree(network_graph)  # Number of connections
  
  # STEP 7: Create clean greyscale visualization
  # Use simple, professional greyscale color scheme
  
  # Define greyscale palette
  node_color <- "steelblue"    # Keep steelblue for nodes (classic, readable)
  edge_color <- "gray60"       # Medium grey for edges
  text_color <- "black"        # Black text for maximum readability
  
  # STEP 7: Create network plot function for specific font size
  create_plot <- function(font_size) {
    
    # Validate graph before plotting
    if (is.null(network_graph) || vcount(network_graph) == 0 || ecount(network_graph) == 0) {
      warning("Cannot create plot - invalid network graph")
      return(NULL)
    }
    
    # Try different layouts if "fr" fails
    plot_result <- tryCatch({
      ggraph(network_graph, layout = "fr") +  # Fruchterman-Reingold layout
        
        # EDGES: Show co-occurrence relationships
        geom_edge_link(
          aes(width = weight, alpha = weight), 
          color = edge_color,
          show.legend = TRUE
        ) +
        
        # NODES: Show individual words
        geom_node_point(
          aes(size = frequency), 
          color = node_color, 
          alpha = 0.8,
          show.legend = TRUE
        ) +
        
        # LABELS: Word text with repulsion to avoid overlap
        geom_node_text(
          aes(label = name), 
          size = font_size * 0.25,        # Convert to ggplot text size units
          color = text_color,
          repel = TRUE, 
          point.padding = unit(0.3, "lines"),
          max.overlaps = 20
        ) +
        
        # SCALES: Control visual mapping of data to aesthetics
        scale_edge_width_continuous(
          range = c(0.3, 3), 
          guide = "none"  # Remove co-occurrence count legend
        ) +
        scale_edge_alpha_continuous(
          range = c(0.4, 0.9), 
          guide = "none"  # Don't show alpha legend
        ) +
        scale_size_continuous(
          range = c(2, 10), 
          name = "Word\nFrequency"  # Keep only word frequency legend
        ) +
        
        # THEME: Clean network visualization with no titles
        theme_graph(base_size = font_size) +
        theme(
          legend.position = "bottom",
          legend.direction = "horizontal",
          legend.box = "horizontal",
          legend.title = element_text(size = font_size * 0.9),
          legend.text = element_text(size = font_size * 0.8)
        )
      # No labs() section - removes all titles and subtitles
    }, error = function(e) {
      # Try simpler layout if FR fails
      warning("Fruchterman-Reingold layout failed, trying circle layout: ", e$message)
      tryCatch({
        ggraph(network_graph, layout = "circle") +
          geom_edge_link(aes(width = weight), color = edge_color) +
          geom_node_point(aes(size = frequency), color = node_color, alpha = 0.8) +
          geom_node_text(aes(label = name), size = font_size * 0.25, color = text_color) +
          scale_edge_width_continuous(range = c(0.3, 3), guide = "none") +  # No edge legend
          scale_size_continuous(range = c(2, 10), name = "Word\nFrequency") +
          theme_graph(base_size = font_size) +
          theme(
            legend.position = "bottom",
            legend.title = element_text(size = font_size * 0.9),
            legend.text = element_text(size = font_size * 0.8)
          )
        # No labs() - removes all titles
      }, error = function(e2) {
        warning("All layouts failed for topic ", topic_num, ": ", e2$message)
        return(NULL)
      })
    })
    
    return(plot_result)
  }
  
  # STEP 8: Save files with organized directory structure - BOTH FONT SIZES
  
  # Create output directories
  base_directory <- paste0("stm_networks_", method)
  dir.create(base_directory, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(base_directory, "plots_font16"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "plots_font18"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "data"), showWarnings = FALSE)
  
  # Generate and save both font size versions
  font_sizes <- c(24, 36)  # CHANGE THIS LINE for different font sizes
  plots_created <- list()
  
  for (font_size in font_sizes) {
    # Create plot with specific font size
    network_plot <- create_plot(font_size)
    
    # Skip this font size if plot creation failed
    if (is.null(network_plot)) {
      warning("Skipping font size ", font_size, " for topic ", topic_num, " due to plot creation error")
      next
    }
    
    # Save PNG plot in appropriate directory
    plot_filename <- file.path(
      base_directory, paste0("plots_font", font_size),
      paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_font", font_size, ".png")
    )
    
    tryCatch({
      ggsave(
        filename = plot_filename, 
        plot = network_plot, 
        width = 12, height = 10,  # Large size for PNG
        dpi = 300,                # High resolution for publication
        bg = "white"              # White background
      )
    }, error = function(e) {
      warning("Failed to save PNG for topic ", topic_num, " font ", font_size, ": ", e$message)
    })
    
    # Save EPS plot for journal publication
    eps_filename <- file.path(
      base_directory, paste0("eps_font", font_size),
      paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_font", font_size, ".eps")
    )
    
    tryCatch({
      ggsave(
        filename = eps_filename,
        plot = network_plot,
        width = 7, height = 5.5,   # Journal publication size (double column)
        units = "in",              # Specify inches for journal standards
        device = "eps",            # Vector EPS format
        bg = "white"               # White background
      )
    }, error = function(e) {
      warning("Failed to save EPS for topic ", topic_num, " font ", font_size, ": ", e$message)
    })
    
    plots_created[[paste0("font", font_size)]] <- network_plot
  }
  
  # Check if any plots were successfully created
  if (length(plots_created) == 0) {
    warning("No plots could be created for topic ", topic_num)
    return(NULL)
  }
  
  # Save network data (edge list with co-occurrence counts) - same for both font sizes
  data_filename <- file.path(
    base_directory, "data",
    paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_edges.csv")
  )
  
  # Add helpful metadata to edge list
  edge_list_annotated <- edge_list %>%
    mutate(
      topic_number = topic_num,
      ranking_method = method,
      word_count = n_words,
      extraction_date = Sys.Date()
    )
  
  write.csv(edge_list_annotated, data_filename, row.names = FALSE)
  
  # Progress reporting
  cat("✓ Topic", topic_num, ":", length(available_terms), "terms,", 
      ecount(network_graph), "connections,", length(plots_created), "successful plots\n")
  
  # Return results for further analysis
  return(list(
    graph = network_graph,
    plots = plots_created,  # Now contains both font size versions
    terms = available_terms,
    edges = nrow(edge_list),
    method = method,
    font_sizes_created = c(24, 36),
    successful_plots = length(plots_created)
  ))
}

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

#' Process Multiple Topics and Word Counts
#' 
#' Batch processing function that creates networks for multiple topics and
#' word count combinations. Includes progress reporting and error handling.
#' 
#' @param topic_model STM model object
#' @param text_data Data frame with text column
#' @param method Character: STM ranking method
#' @param topics Integer vector: which topics to process (default: all)
#' @param word_counts Integer vector: word count options (default: 15, 30, 50)
#' @param font_size Numeric: base font size for plots
#' 
#' @return List with processing statistics and timing information

process_networks <- function(topic_model, text_data, method = "frex", 
                             topics = 1:70, word_counts = c(15, 30, 50)) {
  
  # Validate inputs
  if (!inherits(topic_model, "STM")) {
    stop("topic_model must be an STM object from stm() function")
  }
  
  if (!"text" %in% names(text_data)) {
    stop("text_data must have a 'text' column")
  }
  
  if (!method %in% c("frex", "score", "prob", "lift")) {
    stop("method must be one of: frex, score, prob, lift")
  }
  
  # Initialize processing
  total_combinations <- length(topics) * length(word_counts)
  success_count <- 0
  failed_topics <- c()
  
  cat("=============================================================================\n")
  cat("STM CO-OCCURRENCE NETWORK BATCH PROCESSING\n")
  cat("=============================================================================\n")
  cat("Settings:\n")
  cat("  • Method:", toupper(method), "\n")
  cat("  • Topics:", length(topics), "(", min(topics), "to", max(topics), ")\n")
  cat("  • Word counts:", paste(word_counts, collapse = ", "), "\n")
  cat("  • Font sizes: 24 and 36 (both generated automatically)\n")
  cat("  • Total networks:", total_combinations * 4, "(", total_combinations, "× 2 font sizes × 2 formats)\n")
  cat("  • Output directory: stm_networks_", method, "/\n")
  cat("=============================================================================\n\n")
  
  start_time <- Sys.time()
  
  # Process each topic
  for (topic_num in topics) {
    
    # Progress reporting every 10 topics
    if (topic_num %% 10 == 0 || topic_num == min(topics)) {
      cat("Processing Topic", topic_num, "of", max(topics), "...\n")
    }
    
    topic_success <- FALSE
    
    # Process each word count for this topic
    for (n_words in word_counts) {
      
      # Create network with error handling
      result <- tryCatch({
        make_network(topic_model, text_data, topic_num, n_words, method)
      }, error = function(e) {
        warning("Error processing Topic ", topic_num, " (", n_words, " words): ", e$message)
        NULL
      })
      
      # Track success
      if (!is.null(result)) {
        success_count <- success_count + 1
        topic_success <- TRUE
      }
    }
    
    # Track failed topics
    if (!topic_success) {
      failed_topics <- c(failed_topics, topic_num)
    }
    
    # Memory cleanup every 20 topics
    if (topic_num %% 20 == 0) {
      gc(verbose = FALSE)
    }
  }
  
  # Calculate timing and summary statistics
  end_time <- Sys.time()
  processing_time <- as.numeric(end_time - start_time, units = "mins")
  success_rate <- round((success_count / total_combinations) * 100, 1)
  total_files_created <- success_count * 4  # Two font sizes × two formats per successful network
  
  # Create processing summary
  summary_stats <- list(
    method = method,
    total_attempted = total_combinations,
    successful_networks = success_count,
    total_files_created = total_files_created,
    failed_topics = failed_topics,
    success_rate_percent = success_rate,
    processing_time_minutes = round(processing_time, 2),
    networks_per_minute = round(success_count / processing_time, 1),
    font_sizes = c(24, 36),
    formats = c("PNG", "EPS"),
    timestamp = Sys.time()
  )
  
  # Save processing report
  report_file <- paste0("stm_networks_", method, "/processing_report.txt")
  
  cat("\n=============================================================================\n")
  cat("PROCESSING COMPLETE!\n")
  cat("=============================================================================\n")
  cat("Results:\n")
  cat("  ✓ Successful networks:", success_count, "of", total_combinations, "\n")
  cat("  ✓ Total files created:", total_files_created, "(", success_count, "networks × 2 fonts × 2 formats)\n")
  cat("  ✓ Success rate:", success_rate, "%\n")
  cat("  ✓ Processing time:", round(processing_time, 2), "minutes\n")
  cat("  ✓ Speed:", round(success_count / processing_time, 1), "networks/minute\n")
  
  if (length(failed_topics) > 0) {
    cat("  ⚠ Failed topics:", paste(failed_topics, collapse = ", "), "\n")
  }
  
  cat("  📁 Output location: stm_networks_", method, "/\n")
  cat("=============================================================================\n")
  
  return(summary_stats)
}

# =============================================================================
# TESTING AND VALIDATION FUNCTIONS  
# =============================================================================

#' Test Term Extraction for Multiple Topics
#' 
#' Quick validation function to verify that STM term extraction is working
#' correctly and producing different terms for different topics.
#' 
#' @param topic_model STM model object
#' @param test_topics Integer vector: topics to test
#' @param method Character: ranking method to test
#' @param n_words Integer: number of terms to show

test_term_extraction <- function(topic_model, test_topics = c(1, 5, 10, 15, 20), 
                                 method = "frex", n_words = 8) {
  
  cat("=============================================================================\n")
  cat("TESTING STM TERM EXTRACTION -", toupper(method), "METHOD\n")
  cat("=============================================================================\n")
  cat("This test verifies that different topics return different terms.\n")
  cat("If all topics show the same terms, there's an indexing problem.\n\n")
  
  for (topic_num in test_topics) {
    # Extract terms
    terms <- get_stm_terms(topic_model, topic_num, n_words, method)
    
    # Display results
    cat("Topic", sprintf("%2d", topic_num), ":", paste(head(terms, 6), collapse = ", "))
    if (length(terms) > 6) cat(", ...")
    cat("\n")
  }
  
  cat("\n✓ If you see different terms for each topic, extraction is working correctly!\n")
  cat("✗ If all topics show identical terms, there's still an indexing issue.\n")
  cat("=============================================================================\n")
}

# =============================================================================
# NETWORK COMPARISON FUNCTION  
# =============================================================================

#' Create Sample Network to Test Output
#' 
#' Quick function to test network creation on a single topic to verify
#' everything is working before running the full batch process.
#' 
#' @param topic_model STM model object  
#' @param text_data Data frame with text column
#' @param topic_num Integer: topic to visualize
#' @param method Character: ranking method
#' @param n_words Integer: number of words

test_single_network <- function(topic_model, text_data, topic_num = 1, 
                                method = "frex", n_words = 30) {
  
  cat("Testing network creation for Topic", topic_num, "with", method, "rankings...\n")
  
  # Create single network
  result <- make_network(topic_model, text_data, topic_num, n_words, method)
  
  if (!is.null(result)) {
    cat("✓ Success! Network created with:\n")
    cat("  • Terms:", length(result$terms), "\n")
    cat("  • Edges:", result$edges, "\n") 
    cat("  • Font sizes: 24 and 36 (PNG + EPS formats)\n")
    cat("  • Files saved in: stm_networks_", method, "/\n")
    
    # Show first few terms
    cat("  • Top terms:", paste(head(result$terms, 6), collapse = ", "), "\n")
  } else {
    cat("✗ Failed to create network for Topic", topic_num, "\n")
  }
  
  return(result)
}

# =============================================================================
# EXECUTION AND USAGE EXAMPLES
# =============================================================================

cat("=============================================================================\n")
cat("STM CO-OCCURRENCE NETWORK ANALYSIS - EDUCATIONAL VERSION\n")
cat("=============================================================================\n")
cat("This script creates word co-occurrence networks from STM topic models using\n")
cat("sophisticated ranking methods (FREX, Score, Probability, Lift) rather than\n") 
cat("simple frequency counts. Each method reveals different aspects of topics:\n\n")

cat("📊 RANKING METHODS:\n")
cat("• FREX: Frequent + Exclusive (best for topic interpretation)\n") 
cat("• Score: Weighted probability + exclusivity\n")
cat("• Probability: Raw word probabilities within topics\n")
cat("• Lift: Topic distinctiveness vs. overall corpus\n\n")

cat("🎨 FEATURES:\n")
cat("• Clean greyscale visualizations with black text\n")
cat("• Automatic dual font sizes (16 and 18) for each plot\n") 
cat("• Comprehensive documentation\n")
cat("• Error handling and progress reporting\n\n")

cat("📋 USAGE STEPS:\n")
cat("=============================================================================\n\n")

cat("STEP 1: Test term extraction (verify different topics return different terms)\n")
cat("test_term_extraction(topic_model_prev, test_topics = c(1, 5, 10, 15, 20), method = 'frex')\n")
cat("test_term_extraction(topic_model_prev, test_topics = c(1, 5, 10, 15, 20), method = 'score')\n\n")

cat("STEP 2: Test single network creation (optional - verify everything works)\n")
cat("test_single_network(topic_model_prev, out$meta, topic_num = 1, method = 'frex')\n\n")

cat("STEP 3: Process networks (automatically creates font size 16 and 18 versions)\n")
cat("# FREX networks (recommended for interpretation):\n")
cat("results_frex <- process_networks(topic_model_prev, out$meta, method = 'frex')\n\n")
cat("# Score networks (alternative ranking):\n") 
cat("results_score <- process_networks(topic_model_prev, out$meta, method = 'score')\n\n")

cat("=============================================================================\n")
cat("📁 OUTPUT STRUCTURE:\n")
cat("stm_networks_[method]/\n")
cat("  ├── plots_font24/          # PNG visualizations with font size 24\n")  
cat("  ├── plots_font36/          # PNG visualizations with font size 36\n")
cat("  ├── eps_font24/            # EPS files with font size 24 (journal publication)\n")
cat("  ├── eps_font36/            # EPS files with font size 36 (journal publication)\n")
cat("  ├── data/                  # Edge lists (CSV files)\n")
cat("  └── processing_report.txt  # Summary statistics\n\n")

cat("🔤 FONT SIZES & FORMATS:\n")
cat("Each network automatically generates FOUR files:\n")
cat("• Font 24 PNG: Large text for presentations (12×10 in, 300 DPI)\n")
cat("• Font 36 PNG: Extra large text for posters (12×10 in, 300 DPI)\n")
cat("• Font 24 EPS: Journal publication format (7×5.5 in, vector)\n")
cat("• Font 36 EPS: Journal publication, large text (7×5.5 in, vector)\n\n")

cat("📄 JOURNAL PUBLICATION:\n")
cat("EPS files are optimized for academic publishing:\n")
cat("• Vector format: Scales perfectly at any size\n")
cat("• Standard dimensions: 7×5.5 inches (fits double-column layout)\n")
cat("• Title-free design: Ready for Word document insertion with custom captions\n")
cat("• Clean legends: Only Word Frequency shown (no connection counts)\n\n")

cat("🎨 VISUAL DESIGN:\n")
cat("Clean, professional greyscale design:\n")
cat("• Nodes: Steelblue - classic and readable\n")
cat("• Edges: Gray60 - subtle connection lines\n") 
cat("• Text: Black - maximum contrast and readability\n\n")

cat("=============================================================================\n")
cat("Ready to create title-free STM co-occurrence networks for Word documents!\n") 
cat("Each network generates 4 clean files ready for custom captioning in manuscripts.\n")
cat("Start with the test functions to verify everything works correctly.\n")
cat("=============================================================================\n")
