## Joon Chung
## jxc3388@miami.edu
## The University of Miami, Miller School of Medicine
##  The Department of Informatics and Health Data Science
##
## Structural Topic Models by year
##
## Penultimate update: 7/28/2018
## Last update:        6/13/2025

## This script runs STM, with each year as a PREVALENCE covariate

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
# # Processed
# processed <- textProcessor(full_text_subset$text, striphtml = TRUE, metadata = as.data.frame(full_text_subset))
# 
# plotRemoved(processed$documents, lower.thresh = seq(from = 10, to = 2000, by = 2))
# 
# out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 500) 
# 
# ## Run STM
# set.seed(8675309)
# 
# topic_model_prev <- stm(out$documents,
#                         out$vocab,
#                         data = out$meta,
#                         prevalence = ~year,
#                         K = 70, 
#                         verbose = TRUE, 
#                         init.type = "Spectral")
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
