## Joon Chung
## Contact: see README

## 6/13/2025

## Purpose: Grab top 25 documents per topic (70 topics)

library(stm)
library(dplyr)

# Define all 70 topics (1 through 70)
all_topics <- 1:70

# Function to extract top documents for a topic
extract_top_documents <- function(model, topic_num, n_docs = 25) {
  
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
  
  # Add metadata from full_text_subset
  if (exists("full_text_subset")) {
    # Text content
    if ("text" %in% names(full_text_subset)) {
      top_docs_data$Text <- full_text_subset$text[top_indices]
    } else if ("Text" %in% names(full_text_subset)) {
      top_docs_data$Text <- full_text_subset$Text[top_indices]
    } else if ("content" %in% names(full_text_subset)) {
      top_docs_data$Text <- full_text_subset$content[top_indices]
    }
    
    # Publication date - check for datetime and year columns specifically
    if ("datetime" %in% names(full_text_subset)) {
      top_docs_data$Date <- full_text_subset$datetime[top_indices]
    } else if ("date" %in% names(full_text_subset)) {
      top_docs_data$Date <- full_text_subset$date[top_indices]
    } else if ("Date" %in% names(full_text_subset)) {
      top_docs_data$Date <- full_text_subset$Date[top_indices]
    }
    
    # Year column (separate from datetime)
    if ("year" %in% names(full_text_subset)) {
      top_docs_data$Year <- full_text_subset$year[top_indices]
    } else if ("Year" %in% names(full_text_subset)) {
      top_docs_data$Year <- full_text_subset$Year[top_indices]
    }
    
    # Newspaper/Source
    if ("source" %in% names(full_text_subset)) {
      top_docs_data$Source <- full_text_subset$source[top_indices]
    } else if ("Source" %in% names(full_text_subset)) {
      top_docs_data$Source <- full_text_subset$Source[top_indices]
    } else if ("newspaper" %in% names(full_text_subset)) {
      top_docs_data$Source <- full_text_subset$newspaper[top_indices]
    } else if ("Newspaper" %in% names(full_text_subset)) {
      top_docs_data$Source <- full_text_subset$Newspaper[top_indices]
    } else if ("publication" %in% names(full_text_subset)) {
      top_docs_data$Source <- full_text_subset$publication[top_indices]
    }
    
    # Title (if available)
    if ("title" %in% names(full_text_subset)) {
      top_docs_data$Title <- full_text_subset$title[top_indices]
    } else if ("Title" %in% names(full_text_subset)) {
      top_docs_data$Title <- full_text_subset$Title[top_indices]
    } else if ("headline" %in% names(full_text_subset)) {
      top_docs_data$Title <- full_text_subset$headline[top_indices]
    }
    
    # Document ID (if available)
    if ("id" %in% names(full_text_subset)) {
      top_docs_data$Document_ID <- full_text_subset$id[top_indices]
    } else if ("ID" %in% names(full_text_subset)) {
      top_docs_data$Document_ID <- full_text_subset$ID[top_indices]
    } else if ("doc_id" %in% names(full_text_subset)) {
      top_docs_data$Document_ID <- full_text_subset$doc_id[top_indices]
    }
  }
  
  return(top_docs_data)
}

# Create main directory for all topic outputs
if (!dir.exists("all_topics_documents")) {
  dir.create("all_topics_documents")
}

# Create subdirectories for organization
if (!dir.exists("all_topics_documents/csv_files")) {
  dir.create("all_topics_documents/csv_files")
}
if (!dir.exists("all_topics_documents/txt_files")) {
  dir.create("all_topics_documents/txt_files")
}

# Extract and save top documents for all 70 topics
cat("Processing all 70 topics...\n")
cat("This may take a few minutes.\n\n")

for (topic_num in all_topics) {
  cat("Processing topic", topic_num, "of 70\n")
  
  # Extract top documents
  top_docs <- extract_top_documents(topic_model_prev, topic_num)
  
  # Save as CSV
  csv_filename <- paste0("all_topics_documents/csv_files/topic_", 
                         sprintf("%02d", topic_num), "_top25.csv")
  write.csv(top_docs, csv_filename, row.names = FALSE)
  
  # Save as formatted text file
  txt_filename <- paste0("all_topics_documents/txt_files/topic_", 
                         sprintf("%02d", topic_num), "_top25.txt")
  
  # Create formatted text output
  sink(txt_filename)
  cat("TOP 25 DOCUMENTS FOR TOPIC", topic_num, "\n")
  cat("=" %>% rep(50) %>% paste(collapse = ""), "\n\n")
  
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
}

# Create master summary file
summary_filename <- "all_topics_documents/master_summary_all_70_topics.txt"
sink(summary_filename)
cat("MASTER SUMMARY: TOP 5 DOCUMENTS FOR ALL 70 TOPICS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

for (topic_num in all_topics) {
  top_docs <- extract_top_documents(topic_model_prev, topic_num, n_docs = 5)
  
  cat("TOPIC", sprintf("%02d", topic_num), "\n")
  cat("Top 5 documents (proportion):\n")
  
  for (j in 1:min(5, nrow(top_docs))) {
    cat("  ", j, ". Doc", top_docs$Document_Index[j], 
        "(", top_docs$Topic_Proportion[j], ")")
    
    if ("Year" %in% names(top_docs)) {
      cat(" [", top_docs$Year[j], "]")
    }
    if ("Source" %in% names(top_docs)) {
      cat(" - ", top_docs$Source[j])
    }
    cat("\n")
  }
  cat("\n")
}
sink()

# Create topic index file for easy reference
index_filename <- "all_topics_documents/topic_index.txt"
sink(index_filename)
cat("TOPIC INDEX - ALL 70 TOPICS\n")
cat("Generated:", Sys.time(), "\n")
cat("=" %>% rep(50) %>% paste(collapse = ""), "\n\n")

cat("File Structure:\n")
cat("- csv_files/: Contains CSV data for each topic\n")
cat("- txt_files/: Contains formatted text files for each topic\n")
cat("- master_summary_all_70_topics.txt: Overview of top 5 docs per topic\n\n")

cat("Topics Processed:\n")
for (topic_num in all_topics) {
  cat("Topic", sprintf("%02d", topic_num), 
      "- Files: topic_", sprintf("%02d", topic_num), "_top25.csv/.txt\n")
}

cat("\nTotal Topics:", length(all_topics), "\n")
cat("Documents per Topic: 25\n")
cat("Total Document Extractions:", length(all_topics) * 25, "\n")
sink()

cat("\n=== PROCESSING COMPLETE ===\n")
cat("All 70 topics processed successfully!\n")
cat("Files saved in 'all_topics_documents' directory:\n")
cat("- CSV files in csv_files/ subdirectory\n")
cat("- TXT files in txt_files/ subdirectory\n")
cat("- Master summary file: master_summary_all_70_topics.txt\n")
cat("- Topic index file: topic_index.txt\n")
cat("\nTotal files created:", (length(all_topics) * 2) + 2, "\n")

