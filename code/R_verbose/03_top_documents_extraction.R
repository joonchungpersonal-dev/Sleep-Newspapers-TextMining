# =============================================================================
# SCRIPT:   03_top_documents_extraction.R
# PURPOSE:  Extract and save the top 25 most representative documents for each
#           of the 70 topics in the Structural Topic Model. This enables
#           qualitative validation of topic content and facilitates manual
#           coding or close reading of representative newspaper articles.
#
# AUTHOR:   Joon Chung
# EMAIL:    jxc3388@miami.edu
# AFFIL:    The University of Miami, Miller School of Medicine
#           The Department of Informatics and Health Data Science
#
# DATE:     6/13/2025
#
# INPUTS:
#   - topic_model_prev: A fitted STM model object (must exist in environment)
#   - full_text_subset: A data frame with article text and metadata (must exist
#     in environment). Expected columns may include: text, datetime, year/Year,
#     source/Source/newspaper, title/Title/headline, id/ID/doc_id.
#
# OUTPUTS:
#   - all_topics_documents/csv_files/: CSV files with top 25 docs per topic
#   - all_topics_documents/txt_files/: Formatted text files per topic
#   - all_topics_documents/master_summary_all_70_topics.txt: Quick-ref summary
#   - all_topics_documents/topic_index.txt: File structure documentation
#
# REQUIRED PACKAGES:
#   - stm   : Structural Topic Modeling (for accessing model$theta)
#   - dplyr : Data manipulation (part of tidyverse)
#
# NOTES FOR PYTHON USERS:
#   - In Python, the document-topic matrix (theta) would be accessed as a
#     NumPy array. Sorting to find top documents would use np.argsort().
#   - R's sink() function is similar to Python's contextlib.redirect_stdout(),
#     redirecting all console output to a file.
#   - The %>% pipe operator is equivalent to method chaining in pandas.
#   - sprintf() for string formatting is similar to Python's f-strings or
#     the % operator: sprintf("%02d", 5) -> "05" (Python: f"{5:02d}").
#   - exists() checks if a variable is defined in the current R environment,
#     analogous to checking 'variable_name' in dir() in Python.
# =============================================================================


# =============================================================================
#### SECTION: Load Required Libraries ####
# =============================================================================

## Joon Chung
## jxc3388@miami.edu

## 6/13/2025

## Purpose: Grab top 25 documents per topic (70 topics)

# stm: Structural Topic Model package. We need it here to access the fitted
# model's theta matrix (document-topic proportions).
library(stm)
# dplyr: Data manipulation grammar (filter, select, mutate, etc.)
# Part of the tidyverse, but loaded standalone here for minimal dependencies.
# In Python, the equivalent is pandas.
library(dplyr)


# =============================================================================
#### SECTION: Define Topics to Process ####
# =============================================================================

# Define all 70 topics (1 through 70)
# 1:70 creates an integer sequence from 1 to 70 inclusive.
# In Python: list(range(1, 71)) or np.arange(1, 71)
all_topics <- 1:70


# =============================================================================
#### SECTION: Document Extraction Function ####
# =============================================================================

# This function extracts the top N documents for a given topic based on
# their document-topic proportion (theta value). Documents with the highest
# theta for a topic are the most "representative" of that topic's content.
#
# WHAT IS THETA?
# In topic modeling, theta (also called gamma in some implementations) is a
# matrix of size (N_documents x K_topics). Each cell theta[i, j] represents
# the proportion of document i that is "about" topic j. Values range from 0
# to 1, and each row sums to 1. A document with theta[i, 3] = 0.85 is
# strongly about topic 3.
#
# In Python with gensim, you would access this via:
#   doc_topics = lda_model.get_document_topics(bow, minimum_probability=0)

# Function to extract top documents for a topic
extract_top_documents <- function(model, topic_num, n_docs = 25) {

  # model$theta is the document-topic proportion matrix (N x K).
  # [, topic_num] extracts the column for the specified topic.
  # In Python: theta[:, topic_num]
  # Get document-topic proportions for this topic
  topic_proportions <- model$theta[, topic_num]

  # order() returns the indices that would sort the vector.
  # decreasing = TRUE means highest values first.
  # [1:n_docs] takes the first n_docs indices (the top documents).
  # In Python: np.argsort(topic_proportions)[::-1][:n_docs]
  # Get top document indices
  top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
  top_proportions <- topic_proportions[top_indices]

  # Create the output data frame with core ranking information.
  # stringsAsFactors = FALSE prevents R from converting strings to factors
  # (a legacy R behavior; not relevant in Python).
  # Extract document information
  top_docs_data <- data.frame(
    Rank = 1:n_docs,
    Document_Index = top_indices,
    Topic_Proportion = round(top_proportions, 4),
    stringsAsFactors = FALSE
  )

  # --- Add metadata columns from full_text_subset ---
  # exists() checks whether the object full_text_subset is defined in the
  # current R environment. This defensive check prevents errors if the metadata
  # object is named differently.
  # Add metadata from full_text_subset
  if (exists("full_text_subset")) {
    # Text content - check multiple possible column names
    # names() returns column names of a data frame (like df.columns in pandas).
    # %in% tests set membership (like Python's 'in' operator).
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


# =============================================================================
#### SECTION: Create Output Directory Structure ####
# =============================================================================

# Create main directory for all topic outputs.
# dir.exists() returns TRUE/FALSE. dir.create() makes the directory.
# In Python: os.makedirs("all_topics_documents/csv_files", exist_ok=True)
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


# =============================================================================
#### SECTION: Batch Process All 70 Topics ####
# =============================================================================

# This loop iterates through all 70 topics, extracts the top 25 documents
# for each, and saves the results in both CSV and formatted TXT formats.
# cat() prints progress messages to the console (like Python's print()).

# Extract and save top documents for all 70 topics
cat("Processing all 70 topics...\n")
cat("This may take a few minutes.\n\n")

for (topic_num in all_topics) {
  cat("Processing topic", topic_num, "of 70\n")

  # Extract top documents for this topic
  top_docs <- extract_top_documents(topic_model_prev, topic_num)

  # --- Save as CSV ---
  # sprintf("%02d", topic_num) zero-pads the topic number to 2 digits
  # (01, 02, ..., 70) for consistent file sorting.
  # paste0() concatenates strings without separators.
  # write.csv() writes a data frame to CSV format.
  # In Python: df.to_csv(filename, index=False)
  csv_filename <- paste0("all_topics_documents/csv_files/topic_",
                         sprintf("%02d", topic_num), "_top25.csv")
  write.csv(top_docs, csv_filename, row.names = FALSE)

  # --- Save as formatted text file ---
  # sink() redirects all subsequent cat() output to the specified file.
  # The paired sink() at the end of the block restores output to the console.
  # In Python, you would use: with open(filename, 'w') as f: f.write(...)
  txt_filename <- paste0("all_topics_documents/txt_files/topic_",
                         sprintf("%02d", topic_num), "_top25.txt")

  # Create formatted text output
  sink(txt_filename)
  cat("TOP 25 DOCUMENTS FOR TOPIC", topic_num, "\n")
  # "=" %>% rep(50) creates 50 copies of "=" and paste(collapse="") joins them
  # into a single string. This creates a visual divider line.
  # In Python: "=" * 50
  cat("=" %>% rep(50) %>% paste(collapse = ""), "\n\n")

  for (j in 1:nrow(top_docs)) {
    cat("RANK:", j, "| PROPORTION:", top_docs$Topic_Proportion[j],
        "| DOC INDEX:", top_docs$Document_Index[j], "\n")

    # Conditionally print metadata fields if they exist in the data frame.
    # This defensive approach handles varying metadata availability.
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
      # strwrap() wraps long text strings to a specified character width.
      # This improves readability in the output text files.
      # In Python: textwrap.fill(text, width=80)
      text <- top_docs$Text[j]
      wrapped_text <- strwrap(text, width = 80)
      cat(paste(wrapped_text, collapse = "\n"), "\n")
    }

    cat("\n", "-" %>% rep(80) %>% paste(collapse = ""), "\n\n")
  }
  sink()
}


# =============================================================================
#### SECTION: Create Master Summary File ####
# =============================================================================

# This creates a single overview file showing only the top 5 documents for each
# of the 70 topics. This is useful as a quick reference to see the most
# representative documents across all topics at a glance.

# Create master summary file
summary_filename <- "all_topics_documents/master_summary_all_70_topics.txt"
sink(summary_filename)
cat("MASTER SUMMARY: TOP 5 DOCUMENTS FOR ALL 70 TOPICS\n")
cat("=" %>% rep(70) %>% paste(collapse = ""), "\n\n")

for (topic_num in all_topics) {
  # Re-extract with n_docs=5 for the summary (fewer documents per topic).
  top_docs <- extract_top_documents(topic_model_prev, topic_num, n_docs = 5)

  cat("TOPIC", sprintf("%02d", topic_num), "\n")
  cat("Top 5 documents (proportion):\n")

  # min(5, nrow(top_docs)) is a safety check in case fewer than 5 docs exist.
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


# =============================================================================
#### SECTION: Create Topic Index File ####
# =============================================================================

# This generates a reference file documenting the output directory structure
# and listing all processed topics. Useful for collaborators navigating the
# output files.

# Create topic index file for easy reference
index_filename <- "all_topics_documents/topic_index.txt"
sink(index_filename)
cat("TOPIC INDEX - ALL 70 TOPICS\n")
# Sys.time() returns the current date-time (like Python's datetime.now()).
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


# =============================================================================
#### SECTION: Processing Complete Summary ####
# =============================================================================

# Print final summary to console confirming successful processing.
cat("\n=== PROCESSING COMPLETE ===\n")
cat("All 70 topics processed successfully!\n")
cat("Files saved in 'all_topics_documents' directory:\n")
cat("- CSV files in csv_files/ subdirectory\n")
cat("- TXT files in txt_files/ subdirectory\n")
cat("- Master summary file: master_summary_all_70_topics.txt\n")
cat("- Topic index file: topic_index.txt\n")
# Total files = 70 CSV + 70 TXT + 1 summary + 1 index = 142
cat("\nTotal files created:", (length(all_topics) * 2) + 2, "\n")
