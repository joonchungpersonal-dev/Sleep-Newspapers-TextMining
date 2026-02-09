# =============================================================================
# SCRIPT:   01_stm_model_estimation.R
# PURPOSE:  Estimate Structural Topic Models (STM) with year as a prevalence
#           covariate, visualize topic proportions over time, generate word
#           clouds, and extract top documents for each topic of interest.
#
# AUTHOR:   Joon Chung
# EMAIL:    jxc3388@miami.edu
# AFFIL:    The University of Miami, Miller School of Medicine
#           The Department of Informatics and Health Data Science
#
# CREATED:  7/28/2018 (penultimate update)
# UPDATED:  6/13/2025 (last update)
#
# INPUTS:
#   - full_text_clean: A pre-loaded data frame (from an .Rda file) containing
#     newspaper articles with columns: text, text_id, datetime, Year.
#     The .Rda load is commented out; the object must exist in the environment.
#
# OUTPUTS:
#   - topic_model_prev: A fitted STM object with 70 topics and year prevalence.
#   - prep: An estimateEffect object for plotting temporal trends (b-spline).
#   - PNG plot files for topic proportion time series and word clouds.
#   - CSV and TXT files of top-100 documents per topic in "top_documents/" dir.
#   - A summary file across all topics of interest.
#
# REQUIRED PACKAGES:
#   - stm         : Structural Topic Modeling (estimation, labeling, plotting)
#   - tidyverse    : Data wrangling and ggplot2 visualization (dplyr, tidyr,
#                   ggplot2, stringr, etc.)
#   - gridExtra    : Arranging multiple ggplot objects in a single figure
#   - tidytext     : Tidy text mining; provides tidy() for STM matrices
#   - ggwordcloud  : Word clouds rendered via ggplot2
#
# NOTES FOR PYTHON USERS:
#   - Structural Topic Models (STM) are an extension of LDA (Latent Dirichlet
#     Allocation) that allow document-level covariates to influence topic
#     prevalence and/or word usage. In Python, the closest equivalents are
#     gensim's LdaModel (no covariates) or the guidedlda / corex_topic
#     packages. There is no direct Python port of the full STM framework.
#   - The pipe operator %>% is equivalent to method chaining in pandas:
#       R:      df %>% filter(x > 1) %>% select(y)
#       Python: df.query('x > 1')[['y']]
#   - R uses <- for assignment (equivalent to = in Python).
#   - Formula notation (e.g., prevalence = ~year) specifies model structure,
#     similar to patsy/statsmodels formulas in Python: 'y ~ x'.
# =============================================================================


# =============================================================================
#### SECTION: Load Required Libraries ####
# =============================================================================

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

# library() loads an R package into the current session. If not installed,
# run install.packages("stm") first. In Python, the equivalent would be
# "import stm" -- but stm is R-only.
library(stm)
# tidyverse is a meta-package that loads dplyr (data wrangling), ggplot2
# (plotting), tidyr (reshaping), stringr (strings), readr (CSV I/O), purrr
# (functional programming), tibble, and forcats all at once. In Python, these
# functionalities are split across pandas, matplotlib, seaborn, re, etc.
library(tidyverse)
# gridExtra provides grid.arrange() to place multiple ggplot objects side by
# side in a single figure. In Python, matplotlib's subplot() or fig.add_subplot()
# serve a similar purpose.
library(gridExtra)
# tidytext provides tidy() methods that convert STM model matrices (beta, theta)
# into long-format data frames suitable for ggplot2. In Python, you would
# manually extract these matrices from a gensim model.
library(tidytext)
# ggwordcloud renders word clouds using the ggplot2 grammar of graphics.
# In Python, the wordcloud library (from wordcloud import WordCloud) is the
# standard equivalent.
library(ggwordcloud)


# =============================================================================
#### SECTION: Data Loading and Preprocessing (Commented Out) ####
# =============================================================================

# The original data loading code is commented out below. It loaded a previously
# saved R workspace (.Rda file) containing the cleaned newspaper corpus.
# To reproduce, you would need the original .Rda file. The full_text_clean
# object must already exist in your R environment before running this script.

#
# # Load full_text
# # load("C:/users/jchun/desktop/to do tomorrow/text/clean_full_text_20k.Rda")
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
# # save(topic_model_prev, prep, file = "C:/users/jchun/desktop/to do tomorrow/Text mining sleep/Text mining sleep (7-25-18)/topic_model_prev.Rda")


# =============================================================================
#### SECTION: Topic Labeling and Effect Estimation ####
# =============================================================================

# NOTE: This section assumes that topic_model_prev and out already exist in the
# R environment (loaded from a saved .Rda workspace, or estimated above).

###################################################################################

# sageLabels() prints the top words for each topic using two ranking methods:
#   - Marginal Highest Probability: words with highest P(word | topic)
#   - Marginal Score: a metric that balances frequency and exclusivity
# This is a quick way to see what each of the 70 topics "looks like."
# In Python/gensim, you would call lda_model.print_topics().
sageLabels(topic_model_prev)

# estimateEffect() fits a regression model to examine how topic prevalence
# (the proportion of each document devoted to a topic) varies with covariates.
#
# The formula 1:70 ~ s(year) means:
#   - 1:70: Estimate the effect for all 70 topics simultaneously
#   - s(year): Use a B-spline smoother on year (captures non-linear trends)
#
# A B-spline is a flexible, smooth curve fit; it allows topic prevalence to
# increase, decrease, or follow non-monotonic patterns over time without
# assuming a linear relationship.
#
# In Python/statsmodels, you would use something like:
#   from patsy import dmatrix
#   X = dmatrix("bs(year, df=5)", data=meta)
# and then run a regression for each topic's theta column.
## Run the b-spline model
prep <- estimateEffect(1:70 ~ s(year), topic_model_prev, meta = out$meta)


# =============================================================================
#### SECTION: Custom Plotting Function for STM Topics ####
# =============================================================================

## Function to code topic proportions and terms
stm_plot <- function(topic_select, title){
  ## This function takes a structural topic model and:
  ##  1) extracts meaningful data from a plot.estimateEffect object
  ##  2) plots expected topic proportions by year in ggplot2
  ##  3) tidys the beta matrix of the topic model
  ##  4) plots the top 10 terms in a given topic
  ##  5) returns both plots in grid.arrange

  # plot() on an estimateEffect object returns a list with:
  #   - means: expected topic proportion at each year
  #   - ci: confidence intervals (lower, upper interleaved)
  #   - x: the covariate values (years)
  # method = "continuous" treats year as a continuous variable for smooth curves.
  # In Python, you would manually extract regression predictions and CIs.
  data <- plot(prep, covariate = "year", method = "continuous", topics = topic_select)

  # Extract the expected topic proportions (means) from the plot data.
  # with() evaluates the expression in the context of the 'data' list, so
  # with(data, means) is equivalent to data$means.
  # %>% is the pipe operator: it passes the left-hand result as input to the
  # next function. In Python: result = pd.DataFrame(np.array(data['means']))
  # Expected topic proportion
  means <- with(data, means) %>% unlist() %>% as.matrix() %>% as.data.frame()
  means <- with(means, V1)

  # Extract the 95% confidence intervals. The CI vector interleaves lower and
  # upper bounds, so we use seq() to separate them:
  #   lower: rows 1, 3, 5, ... (every odd index)
  #   upper: rows 2, 4, 6, ... (every even index)
  # 95% Confidence interval
  ci <- with(data, ci) %>% unlist() %>% as.data.frame()

  lower <- ci[seq(1, nrow(ci), 2),]
  upper <- ci[seq(2, nrow(ci), 2),]

  # Extract the year values corresponding to the predictions
  # years
  years <- with(data, x) %>% as.numeric()
  plot_df <- data.frame(years = years, means = means, lower = lower, upper = upper)
  # Filter to keep only years up to 2017 (the study period).
  # The ######## !!!!!!!! comment is the author's emphasis that this filter is
  # important and intentional.
  plot_df <- plot_df %>% filter(years <= 2017) ######## !!!!!!!!!!

  # Create the topic proportion time series plot using ggplot2.
  # - geom_line(): draws the trend line of expected topic proportion over years
  # - geom_ribbon(): adds a shaded band for the 95% confidence interval
  # - scale_x_continuous(): sets x-axis breaks every 3 years from 1983 to 2016
  # - theme_bw(): uses a clean white-background theme
  # In Python, this would be done with matplotlib or seaborn:
  #   plt.plot(years, means)
  #   plt.fill_between(years, lower, upper, alpha=0.2)
  # Plot
  topic_prop <- ggplot(plot_df, aes(x = years, y = means)) + geom_line() +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
    scale_x_continuous(breaks = seq(1983,2016, 3), limits = c(1983, 2017)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle =45)) +
    xlab("Year") +
    ylab("Expected topic proportion")

  # tidy() from the tidytext package converts the STM model's beta matrix
  # (word-topic probabilities) into a long-format data frame with columns:
  #   topic, term, beta
  # beta = P(word | topic), the probability of a word given a topic.
  # In Python/gensim, you would call lda_model.get_topic_terms(topic_id).
  stm_beta <- tidy(topic_model_prev, matrix = "beta")
  # Filter to the selected topic, sort by descending beta, take top 10 terms,
  # and plot as a horizontal bar chart.
  # dplyr::filter() is used with the namespace prefix to avoid conflicts with
  # the base R filter() function. In Python: df[df['topic'] == topic_select].
  # reorder(term, beta) sorts the bars by beta value for readability.
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

  # grid.arrange() from gridExtra places multiple plots side by side.
  # nrow = 1 means both plots appear in a single row (left: terms, right: trend).
  # top = title adds a main title above both panels.
  # In Python, you would use fig, (ax1, ax2) = plt.subplots(1, 2).
  return(grid.arrange(topic_terms, topic_prop, nrow = 1,top = title))
}


# =============================================================================
#### SECTION: Generate Topic Proportion Plots for Topics of Interest ####
# =============================================================================

# topics_of_interest is assumed to be defined earlier or loaded from workspace.
# It is a numeric vector identifying which of the 70 topics are relevant to
# the sleep research study (e.g., sleep disorders, medications, science).
topics_of_interest
# 3  7 15 32 47 51

# Create a two-panel plot for each topic of interest:
#   Left panel:  horizontal bar chart of top 10 words in the topic
#   Right panel: time series of expected topic proportion with 95% CI
sleep_plot <- stm_plot(topic = 3, "Work and sleep")
drug_plot <- stm_plot(topic = 7, "Sleep medicine / drugs")
science_plot <- stm_plot(topic = 15, "Circadian science")
apnea_plot <- stm_plot(topic = 32, "Sleep apnea, hospitals")
health_plot <- stm_plot(topic = 47, "Health research")
academic_research_plot <- stm_plot(topic = 51, "Sleep research")
school_plot <- stm_plot(topic = 59, "School start times")
work_plot <- stm_plot(topic = 22, "Work")
disaster_plot <- stm_plot(topic = 37, "Disaster")


# Validation topics: these topics serve as face-validity checks for the model.
# "Dear Ann" (an advice column) and "Iraq wars" are not sleep-related, so they
# confirm that the model is capturing meaningful thematic distinctions.
dearann_1_plot <- stm_plot(topic = 14, "Dear Ann (validation)")
iraq_2_plot <- stm_plot(topic = 23, "Iraq wars")


# =============================================================================
#### SECTION: Save Topic Proportion Plots to PNG Files ####
# =============================================================================

# ggsave() saves a ggplot/grob object to a file. Width and height are in inches
# by default. In Python, you would use plt.savefig("filename.png", dpi=300).
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


# =============================================================================
#### SECTION: Word Clouds Using Base STM cloud() Function ####
# =============================================================================

# The expanded set of topics of interest (includes validation topics)
topics_of_interest
# 3  7 15 32 45 47 51 59 14 23

# cloud() is an STM package function that generates a simple base-R word cloud
# for a given topic. Words are sized proportional to their probability in the
# topic (beta). In Python, you would use WordCloud from the wordcloud library.
cloud(topic_model_prev, 3)
cloud(topic_model_prev, 7)
cloud(topic_model_prev, 15)
cloud(topic_model_prev, 32)
cloud(topic_model_prev, 47)
cloud(topic_model_prev, 51)
cloud(topic_model_prev, 59)
cloud(topic_model_prev, 14)
cloud(topic_model_prev, 23)


# =============================================================================
#### SECTION: ggplot2-Based Word Clouds (ggwordcloud) ####
# =============================================================================

# This function creates word clouds using ggplot2 syntax via geom_text_wordcloud().
# Unlike the base cloud() function, these can be arranged in grids, themed,
# and saved with ggsave(). The function extracts the top 30 words from a topic
# using the log-beta matrix stored in the STM model object.

# Function to create ggplot wordcloud from STM topic
stm_to_ggwordcloud <- function(model, topic, topic_name) {
  # Get top words and probabilities correctly
  # labelTopics() returns a list with elements like $prob (a matrix of top words).
  # However, for reliable numeric probabilities, we use the beta matrix directly.
  words <- labelTopics(model, topic, n = 30)
  word_probs <- data.frame(
    word = words$prob[1,],  # First row contains the words
    freq = as.numeric(words$prob[2,]),  # Second row contains probabilities
    stringsAsFactors = FALSE
  )

  # Alternative approach: use the topic-word matrix directly
  # This is more reliable for getting actual probabilities
  # model$beta$logbeta[[1]] is a matrix of log(P(word|topic)) values.
  # exp() converts log-probabilities back to probabilities.
  # In Python/gensim: model.get_topic_terms(topic_id, topn=30)
  beta <- exp(model$beta$logbeta[[1]])  # Convert log probabilities to probabilities
  topic_words <- beta[topic, ]
  # order(..., decreasing=TRUE)[1:30] gets the indices of the 30 highest-prob words.
  top_indices <- order(topic_words, decreasing = TRUE)[1:30]

  word_probs <- data.frame(
    word = model$vocab[top_indices],
    freq = topic_words[top_indices],
    stringsAsFactors = FALSE
  )

  # Create ggplot wordcloud using ggwordcloud package.
  # aes(label=word, size=freq) maps words and their probabilities to the plot.
  # scale_size_area() ensures word sizes are proportional to probability.
  # In Python: WordCloud().generate_from_frequencies(word_freq_dict)
  ggplot(word_probs, aes(label = word, size = freq)) +
    geom_text_wordcloud() +
    scale_size_area(max_size = 10) +
    theme_minimal() +
    labs(title = topic_name)
}

# Create ggplot wordcloud objects for each topic of interest
sleep_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 3, "")
drug_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 7, "")
science_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 15, "")
apnea_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 32, "")
health_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 47, "")
academic_research_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 51, "")
school_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 59, "")



dearann_1_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 14, "")
iraq_2_cloud_gg <- stm_to_ggwordcloud(topic_model_prev, 23, "")


# =============================================================================
#### SECTION: Arrange Word Clouds in Grid Layouts ####
# =============================================================================

# grid.arrange() combines multiple ggplot objects into a single multi-panel figure.
# Different layout options are explored below.

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
# layout_matrix lets you specify exactly which plot goes in which cell.
# NA means an empty cell. c(10, 10, NA) makes iraq_2 span 2 columns.
layout_matrix <- rbind(
  c(1, 2, 3),
  c(4, 5, 6),
  c(7, 8, 9),
  c(10, 10, NA)  # iraq_2 spans 2 columns, empty space in corner
)


# =============================================================================
#### SECTION: Save Base-R Word Clouds to PNG Files ####
# =============================================================================

# png() opens a PNG graphics device. All subsequent plotting commands are
# written to the file until dev.off() closes the device. This is necessary
# for base-R plotting functions like cloud().
# In Python: fig.savefig("filename.png") or plt.savefig("filename.png").

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
# This is more concise and avoids repetitive code (DRY principle).
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


# =============================================================================
#### SECTION: Additional Topic Diagnostics ####
# =============================================================================

# Print top words for all 70 topics using both probability and score rankings.
sageLabels(topic_model_prev)

# topicCorr() computes pairwise topic correlations from the document-topic
# proportions (theta). plot() renders a network graph where edges connect
# topics that tend to co-occur in the same documents.
# In Python, you would compute np.corrcoef(theta.T) and use networkx for plotting.
plot(topicCorr(topic_model_prev))


# =============================================================================
#### SECTION: Extract and Analyze Beta and Theta Matrices ####
# =============================================================================

# The STM has two key matrices:
#   beta (word-topic): P(word | topic) -- what words define each topic
#   theta (document-topic): P(topic | document) -- what topics each doc contains
#
# tidy() converts these matrices to long-format data frames for analysis.
# In Python/gensim, these are accessible via model.get_topic_terms() and
# model.get_document_topics().

## Matrices
stm_beta <- tidy(topic_model_prev, matrix = "beta")
# Filter theta to only our topics of interest for focused analysis.
# The 'gamma' column in the tidy theta output represents P(topic | document).
stm_theta <- tidy(topic_model_prev, matrix = "theta") %>%
  dplyr::filter(topic == topics_of_interest)

# Visualize the distribution of document-topic probabilities (gamma/theta).
# Each facet shows one topic; the histogram reveals whether the topic is
# concentrated in a few documents (right-skewed) or spread across many.
## Distribution of document probabilites for each topic
ggplot(stm_theta, aes(gamma, fill = as.factor(topic))) +
  geom_histogram(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  labs(title = "Distribution of document probabilities for each topic",
       y = "Number of stories", x = expression(gamma))

# Merge the document-topic proportions back with the original text metadata.
# mutate(document = row_number()) adds a row ID to match the STM's internal
# document numbering. merge() is an inner join (Python: pd.merge()).
final_text <- full_text_subset %>% mutate(document = row_number())
final_text <- merge(final_text, stm_theta, by = "document")


summary(final_text$gamma)
# qplot() is a quick-plot function from ggplot2 for fast exploratory histograms.
qplot(final_text$gamma)


# plot.STM() is the STM package's built-in summary plot. type = "summary"
# shows expected topic proportions across all documents (bar chart).
# labeltype = "prob" labels each topic with its highest-probability words.
## plot.STM()
plot.STM(topic_model_prev, type = "summary", labeltype = "prob",
         topics = topics_of_interest)


# =============================================================================
#### SECTION: Extract Top 100 Documents per Topic to File ####
# =============================================================================

# This section exports the 100 documents with the highest topic proportion
# (theta) for each topic of interest. This allows qualitative review of
# representative documents for validation and interpretation.

#### Write top 100 articles to file:

# Define your topics of interest and names
topics_of_interest <- c(3, 7, 15, 32, 47, 51)
topic_names <- c("Sleep and Work", "Sleep_medicine_drugs", "Circadian_science",
                 "Sleep_apnea_hospitals", "Sleep_and_health",
                 "Academic_sleep_research")

# Function to extract top documents for a topic.
#
# Parameters:
#   model     : fitted STM object (contains theta matrix)
#   topic_num : which topic number (1-70) to extract for
#   topic_name: human-readable label for the topic (used in filenames)
#   meta_data : data frame with article metadata (text, date, source, etc.)
#   n_docs    : how many top documents to extract (default: 100)
#
# Returns: a data.frame with Rank, Document_Index, Topic_Proportion, and
#          all available metadata columns.
#
# The function uses defensive column-name checking (checking for both lowercase
# and capitalized versions) to handle different metadata formats.
extract_top_documents <- function(model, topic_num, topic_name, meta_data, n_docs = 100) {

  # model$theta is the document-topic proportion matrix (N_docs x K_topics).
  # theta[i, j] = proportion of document i devoted to topic j.
  # Get document-topic proportions for this topic
  topic_proportions <- model$theta[, topic_num]

  # order() returns indices sorted by value. decreasing=TRUE gives highest first.
  # [1:n_docs] takes the top n_docs indices.
  # Get top document indices
  top_indices <- order(topic_proportions, decreasing = TRUE)[1:n_docs]
  top_proportions <- topic_proportions[top_indices]

  # Build the output data frame with core fields
  # Extract document information
  top_docs_data <- data.frame(
    Rank = 1:n_docs,
    Document_Index = top_indices,
    Topic_Proportion = round(top_proportions, 4),
    stringsAsFactors = FALSE
  )

  # Add metadata if available (adjust column names as needed)
  # The series of if/else if checks handle various common column naming
  # conventions (text/Text/content, date/Date/datetime, etc.)
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


# =============================================================================
#### SECTION: Save Top Documents to CSV and Text Files ####
# =============================================================================

# Create directory for output files
# dir.exists() checks if a directory exists; dir.create() creates it.
# In Python: os.makedirs("top_documents", exist_ok=True)
if (!dir.exists("top_documents")) {
  dir.create("top_documents")
}

# Loop over each topic of interest and save results in multiple formats.
# Extract and save top documents for each topic
for (i in 1:length(topics_of_interest)) {
  topic_num <- topics_of_interest[i]
  topic_name <- topic_names[i]

  # cat() prints to console (like Python's print()).
  cat("Processing topic", topic_num, ":", topic_name, "\n")

  # Extract top documents
  # Using full_text_subset as the metadata object
  top_docs <- extract_top_documents(topic_model_prev, topic_num, topic_name, full_text_subset)

  # --- Option 1: Save as CSV ---
  # paste0() concatenates strings without separators (like Python's f-strings or +).
  # write.csv() writes a data frame to CSV. row.names=FALSE omits row numbers.
  # In Python: df.to_csv(filename, index=False)
  csv_filename <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_top100.csv")
  write.csv(top_docs, csv_filename, row.names = FALSE)

  # --- Option 2: Save as readable text file ---
  # sink() redirects all subsequent cat()/print() output to a file (like Python's
  # sys.stdout redirection or contextlib.redirect_stdout).
  # sink() with no arguments restores output to the console.
  txt_filename <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_top100.txt")

  # Create formatted text output
  sink(txt_filename)
  cat("TOP 100 DOCUMENTS FOR TOPIC", topic_num, ":", toupper(gsub("_", " ", topic_name)), "\n")
  # "=" %>% rep(80) %>% paste(collapse = "") creates a string of 80 "=" characters
  # as a visual divider. In Python: "=" * 80
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
      # strwrap() wraps text to a maximum width for readability.
      # In Python: textwrap.fill(text, width=80)
      text <- top_docs$Text[j]
      wrapped_text <- strwrap(text, width = 80)
      cat(paste(wrapped_text, collapse = "\n"), "\n")
    }

    cat("\n", "-" %>% rep(80) %>% paste(collapse = ""), "\n\n")
  }
  sink()

  # --- Option 3: Save individual text files for each document ---
  doc_dir <- paste0("top_documents/topic_", topic_num, "_", topic_name, "_individual/")
  if (!dir.exists(doc_dir)) {
    dir.create(doc_dir, recursive = TRUE)
  }

  for (j in 1:nrow(top_docs)) {
    # sprintf("%03d", j) zero-pads the rank to 3 digits (001, 002, ..., 100)
    # for consistent file sorting. In Python: f"rank_{j:03d}_doc_{idx}.txt"
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


# =============================================================================
#### SECTION: Summary File Across All Topics ####
# =============================================================================

# Create a single summary file showing the top 10 documents for each topic
# of interest (a quick reference for reviewers).
summary_filename <- "top_documents/summary_all_topics.txt"
sink(summary_filename)
cat("SUMMARY: TOP 100 DOCUMENTS FOR ALL TOPICS OF INTEREST\n")
cat("=" %>% rep(60) %>% paste(collapse = ""), "\n\n")

for (i in 1:length(topics_of_interest)) {
  topic_num <- topics_of_interest[i]
  topic_name <- topic_names[i]

  top_docs <- extract_top_documents(topic_model_prev, topic_num, topic_name, full_text_subset)

  # toupper() converts to uppercase. gsub() replaces underscores with spaces.
  # In Python: topic_name.replace("_", " ").upper()
  cat("TOPIC", topic_num, ":", toupper(gsub("_", " ", topic_name)), "\n")
  cat("Top 10 documents (proportion):\n")

  for (j in 1:10) {
    cat("  ", j, ". Doc", top_docs$Document_Index[j],
        "(", top_docs$Topic_Proportion[j], ")", "\n")
  }
  cat("\n")
}
sink()
