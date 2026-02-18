# =============================================================================
# SCRIPT:   02_master_stm_analysis.R
# PURPOSE:  MASTER ANALYSIS SCRIPT for Structural Topic Modeling (STM) of
#           newspaper coverage of sleep. This is the primary analysis script
#           that encompasses the full pipeline:
#             1. Data preprocessing and STM estimation (K=70 topics, year as
#                prevalence covariate)
#             2. Topic labeling, visualization, and temporal trend analysis
#             3. Word clouds (base R and ggplot2-based)
#             4. Top document extraction for qualitative validation
#             5. Topic term extraction using sageLabels (probability and score)
#             6. Word co-occurrence network analysis using FREX/Score/Prob/Lift
#                ranking methods
#
# AUTHOR:   Joon Chung
# EMAIL:    Contact: see README
# AFFIL:    The University of Miami, Miller School of Medicine
#           The Department of Informatics and Health Data Science
#
# CREATED:  7/28/2018 (penultimate update)
# UPDATED:  10/16/2025 (last update, based on Master R script 8_29_2025.R)
#
# INPUTS:
#   - full_text_clean: A pre-loaded R data frame (from .Rda file) containing
#     cleaned newspaper articles. Expected columns: text, text_id, datetime, Year.
#   - The .Rda file path is commented out; the object must exist in the R
#     environment before running.
#
# OUTPUTS:
#   - topic_model_prev: Fitted STM object (70 topics, year prevalence)
#   - prep: estimateEffect object for temporal trend analysis
#   - PNG/EPS plot files for topic proportions, word clouds, and networks
#   - CSV/TXT files with top documents per topic
#   - CSV files with topic term matrices (probability, score)
#   - stm_networks_[method]/ directories with co-occurrence network plots and data
#
# REQUIRED PACKAGES:
#   - stm         : Structural Topic Modeling (core modeling and labeling)
#   - tidyverse   : Data wrangling (dplyr, tidyr, stringr) + ggplot2 visualization
#   - gridExtra   : Multi-panel plot layouts (grid.arrange)
#   - tidytext    : Tidy wrappers for text mining; tidy() for STM matrices
#   - ggwordcloud : Word clouds via ggplot2 grammar of graphics
#   - quanteda    : Text preprocessing and document-feature matrices (for networks)
#   - igraph      : Network/graph objects and analysis
#   - ggraph      : Grammar-of-graphics network visualization
#
# -----------------------------------------------------------------------
# KEY STATISTICAL CONCEPTS:
# -----------------------------------------------------------------------
#
# STRUCTURAL TOPIC MODELS (STM):
#   STM is a probabilistic generative model for text that extends Latent
#   Dirichlet Allocation (LDA) by allowing document-level covariates to
#   influence (a) topic prevalence (how much of each document is about each
#   topic) and/or (b) topic content (which words are used within a topic).
#   Here we use year as a prevalence covariate, meaning topic proportions
#   can change over time while the topics themselves remain stable.
#
#   In Python, the closest equivalents are:
#     - gensim LdaModel (no covariates, but similar generative model)
#     - scikit-learn LatentDirichletAllocation (also no covariates)
#     - guidedlda or corex_topic (partial covariate support)
#   There is no full Python port of STM with prevalence/content covariates.
#
# PREVALENCE COVARIATES:
#   A prevalence covariate allows the frequency of topics to vary with
#   document-level metadata. For example, prevalence = ~year means the
#   model allows some topics to become more or less common over the years.
#   This is specified using R's formula notation (similar to patsy in Python).
#
# BETA MATRIX (word-topic distribution):
#   beta[k, v] = P(word v | topic k). The probability of observing word v
#   given that the text comes from topic k. High-beta words define a topic.
#
# THETA MATRIX (document-topic distribution):
#   theta[d, k] = P(topic k | document d). The proportion of document d
#   devoted to topic k. All values in a row sum to 1.
#
# FREX SCORES:
#   FREX (FREquency + EXclusivity) is a harmonic mean of a word's frequency
#   within a topic and its exclusivity to that topic. Words that are both
#   frequent in a topic AND rare in other topics receive high FREX scores.
#   FREX is often preferred over raw probability for topic labeling because
#   it excludes common "filler" words that appear in many topics.
#
# B-SPLINE SMOOTHING:
#   s(year) in the formula 1:70 ~ s(year) specifies a basis spline (B-spline)
#   smooth function of year. This allows topic prevalence to follow a flexible,
#   non-linear curve over time rather than assuming a strict linear trend.
#   In Python, you would use patsy's bs() function or scipy.interpolate.BSpline.
#
# -----------------------------------------------------------------------
# NOTES FOR PYTHON USERS:
# -----------------------------------------------------------------------
#   - The pipe operator %>% passes the result of the left expression as the
#     first argument of the right function. Equivalent to method chaining:
#       R:      df %>% filter(x > 1) %>% select(y)
#       Python: df.query('x > 1')[['y']]
#   - <- is the assignment operator in R (equivalent to = in Python).
#     Both <- and = work for assignment in R, but <- is conventional.
#   - library() loads a package (like Python's import).
#   - R is 1-indexed (not 0-indexed like Python).
#   - R data frames are similar to pandas DataFrames but use $ for column
#     access (df$col) instead of Python's df['col'] or df.col.
#   - Formula notation like ~year or 1:70 ~ s(year) is unique to R/S.
#     In Python, the equivalent would be patsy formula strings: 'y ~ bs(x)'.
#   - set.seed() ensures reproducible random number generation
#     (Python: np.random.seed() or random.seed()).
# =============================================================================


###############################################################################
#### PART 1: STM MODEL ESTIMATION AND INITIAL DIAGNOSTICS                  ####
###############################################################################


# =============================================================================
#### SECTION: Author Header and Metadata ####
# =============================================================================

## Joon Chung
## Contact: see README
## The University of Miami, Miller School of Medicine
##  The Department of Informatics and Health Data Science
##
## Structural Topic Models by year
##
## Penultimate update: 7/28/2018
## Last update:        10/16/2025

## This script runs STM, with each year as a PREVALENCE covariate


# =============================================================================
#### SECTION: Set Random Seed for Reproducibility ####
# =============================================================================

# set.seed() initializes R's pseudo-random number generator to a fixed state.
# This ensures that stochastic algorithms (like STM's spectral initialization)
# produce identical results every time the script is run.
# 8675309 is the seed value (a memorable phone number from a 1980s pop song).
# In Python: np.random.seed(8675309) or random.seed(8675309)
set.seed(8675309)


# =============================================================================
#### SECTION: Load Required Libraries ####
# =============================================================================

# library() loads an R package into the current session. If a package is not
# installed, you must first run: install.packages("package_name")
# In Python, the equivalent is: import package_name

library(stm)
# stm: Structural Topic Models. Core functions used in this script:
#   stm()            - Estimate a topic model
#   textProcessor()  - Preprocess raw text (tokenize, stem, remove stopwords)
#   prepDocuments()  - Create document-term matrices with vocabulary trimming
#   estimateEffect() - Estimate covariate effects on topic prevalence
#   sageLabels()     - Extract topic labels using probability and score methods
#   labelTopics()    - Get top words per topic (multiple ranking methods)
#   cloud()          - Generate word clouds for topics
#   topicCorr()      - Compute topic correlations
#   plot.STM()       - Built-in STM summary plots

library(tidyverse)
# tidyverse loads: dplyr, ggplot2, tidyr, readr, purrr, tibble, stringr, forcats
# Key functions used:
#   dplyr::select(), filter(), mutate(), arrange(), group_by(), summarise()
#   ggplot2::ggplot(), aes(), geom_line(), geom_ribbon(), geom_col(), etc.
#   tidyr::pivot_longer(), pivot_wider()
#   stringr::str_count(), str_remove()
# In Python, these span pandas, matplotlib, seaborn.

library(gridExtra)
# gridExtra: grid.arrange() combines multiple ggplot objects into one figure.
# In Python: matplotlib subplots -- fig, axes = plt.subplots(1, 2)

library(tidytext)
# tidytext: Provides tidy() method for STM objects, converting beta/theta
# matrices to long-format data frames with columns: topic, term, beta (or gamma).
# In Python, you would manually extract these from the model object.

library(ggwordcloud)
# ggwordcloud: geom_text_wordcloud() renders word clouds using ggplot2 syntax.
# In Python: from wordcloud import WordCloud


# =============================================================================
#### SECTION: Data Loading and Preprocessing ####
# =============================================================================

# Load full_text
# NOTE: The load() call is commented out. The full_text_clean object must
# already exist in your R environment (e.g., from a prior session or script).
# load() restores R objects saved with save(). In Python, the equivalent would
# be pickle.load() or joblib.load().

# Select only the columns needed for modeling, and remove rows with any NA.
# dplyr::select() chooses columns by name.
# na.omit() removes rows containing any NA values (like pandas dropna()).
# The :: syntax (dplyr::select) explicitly calls select from the dplyr package,
# avoiding conflicts with other packages that may define a select() function.
## Remove any NA
full_text_subset <- full_text_clean %>% dplyr::select(text, text_id, datetime, Year) %>%
  na.omit()
# Convert Year (character or factor) to numeric for use as a model covariate.
# as.numeric() coerces the value to a double. In Python: df['year'] = df['Year'].astype(int)
full_text_subset$year <- as.numeric(full_text_subset$Year)


# =============================================================================
#### SECTION: Text Preprocessing with STM ####
# =============================================================================

# textProcessor() tokenizes the text, converts to lowercase, removes stopwords,
# stems words, and creates a vocabulary. It returns:
#   $documents: list of sparse document-term count matrices
#   $vocab: character vector of vocabulary terms
#   $meta: metadata data frame (passed through from input)
#
# striphtml = TRUE removes any HTML tags from the text.
# metadata = as.data.frame(...) attaches document-level metadata for use as
# covariates in the STM model.
#
# In Python, equivalent preprocessing would use:
#   from sklearn.feature_extraction.text import CountVectorizer
#   vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode')
#   dtm = vectorizer.fit_transform(texts)
# Or with gensim:
#   from gensim.utils import simple_preprocess
#   from gensim.corpora import Dictionary
# Processed
processed <- textProcessor(full_text_subset$text, striphtml = TRUE, metadata = as.data.frame(full_text_subset))


# =============================================================================
#### SECTION: Vocabulary Trimming Diagnostics ####
# =============================================================================

# plotRemoved() visualizes how many words and documents are removed at
# different vocabulary frequency thresholds. This helps choose an appropriate
# lower.thresh value for prepDocuments().
#
# lower.thresh = seq(from = 10, to = 2000, by = 2) tests thresholds from 10
# to 2000 (words appearing fewer than threshold times are removed).
#
# In Python, you would vary min_df in CountVectorizer and plot vocabulary sizes.
plotRemoved(processed$documents, lower.thresh = seq(from = 10, to = 2000, by = 2))


# =============================================================================
#### SECTION: Prepare Documents for STM ####
# =============================================================================

# prepDocuments() performs final document preparation:
#   - Removes words appearing in fewer than lower.thresh documents
#   - Removes documents that become empty after vocabulary trimming
#   - Aligns documents, vocabulary, and metadata
#
# lower.thresh = 500 means words must appear in at least 500 documents to be
# retained. This aggressive filtering reduces noise from rare words but may
# remove domain-specific terms. The threshold was chosen based on the
# plotRemoved() diagnostic above.
#
# Returns:
#   out$documents: filtered document list
#   out$vocab: filtered vocabulary
#   out$meta: filtered metadata (rows removed if documents were dropped)
out <- prepDocuments(processed$documents, processed$vocab, processed$meta, lower.thresh = 500)


# =============================================================================
#### SECTION: Estimate Structural Topic Model ####
# =============================================================================

# Re-set the seed immediately before model estimation to ensure exact
# reproducibility even if random operations occurred between the first
# set.seed() and this point.
## Run STM
set.seed(8675309)

# stm() is the main model estimation function. Arguments:
#   documents: sparse document list from prepDocuments()
#   vocab: vocabulary vector from prepDocuments()
#   data: metadata data frame from prepDocuments() (used for covariates)
#   prevalence: formula specifying prevalence covariates
#     ~year means topic proportions vary linearly with year
#   K: number of topics to estimate (70 in this study)
#   verbose: if TRUE, prints progress during estimation
#   init.type: initialization method for the EM algorithm
#     "Spectral" uses a spectral decomposition for deterministic initialization
#     (given the same seed), which tends to produce more stable and reproducible
#     results than random initialization.
#
# The model estimates:
#   - beta: K x V matrix of word-topic probabilities
#   - theta: D x K matrix of document-topic proportions
#   - covariate effects on prevalence
#
# In Python/gensim:
#   from gensim.models import LdaModel
#   lda = LdaModel(corpus=corpus, num_topics=70, id2word=dictionary,
#                   random_state=8675309, passes=20)
# Note: gensim LDA does not support prevalence covariates.
topic_model_prev <- stm(out$documents,
                        out$vocab,
                        data = out$meta,
                        prevalence = ~year,
                        K = 70,
                        verbose = TRUE,
                        init.type = "Spectral")
#


###############################################################################
#### PART 2: TOPIC EXPLORATION AND VISUALIZATION                           ####
###############################################################################


# =============================================================================
#### SECTION: Topic Labeling ####
# =============================================================================

###################################################################################

# sageLabels() prints the top words for all K topics using two ranking methods:
#   1. Marginal Highest Probability: P(word | topic) -- raw probability
#   2. Marginal Score: a metric balancing frequency and exclusivity
# This provides a quick overview of what each topic "looks like."
# In Python/gensim: lda_model.print_topics(num_topics=70, num_words=10)
sageLabels(topic_model_prev)


# =============================================================================
#### SECTION: Estimate Temporal Effects on Topic Prevalence ####
# =============================================================================

# estimateEffect() regresses topic proportions on document-level covariates.
# The formula 1:70 ~ s(year) means:
#   1:70: estimate the effect for all 70 topics
#   s(year): use a B-spline smoother on year
#
# WHAT IS s(year)?
# s() creates a basis spline (B-spline) smooth function. Unlike a linear term
# (just ~year), s(year) allows the relationship between year and topic
# prevalence to be non-linear (curves, peaks, valleys). The number of knots
# (flexibility) is chosen automatically.
#
# meta = out$meta passes the metadata so the function can access the 'year'
# variable for each document.
#
# The result (prep) can be plotted to show how each topic's expected proportion
# changes over time with 95% confidence intervals.
#
# In Python/statsmodels:
#   import statsmodels.api as sm
#   from patsy import dmatrix
#   X = dmatrix("bs(year, df=5)", data=meta_df)
#   for k in range(70):
#       model = sm.OLS(theta[:, k], X).fit()
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

  # plot() on an estimateEffect object extracts prediction data and generates
  # a base R plot. Here we capture the returned data (means, CIs, x-values)
  # to create a custom ggplot2 visualization instead.
  # covariate = "year": the x-axis variable
  # method = "continuous": treat year as continuous (not categorical)
  # topics = topic_select: which topic to plot
  data <- plot(prep, covariate = "year", method = "continuous", topics = topic_select)

  # Extract expected topic proportions.
  # with(data, means) evaluates 'means' in the context of the 'data' list
  # (equivalent to data$means or data['means'] in Python).
  # %>% unlist() flattens nested lists to a vector.
  # %>% as.matrix() %>% as.data.frame() converts to a column data frame.
  # Expected topic proportion
  means <- with(data, means) %>% unlist() %>% as.matrix() %>% as.data.frame()
  means <- with(means, V1)

  # Extract 95% confidence intervals.
  # The CI vector interleaves lower and upper bounds:
  #   [lower_1, upper_1, lower_2, upper_2, ...]
  # seq(1, n, 2) extracts odd-indexed values (lower bounds)
  # seq(2, n, 2) extracts even-indexed values (upper bounds)
  # 95% Confidence interval
  ci <- with(data, ci) %>% unlist() %>% as.data.frame()

  lower <- ci[seq(1, nrow(ci), 2),]
  upper <- ci[seq(2, nrow(ci), 2),]

  # Extract the year values
  # years
  years <- with(data, x) %>% as.numeric()
  plot_df <- data.frame(years = years, means = means, lower = lower, upper = upper)
  # IMPORTANT: Filter to years <= 2017 (the study period end).
  # The author's emphasis marks (######## !!!!!!!) indicate this is a critical
  # analytical decision -- data beyond 2017 may be sparse or unreliable.
  plot_df <- plot_df %>% filter(years <= 2017) ######## !!!!!!!!!!

  # Create the topic proportion time series plot.
  # geom_line(): draws the expected proportion over years
  # geom_ribbon(): adds shaded 95% confidence interval band
  # scale_x_continuous(): x-axis from 1983 to 2017 with 3-year breaks
  # theme_bw(): clean white-background theme
  # In Python: plt.plot(years, means); plt.fill_between(years, lower, upper)
  # Plot
  topic_prop <- ggplot(plot_df, aes(x = years, y = means)) + geom_line() +
    geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
    scale_x_continuous(breaks = seq(1983,2016, 3), limits = c(1983, 2017)) +
    theme_bw() +
    theme(axis.text.x = element_text(angle =45)) +
    xlab("Year") +
    ylab("Expected topic proportion")

  # tidy() from tidytext converts the STM beta matrix to long format:
  #   topic | term | beta
  # where beta = P(word | topic).
  stm_beta <- tidy(topic_model_prev, matrix = "beta")
  # Filter to selected topic, sort by descending probability, take top 10,
  # and create a horizontal bar chart.
  # reorder(term, beta) sorts bars by probability for readability.
  # coord_flip() makes bars horizontal (R's ggplot default is vertical).
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

  # Combine both plots side by side: terms on left, time series on right.
  return(grid.arrange(topic_terms, topic_prop, nrow = 1,top = title))
}


# =============================================================================
#### SECTION: Generate Plots for Topics of Interest ####
# =============================================================================

## Preliminary plots - full plot output later in code.
topics_of_interest
# 3  7 15 32 47 51

# Generate two-panel plots (top terms + temporal trend) for each topic.
sleep_plot <- stm_plot(topic = 3, "Work and sleep")
drug_plot <- stm_plot(topic = 7, "Sleep medicine / drugs")
science_plot <- stm_plot(topic = 15, "Circadian science")
apnea_plot <- stm_plot(topic = 32, "Sleep apnea, hospitals")
health_plot <- stm_plot(topic = 47, "Health research")
academic_research_plot <- stm_plot(topic = 51, "Sleep research")


# =============================================================================
#### SECTION: Save Topic Plots to PNG ####
# =============================================================================

# ggsave() saves a ggplot or grob object to file.
# Width and height are in inches by default.
# In Python: fig.savefig("filename.png", dpi=300, bbox_inches='tight')
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


# =============================================================================
#### SECTION: Word Clouds (Base R) ####
# =============================================================================

# Updated list of topics of interest including additional and validation topics.
topics_of_interest
# 3  7 15 32 45 47 51 59 14 23

# cloud() from the stm package generates a base-R word cloud for a topic.
# Words are sized proportional to P(word | topic).
# In Python: WordCloud().generate_from_frequencies(topic_word_probs)
cloud(topic_model_prev, 3)
cloud(topic_model_prev, 7)
cloud(topic_model_prev, 15)
cloud(topic_model_prev, 32)
cloud(topic_model_prev, 47)
cloud(topic_model_prev, 51)


# =============================================================================
#### SECTION: Additional Diagnostics ####
# =============================================================================

## Terms for topics
sageLabels(topic_model_prev)
# topicCorr() computes pairwise Pearson correlations between topic proportions
# across documents. plot() renders a network where edges connect correlated topics.
# In Python: np.corrcoef(theta.T) then use networkx for graph visualization.
plot(topicCorr(topic_model_prev))

## Matrices
stm_beta <- tidy(topic_model_prev, matrix = "beta")
stm_theta <- tidy(topic_model_prev, matrix = "theta") %>%
  dplyr::filter(topic == topics_of_interest)

# Visualize the distribution of document-topic proportions (theta/gamma).
# Each facet shows one topic. Right-skewed distributions indicate topics
# concentrated in few documents; uniform distributions suggest diffuse topics.
## Distribution of document probabilites for each topic
ggplot(stm_theta, aes(gamma, fill = as.factor(topic))) +
  geom_histogram(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  labs(title = "Distribution of document probabilities for each topic",
       y = "Number of stories", x = expression(gamma))

# Merge theta with original text for downstream analysis.
# mutate(document = row_number()) adds a sequential document ID.
# merge() performs an inner join on 'document'.
final_text <- full_text_subset %>% mutate(document = row_number())
final_text <- merge(final_text, stm_theta, by = "document")


summary(final_text$gamma)
qplot(final_text$gamma)

# plot.STM() is the stm package's built-in summary plot.
# type = "summary": bar chart of expected topic proportions
# labeltype = "prob": label topics with highest-probability words
## plot.STM()
plot.STM(topic_model_prev, type = "summary", labeltype = "prob",
         topics = topics_of_interest)


###############################################################################
#### PART 3: TOP DOCUMENT EXTRACTION                                       ####
###############################################################################


# =============================================================================
#### SECTION: Define Topics and Extract Top 100 Documents ####
# =============================================================================

# This section exports the top 100 documents (highest theta) per topic
# for qualitative validation and close reading.

#### Write top 100 articles to file:

# Define your topics of interest and names
topics_of_interest <- c(3, 7, 15, 32, 47, 51)
topic_names <- c("Sleep and Work", "Sleep_medicine_drugs", "Circadian_science",
                 "Sleep_apnea_hospitals", "Sleep_and_health",
                 "Academic_sleep_research")

# Function to extract top documents for a topic.
# model$theta[, topic_num] gets the column of document-topic proportions.
# order(..., decreasing=TRUE)[1:n_docs] returns indices of top documents.
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
  # The series of if/else if checks handles various column naming conventions.
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
  # sink() redirects cat()/print() output to a file (like Python's sys.stdout redirection).
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


###############################################################################
#### PART 4: STM TOPIC TERMS EXTRACTION - PROBABILITY AND SCORE            ####
###############################################################################


# =============================================================================
#### SECTION: Topic Term Extraction Functions ####
# =============================================================================

# =============================================================================
# STM TOPIC TERMS EXTRACTION - PROBABILITY AND SCORE
# =============================================================================
# Extract marginal probability and score terms for all topics in STM model.
#
# sageLabels() provides two key term-ranking methods:
#   1. Marginal Probability: P(word | topic) -- words most likely in the topic
#   2. Marginal Score: A metric that balances frequency within the topic and
#      rarity across other topics (conceptually similar to TF-IDF).
#
# These are different from FREX scores (used later in network analysis).
# Probability favors common words; Score favors distinctive words.

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
#'
#' NOTE: The #' syntax is roxygen2 documentation format, similar to Python's
#' docstrings. It can be processed to generate help files.

extract_all_topic_terms <- function(topic_model, n_terms = 5) {

  cat("Extracting", n_terms, "terms per method for", topic_model$settings$dim$K, "topics...\n")

  # sageLabels() returns a complex list structure with term rankings for all topics.
  # sage_result$marginal$prob is a matrix: rows = topics, columns = ranked terms.
  # sage_result$marginal$score is a similar matrix using the score ranking.
  sage_result <- sageLabels(topic_model, n = n_terms)

  # Extract probability and score terms
  prob_terms <- sage_result$marginal$prob    # Matrix: topics x terms
  score_terms <- sage_result$marginal$score  # Matrix: topics x terms

  # Convert to long format data frame for easier manipulation.
  # This creates one row per topic-method-rank combination.
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

  # do.call(rbind, list) row-binds all data frames.
  # In Python: pd.concat(results_list, ignore_index=True)
  final_results <- do.call(rbind, results_list)

  cat("Extracted", nrow(final_results), "term entries\n")
  cat("Topics:", topic_model$settings$dim$K, "\n")
  cat("Methods: probability, score\n")
  cat("Terms per method per topic:", n_terms, "\n")

  return(final_results)
}


# =============================================================================
#### SECTION: Wide Format Extraction (Excel-Friendly) ####
# =============================================================================

# =============================================================================
# WIDE FORMAT EXTRACTION (ALTERNATIVE)
# =============================================================================

#' Extract Terms in Wide Format (Excel-friendly)
#'
#' Creates a wide-format table with one row per topic and separate columns
#' for each term rank and method. Better for Excel viewing.

extract_terms_wide_format <- function(topic_model, n_terms = 5) {

  cat("Extracting terms in wide format...\n")

  sage_result <- sageLabels(topic_model, n = n_terms)

  # Create dynamic column names: prob_1, prob_2, ..., score_1, score_2, ...
  prob_cols <- paste0("prob_", 1:n_terms)
  score_cols <- paste0("score_", 1:n_terms)

  wide_results <- data.frame(
    topic = 1:nrow(sage_result$marginal$prob)
  )

  # Add probability columns using [[ ]] for dynamic column creation.
  # In Python: wide_results[f'prob_{i}'] = sage_result.marginal.prob[:, i]
  for (i in 1:n_terms) {
    wide_results[[prob_cols[i]]] <- as.character(sage_result$marginal$prob[, i])
  }

  for (i in 1:n_terms) {
    wide_results[[score_cols[i]]] <- as.character(sage_result$marginal$score[, i])
  }

  cat("Created wide format with", ncol(wide_results), "columns\n")

  return(wide_results)
}


# =============================================================================
#### SECTION: sageLabels-Style Formatted Output Functions ####
# =============================================================================

# These functions generate output formatted exactly like sageLabels() console
# display, making it easy to copy into papers or supplementary materials.

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

  if (is.null(topics)) {
    topics <- 1:topic_model$settings$dim$K
  }

  sage_result <- sageLabels(topic_model, n = n_terms)

  output_lines <- c()

  for (topic_num in topics) {
    if (topic_num <= nrow(sage_result$marginal$prob)) {
      prob_terms <- as.character(sage_result$marginal$prob[topic_num, ])
      score_terms <- as.character(sage_result$marginal$score[topic_num, ])

      topic_line <- paste0("Topic ", topic_num, ":")
      prob_line <- paste0(" \t Marginal Highest Prob: ", paste(prob_terms, collapse = ", "))
      score_line <- paste0(" \t Marginal Score: ", paste(score_terms, collapse = ", "))

      output_lines <- c(output_lines, topic_line, prob_line, score_line, "")
    }
  }

  return(output_lines)
}

#' Print sageLabels Format to Console
print_sagelabels_format <- function(topic_model, n_terms = 5, topics = NULL) {
  output_lines <- create_sagelabels_format(topic_model, n_terms, topics)
  for (line in output_lines) {
    cat(line, "\n")
  }
}

#' Save sageLabels Format to Text File
save_sagelabels_format <- function(topic_model, filename = "stm_sagelabels_format.txt",
                                   n_terms = 5, topics = NULL) {
  output_lines <- create_sagelabels_format(topic_model, n_terms, topics)
  # writeLines() writes a character vector to a file (one element per line).
  # In Python: with open(filename, 'w') as f: f.writelines(output_lines)
  writeLines(output_lines, filename)
  cat("Saved sageLabels format to:", filename, "\n")
  # %||% is the null-coalescing operator: if left side is NULL, use right side.
  # In Python: topics if topics is not None else list(range(1, K+1))
  cat("Topics included:", length(topics %||% 1:topic_model$settings$dim$K), "\n")
  cat("Terms per method:", n_terms, "\n")
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
#### SECTION: Batch Processing and Matrix Output Functions ####
# =============================================================================

# =============================================================================
# BATCH PROCESSING WITH SAGELABELS FORMAT
# =============================================================================

#' Create sageLabels Output for All Topics with File Saving
process_all_sagelabels_format <- function(topic_model, n_terms = 5,
                                          save_file = TRUE,
                                          filename = "stm_all_topics_sagelabels.txt") {
  total_topics <- topic_model$settings$dim$K
  cat("Creating sageLabels format for", total_topics, "topics with", n_terms, "terms each...\n\n")

  if (save_file) {
    save_sagelabels_format(topic_model, filename, n_terms, topics = NULL)
    cat("\n")
  }

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
#' Creates separate CSV files for probability and score matrices.
save_sagelabels_matrices <- function(topic_model, n_terms = 5, prefix = "stm_terms") {
  cat("Extracting matrices in sageLabels format...\n")

  sage_result <- sageLabels(topic_model, n = n_terms)

  prob_matrix <- sage_result$marginal$prob
  score_matrix <- sage_result$marginal$score

  prob_df <- as.data.frame(prob_matrix)
  score_df <- as.data.frame(score_matrix)

  prob_df$topic <- 1:nrow(prob_df)
  score_df$topic <- 1:nrow(score_df)

  prob_df <- prob_df[, c("topic", paste0("V", 1:n_terms))]
  score_df <- score_df[, c("topic", paste0("V", 1:n_terms))]

  colnames(prob_df) <- c("topic", paste0("term_", 1:n_terms))
  colnames(score_df) <- c("topic", paste0("term_", 1:n_terms))

  prob_file <- paste0(prefix, "_probability_matrix.csv")
  score_file <- paste0(prefix, "_score_matrix.csv")

  write.csv(prob_df, prob_file, row.names = FALSE)
  write.csv(score_df, score_file, row.names = FALSE)

  cat("Saved probability matrix:", prob_file, "\n")
  cat("Saved score matrix:", score_file, "\n")

  return(list(probability = prob_df, score = score_df))
}


# =============================================================================
#### SECTION: Term Overlap Analysis ####
# =============================================================================

# =============================================================================
# SUMMARY STATISTICS FUNCTION
# =============================================================================

#' Generate Summary Statistics for Topic Terms
#' Shows overlap between probability and score methods, unique terms, etc.
#'
#' intersect() finds common elements (Python: set(a) & set(b))
#' setdiff() finds elements in one set but not the other (Python: set(a) - set(b))
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
#### SECTION: Topic Terms Usage Examples ####
# =============================================================================

# =============================================================================
# EXECUTION EXAMPLES
# =============================================================================

cat("=============================================================================\n")
cat("STM TOPIC TERMS EXTRACTION - EXACT SAGELABELS FORMAT\n")
cat("=============================================================================\n")
cat("Create output exactly like sageLabels console display with Marginal Prob and Score.\n\n")

cat("SAGELABELS FORMAT OPTIONS:\n\n")

cat("OPTION 1: Display all topics in sageLabels format\n")
cat("process_all_sagelabels_format(topic_model_prev, n_terms = 5)\n")
cat("# Shows preview + saves to file: stm_all_topics_sagelabels.txt\n\n")

cat("OPTION 2: View specific topics in console\n")
cat("print_sagelabels_format(topic_model_prev, n_terms = 5, topics = c(1, 5, 10, 35))\n\n")

cat("OPTION 3: Save specific topics to file\n")
cat("save_sagelabels_format(topic_model_prev, 'selected_topics.txt', n_terms = 5, topics = 1:20)\n\n")

cat("=============================================================================\n")


###############################################################################
#### PART 5: CO-OCCURRENCE NETWORK ANALYSIS                                ####
###############################################################################


# =============================================================================
#### SECTION: Network Analysis Libraries ####
# =============================================================================

# =============================================================================
# STM CO-OCCURRENCE NETWORK ANALYSIS WITH SAGELABELS
# =============================================================================
#
# PURPOSE: Create co-occurrence networks for STM topic models using different
#          word ranking methods (FREX, Score, Probability, Lift)
#
# OVERVIEW:
# This section creates word co-occurrence networks that show which topic-defining
# words tend to appear together in documents. The innovation is using STM's
# sophisticated term-ranking methods (rather than simple frequency) to select
# which words to include in the network.
#
# WHAT IS A CO-OCCURRENCE NETWORK?
# A co-occurrence network is a graph where:
#   - Nodes = words (sized by frequency)
#   - Edges = connections between words that appear in the same documents
#   - Edge weight = how often the word pair co-occurs
# This reveals semantic clusters and relationships within topics.
#
# In Python, you would use:
#   - networkx for graph construction and analysis
#   - matplotlib or pyvis for network visualization
#   - scikit-learn's CountVectorizer for the document-term matrix
# =============================================================================

# Load required libraries for network analysis
library(quanteda)    # Text processing and DTM creation
# quanteda is a comprehensive NLP toolkit for R. Key functions used:
#   corpus() - create a text corpus
#   tokens() - tokenize text
#   dfm()    - create a document-feature matrix
# In Python: from sklearn.feature_extraction.text import CountVectorizer

library(igraph)      # Network analysis and graph objects
# igraph is the standard R package for network/graph analysis.
#   graph_from_data_frame() - create graph from edge list
#   V() and E() - access vertices and edges
#   degree() - count connections per node
# In Python: import networkx as nx; G = nx.from_pandas_edgelist(df)

library(ggraph)      # Grammar of graphics for network visualization
# ggraph extends ggplot2 for network layouts. Key geoms:
#   ggraph() - initialize a network plot with a layout algorithm
#   geom_edge_link() - draw edges
#   geom_node_point() - draw nodes
#   geom_node_text() - label nodes
# In Python: nx.draw() or pyvis for interactive graphs

library(tidyverse)   # Data manipulation (dplyr, tidyr, ggplot2)
library(stm)         # Structural Topic Modeling


# =============================================================================
#### SECTION: STM Term Extraction Function ####
# =============================================================================

# =============================================================================
# TERM EXTRACTION FROM STM MODELS
# =============================================================================

#' Extract Top Terms from STM Model Using Different Ranking Methods
#'
#' RANKING METHODS EXPLAINED:
#'   FREX (FREquency + EXclusivity): Harmonic mean of a word's frequency in
#'     the topic and its exclusivity to that topic. Best for interpretable
#'     topic labels because it avoids common words shared across topics.
#'   Score: Similar to FREX but with different weighting formula.
#'   Probability: Raw P(word | topic). May include common "filler" words.
#'   Lift: Ratio of P(word | topic) to P(word | corpus). Emphasizes words
#'     that are much more common in this topic than overall.
#'
#' @param topic_model STM model object
#' @param topic_num Integer: which topic (1 to K)
#' @param n_words Integer: how many top terms
#' @param method Character: "frex", "score", "prob", or "lift"
#' @return Character vector of top terms

get_stm_terms <- function(topic_model, topic_num, n_words, method = "frex") {

  # sageLabels() returns a nested list with $marginal containing matrices for
  # each ranking method. Each matrix has rows = topics, columns = term ranks.
  sage_result <- sageLabels(topic_model, n = n_words)

  # switch() is R's equivalent of Python's match/case or if/elif chain.
  # It selects the appropriate matrix based on the method argument.
  terms <- switch(method,
                  "frex" = sage_result$marginal$frex[topic_num, ],
                  "score" = sage_result$marginal$score[topic_num, ],
                  "prob" = sage_result$marginal$prob[topic_num, ],
                  "lift" = sage_result$marginal$lift[topic_num, ],
                  stop("Method must be one of: frex, score, prob, lift")
  )

  terms <- as.character(terms)
  terms <- terms[!is.na(terms) & terms != ""]

  return(terms)
}


# =============================================================================
#### SECTION: Document-Term Matrix Creation ####
# =============================================================================

# =============================================================================
# DOCUMENT-TERM MATRIX CREATION
# =============================================================================

#' Create Document-Term Matrix from Raw Text
#'
#' This function preprocesses raw text and creates a quanteda dfm (document-
#' feature matrix) for co-occurrence analysis.
#'
#' PREPROCESSING PIPELINE:
#'   1. corpus() - Wrap texts in a quanteda corpus object
#'   2. tokens() - Tokenize (split into words), remove punctuation and numbers
#'   3. tokens_tolower() - Convert to lowercase
#'   4. tokens_remove(stopwords("en")) - Remove common English stopwords
#'   5. tokens_keep(min_nchar=3) - Remove very short words (< 3 chars)
#'   6. dfm() - Convert tokens to a sparse document-feature matrix
#'   7. dfm_trim() - Remove terms appearing fewer than 2 times
#'
#' In Python:
#'   from sklearn.feature_extraction.text import CountVectorizer
#'   vectorizer = CountVectorizer(stop_words='english', min_df=2,
#'                                 token_pattern=r'\b[a-z]{3,}\b')
#'   dtm = vectorizer.fit_transform(texts)

make_dtm <- function(texts) {
  corpus(texts) %>%                                    # Create quanteda corpus
    tokens(remove_punct = TRUE, remove_numbers = TRUE) %>%  # Tokenize and clean
    tokens_tolower() %>%                               # Convert to lowercase
    tokens_remove(stopwords("en")) %>%                # Remove English stopwords
    tokens_keep(min_nchar = 3) %>%                    # Keep words >= 3 characters
    dfm() %>%                                         # Create document-feature matrix
    dfm_trim(min_termfreq = 2, min_docfreq = 1)       # Remove very rare terms
}


# =============================================================================
#### SECTION: Network Creation and Visualization ####
# =============================================================================

# =============================================================================
# NETWORK CREATION AND VISUALIZATION
# =============================================================================

#' Create Co-occurrence Network for Single Topic
#'
#' ALGORITHM OVERVIEW:
#'   1. Extract the 100 documents most strongly associated with the topic
#'      (highest theta values)
#'   2. Create a document-term matrix from those documents
#'   3. Use STM-ranked terms (FREX/Score/etc.) to filter the DTM
#'   4. Compute co-occurrence: t(DTM) %*% DTM gives a term-term matrix
#'      where cell [i,j] = number of documents containing both term i and j
#'   5. Convert to edge list and build an igraph network
#'   6. Visualize with ggraph using Fruchterman-Reingold force-directed layout
#'   7. Save as PNG and EPS (for journal publication)

make_network <- function(topic_model, text_data, topic_num, n_words, method) {

  # STEP 1: Get top documents for this specific topic.
  # topic_model$theta[, topic_num] is the column of document-topic proportions.
  topic_weights <- topic_model$theta[, topic_num]
  top_doc_indices <- order(topic_weights, decreasing = TRUE)[1:100]  # Top 100 docs

  topic_texts <- text_data$text[top_doc_indices]

  # Filter out invalid documents (NA, empty, or too short to be meaningful)
  valid_texts <- !is.na(topic_texts) & nchar(topic_texts) > 10
  topic_texts <- topic_texts[valid_texts]

  if (length(topic_texts) < 5) {
    warning("Topic ", topic_num, " has insufficient documents (", length(topic_texts), ")")
    return(NULL)
  }

  # STEP 2: Create document-term matrix from topic documents
  dtm <- make_dtm(topic_texts)

  # nfeat() from quanteda returns the number of features (terms) in the DTM.
  if (nfeat(dtm) < 5) {
    warning("Topic ", topic_num, " has insufficient terms after preprocessing")
    return(NULL)
  }

  # STEP 3: Get STM-ranked terms and filter DTM.
  # KEY INNOVATION: Instead of using the most frequent terms in the corpus,
  # we use STM's ranking methods to select topic-defining terms. This focuses
  # the network on semantically meaningful vocabulary.
  stm_terms <- get_stm_terms(topic_model, topic_num, n_words, method)

  # intersect() finds terms that exist in both the STM rankings and the DTM.
  available_terms <- intersect(stm_terms, colnames(dtm))

  if (length(available_terms) < 3) {
    warning("Topic ", topic_num, " has too few available terms (", length(available_terms), ")")
    return(NULL)
  }

  # Filter DTM to only include STM-selected terms
  dtm_filtered <- dtm[, available_terms]

  # STEP 4: Calculate co-occurrence matrix.
  # MATHEMATICAL EXPLANATION:
  #   DTM is a (documents x terms) matrix where DTM[d,t] = count of term t in doc d.
  #   t(DTM) is (terms x documents).
  #   t(DTM) %*% DTM is (terms x terms): cell [i,j] = sum over all docs of
  #     DTM[d,i] * DTM[d,j], which counts how often terms i and j co-occur.
  #
  # In Python: cooccur = dtm.T @ dtm (with scipy sparse matrices)
  cooccur_matrix <- t(dtm_filtered) %*% dtm_filtered

  # STEP 5: Convert to edge list format for network construction.
  cooccur_df <- as.data.frame(as.matrix(cooccur_matrix))
  cooccur_df$term1 <- rownames(cooccur_df)

  # pivot_longer() reshapes from wide (term1, termA, termB, ...) to long
  # (term1, term2, weight) format.
  edge_list <- cooccur_df %>%
    pivot_longer(-term1, names_to = "term2", values_to = "weight") %>%
    filter(
      term1 != term2,      # Remove self-connections
      weight >= 1,         # Minimum co-occurrence threshold
      term1 < term2        # Remove duplicate pairs (keep only A-B, not B-A)
    ) %>%
    arrange(desc(weight))

  if (nrow(edge_list) == 0) {
    warning("Topic ", topic_num, " has no word co-occurrences")
    return(NULL)
  }

  # STEP 6: Create igraph network object.
  # graph_from_data_frame() builds a graph from an edge list.
  # In Python: G = nx.from_pandas_edgelist(edge_list, 'term1', 'term2', 'weight')
  network_graph <- graph_from_data_frame(edge_list, directed = FALSE)

  # Validate the graph
  if (is.null(network_graph) || vcount(network_graph) == 0) {
    warning("Topic ", topic_num, " failed to create valid network graph")
    return(NULL)
  }

  # Add node attributes: word frequency and degree (number of connections).
  # V() accesses the vertices (nodes) of the graph.
  # In Python: nx.set_node_attributes(G, freq_dict, 'frequency')
  word_frequencies <- colSums(dtm_filtered)
  node_names <- V(network_graph)$name
  V(network_graph)$frequency <- word_frequencies[node_names]

  # Handle missing frequencies with a default value
  missing_freq <- is.na(V(network_graph)$frequency)
  if (any(missing_freq)) {
    V(network_graph)$frequency[missing_freq] <- 1
  }

  V(network_graph)$degree <- degree(network_graph)

  # STEP 7: Create visualization.
  # Define a clean, professional color scheme.
  node_color <- "steelblue"
  edge_color <- "gray60"
  text_color <- "black"

  # Inner function to create plots at different font sizes.
  create_plot <- function(font_size) {

    if (is.null(network_graph) || vcount(network_graph) == 0 || ecount(network_graph) == 0) {
      warning("Cannot create plot - invalid network graph")
      return(NULL)
    }

    # tryCatch provides error handling (like Python's try/except).
    plot_result <- tryCatch({
      # ggraph() initializes a network plot with a layout algorithm.
      # "fr" = Fruchterman-Reingold: a force-directed layout where connected
      # nodes are attracted and unconnected nodes are repelled, creating
      # intuitive spatial clusters.
      # In Python: pos = nx.spring_layout(G); nx.draw(G, pos)
      ggraph(network_graph, layout = "fr") +
        geom_edge_link(
          aes(width = weight, alpha = weight),
          color = edge_color,
          show.legend = TRUE
        ) +
        geom_node_point(
          aes(size = frequency),
          color = node_color,
          alpha = 0.8,
          show.legend = TRUE
        ) +
        # repel = TRUE uses the ggrepel algorithm to prevent label overlap.
        geom_node_text(
          aes(label = name),
          size = font_size * 0.25,
          color = text_color,
          repel = TRUE,
          point.padding = unit(0.3, "lines"),
          max.overlaps = 20
        ) +
        scale_edge_width_continuous(range = c(0.3, 3), guide = "none") +
        scale_edge_alpha_continuous(range = c(0.4, 0.9), guide = "none") +
        scale_size_continuous(range = c(2, 10), name = "Word\nFrequency") +
        theme_graph(base_size = font_size) +
        theme(
          legend.position = "bottom",
          legend.direction = "horizontal",
          legend.box = "horizontal",
          legend.title = element_text(size = font_size * 0.9),
          legend.text = element_text(size = font_size * 0.8)
        )
    }, error = function(e) {
      # Fallback to a simpler circular layout if FR fails
      warning("Fruchterman-Reingold layout failed, trying circle layout: ", e$message)
      tryCatch({
        ggraph(network_graph, layout = "circle") +
          geom_edge_link(aes(width = weight), color = edge_color) +
          geom_node_point(aes(size = frequency), color = node_color, alpha = 0.8) +
          geom_node_text(aes(label = name), size = font_size * 0.25, color = text_color) +
          scale_edge_width_continuous(range = c(0.3, 3), guide = "none") +
          scale_size_continuous(range = c(2, 10), name = "Word\nFrequency") +
          theme_graph(base_size = font_size) +
          theme(
            legend.position = "bottom",
            legend.title = element_text(size = font_size * 0.9),
            legend.text = element_text(size = font_size * 0.8)
          )
      }, error = function(e2) {
        warning("All layouts failed for topic ", topic_num, ": ", e2$message)
        return(NULL)
      })
    })

    return(plot_result)
  }

  # STEP 8: Save files in both font sizes and formats.
  base_directory <- paste0("stm_networks_", method)
  dir.create(base_directory, showWarnings = FALSE, recursive = TRUE)
  dir.create(file.path(base_directory, "plots_font24"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "plots_font36"), showWarnings = FALSE)
  dir.create(file.path(base_directory, "data"), showWarnings = FALSE)

  # Generate plots at two font sizes for different use cases:
  #   Font 24: presentations and screen viewing
  #   Font 36: posters and large-format printing
  font_sizes <- c(24, 36)
  plots_created <- list()

  for (font_size in font_sizes) {
    network_plot <- create_plot(font_size)

    if (is.null(network_plot)) {
      warning("Skipping font size ", font_size, " for topic ", topic_num, " due to plot creation error")
      next
    }

    # Save as high-resolution PNG
    plot_filename <- file.path(
      base_directory, paste0("plots_font", font_size),
      paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_font", font_size, ".png")
    )

    tryCatch({
      ggsave(
        filename = plot_filename,
        plot = network_plot,
        width = 12, height = 10,
        dpi = 300,
        bg = "white"
      )
    }, error = function(e) {
      warning("Failed to save PNG for topic ", topic_num, " font ", font_size, ": ", e$message)
    })

    # Save as EPS for journal publication.
    # EPS (Encapsulated PostScript) is a vector format preferred by many
    # academic journals because it scales without quality loss.
    eps_filename <- file.path(
      base_directory, paste0("eps_font", font_size),
      paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_font", font_size, ".eps")
    )

    tryCatch({
      ggsave(
        filename = eps_filename,
        plot = network_plot,
        width = 7, height = 5.5,   # Journal double-column dimensions
        units = "in",
        device = "eps",
        bg = "white"
      )
    }, error = function(e) {
      warning("Failed to save EPS for topic ", topic_num, " font ", font_size, ": ", e$message)
    })

    plots_created[[paste0("font", font_size)]] <- network_plot
  }

  if (length(plots_created) == 0) {
    warning("No plots could be created for topic ", topic_num)
    return(NULL)
  }

  # Save edge list data as CSV for reproducibility and further analysis.
  data_filename <- file.path(
    base_directory, "data",
    paste0("topic_", sprintf("%02d", topic_num), "_", n_words, "words_edges.csv")
  )

  edge_list_annotated <- edge_list %>%
    mutate(
      topic_number = topic_num,
      ranking_method = method,
      word_count = n_words,
      extraction_date = Sys.Date()
    )

  write.csv(edge_list_annotated, data_filename, row.names = FALSE)

  cat("Topic", topic_num, ":", length(available_terms), "terms,",
      ecount(network_graph), "connections,", length(plots_created), "successful plots\n")

  return(list(
    graph = network_graph,
    plots = plots_created,
    terms = available_terms,
    edges = nrow(edge_list),
    method = method,
    font_sizes_created = c(24, 36),
    successful_plots = length(plots_created)
  ))
}


# =============================================================================
#### SECTION: Batch Processing Function ####
# =============================================================================

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

#' Process Multiple Topics and Word Counts
#'
#' Creates networks for all topic/word-count combinations with progress
#' reporting, error handling, and periodic garbage collection.
#' %% is the modulo operator (Python: %). topic_num %% 10 == 0 means
#' "every 10th topic."

process_networks <- function(topic_model, text_data, method = "frex",
                             topics = 1:70, word_counts = c(15, 30, 50)) {

  # inherits() checks if an object belongs to a specific class.
  # In Python: isinstance(topic_model, STM)
  if (!inherits(topic_model, "STM")) {
    stop("topic_model must be an STM object from stm() function")
  }

  if (!"text" %in% names(text_data)) {
    stop("text_data must have a 'text' column")
  }

  if (!method %in% c("frex", "score", "prob", "lift")) {
    stop("method must be one of: frex, score, prob, lift")
  }

  total_combinations <- length(topics) * length(word_counts)
  success_count <- 0
  failed_topics <- c()

  cat("=============================================================================\n")
  cat("STM CO-OCCURRENCE NETWORK BATCH PROCESSING\n")
  cat("=============================================================================\n")
  cat("Settings:\n")
  cat("  Method:", toupper(method), "\n")
  cat("  Topics:", length(topics), "(", min(topics), "to", max(topics), ")\n")
  cat("  Word counts:", paste(word_counts, collapse = ", "), "\n")
  cat("  Font sizes: 24 and 36 (both generated automatically)\n")
  cat("  Total networks:", total_combinations * 4, "(", total_combinations, "x 2 font sizes x 2 formats)\n")
  cat("  Output directory: stm_networks_", method, "/\n")
  cat("=============================================================================\n\n")

  start_time <- Sys.time()

  for (topic_num in topics) {

    if (topic_num %% 10 == 0 || topic_num == min(topics)) {
      cat("Processing Topic", topic_num, "of", max(topics), "...\n")
    }

    topic_success <- FALSE

    for (n_words in word_counts) {
      result <- tryCatch({
        make_network(topic_model, text_data, topic_num, n_words, method)
      }, error = function(e) {
        warning("Error processing Topic ", topic_num, " (", n_words, " words): ", e$message)
        NULL
      })

      if (!is.null(result)) {
        success_count <- success_count + 1
        topic_success <- TRUE
      }
    }

    if (!topic_success) {
      failed_topics <- c(failed_topics, topic_num)
    }

    # gc() triggers garbage collection to free memory.
    # In Python: import gc; gc.collect()
    if (topic_num %% 20 == 0) {
      gc(verbose = FALSE)
    }
  }

  end_time <- Sys.time()
  processing_time <- as.numeric(end_time - start_time, units = "mins")
  success_rate <- round((success_count / total_combinations) * 100, 1)
  total_files_created <- success_count * 4

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

  cat("\n=============================================================================\n")
  cat("PROCESSING COMPLETE!\n")
  cat("=============================================================================\n")
  cat("Results:\n")
  cat("  Successful networks:", success_count, "of", total_combinations, "\n")
  cat("  Total files created:", total_files_created, "(", success_count, "networks x 2 fonts x 2 formats)\n")
  cat("  Success rate:", success_rate, "%\n")
  cat("  Processing time:", round(processing_time, 2), "minutes\n")
  cat("  Speed:", round(success_count / processing_time, 1), "networks/minute\n")

  if (length(failed_topics) > 0) {
    cat("  Failed topics:", paste(failed_topics, collapse = ", "), "\n")
  }

  cat("  Output location: stm_networks_", method, "/\n")
  cat("=============================================================================\n")

  return(summary_stats)
}


# =============================================================================
#### SECTION: Testing and Validation Functions ####
# =============================================================================

# =============================================================================
# TESTING AND VALIDATION FUNCTIONS
# =============================================================================

#' Test Term Extraction for Multiple Topics
#' Verifies that different topics return different terms (no indexing bugs).

test_term_extraction <- function(topic_model, test_topics = c(1, 5, 10, 15, 20),
                                 method = "frex", n_words = 8) {

  cat("=============================================================================\n")
  cat("TESTING STM TERM EXTRACTION -", toupper(method), "METHOD\n")
  cat("=============================================================================\n")
  cat("This test verifies that different topics return different terms.\n")
  cat("If all topics show the same terms, there's an indexing problem.\n\n")

  for (topic_num in test_topics) {
    terms <- get_stm_terms(topic_model, topic_num, n_words, method)
    cat("Topic", sprintf("%2d", topic_num), ":", paste(head(terms, 6), collapse = ", "))
    if (length(terms) > 6) cat(", ...")
    cat("\n")
  }

  cat("\nIf you see different terms for each topic, extraction is working correctly!\n")
  cat("If all topics show identical terms, there's still an indexing issue.\n")
  cat("=============================================================================\n")
}

#' Test Single Network Creation
#' Quick test on one topic before running the full batch.

test_single_network <- function(topic_model, text_data, topic_num = 1,
                                method = "frex", n_words = 30) {

  cat("Testing network creation for Topic", topic_num, "with", method, "rankings...\n")

  result <- make_network(topic_model, text_data, topic_num, n_words, method)

  if (!is.null(result)) {
    cat("Success! Network created with:\n")
    cat("  Terms:", length(result$terms), "\n")
    cat("  Edges:", result$edges, "\n")
    cat("  Font sizes: 24 and 36 (PNG + EPS formats)\n")
    cat("  Files saved in: stm_networks_", method, "/\n")
    cat("  Top terms:", paste(head(result$terms, 6), collapse = ", "), "\n")
  } else {
    cat("Failed to create network for Topic", topic_num, "\n")
  }

  return(result)
}


# =============================================================================
#### SECTION: Network Analysis Usage Examples ####
# =============================================================================

# =============================================================================
# EXECUTION AND USAGE EXAMPLES
# =============================================================================

cat("=============================================================================\n")
cat("STM CO-OCCURRENCE NETWORK ANALYSIS\n")
cat("=============================================================================\n")
cat("This script creates word co-occurrence networks from STM topic models using\n")
cat("sophisticated ranking methods (FREX, Score, Probability, Lift) rather than\n")
cat("simple frequency counts. Each method reveals different aspects of topics:\n\n")

cat("RANKING METHODS:\n")
cat("  FREX: Frequent + Exclusive (best for topic interpretation)\n")
cat("  Score: Weighted probability + exclusivity\n")
cat("  Probability: Raw word probabilities within topics\n")
cat("  Lift: Topic distinctiveness vs. overall corpus\n\n")

cat("USAGE STEPS:\n")
cat("=============================================================================\n\n")

cat("STEP 1: Test term extraction\n")
cat("test_term_extraction(topic_model_prev, test_topics = c(1, 5, 10, 15, 20), method = 'frex')\n\n")

cat("STEP 2: Test single network creation\n")
cat("test_single_network(topic_model_prev, out$meta, topic_num = 1, method = 'frex')\n\n")

cat("STEP 3: Process all networks\n")
cat("results_frex <- process_networks(topic_model_prev, out$meta, method = 'frex')\n")
cat("results_score <- process_networks(topic_model_prev, out$meta, method = 'score')\n\n")

cat("OUTPUT STRUCTURE:\n")
cat("stm_networks_[method]/\n")
cat("  plots_font24/          # PNG visualizations with font size 24\n")
cat("  plots_font36/          # PNG visualizations with font size 36\n")
cat("  eps_font24/            # EPS files with font size 24 (journal publication)\n")
cat("  eps_font36/            # EPS files with font size 36 (journal publication)\n")
cat("  data/                  # Edge lists (CSV files)\n")
cat("  processing_report.txt  # Summary statistics\n\n")
cat("=============================================================================\n")
