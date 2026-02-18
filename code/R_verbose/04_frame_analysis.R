# =============================================================================
# SCRIPT:   04_frame_analysis.R
# PURPOSE:  Perform a deep-dive frame analysis on the top 100 documents from
#           STM Topic 3 ("Work and Sleep"). Frames are predefined dictionaries
#           of keywords representing different rhetorical or conceptual lenses
#           (health, productivity, moral, scientific, economic, lifestyle).
#           The script counts keyword matches per frame per document, examines
#           temporal trends, computes frame co-occurrence and dominance, and
#           generates publication-quality visualizations.
#
# AUTHOR:   Joon Chung
# EMAIL:    Contact: see README
# AFFIL:    The University of Miami, Miller School of Medicine
#           The Department of Informatics and Health Data Science
#
# INPUTS:
#   - top_100_topic3: A data frame (must exist in the R environment) containing
#     the top 100 documents for Topic 3 with at least a 'text' column and
#     optional 'year', 'rank', 'doc_index', 'source' columns.
#
# OUTPUTS:
#   - frame_analysis_results/ directory containing:
#     - PNG visualizations (distribution, prevalence, temporal trends, etc.)
#     - CSV summaries (frame statistics, decade breakdowns, trend tests)
#     - Frame dominance by article CSV
#
# REQUIRED PACKAGES:
#   - tidyverse  : Data wrangling (dplyr, tidyr, stringr) and plotting (ggplot2)
#   - ggplot2    : Grammar of graphics visualization (loaded via tidyverse)
#   - viridis    : Color-blind-friendly color palettes for plots
#   - patchwork  : Combining ggplot objects (loaded but not directly used here)
#   - corrplot   : Correlation matrix visualization (loaded for potential use)
#
# STATISTICAL CONCEPTS:
#   Frame Analysis: A qualitative/quantitative content analysis method that
#   examines how issues are "framed" in media coverage. Each frame represents
#   a particular lens (e.g., health, economic, moral) through which an issue
#   is presented. This script operationalizes frames via keyword dictionaries
#   and counts matches in article text.
#
# NOTES FOR PYTHON USERS:
#   - str_count() from stringr counts regex matches in text, similar to
#     Python's len(re.findall(pattern, text)) or sum(1 for _ in re.finditer(...))
#   - pivot_longer() reshapes data from wide to long format, like
#     pd.melt() in pandas.
#   - rowwise() + c_across() applies operations across columns row by row,
#     similar to df.apply(func, axis=1) in pandas.
#   - case_when() is R's vectorized if-else chain, similar to np.select()
#     or pd.cut() in Python.
#   - The formula notation in lm(frame_score ~ year) fits a linear regression
#     of frame_score on year, like sklearn.linear_model.LinearRegression() or
#     statsmodels.OLS.from_formula('frame_score ~ year', data=df).
# =============================================================================


# =============================================================================
#### SECTION: Load Required Libraries ####
# =============================================================================

# =============================================================================
# DEEP DIVE FRAME ANALYSIS WITH TEMPORAL TRENDS
# =============================================================================

# tidyverse: meta-package loading dplyr, ggplot2, tidyr, stringr, etc.
library(tidyverse)
# ggplot2: already loaded via tidyverse, but explicitly listed for clarity.
library(ggplot2)
# viridis: provides color-blind-friendly color scales (viridis, magma, plasma).
# scale_fill_viridis_d() and scale_color_viridis_d() use these palettes.
# In Python: matplotlib's plt.cm.viridis or seaborn palettes.
library(viridis)
# patchwork: allows combining ggplot objects with + and / operators.
# (e.g., plot1 + plot2 creates a side-by-side layout)
# In Python: matplotlib subplots or fig.add_subplot().
library(patchwork)
# corrplot: specialized package for visualizing correlation matrices.
# In Python: seaborn.heatmap(corr_matrix).
library(corrplot)

cat("=== DEEP DIVE FRAME ANALYSIS ===\n")


# =============================================================================
#### SECTION 1: Enhanced Frame Definitions and Calculation ####
# =============================================================================

# Frame definitions are dictionaries (R lists) mapping frame names to keyword
# vectors. Each keyword is searched in the article text using regex matching.
# The count of matches serves as the "frame score" for that article.
#
# WHY KEYWORD-BASED FRAMES?
# This is a dictionary-based approach to content analysis. Each frame has a
# manually curated list of keywords that indicate the presence of that frame
# in media coverage. Higher scores mean the article uses more language
# associated with that particular framing.
#
# ALTERNATIVE APPROACHES:
# In Python, you might use scikit-learn's CountVectorizer with a custom
# vocabulary, or spaCy's PhraseMatcher for more sophisticated matching.

cat("\n1. ENHANCED FRAME DEFINITIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Define comprehensive frames with more terms.
# list() creates a named list (similar to a Python dictionary).
# Each element is a character vector of keywords for that frame.
frames_detailed <- list(
  health = c("health", "wellness", "medical", "doctor", "disease", "risk", "benefits",
             "harmful", "illness", "sick", "recovery", "healing", "treatment", "therapy",
             "medication", "symptoms", "diagnosis", "patient", "clinic", "hospital"),

  productivity = c("efficiency", "performance", "output", "profit", "competitive",
                   "productive", "effective", "success", "achievement", "goals", "results",
                   "optimization", "maximize", "improve", "enhance", "boost", "increase",
                   "economic", "business", "profit", "revenue", "growth"),

  moral = c("lazy", "disciplined", "responsible", "work ethic", "character", "virtue",
            "vice", "dedication", "commitment", "diligent", "hardworking", "slacker",
            "irresponsible", "duty", "obligation", "values", "integrity", "shame",
            "guilt", "pride", "honor", "respect"),

  scientific = c("research", "study", "evidence", "data", "findings", "scientist",
                 "experiment", "analysis", "investigation", "hypothesis", "theory",
                 "methodology", "peer-reviewed", "journal", "publication", "statistics",
                 "correlation", "causation", "sample", "survey", "clinical", "trial"),

  economic = c("cost", "expensive", "cheap", "money", "financial", "budget", "savings",
               "investment", "return", "roi", "economics", "market", "competition",
               "industry", "corporate", "business", "commerce", "trade", "capitalism"),

  lifestyle = c("balance", "wellbeing", "quality of life", "happiness", "satisfaction",
                "fulfillment", "personal", "individual", "choice", "preference", "comfort",
                "convenience", "leisure", "recreation", "hobby", "family", "relationships")
)

# Remove any existing frame columns from a previous run to ensure clean calculation.
# grep("_frame$|_score$", ...) uses regex to find column names ending in
# "_frame" or "_score". The $ anchors the match to the end of the string.
# value = TRUE returns the matching names (not indices).
existing_frame_cols <- grep("_frame$|_score$", names(top_100_topic3), value = TRUE)
# Exclude sentiment_score from removal (it may have been computed separately)
existing_frame_cols <- existing_frame_cols[existing_frame_cols != "sentiment_score"]

if (length(existing_frame_cols) > 0) {
  # select(-all_of(...)) drops the specified columns from the data frame.
  # all_of() is used within tidyselect to reference a character vector of names.
  # The negative sign (-) means "remove these columns."
  top_100_topic3 <- top_100_topic3 %>% select(-all_of(existing_frame_cols))
  cat("Removed existing frame columns:", paste(existing_frame_cols, collapse = ", "), "\n")
}

# Calculate frame scores for each article.
# For each frame, we create a regex pattern that matches any of the keywords
# (joined by | which means OR in regex), then count occurrences in each article.
frame_results <- data.frame(
  frame = character(),
  total_matches = numeric(),
  articles_with_frame = numeric(),
  mean_score = numeric(),
  max_score = numeric(),
  stringsAsFactors = FALSE
)

for (frame_name in names(frames_detailed)) {
  frame_words <- frames_detailed[[frame_name]]
  # paste(frame_words, collapse = "|") creates a single regex pattern like:
  # "health|wellness|medical|doctor|..." (matches any of these words).
  # In Python: pattern = "|".join(frame_words)
  pattern <- paste(frame_words, collapse = "|")

  # str_count() from stringr counts the number of regex matches in each string.
  # tolower() converts text to lowercase for case-insensitive matching.
  # In Python: [len(re.findall(pattern, text.lower())) for text in texts]
  scores <- str_count(tolower(top_100_topic3$text), pattern)
  # Add the scores as a new column named "{frame_name}_frame"
  # [[...]] notation creates or assigns to a column by name (dynamic column naming).
  top_100_topic3[[paste0(frame_name, "_frame")]] <- scores

  # Accumulate summary statistics using rbind() to append rows.
  # rbind() is like pd.concat() in pandas.
  frame_results <- rbind(frame_results, data.frame(
    frame = frame_name,
    total_matches = sum(scores),
    articles_with_frame = sum(scores > 0),
    mean_score = round(mean(scores), 2),
    max_score = max(scores),
    stringsAsFactors = FALSE
  ))

  cat("Frame:", frame_name, "- Total matches:", sum(scores),
      "- Articles with frame:", sum(scores > 0), "\n")
}

print(frame_results)


# =============================================================================
#### SECTION 2: Frame Comparison Analysis ####
# =============================================================================

cat("\n2. FRAME COMPARISON ANALYSIS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Identify frame columns for analysis (those ending in "_frame").
# grep() with value=TRUE returns matching column names.
frame_cols <- grep("_frame$", names(top_100_topic3), value = TRUE)
cat("Frame columns identified:", paste(frame_cols, collapse = ", "), "\n")

# Reshape data from wide to long format for grouped analysis.
# pivot_longer() converts multiple columns into key-value pairs:
#   BEFORE: rank, doc_index, year, health_frame, productivity_frame, ...
#   AFTER:  rank, doc_index, year, frame_type, frame_score
# any_of() is a tidyselect helper that selects columns if they exist (no error
# if missing). all_of() requires all columns to exist.
# str_remove() strips the "_frame" suffix for cleaner labels.
# In Python: pd.melt(df, id_vars=[...], value_vars=frame_cols)
frame_data <- top_100_topic3 %>%
  select(rank, doc_index, any_of("year"), any_of("source"), all_of(frame_cols)) %>%
  pivot_longer(cols = all_of(frame_cols), names_to = "frame_type", values_to = "frame_score") %>%
  mutate(frame_type = str_remove(frame_type, "_frame"))

# Compute summary statistics grouped by frame type.
# summarise() (or summarize()) collapses grouped data to one row per group.
# .groups = "drop" removes the grouping after summarizing.
frame_summary <- frame_data %>%
  group_by(frame_type) %>%
  summarise(
    total_score = sum(frame_score),
    mean_score = round(mean(frame_score), 3),
    median_score = median(frame_score),
    articles_with_frame = sum(frame_score > 0),
    percentage_articles = round(sum(frame_score > 0) / n() * 100, 1),
    max_score = max(frame_score),
    .groups = "drop"
  ) %>%
  arrange(desc(total_score))

cat("\nFrame Summary Statistics:\n")
print(frame_summary)


# =============================================================================
#### SECTION 3: Comprehensive Temporal Frame Analysis ####
# =============================================================================

# This section examines how frame usage changes over time (by year and decade).
# Only runs if a 'year' column exists in the data.

if ("year" %in% names(top_100_topic3)) {
  cat("\n3. TEMPORAL FRAME ANALYSIS\n")
  cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

  # 3.1 Frame evolution by year: average frame score per year per frame type.
  frame_temporal <- frame_data %>%
    filter(!is.na(year)) %>%
    group_by(year, frame_type) %>%
    summarise(
      mean_frame_score = mean(frame_score),
      total_frame_score = sum(frame_score),
      articles_with_frame = sum(frame_score > 0),
      .groups = "drop"
    )

  # 3.2 Frame evolution by decade.
  # floor(year/10)*10 converts a year to its decade (e.g., 1993 -> 1990).
  # In Python: (year // 10) * 10
  frame_decade <- top_100_topic3 %>%
    filter(!is.na(year)) %>%
    mutate(decade = floor(year/10)*10) %>%
    select(decade, all_of(frame_cols)) %>%
    pivot_longer(cols = all_of(frame_cols), names_to = "frame_type", values_to = "frame_score") %>%
    mutate(frame_type = str_remove(frame_type, "_frame")) %>%
    group_by(decade, frame_type) %>%
    summarise(
      mean_score = mean(frame_score),
      total_score = sum(frame_score),
      prevalence = sum(frame_score > 0) / n() * 100,
      .groups = "drop"
    )

  cat("Frame evolution by decade:\n")
  # pivot_wider() converts from long to wide format (opposite of pivot_longer).
  # In Python: df.pivot_table(index='decade', columns='frame_type', values='mean_score')
  decade_summary <- frame_decade %>%
    select(decade, frame_type, mean_score) %>%
    pivot_wider(names_from = frame_type, values_from = mean_score, values_fill = 0) %>%
    arrange(decade)

  print(decade_summary)

  # Also show prevalence by decade
  cat("\nFrame prevalence by decade (% of articles):\n")
  decade_prevalence <- frame_decade %>%
    select(decade, frame_type, prevalence) %>%
    pivot_wider(names_from = frame_type, values_from = prevalence, values_fill = 0) %>%
    arrange(decade)

  print(decade_prevalence)

  # 3.3 Frame dominance over time: what proportion of all frame mentions
  # in a given year belongs to each frame type.
  frame_dominance <- frame_temporal %>%
    group_by(year) %>%
    mutate(
      total_frames_year = sum(total_frame_score),
      # if_else() is a type-strict version of ifelse(). It requires both
      # outcomes to be the same type.
      frame_proportion = if_else(total_frames_year > 0, total_frame_score / total_frames_year * 100, 0)
    ) %>%
    ungroup()

  # 3.4 Statistical tests for temporal trends.
  # For each frame, fit a simple linear regression: frame_score ~ year
  # to test whether frame usage is significantly increasing or decreasing
  # over time. The slope indicates the direction and magnitude of change.
  cat("\nTesting for temporal trends in frame usage:\n")

  frame_trend_tests <- data.frame(
    frame_type = character(),
    slope = numeric(),
    p_value = numeric(),
    r_squared = numeric(),
    stringsAsFactors = FALSE
  )

  for (frame in unique(frame_data$frame_type)) {
    frame_subset <- frame_data %>%
      filter(frame_type == frame, !is.na(year))

    if (nrow(frame_subset) > 5) {  # Need minimum observations
      # tryCatch() is R's error-handling mechanism (like Python's try/except).
      # If lm() fails for any reason, the error is caught and reported.
      tryCatch({
        # lm() fits a linear model. frame_score ~ year means:
        #   frame_score = intercept + slope * year + error
        # coef(model)[2] extracts the slope coefficient.
        # summary()$coefficients[2, 4] extracts the p-value for the slope.
        # summary()$r.squared gives the R-squared (proportion of variance explained).
        # In Python: from sklearn.linear_model import LinearRegression
        #   model = LinearRegression().fit(X, y); slope = model.coef_[0]
        model <- lm(frame_score ~ year, data = frame_subset)
        model_summary <- summary(model)

        frame_trend_tests <- rbind(frame_trend_tests, data.frame(
          frame_type = frame,
          slope = round(coef(model)[2], 6),
          p_value = round(model_summary$coefficients[2, 4], 4),
          r_squared = round(model_summary$r.squared, 4),
          stringsAsFactors = FALSE
        ))
      }, error = function(e) {
        cat("Error analyzing frame:", frame, "-", e$message, "\n")
      })
    }
  }

  frame_trend_tests <- frame_trend_tests %>% arrange(p_value)
  print(frame_trend_tests)

  # 3.5 Frame correlation over time periods.
  # Compare how frames co-occur in early vs. late periods.
  # cor() computes the Pearson correlation matrix.
  # use = "complete.obs" handles missing values by using only complete cases.
  # In Python: df[frame_cols].corr()
  early_period <- top_100_topic3 %>% filter(year <= 2000)
  late_period <- top_100_topic3 %>% filter(year > 2000)

  if (nrow(early_period) > 10 && nrow(late_period) > 10) {
    early_corr <- early_period %>%
      select(all_of(frame_cols)) %>%
      cor(use = "complete.obs")

    late_corr <- late_period %>%
      select(all_of(frame_cols)) %>%
      cor(use = "complete.obs")

    cat("\nFrame correlations - Early period (<=2000):\n")
    print(round(early_corr, 3))

    cat("\nFrame correlations - Late period (>2000):\n")
    print(round(late_corr, 3))
  }
}


# =============================================================================
#### SECTION 4: Comprehensive Frame Visualizations ####
# =============================================================================

cat("\n4. CREATING FRAME VISUALIZATIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Store all plots in a named list for batch display and saving.
frame_plots <- list()

# 4.1 Frame distribution comparison (box plots).
# reorder(frame_type, frame_score, median) sorts frames by their median score.
# coord_flip() makes horizontal box plots (easier to read frame names).
# In Python: sns.boxplot(x='frame_score', y='frame_type', data=frame_data)
frame_plots$distribution <- frame_data %>%
  ggplot(aes(x = reorder(frame_type, frame_score, median), y = frame_score, fill = frame_type)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.5) +
  coord_flip() +
  scale_fill_viridis_d() +
  labs(title = "Frame Usage Distribution",
       subtitle = "Comparison across all frame types",
       x = "Frame Type", y = "Frame Score") +
  theme_minimal() +
  theme(legend.position = "none")

# 4.2 Frame prevalence (percentage of articles containing each frame).
frame_prevalence <- frame_data %>%
  group_by(frame_type) %>%
  summarise(
    prevalence = sum(frame_score > 0) / n() * 100,
    .groups = "drop"
  )

frame_plots$prevalence <- frame_prevalence %>%
  ggplot(aes(x = reorder(frame_type, prevalence), y = prevalence, fill = frame_type)) +
  geom_col(alpha = 0.8) +
  coord_flip() +
  scale_fill_viridis_d() +
  labs(title = "Frame Prevalence",
       subtitle = "Percentage of articles containing each frame",
       x = "Frame Type", y = "Prevalence (%)") +
  theme_minimal() +
  theme(legend.position = "none")

# 4.3 Temporal frame evolution (smooth trends over time).
if ("year" %in% names(top_100_topic3)) {

  # geom_smooth(method = "loess") fits a LOcally Estimated Scatterplot Smoothing
  # curve, which is a non-parametric regression that adapts to local patterns.
  # se = FALSE hides the confidence band for cleaner visualization.
  # In Python: sns.lmplot() or scipy.signal.savgol_filter() for smoothing.
  frame_plots$temporal_trends <- frame_temporal %>%
    ggplot(aes(x = year, y = mean_frame_score, color = frame_type)) +
    geom_smooth(method = "loess", se = FALSE, size = 1.2) +
    geom_point(alpha = 0.6, size = 1) +
    scale_color_viridis_d() +
    labs(title = "Frame Usage Evolution Over Time",
         subtitle = "Smoothed trends showing changing frame prominence",
         x = "Year", y = "Mean Frame Score", color = "Frame") +
    theme_minimal() +
    theme(legend.position = "bottom")

  # Frame dominance over time (proportional stacked area or line plot)
  frame_plots$dominance_trends <- frame_dominance %>%
    ggplot(aes(x = year, y = frame_proportion, color = frame_type)) +
    geom_smooth(method = "loess", se = FALSE, size = 1) +
    scale_color_viridis_d() +
    labs(title = "Frame Dominance Over Time",
         subtitle = "Relative proportion of each frame type by year",
         x = "Year", y = "Frame Proportion (%)", color = "Frame") +
    theme_minimal() +
    theme(legend.position = "bottom")

  # Decade comparison (grouped bar chart)
  frame_plots$decade_comparison <- frame_decade %>%
    ggplot(aes(x = factor(decade), y = mean_score, fill = frame_type)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(title = "Frame Usage by Decade",
         subtitle = "Mean frame scores across time periods",
         x = "Decade", y = "Mean Frame Score", fill = "Frame") +
    theme_minimal() +
    theme(legend.position = "bottom")

  # Frame prevalence over time
  frame_prevalence_time <- frame_data %>%
    filter(!is.na(year)) %>%
    mutate(decade = floor(year/10)*10) %>%
    group_by(decade, frame_type) %>%
    summarise(prevalence = sum(frame_score > 0) / n() * 100, .groups = "drop")

  frame_plots$prevalence_time <- frame_prevalence_time %>%
    ggplot(aes(x = factor(decade), y = prevalence, fill = frame_type)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_viridis_d() +
    labs(title = "Frame Prevalence by Decade",
         subtitle = "Percentage of articles containing each frame",
         x = "Decade", y = "Prevalence (%)", fill = "Frame") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# 4.4 Frame correlation heatmap.
# cor() computes pairwise Pearson correlations between frame score columns.
# expand.grid() creates all combinations of row and column names.
# In Python: corr_matrix = df[frame_cols].corr(); sns.heatmap(corr_matrix)
frame_correlation_data <- top_100_topic3 %>%
  select(all_of(frame_cols)) %>%
  cor(use = "complete.obs")

# Convert correlation matrix to long format for ggplot heatmap
frame_corr_long <- expand.grid(
  Frame1 = factor(rownames(frame_correlation_data), levels = rownames(frame_correlation_data)),
  Frame2 = factor(colnames(frame_correlation_data), levels = colnames(frame_correlation_data))
) %>%
  mutate(
    correlation = as.vector(frame_correlation_data),
    Frame1_clean = str_remove(as.character(Frame1), "_frame"),
    Frame2_clean = str_remove(as.character(Frame2), "_frame")
  )

# geom_tile() creates a heatmap (colored grid).
# scale_fill_gradient2() creates a diverging color scale (blue-white-red)
# centered at zero, which is standard for correlation heatmaps.
frame_plots$correlation_heatmap <- frame_corr_long %>%
  ggplot(aes(x = Frame1_clean, y = Frame2_clean, fill = correlation)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, name = "Correlation") +
  labs(title = "Frame Correlation Matrix",
       subtitle = "How different frames co-occur in articles",
       x = "Frame Type", y = "Frame Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 4.5 Frame intensity distribution (only for articles containing the frame).
frame_plots$intensity <- frame_data %>%
  filter(frame_score > 0) %>%  # Only articles with the frame
  ggplot(aes(x = frame_score, fill = frame_type)) +
  geom_histogram(bins = 15, alpha = 0.7) +
  facet_wrap(~frame_type, scales = "free") +
  scale_fill_viridis_d() +
  labs(title = "Frame Intensity Distribution",
       subtitle = "Distribution of frame scores when frame is present",
       x = "Frame Score", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")


# =============================================================================
#### SECTION 5: Advanced Frame Analysis ####
# =============================================================================

cat("\n5. ADVANCED FRAME ANALYSIS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# 5.1 Frame co-occurrence analysis.
# Convert frame scores to binary (0/1) to count how often frames co-occur.
# mutate_all() applies a function to every column.
# t() transposes a matrix. The matrix multiplication t(A) %*% A produces
# a co-occurrence matrix where cell [i,j] counts how many articles have
# both frame i and frame j present.
# In Python: binary = (df[frame_cols] > 0).astype(int)
#             cooccur = binary.T @ binary
frame_cooccurrence <- top_100_topic3 %>%
  select(all_of(frame_cols)) %>%
  mutate_all(~ ifelse(. > 0, 1, 0)) %>%  # Convert to binary
  as.matrix()

cooccur_matrix <- t(frame_cooccurrence) %*% frame_cooccurrence
diag(cooccur_matrix) <- 0  # Remove self-cooccurrence

cat("Frame co-occurrence counts (how often frames appear together):\n")
print(cooccur_matrix)

# 5.2 Frame dominance patterns.
# For each article, determine which frame has the highest score (dominant frame).
# rowwise() makes subsequent operations apply per-row (like pandas .apply(axis=1)).
# c_across() collects values across specified columns for the current row.
# which.max() returns the index of the maximum value.
frame_dominance_analysis <- top_100_topic3 %>%
  rowwise() %>%
  mutate(
    total_frame_score = sum(c_across(all_of(frame_cols))),
    dominant_frame = if_else(
      total_frame_score > 0,
      frame_cols[which.max(c_across(all_of(frame_cols)))],
      "none"
    )
  ) %>%
  ungroup() %>%
  mutate(dominant_frame = str_remove(dominant_frame, "_frame"))

cat("\nDominant frame distribution:\n")
# table() creates a frequency table (like Python's pd.Series.value_counts()).
dominant_frame_summary <- table(frame_dominance_analysis$dominant_frame)
print(dominant_frame_summary)

# 5.3 Frame complexity: how many different frames are used per article.
# rowSums(select(., ...) > 0) counts how many frame columns have non-zero values.
frame_complexity <- top_100_topic3 %>%
  mutate(
    frames_used = rowSums(select(., all_of(frame_cols)) > 0),
    frame_diversity = case_when(
      frames_used == 0 ~ "No frames",
      frames_used == 1 ~ "Single frame",
      frames_used == 2 ~ "Two frames",
      frames_used >= 3 ~ "Multiple frames"
    )
  )

cat("\nFrame complexity distribution:\n")
complexity_summary <- table(frame_complexity$frame_diversity)
print(complexity_summary)


# =============================================================================
#### SECTION 6: Display All Frame Visualizations ####
# =============================================================================

cat("\n6. DISPLAYING FRAME VISUALIZATIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Display all plots stored in the frame_plots list.
# Iterating over a named list with names() is like iterating over
# dict.items() in Python.
for (plot_name in names(frame_plots)) {
  cat("Displaying:", plot_name, "\n")
  print(frame_plots[[plot_name]])
  cat("\n")
}


# =============================================================================
#### SECTION 7: Save Frame Analysis Results ####
# =============================================================================

cat("\n7. SAVING FRAME ANALYSIS RESULTS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Save all plots as high-resolution PNG files.
if (!dir.exists("frame_analysis_results")) {
  dir.create("frame_analysis_results")
}

for (plot_name in names(frame_plots)) {
  filename <- paste0("frame_analysis_results/", plot_name, ".png")
  # ggsave() saves a ggplot object to a file.
  # width/height in inches, dpi = dots per inch (300 is publication quality).
  # In Python: fig.savefig(filename, dpi=300, bbox_inches='tight')
  ggsave(filename, frame_plots[[plot_name]], width = 12, height = 8, dpi = 300)
  cat("Saved:", filename, "\n")
}

# Save summary data as CSV files for further analysis or inclusion in papers.
write.csv(frame_results, "frame_analysis_results/frame_summary_statistics.csv", row.names = FALSE)
write.csv(frame_summary, "frame_analysis_results/frame_detailed_summary.csv", row.names = FALSE)

if ("year" %in% names(top_100_topic3)) {
  write.csv(decade_summary, "frame_analysis_results/frame_by_decade.csv", row.names = FALSE)
  write.csv(frame_trend_tests, "frame_analysis_results/frame_temporal_trends.csv", row.names = FALSE)
}

write.csv(frame_dominance_analysis, "frame_analysis_results/frame_dominance_by_article.csv", row.names = FALSE)

cat("\nFrame analysis complete! Results saved to 'frame_analysis_results/' directory\n")
cat("Total visualizations created:", length(frame_plots), "\n")


# =============================================================================
#### SECTION: Final Summary ####
# =============================================================================

cat("\n=== FRAME ANALYSIS SUMMARY ===\n")
cat("Frames analyzed:", length(frame_cols), "\n")
cat("Articles analyzed:", nrow(top_100_topic3), "\n")
if ("year" %in% names(top_100_topic3)) {
  cat("Time span:", min(top_100_topic3$year, na.rm = TRUE), "-", max(top_100_topic3$year, na.rm = TRUE), "\n")
}
cat("Most prevalent frame:", frame_summary$frame_type[1],
    "(", frame_summary$percentage_articles[1], "% of articles)\n")
# tail() returns the last element(s), opposite of head().
# In Python: frame_summary['frame_type'].iloc[-1]
cat("Least prevalent frame:", tail(frame_summary$frame_type, 1),
    "(", tail(frame_summary$percentage_articles, 1), "% of articles)\n")
