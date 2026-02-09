# =============================================================================
# DEEP DIVE FRAME ANALYSIS WITH TEMPORAL TRENDS
# =============================================================================

library(tidyverse)
library(ggplot2)
library(viridis)
library(patchwork)
library(corrplot)

cat("=== DEEP DIVE FRAME ANALYSIS ===\n")

# =============================================================================
# 1. ENHANCED FRAME DEFINITIONS AND CALCULATION
# =============================================================================

cat("\n1. ENHANCED FRAME DEFINITIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Define comprehensive frames with more terms
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

# Remove any existing frame columns to start fresh
existing_frame_cols <- grep("_frame$|_score$", names(top_100_topic3), value = TRUE)
# Exclude sentiment_score from removal
existing_frame_cols <- existing_frame_cols[existing_frame_cols != "sentiment_score"]

if (length(existing_frame_cols) > 0) {
  top_100_topic3 <- top_100_topic3 %>% select(-all_of(existing_frame_cols))
  cat("Removed existing frame columns:", paste(existing_frame_cols, collapse = ", "), "\n")
}

# Calculate frame scores with detailed reporting
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
  pattern <- paste(frame_words, collapse = "|")
  
  # Calculate scores
  scores <- str_count(tolower(top_100_topic3$text), pattern)
  top_100_topic3[[paste0(frame_name, "_frame")]] <- scores
  
  # Record statistics
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
# 2. FRAME ANALYSIS EXCLUDING SENTIMENT_SCORE
# =============================================================================

cat("\n2. FRAME COMPARISON ANALYSIS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Get only frame columns (excluding sentiment_score)
frame_cols <- grep("_frame$", names(top_100_topic3), value = TRUE)
cat("Frame columns identified:", paste(frame_cols, collapse = ", "), "\n")

# Create frame data for analysis
frame_data <- top_100_topic3 %>%
  select(rank, doc_index, any_of("year"), any_of("source"), all_of(frame_cols)) %>%
  pivot_longer(cols = all_of(frame_cols), names_to = "frame_type", values_to = "frame_score") %>%
  mutate(frame_type = str_remove(frame_type, "_frame"))

# Summary statistics by frame
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
# 3. COMPREHENSIVE TEMPORAL FRAME ANALYSIS
# =============================================================================

if ("year" %in% names(top_100_topic3)) {
  cat("\n3. TEMPORAL FRAME ANALYSIS\n")
  cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")
  
  # 3.1 Frame evolution by year
  frame_temporal <- frame_data %>%
    filter(!is.na(year)) %>%
    group_by(year, frame_type) %>%
    summarise(
      mean_frame_score = mean(frame_score),
      total_frame_score = sum(frame_score),
      articles_with_frame = sum(frame_score > 0),
      .groups = "drop"
    )
  
  # 3.2 Frame evolution by decade
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
  # Fix: Create proper wide format for decade comparison
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
  
  # 3.3 Frame dominance over time
  frame_dominance <- frame_temporal %>%
    group_by(year) %>%
    mutate(
      total_frames_year = sum(total_frame_score),
      frame_proportion = if_else(total_frames_year > 0, total_frame_score / total_frames_year * 100, 0)
    ) %>%
    ungroup()
  
  # 3.4 Statistical tests for temporal trends - FIXED
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
      tryCatch({
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
  
  # 3.5 Frame correlation over time periods
  early_period <- top_100_topic3 %>% filter(year <= 2000)
  late_period <- top_100_topic3 %>% filter(year > 2000)
  
  if (nrow(early_period) > 10 && nrow(late_period) > 10) {
    early_corr <- early_period %>% 
      select(all_of(frame_cols)) %>% 
      cor(use = "complete.obs")
    
    late_corr <- late_period %>% 
      select(all_of(frame_cols)) %>% 
      cor(use = "complete.obs")
    
    cat("\nFrame correlations - Early period (≤2000):\n")
    print(round(early_corr, 3))
    
    cat("\nFrame correlations - Late period (>2000):\n")
    print(round(late_corr, 3))
  }
}

# =============================================================================
# 4. COMPREHENSIVE FRAME VISUALIZATIONS
# =============================================================================

cat("\n4. CREATING FRAME VISUALIZATIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

frame_plots <- list()

# 4.1 Frame distribution comparison (excluding sentiment)
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

# 4.2 Frame prevalence (articles with non-zero scores)
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

# 4.3 Temporal frame evolution (if year available)
if ("year" %in% names(top_100_topic3)) {
  
  # Smooth temporal trends
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
  
  # Frame dominance over time (proportional)
  frame_plots$dominance_trends <- frame_dominance %>%
    ggplot(aes(x = year, y = frame_proportion, color = frame_type)) +
    geom_smooth(method = "loess", se = FALSE, size = 1) +
    scale_color_viridis_d() +
    labs(title = "Frame Dominance Over Time",
         subtitle = "Relative proportion of each frame type by year",
         x = "Year", y = "Frame Proportion (%)", color = "Frame") +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Decade comparison
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

# 4.4 Frame correlation heatmap
frame_correlation_data <- top_100_topic3 %>%
  select(all_of(frame_cols)) %>%
  cor(use = "complete.obs")

# Convert to long format for ggplot
frame_corr_long <- expand.grid(
  Frame1 = factor(rownames(frame_correlation_data), levels = rownames(frame_correlation_data)),
  Frame2 = factor(colnames(frame_correlation_data), levels = colnames(frame_correlation_data))
) %>%
  mutate(
    correlation = as.vector(frame_correlation_data),
    Frame1_clean = str_remove(as.character(Frame1), "_frame"),
    Frame2_clean = str_remove(as.character(Frame2), "_frame")
  )

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

# 4.5 Frame intensity distribution
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
# 5. ADVANCED FRAME ANALYSIS
# =============================================================================

cat("\n5. ADVANCED FRAME ANALYSIS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# 5.1 Frame co-occurrence analysis
frame_cooccurrence <- top_100_topic3 %>%
  select(all_of(frame_cols)) %>%
  mutate_all(~ ifelse(. > 0, 1, 0)) %>%  # Convert to binary
  as.matrix()

cooccur_matrix <- t(frame_cooccurrence) %*% frame_cooccurrence
diag(cooccur_matrix) <- 0  # Remove self-cooccurrence

cat("Frame co-occurrence counts (how often frames appear together):\n")
print(cooccur_matrix)

# 5.2 Frame dominance patterns
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
dominant_frame_summary <- table(frame_dominance_analysis$dominant_frame)
print(dominant_frame_summary)

# 5.3 Frame complexity (articles using multiple frames)
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
# 6. DISPLAY ALL FRAME VISUALIZATIONS
# =============================================================================

cat("\n6. DISPLAYING FRAME VISUALIZATIONS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Display all plots
for (plot_name in names(frame_plots)) {
  cat("Displaying:", plot_name, "\n")
  print(frame_plots[[plot_name]])
  cat("\n")
}

# =============================================================================
# 7. SAVE FRAME ANALYSIS RESULTS
# =============================================================================

cat("\n7. SAVING FRAME ANALYSIS RESULTS\n")
cat("=" %>% rep(40) %>% paste(collapse = ""), "\n")

# Save plots
if (!dir.exists("frame_analysis_results")) {
  dir.create("frame_analysis_results")
}

for (plot_name in names(frame_plots)) {
  filename <- paste0("frame_analysis_results/", plot_name, ".png")
  ggsave(filename, frame_plots[[plot_name]], width = 12, height = 8, dpi = 300)
  cat("Saved:", filename, "\n")
}

# Save data summaries
write.csv(frame_results, "frame_analysis_results/frame_summary_statistics.csv", row.names = FALSE)
write.csv(frame_summary, "frame_analysis_results/frame_detailed_summary.csv", row.names = FALSE)

if ("year" %in% names(top_100_topic3)) {
  write.csv(decade_summary, "frame_analysis_results/frame_by_decade.csv", row.names = FALSE)
  write.csv(frame_trend_tests, "frame_analysis_results/frame_temporal_trends.csv", row.names = FALSE)
}

write.csv(frame_dominance_analysis, "frame_analysis_results/frame_dominance_by_article.csv", row.names = FALSE)

cat("\nFrame analysis complete! Results saved to 'frame_analysis_results/' directory\n")
cat("Total visualizations created:", length(frame_plots), "\n")

# Final summary
cat("\n=== FRAME ANALYSIS SUMMARY ===\n")
cat("Frames analyzed:", length(frame_cols), "\n")
cat("Articles analyzed:", nrow(top_100_topic3), "\n")
if ("year" %in% names(top_100_topic3)) {
  cat("Time span:", min(top_100_topic3$year, na.rm = TRUE), "-", max(top_100_topic3$year, na.rm = TRUE), "\n")
}
cat("Most prevalent frame:", frame_summary$frame_type[1], 
    "(", frame_summary$percentage_articles[1], "% of articles)\n")
cat("Least prevalent frame:", tail(frame_summary$frame_type, 1), 
    "(", tail(frame_summary$percentage_articles, 1), "% of articles)\n")