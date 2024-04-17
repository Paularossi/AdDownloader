library(dplyr)
library(readxl)
library(ggplot2)
library(openxlsx)


#### US Elections Study Case:
df <- read_excel("output/uselections/ads_data/uselections_processed_data.xlsx")
summary(df)
df$ad_delivery_start_time <- as.Date(df$ad_delivery_start_time, format = "%Y-%m-%d")
df$ad_delivery_stop_time <- as.Date(df$ad_delivery_stop_time, format = "%Y-%m-%d")

periods <- list(
  period1 = c(as.Date("2020-10-03"), as.Date("2020-10-10")),
  period2 = c(as.Date("2020-10-11"), as.Date("2020-10-18")),
  period3 = c(as.Date("2020-10-19"), as.Date("2020-10-26")),
  period4 = c(as.Date("2020-10-27"), as.Date("2020-11-03"))
)
pages <- c("Democratic Party", "GOP")


df <- df %>%
  filter(ad_delivery_start_time >= as.Date("2020-10-03") & ad_delivery_stop_time <= as.Date("2020-11-03")) %>%
  filter(languages == "['en']")


sampled_dfs <- lapply(pages, function(page) {
  lapply(periods, function(period) {
    df %>%
      filter(page_name == page) %>%
      filter(ad_delivery_start_time >= period[1] & ad_delivery_stop_time <= period[2]) %>%
      filter(!is.na(demographic_distribution)) %>%
      sample_n(size = 250, replace = FALSE)
  }) %>%
  bind_rows() # combine samples from all periods for the current page
}) %>%
bind_rows()
.

df_sampled <- read_excel("output/uselections/ads_data/uselections-sample.xlsx")
# sampled_df <- df %>%
#   filter(page_name %in% c("Democratic Party", "GOP")) %>%
#   filter(!is.na(demographic_distribution)) %>%
#   # distinct(ad_creative_bodies, .keep_all = TRUE) %>%
#   group_by(page_name) %>%
#   sample_n(size = 1000, replace = FALSE) %>%
#   ungroup()





# ========= Impressions modelling =========

library(randomForest)
library(ggplot2)
library(stargazer)
library(broom)

# set decimals to digits instead of scientific
options(scipen = 999)

data <- read_excel("output/uselections/us-elections-final.xlsx")
# log transform impressions_avg
data$log_impressions_avg <- log(data$impressions_avg)
data$dom_topic <- as.factor(data$dom_topic)
data$dom_topic_caption <- as.factor(data$dom_topic_caption)
data$people <- factor(data$people, levels = c("no_people", "woman", "man", "both"))
data$emotion <- factor(data$emotion, levels = c("none", "anger", "happiness", "sadness"))
data$masks <- as.factor(data$masks)
data$`afr-amer` <- as.factor(data$`afr-amer`)
data$asian <- as.factor(data$asian)
data$white <- as.factor(data$white)
str(data)

# normalize the img data
min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

data$brightness_norm <- min_max_normalize(data$brightness)
data$sharpness_norm <- min_max_normalize(data$sharpness)
data$contrast_norm <- min_max_normalize(data$contrast)
data$resolution_norm <- min_max_normalize(data$resolution)
data$spend_avg_norm <- min_max_normalize(data$spend_avg)

hist(data$log_impressions_avg)
# model 1 - ad library features
model1 <- glm(log_impressions_avg ~ page_name + campaign_duration + log(spend_avg), data = data)
summary(model1) 

# model 2 - ad library + text analysis
model2 <- lm(log_impressions_avg ~ page_name + campaign_duration + log(spend_avg) + dom_topic + textblb_sent, data = data)
summary(model2)

# model 3 - ad library features + text anlaysis + image analysis
model3 <- lm(log_impressions_avg ~ page_name + campaign_duration + log(spend_avg) + dom_topic + textblb_sent +
              resolution_norm + brightness_norm + contrast_norm + sharpness_norm, data = data)
summary(model3)


# model 4 - ad library features + text anlaysis + image analysis + BLIP
model4 <- lm(log_impressions_avg ~ page_name + campaign_duration + log(spend_avg) + dom_topic + textblb_sent +
              resolution_norm + brightness_norm + contrast_norm + sharpness_norm + dom_topic_caption + 
              `afr-amer` + asian + white + masks + people + emotion, data = data)
summary(model4)

model1_tidy <- tidy(model1)
model2_tidy <- tidy(model2)
model3_tidy <- tidy(model3)
model4_tidy <- tidy(model4)
output <- merge(model3_tidy, model4_tidy, by='term', all.x=T, all.y=T)
stargazer(output, type='latex', summary=FALSE, single.row=TRUE)

anova(model1, model2, test = "F")
anova(model2, model3, test = "F")
anova(model3, model4, test = "F")

# model 5 - significant parameters only
model5 <- lm(log_impressions_avg ~ page_name + campaign_duration + spend_avg + dom_topic +
              contrast + dom_topic_caption + `afr-amer`, data = data)
summary(model5)

# ad text:
# 0 - early voting
# 1 - urgency
# 2 - patriotic appeal

# ad image caption:
# 0 - civic duty
# 1 - progressive (left) activist
# 2 - republican poster other (no logo)
# 3 - right activist
# 4 - republic poster logo

cor(cbind(data$is_man, data$is_woman, data$is_no_people, data$is_both))


summary(data$people)


########### Radlibrary
library(devtools)
devtools::install_github("facebookresearch/Radlibrary")
library(Radlibrary)

token <- readline()

query <- adlib_build_query(
  ad_reached_countries = "BE",
  ad_active_status = "ALL",
  ad_type = "ALL",
  search_terms = "pizza",
  ad_delivery_date_max = '2024-03-11',
  ad_delivery_date_min = '2024-01-01',
  fields = "ad_data",
  limit=9000
)

response <- adlib_get(params = query, token = token)

results.tibble <- as_tibble(response, censor_access_token = TRUE)
head(results.tibble)