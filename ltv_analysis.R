library(tidyverse)
library(stringr)
library(urltools)
library(caret)
library(RANN)
library(Matrix)
library(xgboost)
library(data.table)

## import 2 countries' data and combine into one 
us_data <- read_csv("~/Documents/abc/results_US_nooutliers_50days.csv")
de_data <- read_csv("~/Documents/abc/results_DE_nooutliers_50days.csv")
us_data <- us_data %>% mutate(country="US")
de_data <- de_data %>% mutate(country="DE")
usde_data <- bind_rows(us_data, de_data)

# available columns in the table include (only 1-3, and 24 are categorical variables, all else are numerical based on aggregation data of clients based on group by source,medium,campaign,country)
# (1) source
# (2) medium
# (3) campaign
# (4) clients
# (5) avg_days_since_last_active
# (6) avg_historical_searches
# (7) avg_profile_age
# (8) avg_addos
# (9) avg_bookmarks
# (10) avg_alive_probability
# (11) avg_LTV
# (12) sum_LTV
# (13) avg_pLTV
# (14) active_clients
# (15) active_client_pct
# (16) e_10s
# (17) e_10s_pct
# (18) FxA
# (19) FxA_pct
# (20) nocodes
# (21) nocodes_pct
# (22) default_browser
# (23) default_browser_pct
# (24) country

########################################################################################################################
####### data quality check   ###########################################################################################
########################################################################################################################

# take a glimpse of data
glimpse(usde_data)
str(usde_data)
View(usde_data)

# check min and max value for numeric variables
sapply(usde_data %>% select_if(is.numeric), function(x) c(min(x), max(x)))
# > sapply(usde_data %>% select_if(is.numeric), function(x) c(min(x), max(x)))
# clients avg_days_since_last_active avg_historical_searches avg_profile_age avg_addos avg_bookmarks avg_alive_probability      avg_LTV
# [1,]       1                          0                       1              -1         4            NA             0.0000000  0.002169135
# [2,]    2537                         48                     437            2856        15            NA             0.9927989 74.221679700
# sum_LTV avg_pLTV active_clients active_client_pct e_10s e_10s_pct FxA FxA_pct nocodes nocodes_pct default_browser default_browser_pct
# [1,] 2.169135e-03  0.00000              0                 0     0         0   0       0       0           0               0                   0
# [2,] 1.070534e+04 68.75113           1989                 1  2530         1 439       1       7           1            1079                   1

# view records with negative "avg_profile_age"
View(usde_data %>% filter(avg_profile_age < 0))


# check missing value percentage for each country, get missing 27.3% of avg_bookmarks for DE, and 25.9% of avg_bookmarks for US
data_missing <- usde_data %>% group_by(country) %>% summarise_all(funs(sum(is.na(.)/n())));
data_missing <- gather(data_missing, key="feature", value="missing_pcnt", - country) %>%
  filter(missing_pcnt > 0);
data_missing
# # A tibble: 2 x 3
# country feature       missing_pcnt
# <chr>   <chr>                <dbl>
#   1 DE      avg_bookmarks        0.273
#   2 US      avg_bookmarks        0.259

# checking how many clients used to represent each combination of (source, medium, campaign, content) 
# find around 50% of these combinations are only represented by <= 2 clients -- the reliability is questionable
summary(usde_data%>% select(clients))
# clients       
# Min.   :   1.00  
# 1st Qu.:   1.00  
# Median :   2.00  
# Mean   :  38.56  
# 3rd Qu.:   6.00  
# Max.   :2537.00 


########################################################################################################################
########################################################################################################################
########################################################################################################################

# boxplot for avg_LTV per source for both DE/US
usde_data %>% 
  ggplot() + 
  geom_boxplot(aes(x=str_wrap(source,70), y=avg_LTV, color= country), outlier.color="blue" ) + 
  coord_flip()+
  theme(axis.text.y = element_text(size=10)) + 
  xlab("Source") + 
  ylab("Avergage LTV")
ggsave("~/Documents/abc/ltv_by_source.png", width = 10, height = 6)


# boxplot for avg_LTV per medium for both DE/US
usde_data %>% 
  ggplot() + 
  geom_boxplot(aes(x=str_wrap(medium,70), y=avg_LTV, color= country), outlier.color="blue" ) + 
  coord_flip()+
  theme(axis.text.y = element_text(size=10)) + 
  xlab("Medium") + 
  ylab("Avergage LTV")
ggsave("~/Documents/abc/ltv_by_medium.png", width = 10, height = 6)


# boxplot of avg_LTV per campaign for US data
usde_data %>% filter(country == "US") %>%
  ggplot() + 
   geom_boxplot(aes(x=str_wrap(campaign,70), y=avg_LTV, color=country), outlier.color="blue" ) + 
   coord_flip()+
   theme(axis.text.y = element_text(size=10)) + 
   xlab("Campaigns") + 
   ylab("Avergage LTV")
ggsave("~/Documents/abc/ltv_by_campaign_us.png", width = 11, height = 7)


# boxplot of avg_LTV per campaign for DE data
usde_data %>% filter(country == "DE") %>%
  ggplot() + 
  geom_boxplot(aes(x=str_wrap(campaign,70), y=avg_LTV, color=country), outlier.color="blue" ) + 
  coord_flip()+
  theme(axis.text.y = element_text(size=10)) + 
  xlab("Campaigns") + 
  ylab("Avergage LTV")
ggsave("~/Documents/abc/ltv_by_campaign_de.png", width = 11, height = 7)

########################################################################################################################
####### Find important predictors of high value users (avg_LTV)            #############################################
########################################################################################################################

# First, impute missing data with Caret package, for missing avg_bookmarks, 
# and for 2 rows with -1 for avg_profile_age, change it to NA first, then impute with NA's in avg_bookmarks similarly
# and for pairs of variables like (active_clients, active_clients_pct), I will only include *pct 
# and remove 2 columns that highly-correlated to avg_LTV (sum_LTV, and avg_pLTV)

usde_data_temp <- usde_data %>% 
  mutate(avg_profile_age = replace(avg_profile_age, avg_profile_age < 0, NA), 
         source     = as.factor(source), 
         medium     = as.factor(medium), 
         campaign   = as.factor(campaign),
         country    = as.factor(country), 
         content    = as.factor(content)) %>%
  select(avg_LTV, source, medium, campaign, country, content, clients, avg_days_since_last_active, avg_historical_searches, 
         avg_profile_age, avg_addos, avg_bookmarks, avg_alive_probability,  active_client_pct, 
         e_10s_pct, FxA_pct, nocodes_pct, default_browser_pct) 

# use caret package to impute the missing value in avg_bookmarks
usde_data_pre <- preProcess(usde_data_temp, method = "bagImpute")
usde_data_impt <- predict(usde_data_pre, usde_data_temp, na.action=na.pass)
# compare before and after the min-max for each column
sapply(usde_data_impt %>% select_if(is.numeric), function(x) c(min(x), max(x)))
sapply(usde_data %>% select_if(is.numeric), function(x) c(min(x, na.rm = TRUE), max(x, na.rm = TRUE)))

# split data into training data set and test data set
set.seed(1)
inTrain <- createDataPartition(y=usde_data_impt$avg_LTV, p = 0.8, list = FALSE)
train.data <- usde_data_impt[inTrain,]
test.data  <- usde_data_impt[-inTrain,]
xgboost_data <- sparse.model.matrix(avg_LTV ~ . -1, data = train.data)
xgboost_label <- as.matrix(train.data$avg_LTV)
dim(xgboost_data)
dim(xgboost_label)
bst <- xgboost(data = xgboost_data, label=xgboost_label,  
               objective = "reg:linear", max.depth = 3, nthread = 2, nround = 50, eta= 0.2)
var.imp.xgboost <- xgb.importance(feature_names = colnames(xgboost_data), model = bst)
print(var.imp.xgboost[order(var.imp.xgboost$Gain, decreasing=T),])
xgb.plot.importance(importance_matrix = var.imp.xgboost[1:10], xlab = "Feature Importance", left_margin = 20)

pred.xgboost <- predict(bst, newdata=sparse.model.matrix(avg_LTV ~ . -1, data = test.data))
RMSE.xgboost <- sqrt(colMeans((test.data[,1]-pred.xgboost)^2))
RMSE.xgboost
cbind(test.data[,1], pred.xgboost)

