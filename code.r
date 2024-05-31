# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)


##########################################################
# Create edx and final_holdout_test sets - 125.9 Class Code
##########################################################

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#########################################################
# Data Exploration & Analysis
#########################################################

### Plots counts of each rating ###
edx %>% ggplot(aes(x = rating)) +
  geom_histogram(binwidth = .25, color = "black")

### Plots average rating per movie ###
edx %>% group_by(movieId) %>%
  summarise(avg_rating_per_movie = mean(rating)) %>%
  ggplot(aes(avg_rating_per_movie)) +
  geom_histogram(bins=30, color = "black")

### Plots average rating per user ###
edx %>% group_by(userId) %>%
  summarise(avg_rating_per_user = mean(rating)) %>%
  ggplot(aes(avg_rating_per_user)) +
  geom_histogram(bins=30, color = "black")

### See how many different genres there are, and the average rating for this genre###
library(knitr)
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n(), avg_rating = mean(rating)) %>%
  kable(caption = "Genre Check")

### Plot average genre combination rating for genre combinations with more than 50,000 ratings
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 50000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

### Creates a new column in edx dataset for the movie release year ###
library(stringr)
edx$movieYear <- str_sub(edx$title, start = -6, end = -1) 
edx$movieYear <- as.numeric(gsub("\\(|\\)", "", edx$movieYear))

### Plots the average movie rating per movie release year ###
edx %>% group_by(movieYear) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(x = movieYear, y = rating)) +
  geom_point() +
  geom_smooth()

### Changes timestamp into a date object and plots the average rating per timestamp date ###
edx$timestamp <- as_date(as_datetime(edx$timestamp))

edx %>% group_by(timestamp) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(x = timestamp, y = rating)) +
  geom_point() +
  geom_smooth()


##########################################################
# Split data into training and test set
##########################################################

set.seed(1)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_edx <- edx[-test_index,]
test_edx_1 <- edx[test_index,]

### Ensure that the test data only contains data present in the train ###

test_edx <- test_edx_1 %>%
  semi_join(train_edx, by="userId") %>%
  semi_join(train_edx, by="movieId")

### Adds removed test rows to training data ###

removed_data <- anti_join(test_edx_1,test_edx)
train_edx <- rbind(train_edx,removed_data)

rm(removed_data, test_edx_1, test_index)

### Defines an RMSE & MAE function ###
RMSE <- function(ratings, pred_ratings){
  sqrt(mean((ratings-pred_ratings)^2))
}

MAE <- function(ratings, pred_ratings){
  mean(abs(ratings - pred_ratings))
}

goal_rmse <- 0.86490


##########################################################
# Start creating and fitting regression models
##########################################################

### Finds training set average mu_hat ###

mu <- mean(train_edx$rating)
rmse_basic_model <- RMSE(test_edx$rating,mu)
mae_basic_model <- MAE(test_edx$rating,mu)
print(rmse_basic_model)

### Adds in a variable for movies to account for movie by movie variability "b_m" ###
### Define b_m ###
avg_by_movie <- train_edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating - mu))

### Calculates the predicted ratings & RMSE using the Y_{u,m} = mu + b_m + e{u,m} model ###
predicted_ratings <- mu + test_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  pull(b_m)

b_m_model_rmse <- RMSE(test_edx$rating,predicted_ratings)
b_m_model_mae <- MAE(test_edx$rating,predicted_ratings)

b_m_model_rmse
rm(predicted_ratings)

### Adds a variable for users "b_u" to account for user by user variability ###
### Defines b_u estimate ###
avg_by_user <- train_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))

### Calculates the predicted ratings & RMSE using the Y_{u,m} = mu + b_m + b_u + e{u,m} model ###
predicted_ratings <- test_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  mutate(predicted_ratings = mu + b_m + b_u) %>%
  pull(predicted_ratings)

b_u_model_rmse <- RMSE(test_edx$rating,predicted_ratings)
b_u_model_mae <- MAE(test_edx$rating,predicted_ratings)

b_u_model_rmse
rm(predicted_ratings)

### Adds a genre variable "b_g" to account for the genre variability by grouped genres ###
### Defines b_g estimate ###
avg_by_genre <- train_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_m - b_u))

### Calculates the predicted ratings & RMSE using the Y_{u,m} = mu + b_m + b_u + b_g + e{u,m} model ###
predicted_ratings <- test_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  left_join(avg_by_genre, by = "genres") %>%
  mutate(predicted_ratings = mu + b_m + b_u + b_g) %>%
  pull(predicted_ratings)

b_g_model_rmse <- RMSE(test_edx$rating,predicted_ratings)
b_g_model_mae <- MAE(test_edx$rating,predicted_ratings)

b_g_model_rmse
rm(predicted_ratings)

### Adds a release year variable "b_y" to account for the release year by release year variability ###
### Defines b_y estimate ###
avg_by_year <- train_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  left_join(avg_by_genre, by = "genres") %>%
  group_by(movieYear) %>%
  summarise(b_y = mean(rating - mu - b_m - b_u - b_g))

### Calculates the predicted ratings & RMSE using the Y_{u,m} = mu + b_m + b_u + b_g + b_y + e{u,m} model ###
predicted_ratings <- test_edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  left_join(avg_by_genre, by = "genres") %>%
  left_join(avg_by_year, by = "movieYear") %>%
  mutate(predicted_ratings = mu + b_m + b_u + b_g + b_y) %>%
  pull(predicted_ratings)

b_y_model_rmse <- RMSE(test_edx$rating,predicted_ratings)
b_y_model_mae <- MAE(test_edx$rating,predicted_ratings)

b_y_model_rmse
rm(predicted_ratings)

##########################################################
# Regularize model
##########################################################

### Define lambda, our tuning parameter ###
lambda <- seq(1,5, by = .1)

### Creates a function that uses all values of lambda in our final model Y_{u,m} = mu + b_m + b_u + b_g + b_y + e{u,m} to predict ratings and find RMSE ###
reg_b_y_model_rmses <- sapply(lambda, function(lambda){
  
  ### redefines b_m ###
  avg_by_movie <- train_edx %>% 
    group_by(movieId) %>% 
    summarize(b_m = sum(rating - mu)/(n() + lambda))
  
  ### Redefine b_u ###
  avg_by_user <- train_edx %>%
    left_join(avg_by_movie, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_m)/(n() + lambda))
  
  ### Redefines b_g ###
  avg_by_genre <- train_edx %>%
    left_join(avg_by_movie, by = "movieId") %>%
    left_join(avg_by_user, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_m - b_u)/(n() + lambda))
  
  ### Redefines b_y ###
  avg_by_year <- train_edx %>%
    left_join(avg_by_movie, by = "movieId") %>%
    left_join(avg_by_user, by = "userId") %>%
    left_join(avg_by_genre, by = "genres") %>%
    group_by(movieYear) %>%
    summarise(b_y = sum(rating - mu - b_m - b_u - b_g)/(n() + lambda))
  
  
  ### Calculates the predicted ratings & RMSE using the regularized Y_{u,m} = mu + b_m + b_u + b_g + b_y + e{u,m} model ###
  predicted_ratings <- test_edx %>%
    left_join(avg_by_movie, by = "movieId") %>%
    left_join(avg_by_user, by = "userId") %>%
    left_join(avg_by_genre, by = "genres") %>%
    left_join(avg_by_year, by = "movieYear") %>%
    mutate(predicted_ratings = mu + b_m + b_u + b_g + b_y) %>%
    pull(predicted_ratings)
  
  reg_b_y_model_rmses <- RMSE(test_edx$rating,predicted_ratings)
  
  return(reg_b_y_model_rmses)
}) 

### Creates a graph of RMSE plotted against Lambda ###
ggplot() + geom_point(aes(x = lambda, y = reg_b_y_model_rmses))

### Finds the best RMSE & tuning parameter ###
best_lambda <- lambda[which.min(reg_b_y_model_rmses)]
best_lambda

best_rmse <- min(reg_b_y_model_rmses)
best_rmse


##########################################################
# Creates RMSE table
##########################################################

RMSE_table <- tibble(Regression_Model = c("rmse_basic_average_model", "b_m_model_rmse", "b_u_model_rmse", "b_g_model_rmse", "b_y_model_rmse", "regularized_b_y_model_rmse"),
                     RMSE = c(rmse_basic_model, b_m_model_rmse, b_u_model_rmse, b_g_model_rmse, b_y_model_rmse, best_rmse),
                     Diff_from_goal = c(goal_rmse - rmse_basic_model, goal_rmse - b_m_model_rmse, goal_rmse - b_u_model_rmse, goal_rmse - b_g_model_rmse, goal_rmse - b_y_model_rmse, goal_rmse - best_rmse))

kable(x = RMSE_table, caption = "RMSE Table")


##########################################################
# Creates final model and tests it on the test data
##########################################################

### Mutates final_holdout_test data to match the mutations made to the edx data set ###
final_holdout_test$movieYear <- str_sub(final_holdout_test$title, start = -6, end = -1) 
final_holdout_test$movieYear <- as.numeric(gsub("\\(|\\)", "", final_holdout_test$movieYear))


##########################################################
# Trains the regularized regression model on the edx data set
##########################################################

### Y_{u,m} = mu + b_m + b_u + b_g + b_y + e{u,m} ###

### redefines b_m ###
b_m <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = sum(rating - mu)/(n() + best_lambda))

### Redefine b_u ###
b_u <- edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_m)/(n() + best_lambda))

### Redefines b_g ###
b_g <- edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_m - b_u)/(n() + best_lambda))

### Redefines b_y ###
b_y <- edx %>%
  left_join(avg_by_movie, by = "movieId") %>%
  left_join(avg_by_user, by = "userId") %>%
  left_join(avg_by_genre, by = "genres") %>%
  group_by(movieYear) %>%
  summarise(b_y = sum(rating - mu - b_m - b_u - b_g)/(n() + best_lambda))


### Calculates the predicted ratings & RMSE using the regularized Y_{u,m} = mu + b_m + b_u + b_g + b_y + e{u,m} model ###
predicted_ratings <- final_holdout_test %>%
  left_join(b_m, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_y, by = "movieYear") %>%
  mutate(predicted_ratings = mu + b_m + b_u + b_g + b_y) %>%
  pull(predicted_ratings)

final_regularized_b_y_model_rmse <- RMSE(final_holdout_test$rating,predicted_ratings)
final_regularized_b_y_model_rmse

### Adds final RMSE to RMSE table ###
RMSE_table <- rbind(RMSE_table, data.frame(Regression_Model = "final_regularized_b_y_model_rmse", 
                                           RMSE = final_regularized_b_y_model_rmse,
                                           Diff_from_goal = goal_rmse - final_regularized_b_y_model_rmse))

RMSE_table %>% kable(caption = "Final RMSE Table")
