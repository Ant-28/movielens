##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# # if using R 3.6 or earlier:
# movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                             title = as.character(title),
#                                             genres = as.character(genres))


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



# Loading the required libraries, can add required libraries later
required_libraries <- c("caret", "matrixStats", "tidyverse", "knitr", "lubridate", "broom", "ranger", "knitr", "rmarkdown")
# The for loop installs and loads libraries one by one from required_libraries
for(i in 1:length(required_libraries)){
if(!require(required_libraries[i], character.only = TRUE)){
  install.packages(required_libraries[i], repos = "http://cran.us.r-project.org")
  library(required_libraries[i], character.only = TRUE)
}
  else{
require(required_libraries[i], character.only = TRUE)}}

# Run this line only if necessary:
# update.packages()

# Splits the edx set into training and validation+debug sets
# Set seed so that results are replicable
set.seed(2000, sample.kind = "Rounding")

#Dataset creation: This is similar to how the edx and validation sets are created

edx_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
#edx_train is the training set
edx_train <- edx[-edx_index,]
#edx_foo is a temporary set which needs to be modified
edx_foo <- edx[edx_index,]


# Remove all rows from edx_foo that aren't in edx_train as 
# a show cannot be recommended without prior knowledge of the user
# and movie biases
edx_temp <- edx_foo %>% semi_join(edx_train, by = "movieId") %>% 
  semi_join(edx_train, by = "userId") %>% as.data.frame()
# edx_removed includes items in edx_foo that are not in edx_train
# if you have the latest version of tidyverse with dtplyr use the as.data.frame command


edx_removed <- anti_join(edx_foo, edx_temp) %>% as.data.frame()
#edx_removed is added to edx_train otherwise it would go to waste.
edx_train <- bind_rows(edx_train, edx_removed)

# Split edx_temp into two halves, the first half is used for crossvalidation,
# Such as choosing regularization parameters while the other half acts like a
# 'debug' set, which is a type of test set to see the efficacy of the algorithm 
# before training on the entire edx set and testing on validation.

# Again, set.seed for consistency
set.seed(5000, sample.kind = "Rounding")
debug_ind <- createDataPartition(y = edx_temp$rating, 
                                 time = 1, p = 0.5, list = FALSE)
#creating the validation and debug sets
edx_validation <- edx_temp[-debug_ind,]
edx_debug <- edx_temp[debug_ind,] 



# check if the the split datasets and edx contain the exact same elements
#edx_extra contains the split datasets combines
edx_extra <- bind_rows(edx_train, edx_validation,edx_debug)
# order by user and movie ID as those are the only unique orderings
# since users wouldn't rate the same movie twice
order1 <- order(edx_extra$userId, edx_extra$movieId)
order2 <- order(edx$userId, edx$movieId)
# ordered versions of edx_extra and edx for identical to work
edx_extra_sorted <- edx_extra[order1,]
edx_sorted <- edx[order2,]

identical(edx_extra_sorted, edx_sorted)



# Clear memory because R isn't good at it, you'll see this quite a few times
# across the code

gc()

# Remove redundant tables
remove(edx_extra, edx_sorted, edx_index, edx_removed, edx_extra_sorted, edx_foo, edx_temp, order1, order2)

mu <- mean(edx_train$rating)

# first prediction
only_the_mean <- edx_debug %>% mutate(pred = mu) %>% .$pred

rmse0 <- RMSE(only_the_mean, edx_debug$rating)
rmse0

# Adding movie effects
movie_effects <- edx_train %>% group_by(movieId) %>% 
  summarise(effect_bi = mean(rating - mu)) 

mean_and_movie <- mu + edx_debug %>%
 left_join(movie_effects, by = 'movieId') %>%
  pull(effect_bi)

rmse1 <- RMSE(mean_and_movie, edx_debug$rating)
rmse1

# Adding user effects
user_effects <- edx_train %>% left_join(movie_effects, by = "movieId") %>% 
  group_by(userId) %>%
  summarize(effect_bu = mean(rating-mu-effect_bi)) 

mean_movie_user <- mu + edx_debug %>% left_join(movie_effects, by = "movieId") %>% 
  left_join(user_effects, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)

rmse2 <- RMSE(edx_debug$rating, mean_movie_user)
rmse2

edx_train$genres %>% head(5)

maxgenre <- 1 + str_count(edx_train$genres, "\\|") %>% max()
maxgenre

# paste0 concatenates strings without any space.
genrecols <- paste0("genre",as.character(1:maxgenre))

# invisible function hides gc output
invisible(gc())
#this takes a while
edx_train_genres <- edx_train %>% separate(col = genres, into = genrecols, sep = "\\|")

invisible(gc())

edx_train_genres <- edx_train_genres %>% left_join(movie_effects, by = "movieId") 

invisible(gc())

edx_train_genres <- edx_train_genres %>% left_join(user_effects, by = "userId")

biascols <- paste0("effect_bg", as.character(1:(maxgenre)))
numcols <- paste0("n",as.character(1:(maxgenre)))

str(edx_train_genres)

ind <- -1 + which(colnames(edx_train_genres) == 'genre1') 
genre_effects <- list()
for(i in 1:maxgenre){
                     
  genre_effects[[i]] <- setNames(edx_train_genres %>% group_by_at(ind+i) %>%
                                   summarize(effect_bg = mean(rating - mu - effect_bi - effect_bu), n = n()), 
                                 c(genrecols[i],biascols[i], numcols[i]))
  
}


join_genres <- function(...){
  df1 <- list(...)[[1]]
  df2 <- list(...)[[2]]
  # Left join a la merge
  merge(df1, df2, by.x = 1, by.y = 1, all.x = TRUE)
  
}

total_genre_effects <- Reduce(join_genres, genre_effects)

head(total_genre_effects)

total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices<- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})



head(dat)


genre_bias <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))

remove(genre_effects, dat)

# Testing on the debug set

debug_genre_count <- 1 + str_count(edx_debug$genres, "\\|") %>% max()

genrecols_debug <- paste0('genre', 1:debug_genre_count)

edx_debug_genres <- edx_debug %>% separate(genres, genrecols_debug, sep =  "\\|")

genrecols_debug

invisible(gc())
genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug))


for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias, by.x = (genre_ind + i), by.y = 1, all.x = TRUE) 
 debug_genrebias[,i] <- foo$effect_bg
}

invisible(gc())
# Contains NAs
noquote("NAs by genre column")
apply(debug_genrebias,2,function(x){sum(is.na(x))})

#Does not contain NAs
noquote(" ")
noquote("Not NAs by genre column")
apply(debug_genrebias,2,function(x){sum(!is.na(x))})

debug_genrebias[is.na(debug_genrebias)] <- 0

debug_genrebias <- rowSums(debug_genrebias)

by_movie_user_and_genre <-  mu + {edx_debug %>% left_join(movie_effects, by = "movieId") %>% 
  left_join(user_effects, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias

rmse3 <- RMSE(edx_debug$rating, by_movie_user_and_genre)
rmse3

# Looking at what went wrong
genre_bias %>% mutate(n = rowSums(total_genre_effects[,n_indices])) %>% arrange(desc(abs(effect_bg)))


params_lambda <- 1/30 * (3^seq(0,9,1))

invisible(gc())
#Remove unregularized movie and user effects in training set
edx_train_genres <- edx_train_genres %>% select(-effect_bi, -effect_bu)


#Creating edx_validation_genres
validation_genre_count <- 1 + str_count(edx_validation$genres, "\\|") %>% max()

genrecols_validation <- paste0('genre', 1:validation_genre_count)

edx_validation_genres <- edx_validation %>% separate(genres, genrecols_validation, sep =  "\\|")

# Temporarily unload useless data frames
save(mu, rmse0, rmse1, rmse2, rmse3, edx, validation, edx_train, edx_validation, edx_debug, edx_train_genres, edx_validation_genres, edx_debug_genres, join_genres, maxgenre, genrecols, numcols, biascols, file = 'importants.Rdata')
remove(rmse0, rmse1, rmse2, rmse3, edx, validation, edx_train, edx_validation, edx_debug)



invisible(gc())
# Calculating best regularization parameter
# Same as debug set testing with genre effects, but this uses the validation set
# and regularization parameters
# this can take upwards of 30 minutes 

rmses <- sapply(1:length(params_lambda), function(x){
  # remove(movie_bias, genre_effects, genre_bias, user_bias, edx_train_genres_movie, edx_train_genres_user)

movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i], numcols[i], biascols[i]))
  # Bias column comes after frequency column in SetNames as mutate comes after summarize.
}


total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices<- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))

validation_genre_ind <- -1 + which(colnames(edx_validation_genres) == "genre1")

validation_genrebias <- matrix(ncol = validation_genre_count, nrow = nrow(edx_validation_genres))



for(i in 1:validation_genre_count){
  
 foo <- merge(edx_validation_genres, genre_bias_regularized, by.x = (validation_genre_ind + i), by.y = 1, all.x = TRUE) 
validation_genrebias[,i] <- foo$effect_bg
}

validation_genrebias[is.na(validation_genrebias)] <- 0
validation_genrebias <- rowSums(validation_genrebias)

by_movie_user_and_genre_reg <- mu + {edx_validation_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + validation_genrebias
gc()
return(RMSE(edx_validation_genres$rating, by_movie_user_and_genre_reg))
})


params_lambda[which.min(rmses)]


invisible(gc())
# x is the index of the lambda parameter with the lowest rmse in the valid
x <- which.min(rmses)
movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)
edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i],numcols[i], biascols[i]))
  
}


total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices<- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))



for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg[,i] <- foo$effect_bg
}
debug_genrebias_reg[is.na(debug_genrebias_reg)] <- 0
debug_genrebias_reg <- rowSums(debug_genrebias_reg)

by_movie_user_and_genre_reg <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg

rmse4 <- RMSE(edx_debug_genres$rating, by_movie_user_and_genre_reg)
rmse4

remove(genre_effects_reg, edx_train_genres_movie, edx_train_genres_user)

invisible(gc())
load(paste0(getwd(),"/importants.Rdata"))
file.remove("importants.Rdata")

by_movie_user_reg <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
        left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
        pull(total_effect)} 
rmsea <- RMSE(edx_debug_genres$rating, by_movie_user_reg)
rmsea

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg_2 <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))


for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg_2[,i] <- foo$effect_bg
}

# na.rm = TRUE because many entries will have NAs. Switching the NAs to zeros will only result in the zeros being added in the mean, and skewing the bias towards zero.
debug_genrebias_reg_2 <- rowMeans(debug_genrebias_reg_2, na.rm = TRUE)

by_movie_user_and_genre_reg_2 <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg_2
# identical(edx_debug_genres$rating, edx_debug$rating) is TRUE
# so it does not matter which vector one uses.
rmse5 <- RMSE(edx_debug_genres$rating, by_movie_user_and_genre_reg_2)
rmse5



rmses_2 <- sapply(1:length(params_lambda), function(x){
  # optional line of code
  # remove(movie_bias, genre_effects, genre_bias, user_bias, edx_train_genres_movie, edx_train_genres_user)

movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i], numcols[i], biascols[i]))
  # Bias column comes after frequency column in SetNames as mutate comes after summarize.
}


total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices<- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))

val_genre_ind <- -1 + which(colnames(edx_validation_genres) == "genre1")

validation_genrebias_2 <- matrix(ncol = validation_genre_count, nrow = nrow(edx_validation_genres))



for(i in 1:validation_genre_count){
  
 foo <- merge(edx_validation_genres, genre_bias_regularized, by.x = (val_genre_ind + i), by.y = 1, all.x = TRUE) 
validation_genrebias_2[,i] <- foo$effect_bg
}

# sum(!is.na(u)) sums all non-NA terms in vector u and adds them to the lambda parameter
# 1 in apply means the operation is applied to each ROW, not COLUMN
validation_genrebias_2 <- apply(validation_genrebias_2,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u))+params_lambda[x])})

by_movie_user_and_genre_reg_2 <- mu + {edx_validation_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + validation_genrebias_2
gc()
return(RMSE(edx_validation_genres$rating, by_movie_user_and_genre_reg_2))
})



params_lambda[which.min(rmses_2)]



invisible(gc())
x <- which.min(rmses_2)

movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i],numcols[i], biascols[i]))
  
}


total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices<- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg_3 <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))



for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg_3[,i] <- foo$effect_bg
}

debug_genrebias_reg_3 <- apply(debug_genrebias_reg_3,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u))+params_lambda[x])})

by_movie_user_and_genre_reg_3 <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg_3

rmse6 <- RMSE(edx_debug_genres$rating, by_movie_user_and_genre_reg_3)
rmse6




rmse6 < rmsea

str(edx_train)

gc()
edx_train_date <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')
gc()
edx_train_date <- edx_train_date %>% left_join(user_bias, by = 'userId')

save(mu, movie_bias, user_bias, rmse0, rmse1, rmse2, rmse3, rmse4,rmse5, rmse6, rmsea, rmses,rmses_2,  edx, validation, edx_train, edx_validation, edx_debug, edx_train_genres, edx_validation_genres, edx_debug_genres, join_genres, maxgenre, genrecols, numcols, biascols, file = 'importants.Rdata')
remove(rmse0, rmse1, rmse2, rmse3, rmse4, edx, validation, edx_train, edx_validation, edx_debug)

invisible(gc())

x <- which.min(rmses)
movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")

genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i],numcols[i], biascols[i]))
  
}

remove(edx_train_genres_movie, edx_train_genres_user)

total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices <- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))





edx_train_date <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')
edx_train_date <- edx_train_date %>%
  left_join(user_bias, by = 'userId')

invisible(gc())

train_genre_ind <- -1 + which(colnames(edx_train_date) == "genre1") %>% as.integer()

train_genrebias_reg <- matrix(ncol = maxgenre, nrow = nrow(edx_train_date))
edx_train_date <- as_tibble(edx_train_date)
invisible(gc())

for(i in 1:maxgenre){
  
 foo <- merge.data.frame(edx_train_date, genre_bias_regularized, by.x = (train_genre_ind + i), by.y = 1, all.x = TRUE) 
train_genrebias_reg[,i] <- foo$effect_bg
remove(foo)
invisible(gc())
}


train_genrebias_reg[is.na(train_genrebias_reg)] <- 0
train_genrebias_reg <- rowSums(train_genrebias_reg)
  
  

edx_train_date <- edx_train_date %>% mutate(effect_bg = train_genrebias_reg)
remove(train_genrebias_reg)


edx_train_date <- edx_train_date %>% mutate(error = rating - mu -effect_bi - effect_bu - effect_bg)

linear_date <- edx_train_date %>% lm(error ~ timestamp,data = .)

debug_time <- predict(linear_date, newdata = data.frame(timestamp = edx_debug_genres$timestamp))

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg_temp <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))

for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg_temp[,i] <- foo$effect_bg
}

debug_genrebias_reg_temp[is.na(debug_genrebias_reg_temp)] <- 0
debug_genrebias_reg_temp <- rowSums(debug_genrebias_reg_temp)




time_effects <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg_temp + debug_time

rmse7 <- RMSE(edx_debug_genres$rating, time_effects)
rmse7
remove(debug_time, debug_genrebias_reg_temp)

# Average of genre biases (copy-pasting the cumulative genre biases model with one small change)

remove(linear_date, movie_bias, user_bias)
invisible(gc())
# x is index for parameter of best lambda
x <- which.min(rmses)

# movie bias
movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')



# user bias
user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

edx_train_genres_user <- edx_train_genres_movie %>% left_join(user_bias, by = "userId")


# genre bias
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_genres_user) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_genres_user %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i],numcols[i], biascols[i]))
  
}


total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices <- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))


remove(edx_train_genres_movie, edx_train_genres_user)

# joining the movie and user biases to the table
edx_train_date <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')
edx_train_date <- edx_train_date %>%
  left_join(user_bias, by = 'userId')

invisible(gc())

# adding genre biases to edx_train_date
train_genre_ind <- -1 + which(colnames(edx_train_date) == "genre1") %>% as.integer()

train_genrebias_reg <- matrix(ncol = maxgenre, nrow = nrow(edx_train_date))
edx_train_date <- as_tibble(edx_train_date)
invisible(gc())

for(i in 1:maxgenre){
  
 foo <- merge.data.frame(edx_train_date, genre_bias_regularized, by.x = (train_genre_ind + i), by.y = 1, all.x = TRUE) 
train_genrebias_reg[,i] <- foo$effect_bg
remove(foo)
invisible(gc())
}


train_genrebias_reg <- rowMeans(train_genrebias_reg, na.rm = TRUE)

edx_train_date <- edx_train_date %>% mutate(effect_bg = train_genrebias_reg)
remove(train_genrebias_reg)

# error
edx_train_date <- edx_train_date %>% mutate(error = rating - mu -effect_bi - effect_bu - effect_bg)

# linear model

linear_date <- edx_train_date %>% lm(error ~ timestamp,data = .)


# testing
debug_time <- predict(linear_date, newdata = data.frame(timestamp = edx_debug_genres$timestamp))

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg_temp <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))

for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg_temp[,i] <- foo$effect_bg
}


debug_genrebias_reg_temp <- rowMeans(debug_genrebias_reg_temp, na.rm = TRUE)

time_effects <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg_temp + debug_time

rmse8 <- RMSE(edx_debug_genres$rating, time_effects)
rmse8

remove(debug_time, debug_genrebias_reg_temp)

remove(linear_date, movie_bias, user_bias)

invisible(gc())
# x is index for parameter of best lambda, rmses_2 used
# as lambda parameter also applies when averaging genre biases.
x <- which.min(rmses_2)

# movie bias
movie_bias <- edx_train_genres %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+params_lambda[x])) %>% select(-test, -n)

invisible(gc())

edx_train_genres_movie <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')

# user bias
user_bias <- edx_train_genres_movie %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+params_lambda[x])) %>% select(-test, -n)

remove(edx_train_genres_movie)

# joining movie and user effects to the table
edx_train_date <- edx_train_genres %>% left_join(movie_bias, by = 'movieId')
edx_train_date <- edx_train_date %>%
  left_join(user_bias, by = 'userId')


# genre bias
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_train_date) == 'genre1')
for(i in 1:maxgenre){
                     
  genre_effects_reg[[i]] <- setNames(edx_train_date %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+params_lambda[x])) %>% select(-test), 
                                 c(genrecols[i],numcols[i], biascols[i]))
  
}

remove(edx_train_genres_movie, edx_train_genres_user)

total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

n_indices <- which(colnames(total_genre_effects) %in% numcols)

bias_indices <- which(colnames(total_genre_effects) %in% biascols)

dat <- sapply(1:maxgenre, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))



invisible(gc())

# adding genre biases to edx_train_date 
train_genre_ind <- -1 + which(colnames(edx_train_date) == "genre1") %>% as.integer()

train_genrebias_reg <- matrix(ncol = maxgenre, nrow = nrow(edx_train_date))
edx_train_date <- as_tibble(edx_train_date)
invisible(gc())

for(i in 1:maxgenre){
  
 foo <- merge.data.frame(edx_train_date, genre_bias_regularized, by.x = (train_genre_ind + i), by.y = 1, all.x = TRUE) 
train_genrebias_reg[,i] <- foo$effect_bg
remove(foo)
invisible(gc())
}

train_genrebias_reg <- apply(train_genrebias_reg,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u)) + params_lambda[x])})

edx_train_date <- edx_train_date %>% mutate(effect_bg = train_genrebias_reg)
remove(train_genrebias_reg)

# error
edx_train_date <- edx_train_date %>% mutate(error = rating - mu -effect_bi - effect_bu - effect_bg)

invisible(gc())

# linear model
linear_date <- edx_train_date %>% lm(error ~ timestamp,data = .)

# testing 
debug_time <- predict(linear_date, newdata = data.frame(timestamp = edx_debug_genres$timestamp))

debug_genre_ind <- -1 + which(colnames(edx_debug_genres) == "genre1")

debug_genrebias_reg_temp <- matrix(ncol = debug_genre_count, nrow = nrow(edx_debug_genres))

for(i in 1:debug_genre_count){
  
 foo <- merge(edx_debug_genres, genre_bias_regularized, by.x = (debug_genre_ind + i), by.y = 1, all.x = TRUE) 
debug_genrebias_reg_temp[,i] <- foo$effect_bg
}


debug_genrebias_reg_temp <- apply(debug_genrebias_reg_temp,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u)) + params_lambda[x])})

time_effects <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg_temp + debug_time

rmse9 <- RMSE(edx_debug_genres$rating, time_effects)
rmse9


# this can be done because the third dataset's genre biases is the one that 
# is currently loaded in the environment. So the temporary genre biases can be used going forward.
debug_genrebias_reg <- debug_genrebias_reg_temp

# code to recreate edx_train_date not used since the required model's error values are already loaded in edx_train_date 

noquote("R Squared")
glance(linear_date)$r.squared

noquote("P Value")
glance(linear_date)$p.value

edx_train_date %>% ggplot(aes(as_datetime(timestamp), error)) + geom_point() + xlab("Time") + ylab("Error") 


# Date format is YYYY-MM-DD
startdate <- as.integer(as_datetime("1996-01-01 00:00:00 UTC"))
enddate <- as.integer(as_datetime("1996-12-31 23:59:59 UTC"))
edx_train_date %>% filter(timestamp >= startdate & timestamp <= enddate) %>% ggplot(aes(as_datetime(timestamp), error)) + geom_point() + xlab("Time") + ylab("Error")

# RAM Optimization, linear date is a large lm which, if saved, cannot be reloaded due to its large size
save(linear_date, file = "linear_date.RData")
remove(linear_date)

# group_split splits the train set into multiple data frames, each with one movieId.

movie_split <- group_split(edx_train_date, movieId)
# map_dbl and map are used to store the moveId and the linear model, map stores lists so this is why it can be used to store lms, as lm class is a type of list.
movie_lm <- tibble(movieId = map_dbl(movie_split, function(x) return(x$movieId[1])),
                   lm = map(movie_split, ~ lm(error ~ timestamp, data = .x)))
remove(movie_split)

invisible(gc())

# map2_dbl can take two inputs and produce one output

# Suppresswarnings used to prevent the following warning:
# Warning: Problem with `mutate()` column `foo`.
# i `foo = map2_dbl(.x = lm, .y = timestamp, ~predict(.x, newdata = tibble(timestamp = .y)))`.
# i prediction from a rank-deficient fit may be misleading
suppressWarnings(debug_movie_time <- edx_debug_genres %>% left_join(movie_lm, by = "movieId") %>% mutate(foo = map2_dbl(.x = lm, .y = timestamp, ~ predict(.x, newdata = tibble(timestamp = .y)))) %>% .$foo)

# testing
invisible(gc())
time_effects_movie <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg + debug_movie_time

rmse10 <- RMSE(edx_debug_genres$rating, time_effects_movie)
rmse10



rsq = map_dbl(.x = movie_lm$lm, function(u) {glance(u)$r.squared})
rsq %>% head(10)
remove(rsq, movie_lm)

edx_train_date %>% filter(movieId == 1) %>% ggplot(aes(as_datetime(timestamp), error)) + geom_point() + xlab("Time") + ylab("Error")


edx_train_date %>% filter(userId == 1) %>% ggplot(aes(as_datetime(timestamp), error)) + geom_point() + xlab("Time") + ylab("Error")



# same as movie_lm, but with users
invisible(gc())
user_split <- group_split(edx_train_date, userId)
user_lm <- tibble(userId = map_dbl(user_split, function(x) return(x$userId[1])),
                   lm = map(user_split, ~ lm(error ~ timestamp, data = .x)))
remove(user_split)

# testing, but with user time effects
invisible(gc())
suppressWarnings(debug_user_time <- edx_debug_genres %>% left_join(user_lm, by = "userId") %>% mutate(foo = map2_dbl(.x = lm, .y = timestamp, ~ predict(.x, newdata = tibble(timestamp = .y)))) %>% .$foo)


time_effects_user <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg + debug_user_time

rmse11 <- RMSE(edx_debug_genres$rating, time_effects_user)
rmse11

rsq = map_dbl(.x = user_lm$lm, function(u) {glance(u)$r.squared})
rsq %>% head(10)

remove(rsq, user_lm, time_effects, time_effects_movie, time_effects_user)

invisible(gc())
edx_train_date <- edx_train_date %>% mutate(t2 = timestamp^2)

quadratic_date <- edx_train_date %>% lm(error ~ timestamp + t2,data = .)

debug_time_2 <- predict(quadratic_date, newdata = data.frame(timestamp = edx_debug_genres$timestamp, t2 = (edx_debug_genres$timestamp)^2))

time_effects_2 <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg + debug_time_2

rmse12 <- RMSE(time_effects_2, edx_debug_genres$rating)
rmse12

noquote("R Squared")
glance(quadratic_date)$r.squared

noquote("P Value")
glance(quadratic_date)$p.value

remove(quadratic_date)

edx_train_date <- edx_train_date %>% mutate(t3 = timestamp^3)

cubic_date <- edx_train_date %>% lm(error ~ timestamp + t2 + t3,data = .)

debug_time_3 <- predict(cubic_date, newdata = data.frame(timestamp = edx_debug_genres$timestamp, t2 = (edx_debug_genres$timestamp)^2, t3 = (edx_debug_genres$timestamp)^3))

time_effects_3 <- mu + {edx_debug_genres %>% left_join(movie_bias, by = "movieId") %>% 
  left_join(user_bias, by = "userId") %>% mutate(total_effect = effect_bi + effect_bu) %>%
  pull(total_effect)}  + debug_genrebias_reg + debug_time_3

rmse13 <- RMSE(time_effects_3, edx_debug_genres$rating)
rmse13


noquote("R Squared")
glance(cubic_date)$r.squared

noquote("P Value")
glance(cubic_date)$p.value

remove(cubic_date)


dir <- getwd()
load(paste0(dir,"/importants.Rdata"))

# Show that ratings can be considered a classification problem
levels(as.factor(edx_train$rating))


# as.factor to ensure ranger uses classification
edx_train_2 <- edx_train %>% mutate(rating = as.factor(rating)) 


fit_ranger <- ranger(formula = rating ~ ., data = edx_train_2, num.trees = 100, max.depth = 10)




edx_debug_2 <- edx_debug %>% mutate(rating = as.factor(rating))
test <- predict(fit_ranger, data = edx_debug_2)

rmse14 <- RMSE(as.numeric(test$predictions), as.numeric(edx_debug_2$rating))
rmse14


table(test$predictions)
table(edx_debug_2$rating)

# rating is a numeric in edx_train and edx_debug so ranger will default to regression
remove(fit_ranger)
fit_ranger <- ranger(formula = rating ~ ., data = edx_train, num.trees = 100, max.depth = 10)

test <- predict(fit_ranger, data = edx_debug)
rmse15 <- RMSE(test$predictions, edx_debug$rating)
rmse15

# try again because why not?

# rating is a numeric in edx_train and edx_debug so ranger will default to regression
remove(fit_ranger)
invisible(gc())
fit_ranger <- ranger(formula = rating ~ ., data = edx_train, num.trees = 200, max.depth = 10)

test <- predict(fit_ranger, data = edx_debug)
rmse16 <- RMSE(test$predictions, edx_debug$rating)
rmse16
remove(fit_ranger)



results <- data.frame(method = c("Only the Mean","Movie Effects", "Movie and User Effects", "Movie, User and Genre Effects",
                  "Regularized Movie and User Effects",
                  "Regularized Movie, User and Cumulative Genre Effects", "Regularized Movie, User and Averaged Genre Effects", "Regularized Movie, User and Penalized Average Genre Effects", "Time Effects + Regularized Movie, User and Cumulative Genre Effects", "Time Effects + Regularized Movie, User and Averaged Genre Effects", "Time Effects + Regularized Movie, User and Penalized Average Genre Effects (Linear)","Time Effects + Regularized Movie, User and Penalized Average Genre Effects (Quadratic)","Time Effects + Regularized Movie, User and Penalized Average Genre Effects (Cubic)" ,"Movie Effects over time + Regularized Movie, User and Penalized Average Genre Effects", "User Effects over time + Regularized Movie, User and Penalized Average Genre Effects", "Ranger Random Forest (Classification, 100 Trees)", "Ranger Random Forest (Regression, 100 Trees)", "Ranger Random Forest (Regression, 200 Trees)"), RMSE = c(rmse0,rmse1, rmse2,  rmse3, rmsea,rmse4, rmse5, rmse6, rmse7, rmse8, rmse9, rmse12,rmse13,rmse10, rmse11, rmse14, rmse15, rmse16))
print(results, right = F)


results$method[which.min(results$RMSE)]

# Saving all data frames to free up RAM
stuff <- ls()

ind <- which(!stuff %in% c("edx", "validation", "rmses_2", "params_lambda", "stuff", "join_genres"))

stuff <- stuff[ind]

# save(list = stuff,file = "all_non_main.RData")
remove(list = stuff)

#Storing the lambda parameter
lambda <- params_lambda[which.min(rmses_2)]

invisible(gc())
# training on all of edx
edx_2 <- edx 


save(edx, file = 'edx.Rdata')
remove(edx)


#maximum number of genres in edx set


edx_2_genrecount <- 1+ str_count(edx_2$genres, "\\|") %>% max()


edx_2_genrecols <- paste0("genre", 1:edx_2_genrecount)
edx_2_numcols <- paste0("n", 1:edx_2_genrecount)
edx_2_biascols <- paste0("bias", 1:edx_2_genrecount)

invisible(gc())
# edx_2 can't be split directly due to ram constraints so it will be divided first
rows_edx <- nrow(edx_2)
half_rows <- round(rows_edx/2,0)
edx_2_a <- edx_2[(1:half_rows),]
edx_2_b <- edx_2[((half_rows+1):rows_edx),]
remove(edx_2)
edx_2_a <- edx_2_a %>% separate(data = ., col = genres, into = edx_2_genrecols, sep = "\\|")

edx_2_b <- edx_2_b %>% separate(data = ., col = genres, into = edx_2_genrecols, sep = "\\|")
edx_2 <- bind_rows(edx_2_a, edx_2_b)
# Mean

mu <- mean(edx_2$rating)
# Recreating Movie Bias
movie_bias <- edx_2 %>% group_by(movieId) %>% summarize(test = sum(rating - mu), n = n()) %>% mutate(effect_bi = test/(n+lambda)) %>% select(-test, -n)


edx_2 <- edx_2 %>% left_join(movie_bias, by = "movieId")
# Recreating user bias

user_bias <- edx_2 %>% group_by(userId) %>% summarize(test = sum(rating - mu - effect_bi), n = n()) %>% mutate(effect_bu = test/(n+lambda)) %>% select(-test, -n)

edx_2 <- edx_2 %>% left_join(user_bias, by = "userId")
#Recreating genre bias
genre_effects_reg <- list()
ind <- -1 + which(colnames(edx_2) == 'genre1')
for(i in 1:edx_2_genrecount){
                     
  genre_effects_reg[[i]] <- setNames(edx_2 %>% group_by_at(ind+i) %>%
                                   summarize(test = sum(rating - mu - effect_bi - effect_bu), n = n()) %>% mutate(effect_bg = test/(n+lambda)) %>% select(-test), 
                                 c(edx_2_genrecols[i],edx_2_numcols[i], edx_2_biascols[i]))
  
}



total_genre_effects <- Reduce(join_genres, genre_effects_reg)
total_genre_effects[is.na(total_genre_effects)] <- 0

remove(genre_effects_reg)
n_indices <- which(colnames(total_genre_effects) %in% edx_2_numcols)

bias_indices <- which(colnames(total_genre_effects) %in% edx_2_biascols)

dat <- sapply(1:edx_2_genrecount, function(i){
  biascol <- total_genre_effects[, bias_indices[i]]
  numcol <-  total_genre_effects[, n_indices[i]]
  return(biascol*numcol)
})

genre_bias_regularized <- data.frame(genre = total_genre_effects$genre1, effect_bg = rowSums(dat)/rowSums(total_genre_effects[, n_indices]))



edx2_genre_bias <- matrix(ncol = edx_2_genrecount, nrow = nrow(edx_2))

invisible(gc())

for(i in 1:edx_2_genrecount){
  
 foo <- merge.data.frame(edx_2, genre_bias_regularized, by.x = (ind + i), by.y = 1, all.x = TRUE) 
edx2_genre_bias[,i] <- foo$effect_bg
remove(foo)
invisible(gc())
}

edx2_genre_bias <- apply(edx2_genre_bias,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u)) + lambda)})

edx_2 <- edx_2 %>% mutate(effect_bg = edx2_genre_bias)
remove(edx2_genre_bias)

edx_2 <- edx_2 %>% mutate(error = rating - mu - effect_bi - effect_bu - effect_bg)

edx_2 <- edx_2 %>% mutate(t2 = timestamp^2)

quadratic_date <- lm(error ~ timestamp + t2, data = edx_2)


# testing on validation

validation_2 <- validation

validation_2_genrecount <- 1 + str_count(validation_2$genres, "\\|") %>% max()

validation_2_genrecols <- paste0("genre", 1:validation_2_genrecount)


validation_2 <- separate(data = validation_2, col = genres, into = validation_2_genrecols, sep = "\\|")


validation_2_time <- predict(quadratic_date, newdata = data.frame(timestamp = validation_2$timestamp, t2 = (validation_2$timestamp)^2))

remove(quadratic_date)

validation_index <- -1 + which(colnames(validation_2) == "genre1")

validation_genre_bias <- matrix(ncol = validation_2_genrecount, nrow = nrow(validation_2))

for(i in 1:validation_2_genrecount){
  
 foo <- merge.data.frame(validation_2, genre_bias_regularized, by.x = (validation_index + i), by.y = 1, all.x = TRUE) 
validation_genre_bias[,i] <- foo$effect_bg
}


validation_final_genre_bias <- apply(validation_genre_bias,1,FUN = function(u){sum(u, na.rm = TRUE)/(sum(!is.na(u)) + lambda)})

yhat <- mu + {validation %>% left_join(movie_bias, by = 'movieId') %>% left_join(user_bias, by = 'userId') %>% mutate(total = effect_bi + effect_bu) %>% .$total} + validation_2_time + validation_final_genre_bias

# identical(validation_2$rating, validation$rating)
# the two vectors are the same so it doesn't make a difference
# as to which one is used to calculate the RMSE.

rmse_final <- RMSE(yhat, validation_2$rating)
rmse_final


# remove all Rdata files

file.remove('importants.Rdata')
file.remove('linear_date.Rdata')
# file.remove('all_non_main.Rdata')
file.remove('edx.Rdata')
