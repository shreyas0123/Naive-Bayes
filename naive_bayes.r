############### problem1 ######################
#1.) Prepare a classification model using Naive Bayes for Salary dataset, train and test datasets are given separately use both datasets for model building. 	

# Import  dataset
library(readr)
salary_test <- read.csv(file.choose())
salary_train <- read.csv(file.choose())

#Data preprocessing for test data
salary_test$Salary <- ifelse(salary_test$Salary == " <=50K",0,1)

str(salary_test)
colnames(salary_test)

#factor analysis for test data
cols <- c("age","workclass","education","educationno","maritalstatus","occupation","relationship","race","sex","capitalgain","capitalloss","hoursperweek","native","Salary")
salary_test[cols] <- lapply(salary_test[cols],factor)
str(salary_test)


#Data preprocessing for train data
salary_train$Salary <- ifelse(salary_train$Salary == " <=50K",0,1)

str(salary_train)
colnames(salary_train)

#factor analysis for train data
col <- c("age","workclass","education","educationno","maritalstatus","occupation","relationship","race","sex","capitalgain","capitalloss","hoursperweek","native","Salary")
salary_train[col] <- lapply(salary_train[col],factor)
str(salary_train)


#viewing str and table
str(salary_test$Salary)
table(salary_test$Salary)

str(salary_train$Salary)
table(salary_train$Salary)

# proportion of salaries 0(<=50k) and 1(>50k)
prop.table(table(salary_test$Salary))
prop.table(table(salary_train$Salary))

##  Training a model on the data ----
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
salary_classifier <- naiveBayes(Salary~., data = salary_train)
salary_classifier

### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
salary_lap <- naiveBayes(Salary~., data = salary_train,laplace = 3)
salary_lap

##  Evaluating model performance without laplace
salary_test_pred <- predict(salary_classifier, salary_test)

# Evaluating model performance after applying laplace smoothing
salary_test_pred_lap <- predict(salary_lap, salary_test)

## crosstable without laplace
install.packages("gmodels")
library(gmodels)

CrossTable(salary_test_pred, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
## test accuracy
test_acc <- mean(salary_test_pred == salary_test$Salary)
test_acc

## crosstable of laplace smoothing model
CrossTable(salary_test_pred_lap, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(salary_test_pred_lap == salary_test$Salary)
test_acc_lap

# On Training Data without laplace 
salary_train_pred <- predict(salary_classifier, salary_train)
salary_train_pred

# train accuracy
train_acc = mean(salary_train_pred == salary_train$Salary)
train_acc

# prediction on train data for laplace model
salary_train_pred_lap <- predict(salary_lap,salary_train)
salary_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(salary_train_pred_lap == salary_train$Salary)
train_acc_lap

####################### problem2 ##################################
#Problem Statement: -
#This dataset contains information of users in social network. This social network has several business clients which can put their ads on social network and one of the Client has a car company who has just launched a luxury SUV for ridiculous price. Build the Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV.
#Purchased: - 1 and Not Purchased: - 0

# Import the dataset
library(readr)
suv_car <- read.csv(file.choose())
suv_car <- suv_car[c(-1)]
View(suv_car)
str(suv_car)
colnames(suv_car)

#data preprocessing
col <- c("Gender","Age","EstimatedSalary","Purchased")
col <- c("Purchased")
suv_car[col] <- lapply(suv_car[col],factor)
suv_car[c(3)] <- scale(suv_car[c(3)])
str(suv_car)

# examine the type variable more carefully
str(suv_car$Purchased)
table(suv_car$Purchased)

# proportion of 0(not purchased) and 1(purchased) messages
prop.table(table(suv_car$Purchased))

str(suv_car)

# creating training and test datasets
suv_car_train <- suv_car[1:280, ]
suv_car_test  <- suv_car[280:400, ]


# check that the proportion of 0(not purchased) and 1(purchased)  is similar
prop.table(table(suv_car$Purchased))

prop.table(table(suv_car_train$Purchased))
prop.table(table(suv_car_test$Purchased))

##  Training a model on the data ----
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
suv_car_classifier <- naiveBayes(Purchased~., data = suv_car_train)
suv_car_classifier

?naiveBayes
### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
suv_car_lap <- naiveBayes(Purchased~.,data = suv_car_train ,laplace = 3)
suv_car_lap

##  Evaluating model performance with out laplace
suv_car_test_pred <- predict(suv_car_classifier, suv_car_test)

# Evaluating model performance after applying laplace smoothing
suv_car_test_pred_lap <- predict(suv_car_lap, suv_car_test)

## crosstable without laplace
install.packages("gmodels")
library(gmodels)

CrossTable(suv_car_test_pred, suv_car_test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy
test_acc <- mean(suv_car_test_pred ==suv_car_test$Purchased)
test_acc

## crosstable of laplace smoothing model
CrossTable(suv_car_test_pred_lap, suv_car_test$Purchased,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(suv_car_test_pred_lap == suv_car_test$Purchased)
test_acc_lap

# On Training Data without laplace 
suv_car_train_pred <- predict(suv_car_classifier, suv_car_train)
suv_car_train_pred


# train accuracy
train_acc = mean(suv_car_train_pred == suv_car_train$Purchased)
train_acc


# prediction on train data for laplace model
suv_car_train_pred_lap <- predict(suv_car_lap,suv_car_train)
suv_car_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(suv_car_train_pred_lap == suv_car_train$Purchased)
train_acc_lap

################### problem3 ####################################
#Problem Statement: -
#In this case study you have been given with tweeter data collected from an anonymous twitter handle, with the help of Naïve Bayes algorithm predict a given tweet is Fake or Real about real disaster occurring. 
#Real tweet: - 1 and Fake tweet: - 0

# Import the twitter dataset
library(readr)
tweet <- read.csv(file.choose())
View(sms_raw)

str(tweet)

tweet$target <- factor(tweet$target)
# examine the type variable more carefully
str(tweet$target)
table(tweet$target)

# proportion of ham and spam messages
prop.table(table(tweet$target))

# build a corpus using the text mining (tm) package
install.packages("tm")
library(tm)
str(tweet$text)
tweet_corpus <- Corpus(VectorSource(tweet$text))
# str(sms_corpus)

tweet_corpus <- tm_map(tweet_corpus, function(x) iconv(enc2utf8(x), sub='byte'))
# clean up the corpus using tm_map()
corpus_clean <- tm_map(tweet_corpus, tolower)
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# create a document-term sparse matrix
tweet_dtm <- DocumentTermMatrix(corpus_clean)
tweet_dtm
View(tweet_dtm[1:10, 1:30])
# To view DTM we need to convert it into matrix first
dtm_matrix <- as.matrix(tweet_dtm)
str(dtm_matrix)

View(dtm_matrix[1:10, 1:20])

colnames(tweet_dtm)[1:50]
# creating training and test datasets
tweets_train <- tweet[1:5329, ]
tweets_test  <- tweet[5330:7613, ]

tweet_corpus_train <- corpus_clean[1:5329]
tweet_corpus_test  <- corpus_clean[5330:7613]

tweet_dtm_train <-tweet_dtm[1:5329, ]
tweet_dtm_test  <- tweet_dtm[5330:7613, ]

# check that the proportion of spam is similar
prop.table(table(tweet$target)) 

prop.table(table(tweet_train$target))
prop.table(table(tweet_test$target))

# indicator features for frequent words
# dictionary of words which are used more than 5 times
tweet_dict <- findFreqTerms(tweet_dtm_train, 5)
tweet_train <- DocumentTermMatrix(tweet_corpus_train, list(dictionary = tweet_dict))
tweet_test  <- DocumentTermMatrix(tweet_corpus_test, list(dictionary = tweet_dict))

tweet_test_matrix <- as.matrix(tweet_test)
View(tweet_test_matrix[1:10,1:10])

# convert counts to a factor
# custom function: if a word is used more than 0 times then mention 1 else mention 0
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
# Margin = 2 is for columns
# Margin = 1 is for rows
tweet_train <- apply(tweet_train, MARGIN = 2, convert_counts)
tweet_test  <- apply(tweet_test, MARGIN = 2, convert_counts)
?apply()

View(tweet_test[1:10,1:10])

##  Training a model on the data ----
install.packages("e1071")
library(e1071)
## building naiveBayes classifier.
tweet_classifier <- naiveBayes(tweet_train, tweets_train$target)
tweet_classifier

### laplace smoothing, by default the laplace value = 0
## naiveBayes function has laplace parameter, the bigger the laplace smoothing value, 
# the models become same.
tweet_lap <- naiveBayes(tweet_train, tweets_train$target,laplace = 3)
tweet_lap

##  Evaluating model performance with out laplace
tweet_test_pred <- predict(tweet_classifier, tweet_test)

# Evaluating model performance after applying laplace smoothing
tweet_test_pred_lap <- predict(tweet_lap, tweet_test)

## crosstable without laplace
install.packages("gmodels")
library(gmodels)

CrossTable(tweet_test_pred, tweets_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
## test accuracy
test_acc <- mean(tweet_test_pred == tweets_test$target)
test_acc

## crosstable of laplace smoothing model
CrossTable(tweet_test_pred_lap, tweets_test$target,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## test accuracy after laplace 
test_acc_lap <- mean(tweet_test_pred_lap == tweets_test$target)
test_acc_lap

# On Training Data without laplace 
tweet_train_pred <- predict(tweet_classifier, tweet_train)
tweet_train_pred

# train accuracy
train_acc = mean(tweet_train_pred == tweets_train$target)
train_acc


# prediction on train data for laplace model
tweet_train_pred_lap <- predict(tweet_lap,tweet_train)
tweet_train_pred_lap

# train accuracy after laplace
train_acc_lap = mean(tweet_train_pred_lap == tweets_train$target)
train_acc_lap


