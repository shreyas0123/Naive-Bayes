##################### problem1 ###########################
#1.) Prepare a classification model using Naive Bayes for Salary dataset, train and test datasets are given separately use both datasets for model building. 	

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
salary_test = pd.read_csv("C:/Users/DELL/Downloads/SalaryData_Test.csv",encoding = "ISO-8859-1")
salary_train = pd.read_csv("C:/Users/DELL/Downloads/SalaryData_Train.csv",encoding = "ISO-8859-1")

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
x = {" <=50K" :1," >50K" :2}
salary_test.Salary = [x[item] for item in salary_test.Salary]
salary_train.Salary = [x[item] for item in salary_train.Salary]

#dummy variables for test data
dummies_salary_test = pd.get_dummies(salary_test)
dummies_salary_test.drop(['Salary'],axis = 1,inplace =True)
dummies_salary_test.isna()

#dummy variables for train data
dummies_salary_train = pd.get_dummies(salary_train)
dummies_salary_train.drop(['Salary'],axis = 1,inplace =True)
dummies_salary_train.isna()

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(dummies_salary_train, salary_train.Salary)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(dummies_salary_test)
accuracy_test_m = np.mean(test_pred_m == salary_test.Salary)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, salary_test.Salary) 

pd.crosstab(test_pred_m, salary_test.Salary)

# Training Data accuracy
train_pred_m = classifier_mb.predict(dummies_salary_train)
accuracy_train_m = np.mean(train_pred_m == salary_train.Salary)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(dummies_salary_train, salary_train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(dummies_salary_test)
accuracy_test_lap = np.mean(test_pred_lap == salary_test.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, salary_test.Salary) 

pd.crosstab(test_pred_lap, salary_test.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(dummies_salary_train)
accuracy_train_lap = np.mean(train_pred_lap == salary_train.Salary)
accuracy_train_lap


######################## problem2 #####################
#Problem Statement: -
#This dataset contains information of users in social network. This social network has several business clients which can put their ads on social network and one of the Client has a car company who has just launched a luxury SUV for ridiculous price. Build the Bernoulli Naïve Bayes model using this dataset and classify which of the users of the social network are going to purchase this luxury SUV.
#Purchased: - 1 and Not Purchased: - 0

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
suv_car = pd.read_csv("C:/Users/DELL/Downloads/NB_Car_Ad.csv",encoding = "ISO-8859-1")

#dropping first column for analysis
suv_car.drop(['User ID'],axis = 1 ,inplace = True)

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
suv_car [['Age','EstimatedSalary']] = scaler.fit_transform(suv_car [['Age','EstimatedSalary']])


# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

suv_car_train, suv_car_test = train_test_split(suv_car, test_size = 0.2)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

#dummies for test and train data
suv_car_dumm_train = pd.get_dummies(suv_car_train)
suv_car_dumm_train.drop(["Purchased"], axis = 1, inplace = True)

suv_car_dumm_test = pd.get_dummies(suv_car_test)
suv_car_dumm_test.drop(["Purchased"], axis = 1, inplace = True)

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(suv_car_dumm_train, suv_car_train.Purchased)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(suv_car_dumm_test)
accuracy_test_m = np.mean(test_pred_m == suv_car_test.Purchased)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, suv_car_test.Purchased) 

pd.crosstab(test_pred_m, suv_car_test.Purchased)

# Training Data accuracy
train_pred_m = classifier_mb.predict(suv_car_dumm_train)
accuracy_train_m = np.mean(train_pred_m == suv_car_train.Purchased)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(suv_car_dumm_train, suv_car_train.Purchased)


# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(suv_car_dumm_test)
accuracy_test_lap = np.mean(test_pred_lap == suv_car_test.Purchased)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, suv_car_test.Purchased) 

pd.crosstab(test_pred_lap, suv_car_test.Purchased)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(suv_car_dumm_train)
accuracy_train_lap = np.mean(train_pred_lap == suv_car_train.Purchased)
accuracy_train_lap

################### problem3 #################################
#Problem Statement: -
#In this case study you have been given with tweeter data collected from an anonymous twitter handle, with the help of Naïve Bayes algorithm predict a given tweet is Fake or Real about real disaster occurring. 
#Real tweet: - 1 and Fake tweet: - 0

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
tweet = pd.read_csv("C:/Users/DELL/Downloads/Disaster_tweets_NB.csv",encoding = "ISO-8859-1")
tweet = tweet.iloc[:,3:5]
# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("C:/Users/DELL/Downloads/stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, numbers
tweet.text = tweet.text.apply(cleaning_text)

# removing empty rows
tweet = tweet.loc[tweet.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

tweet_train, tweet_test = train_test_split(tweet, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words
tweet_bow = CountVectorizer(analyzer = split_into_words).fit(tweet.text)

# Defining BOW for all messages
all_tweet_matrix = tweet_bow.transform(tweet.text)

# For training messages
train_tweet_matrix = tweet_bow.transform(tweet_train.text)

# For testing messages
test_tweet_matrix = tweet_bow.transform(tweet_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweet_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_tweet_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_tweet_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == tweet_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, tweet_test.target) 

pd.crosstab(test_pred_m, tweet_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == tweet_train.target)
accuracy_train_m

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, tweet_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == tweet_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, tweet_test.target) 

pd.crosstab(test_pred_lap, tweet_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == tweet_train.target)
accuracy_train_lap

################# END ########################################









