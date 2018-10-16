# Sci-kit_learn_medical_tweets_classification
Classification of personal medication intake 

OBJECTIVE:
•	To read tweets from twitter and save it in a text file along with classes.
•	To read data from text file and perform vectorization.
•	To perform Sk-learn algorithms and compare results with actual data with the help of accuracy scores and confusion matrix

Procedure:
1.	I used the file "tweets" to read data from twitter using tweed ids. 
2.  I used the code “get_twitter” to get data from the "tweets" file.
3.  Then I saved the output in data.text which contains tweets along with their id and classes.
4.  I read through the file “data.txt” line by line , and its columns are separated by tab, I split every line into tabs and stored   classes which are at index 3 in “list_of_classes” and tweets which are at index 4 in “list_of_tweets” 
5.	After performing SVD and saved it in file “lsafinal.txt”
6.  I used train_test_split to to split data into training and test sets.
7.	Then I used “Count Vectorizer”  to covert tweets into features and then I used “Truncated Svd” for decomposing that features into 300.
8.	Then I used 10_fold_cross_validation technique to randomly split training set into k folds without replacement,(k-1)folds for model training & 1 fold for performance evaluation.
9.	I have used f scores, confusion matrix and accuracy score to test the performance of every model.
10.	I have done evaluation using algorithms knn, svm, logistic regression, decision tree and random forest



