import warnings
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
import re
import sklearn
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import decomposition
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=DeprecationWarning)


list_of_classes = []
list_of_tweets = []
f_open = open('data.txt','r')

for line in f_open:
    temp = line.split("\t")
    num_parts = int(len(temp))
    the_class = temp[3]
    text_temp = temp[4:num_parts]
    tweet_string = ' '.join(text_temp)
    tweet_string = re.sub(r'[^\x00-\x7f]',r' ',tweet_string)
    string = re.sub(r'[^\x00-\x7F]', ' ', tweet_string) #remove unicodes
    string = string.replace(","," ")
    string = string.replace(".", " ")
    string = string.replace("!", " ")
    string = string.replace('*', " ")
    string = string.replace('"', " ")
    string  =string.replace("\n"," ")
    string = string.replace("\t"," ")
    string = re.sub('\s+', ' ', string)
    string = re.sub('[^0-9a-zA-Z]+', ' ', string)
    #tokens = word_tokenize(string)
    #string_as_tokens = [stemmer.stem(i.lower()) for i in tokens if i not in stop_words_list]
    list_of_classes.append(the_class)
    list_of_tweets.append(string)
X = list_of_tweets
y = list_of_classes
f_open.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

vectorizer = CountVectorizer(min_df=1, stop_words='english')
dtm = vectorizer.fit(X_train)
dtm_train = dtm.transform(X_train)
dtm_test  = dtm.transform(X_test)


lsa = TruncatedSVD(300)
dtm_lsa = lsa.fit(dtm_train)
dtm_lsa_train = dtm_lsa.transform(dtm_train)
dtm_lsa_test = dtm_lsa.transform(dtm_test)


#############################################################
def print_stats_10_fold_crossvalidation(algo_name, model, dtm_lsa_train, y_train ):
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    kfold = StratifiedKFold(y=y_train,
                        n_folds=10,
                        random_state=1)
    print "----------------------------------------------"
    print "Start of 10 fold crossvalidation results"
    print "the algorithm is: ", algo_name
    #################################
    #roc
    fig = plt.figure(figsize=(7,5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    ################################
    scores = []
    f_scores = []
    for k, (train, test) in enumerate(kfold):
        model.fit(dtm_lsa_train[train], y_train[train])
        y_pred = model.predict(dtm_lsa_train[test])
        ########################
        #roc
        probas = model.predict_proba(dtm_lsa_train[test])
        #pos_label in the roc_curve function is very important. it is the value
        #of your classes such as 1 or 2, for versicolor or setosa
        fpr, tpr, thresholds = roc_curve(y_train[test],probas[:,300], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, 
                 label='ROC fold %d (area = %0.2f)' % (k+1, roc_auc))

        ########################
        ## print results
        print('Accuracy: %.2f' % accuracy_score(y_train[test], y_pred))
        confmat = confusion_matrix(y_true=y_train[test], y_pred=y_pred)
        print "confusion matrix"
        print(confmat)
        print pd.crosstab(y_train[test], y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

        print('Precision: %.3f' % precision_score(y_true=y_train[test], y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=y_train[test], y_pred=y_pred))
        f_score = f1_score(y_true=y_train[test], y_pred=y_pred)
        print('F1-measure: %.3f' % f_score)
        f_scores.append(f_score)
        score = model.score(X_train[test], y_train[test])
        scores.append(score)
        print('fold : %s, Accuracy: %.3f' % (k+1, score))
        print "****************************************************************************************************************"
    ######################################
    #roc
    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plot_ROC_curve(plt, mean_fpr, mean_tpr, mean_auc )
    ######################################
    print('overall accuracy: %.3f and +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('overall f1_score: %.3f' % np.mean(f_scores))


#########################################################
## print stats train % and test percentage (i.e. 70% train
## and 30% test
## comparison function

def print_stats_percentage_train_test(algorithm_name, y_test, y_pred):
    print "---------   -------   --------------"
    print "the algorithm is: ", algorithm_name
    print('Accuracy is : %.2f' % accuracy_score(y_test, y_pred))
    print('Precision: %.3f' % precision_score(y_test, y_pred,average='weighted'))
    print('Recall: %.3f' % recall_score(y_test, y_pred,average='weighted'))
    print('F1-measure: %.3f' % f1_score(y_test, y_pred,average='weighted'))
    confmat = confusion_matrix(y_test, y_pred)
    print "confusion matrix"
    print(confmat)


###################################################
## knn

def knn_rc(X_train_std, y_train, X_test_std, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    #print y_pred
    #print "actual data"
    #print y_test
    print_stats_percentage_train_test("knn", y_test, y_pred) #  comparison real and predict

#######################################
## logistic regression
def logistic_regression_rc(X_train_std, y_train,X_test_std,y_test):
    lr = LogisticRegression(C=1000.0, random_state=1)
    lr.fit(X_train_std, y_train)
    #lr_result = lr.predict_proba(X_test_std[0, :])
    #print lr_result
    y_pred = lr.predict(X_test_std)
    print_stats_percentage_train_test("logitic regression", y_test, y_pred)
    #plot_2d_graph_model(lr, 398, 569, pca_X_train, pca_X_test, y_train, y_test)
    #print_stats_10_fold_crossvalidation("logistic_regr", lr, X_train_std, y_train)


#####################################################
## random forest

def random_forest_rc(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier( criterion='entropy',
                                     n_estimators=10,
                                     random_state=1,
                                     n_jobs=2)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    print_stats_percentage_train_test("random forest", y_test, y_pred)
#    print_stats_10_fold_crossvalidation("random forest", forest,X_train,y_train)
#######################################
## svm
def svm_rc(X_train_std, y_train, X_test_std, y_test):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    print_stats_percentage_train_test("svm", y_test,y_pred)
    #print_stats_10_fold_crossvalidation("svm (rbf)",svm,X_train_std,y_train )

#######################################
## decision trees

def decision_trees(X_train_std, y_train, X_test_std, y_test):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=30, random_state=0)
    tree.fit(X_train_std, y_train)
    y_pred = tree.predict(X_test_std)
    print_stats_percentage_train_test("decision trees", y_test, y_pred)
    #print_stats_10_fold_crossvalidation("decision trees",tree,X_train,y_train)

#######################################
# A perceptron

def simple_perceptron_rc(X_train_std, y_train, X_test_std, y_test):
    from sklearn.linear_model import Perceptron
    ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_test_std)
    print_stats_percentage_train_test("simple perceptron", y_test, y_pred) 
    #print_stats_10_fold_crossvalidation("simple_percp",ppn,X_train_std,y_train)

###################################################
## feature scaling (only scale X NOT Y)
sc = StandardScaler()
sc.fit(dtm_lsa_train)
X_train_std = sc.transform(dtm_lsa_train)
X_test_std = sc.transform(dtm_lsa_test)

#######################################
## ML_MAIN()
####### Calling all algorithms#############
knn_rc(X_train_std, y_train, X_test_std, y_test)
decision_trees(X_train_std, y_train, X_test_std, y_test)
random_forest_rc(X_train_std, y_train, X_test_std, y_test)
logistic_regression_rc(X_train_std,y_train, X_test_std,y_test)
svm_rc(X_train_std, y_train, X_test_std, y_test)
simple_perceptron_rc(X_train_std, y_train, X_test_std, y_test)




