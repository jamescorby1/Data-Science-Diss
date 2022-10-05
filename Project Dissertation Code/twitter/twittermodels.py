import pandas as pd
from sklearn.svm import OneClassSVM
import re 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#STOPWORDS 
stopwords = nltk.corpus.stopwords.words('english')
added_words = ['CVE', 'CVE-', 'teach', 'teaching']
stopwords.extend(added_words)
ps = nltk.PorterStemmer()

#Import data
data = pd.read_csv('/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/Datasets(CSV)/Imbalanced_twitter_data.csv')
data['Target'] = data['Target'].replace(['Critical', 'Non Critical'], [0 , 1])

#Split the data 
X_train, X_test, y_train, y_test = train_test_split(data['Title'], data['Target'], test_size=0.2)

#Clean data 
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text 

#Vectorize data
tfidf_vect = TfidfVectorizer(analyzer= clean_text)
X_train = X_train[y_train == 0]
tfidf_vect_fit = tfidf_vect.fit(X_train)
tfidf_train = tfidf_vect_fit.transform(X_train)
tfidf_test = tfidf_vect_fit.transform(X_test)

#One class support vector machine
svm_model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.1)
svm_model.fit(tfidf_train, y_train)
svm_predict = svm_model.predict(tfidf_test)

#Mark outliers 
y_test[y_test == 1] = -1
y_test[y_test == 0] = 1

print("Accuracy score is {}".format(round(accuracy_score(y_test, svm_predict),3)))
print("F1-Score is {}".format(round(f1_score(y_test, svm_predict, pos_label=-1),3)))

svm_fpr1, svm_tpr1, svm_thresh1 = roc_curve(y_test, svm_predict, pos_label=1)
svm_random_probs = [0 for i in range(len(y_test))] 
svm_p_fpr, svm_p_tpr, _ = roc_curve(y_test, svm_random_probs, pos_label=1)

plt.style.use('seaborn')
plt.plot(svm_fpr1, svm_tpr1, linestyle='--',color='orange')
plt.plot(svm_p_fpr, svm_p_tpr, linestyle='--', color='blue')
plt.title('Multinomial Bayes ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()

# ------------------------------------------------------------------------------------

#Isolation Forest 
if_model = IsolationForest()
if_model.fit(tfidf_train, y_train)
if_prediction = if_model.predict(tfidf_test)

print("Accuracy score is {}".format(round(accuracy_score(y_test, if_prediction),3)))
print("F1-Score is {}".format(round(f1_score(y_test, if_prediction, pos_label=-1),5)))


mb_fpr1, mb_tpr1, mb_thresh1 = roc_curve(y_test, if_prediction, pos_label=1)
mb_random_probs = [0 for i in range(len(y_test))] 
mb_p_fpr, mb_p_tpr, _ = roc_curve(y_test, mb_random_probs, pos_label=1)

plt.style.use('seaborn')
plt.plot(mb_fpr1, mb_tpr1, linestyle='--',color='orange')
plt.plot(mb_p_fpr, mb_p_tpr, linestyle='--', color='blue')
plt.title('Multinomial Bayes ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()






#post = "New personal best Cayo run"
#tweet_vect = tfidf_vect_fit.transform([post])
#tweet_pred = if_model.predict(tweet_vect)
#print(tweet_pred)

"""Opted for One class as there was alot of overfitting due to only the 
critical data containing words such as CVE- etc"""
