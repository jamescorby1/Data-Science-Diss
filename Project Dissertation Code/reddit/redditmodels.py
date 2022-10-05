import re 
import pandas as pd 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import string
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

#STOPWORDS 
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
added_words = ['learn', 'learning', 'teach', 'teaching']
stopwords.extend(added_words)

#Import data
data = pd.read_csv('/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/Datasets(CSV)/Reddit-scraped-data.csv', index_col=[0])

#Clean data 
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text 

# split data
X_train, X_test, y_train, y_test = train_test_split(data['Title'], data['Target'], test_size=0.2)

#Vectorize data
tfidf_vect = TfidfVectorizer(analyzer= clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train)
tfidf_train = tfidf_vect_fit.transform(X_train)
tfidf_test = tfidf_vect_fit.transform(X_test)
pickle.dump(tfidf_vect, open("ctiVectorizer.pickle", "wb"))

#Random Forest model
rf_model = RandomForestClassifier(n_estimators= 150, max_depth=20, n_jobs=-1)
rf_model.fit(tfidf_train, y_train)
rf_y_pred = rf_model.predict(tfidf_test)
rf_predict_proba = rf_model.predict_proba(tfidf_test)
pickle.dump(rf_model, open("ctiRandomForest_model.pickle", "wb"))

#RF Accuracy Scores 
rf_precision, rf_recall, rf_fscore, rf_support = score(y_test, rf_y_pred, pos_label = 'Critical', average = 'binary')
print("Random Forest Classifier results... ")
rf_f1score = 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall)
print('Precision: {} / Recall: {} / Accuraccy: {} / F1-Score: {}'.format(
    round(rf_precision, 3), round(rf_recall, 3), round((rf_y_pred==y_test).sum() / len(rf_y_pred), 3), round(rf_f1score, 3)))

#RF Confusion matrix 
rf_matrix = confusion_matrix(y_test, rf_y_pred)
rf_group_names = ["True Neg","False Pos","False Neg","True Pos"]
rf_group_percentages = ["{0:.2%}".format(value) for value in
                     rf_matrix.flatten()/np.sum(rf_matrix)]
rf_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(rf_group_names,rf_group_percentages)]
rf_label = np.asarray(rf_labels).reshape(2,2)
sns.heatmap(rf_matrix, annot=rf_label, fmt="", cmap='Blues')
plt.title("Random Forest Confusion matrix")
plt.show()

#RF ROC Curve 
rf_fpr1, rf_tpr1, rf_thresh1 = roc_curve(y_test, rf_predict_proba[:,1], pos_label='Non Critical')
rf_random_probs = [0 for i in range(len(y_test))] 
rf_p_fpr, rf_p_tpr, _ = roc_curve(y_test, rf_random_probs, pos_label='Non Critical')

#AUC Score
auc_score1 = roc_auc_score(y_test, rf_predict_proba[:,1])
print('Area under Curve Score: {}'.format(round(auc_score1, 3)))
#RF ROC Curve 
plt.style.use('seaborn')
plt.plot(rf_fpr1, rf_tpr1, linestyle='--',color='orange')
plt.plot(rf_p_fpr, rf_p_tpr, linestyle='--', color='blue')
plt.title('Random Forest ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()

#-------------------------------------------------------------------------


#Gradient Boosting model  
gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=15, learning_rate=0.1)
gb_model = gb_model.fit(tfidf_train, y_train)
gb_y_pred = gb_model.predict(tfidf_test)
gb_predict_proba = gb_model.predict_proba(tfidf_test)
pickle.dump(rf_model, open("ctiGradientBoosting_model.pickle", "wb"))
#GB Accuracy 
gb_precision, gb_recall, gb_fscore, gb_support = score(y_test, gb_y_pred, pos_label = 'Critical', average = 'binary')
print("Gradient Boosting results... ")
gb_f1score = 2 * (gb_precision * gb_recall) / (gb_precision + gb_recall)
print('Precision: {} / Recall: {} / Accuraccy: {} / F1-Score: {}'.format(
    round(gb_precision, 3), round(gb_recall, 3), round((gb_y_pred==y_test).sum() / len(gb_y_pred), 3), round(gb_f1score, 3)))
#GB Confusion matrix 
gb_matrix = confusion_matrix(y_test, gb_y_pred)
gb_group_names = ["True Neg","False Pos","False Neg","True Pos"]
gb_group_percentages = ["{0:.2%}".format(value) for value in
                     gb_matrix.flatten()/np.sum(gb_matrix)]
gb_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(gb_group_names,gb_group_percentages)]
gb_label = np.asarray(gb_labels).reshape(2,2)
sns.heatmap(gb_matrix, annot=gb_label, fmt="", cmap='Blues')
plt.title("Gradient Boosting Confusion matrix")
plt.show()

#GB ROC Curve 
gb_fpr1, gb_tpr1, gb_thresh1 = roc_curve(y_test, gb_predict_proba[:,1], pos_label='Non Critical')
gb_random_probs = [0 for i in range(len(y_test))] 
gb_p_fpr, gb_p_tpr, _ = roc_curve(y_test, gb_random_probs, pos_label='Non Critical')
#GB AUC Score
gbauc_score = roc_auc_score(y_test, gb_predict_proba[:,1])
print('Area under Curve Score: {}'.format(round(gbauc_score, 3)))
#GB ROC Curve 
plt.style.use('seaborn')
plt.plot(gb_fpr1, gb_tpr1, linestyle='--',color='orange')
plt.plot(gb_p_fpr, gb_p_tpr, linestyle='--', color='blue')
plt.title('Gradient Boosting ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()

# ----------------------------------------------------------------------------------


#Multinomial Bayes model
mb_model = MultinomialNB()
mb_model = mb_model.fit(tfidf_train, y_train)
mb_y_pred = mb_model.predict(tfidf_test)
mb_predict_proba = mb_model.predict_proba(tfidf_test)
pickle.dump(mb_model, open("ctiBayes_model.pickle", "wb"))

mb_precision, mb_recall, mb_fscore, mb_support = score(y_test, mb_y_pred, pos_label = 'Critical', average='binary')
print("Multinomial Bayes results... ")
mb_f1score = 2 * (mb_precision * mb_recall) / (mb_precision + mb_recall)
print('Precision: {} / Recall: {} / Accuraccy: {} / F1-Score: {}'.format(
    round(mb_precision, 3), round(mb_recall, 3), round((mb_y_pred==y_test).sum() / len(mb_y_pred), 3), round(mb_f1score, 3)))
#MB Confusion matrix 
mb_matrix = confusion_matrix(y_test, mb_y_pred)
mb_group_names = ["True Neg","False Pos","False Neg","True Pos"]
mb_group_percentages = ["{0:.2%}".format(value) for value in
                     mb_matrix.flatten()/np.sum(mb_matrix)]
mb_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(mb_group_names,mb_group_percentages)]
mb_label = np.asarray(mb_labels).reshape(2,2)
sns.heatmap(mb_matrix, annot=mb_label, fmt="", cmap='Blues')
plt.title("Multinomial Bayes Confusion matrix")
plt.show()
#MB ROC Curve 
mb_fpr1, mb_tpr1, mb_thresh1 = roc_curve(y_test, mb_predict_proba[:,1], pos_label='Non Critical')
mb_random_probs = [0 for i in range(len(y_test))] 
mb_p_fpr, mb_p_tpr, _ = roc_curve(y_test, mb_random_probs, pos_label='Non Critical')
#MB AUC Score
mbauc_score = roc_auc_score(y_test, mb_predict_proba[:,1])
print('Area under Curve Score: {}'.format(round(mbauc_score, 3)))
#MB ROC Curve 
plt.style.use('seaborn')
plt.plot(mb_fpr1, mb_tpr1, linestyle='--',color='orange')
plt.plot(mb_p_fpr, mb_p_tpr, linestyle='--', color='blue')
plt.title('Multinomial Bayes ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()


# --------------------------------------------------------------------------------


#Linear SVM model
lsvc_model = LinearSVC()
lsvc_model = lsvc_model.fit(tfidf_train, y_train)
lsvc_y_pred = lsvc_model.predict(tfidf_test)
#lsvc_predict_proba = lsvc_model.predict_proba(tfidf_test)
pickle.dump(rf_model, open("ctiLinear_SVM_model.pickle", "wb"))
#Accuracy 
lsvc_precision, lsvc_recall, lsvc_fscore, lsvc_support = score(y_test, lsvc_y_pred, pos_label = 'Critical', average='binary')
print("Linear SVC results... ")
lsvc_f1score = 2 * (lsvc_precision * lsvc_recall) / (lsvc_precision + lsvc_recall)
print('Precision: {} / Recall: {} / Accuraccy: {} / F1-Score: {}'.format(
    round(lsvc_precision, 3), round(lsvc_recall, 3), round((lsvc_y_pred==y_test).sum() / len(lsvc_y_pred), 3), round(lsvc_f1score, 3)))

#LSVC Confusion matrix 
lsvc_matrix = confusion_matrix(y_test, lsvc_y_pred)
lsvc_group_names = ["True Neg","False Pos","False Neg","True Pos"]
lsvc_group_percentages = ["{0:.2%}".format(value) for value in
                     lsvc_matrix.flatten()/np.sum(lsvc_matrix)]
lsvc_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(lsvc_group_names,lsvc_group_percentages)]
lsvc_label = np.asarray(lsvc_labels).reshape(2,2)
sns.heatmap(lsvc_matrix, annot=lsvc_label, fmt="", cmap='Blues')
plt.title("Linear Support Vector Machine Confusion matrix")
plt.show()

