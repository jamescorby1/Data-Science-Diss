from keras.layers.core import Activation
import pandas as pd 
import string
from sklearn.model_selection import train_test_split
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
from collections import Counter
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant 
from keras.optimizers import Adam
import pickle
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/Datasets(CSV)/Reddit-scraped-data.csv.csv')
data['Target'] = data['Target'].replace("Critical", 1)
data['Target'] = data['Target'].replace("Non Critical", 0)
data['Target'] = pd.to_numeric(data.Target)
print(data.dtypes)
train_text, test_text, y_train, y_test = train_test_split(data['Title'], data['Target'], test_size=0.2)

#joining training labels and test labes with data
train_data = pd.concat([train_text, y_train], axis=1)
test_data = pd.concat([test_text, y_test], axis=1)

#STOPWORDS 
stopwords = nltk.corpus.stopwords.words('english')

#Clean data 
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = " ".join([word for word in tokens if word not in stopwords])
    return text 

train_data['Title'] = train_data.Title.map(lambda x: clean_text(x))

#Counter 
def counter_words(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count

posts = train_data['Title']
counter = counter_words(posts)
print(len(counter))

num_words = len(counter)
max_length = 50

#Splitting the data
train_posts = train_data.Title
train_labels = train_data.Target

test_posts = test_data.Title
test_labels = test_data.Target

#Tokenizing the data
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_posts)
pickle.dump(tokenizer, open("ctitokenizer.pickle", "wb"))

word_index = tokenizer.word_index

#creating sequences
train_sequences = tokenizer.texts_to_sequences(train_posts)
train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding='post', truncating='post'
)

test_sequences = tokenizer.texts_to_sequences(test_posts)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding='post', truncating='post'
)

#Building and fitting the model
model = Sequential()
model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=3e-4)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

history = model.fit(
    train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels)
)

loss, accuracy = model.evaluate(train_padded, train_labels, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(test_padded, test_labels, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

#Plotting accuracies 
plt.style.use('ggplot')
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)


