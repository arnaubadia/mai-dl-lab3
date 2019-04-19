import numpy as np
import pandas as pd
import re
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


DATASET_PATH = 'Tweets.csv'
embedding_dim = 100

def preprocess_tweet(t):
    # remove mention of the airline twitter account
    t = re.sub(r"(@VirginAmerica|@united|@SouthwestAir|@JetBlue|@USAirways|@AmericanAir)", "", t)
    # remove "#" and "@" characters, but we still want to keep the words
    t = re.sub(r"(@|#)", "", t)
    # convert to lowercase
    t = t.lower()
    # in informal context such as twitter it is common to remove
    t = re.sub(r"[?]+", '?', t)
    t = re.sub(r"[!]+", '!', t)
    return t

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv(DATASET_PATH)
tweets = df['text'].apply(preprocess_tweet).values
x = tweets
labels = df['airline_sentiment'].values

# random train-val-test partition (80-10-10)
indices = list(range(len(x)))
np.random.shuffle(indices)
train_indices = indices[:int(0.8*len(x))]
val_indices = indices[int(0.8*len(x)):int(0.9*len(x))]
test_indices = indices[int(0.9*len(x)):]

x_train = x[train_indices]
y_train = labels[train_indices]
x_val = x[val_indices]
y_val = labels[val_indices]
x_test = x[test_indices]
y_test = labels[test_indices]

from sklearn.feature_extraction.text import CountVectorizer
clf = CountVectorizer()
X_train_one_hot =  clf.fit_transform(x_train).toarray()
X_test_one_hot = clf.transform(x_test).toarray()

from sklearn.naive_bayes import MultinomialNB
nbc = MultinomialNB()
nbc.fit(X_train_one_hot, y_train)

from sklearn.metrics import classification_report,confusion_matrix
y_pred = nbc.predict(X_test_one_hot)
score = nbc.score(X_test_one_hot, y_test)
print(score)

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
