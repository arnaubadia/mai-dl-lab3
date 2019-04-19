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

import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import gensim

def train_embedding(tweets):
    # extract all sentences from tweets
    sentences = []
    for tweet in tweets:
        tweet_sentences = sent_tokenize(tweet)
        for sentence in tweet_sentences:
            words = word_tokenize(sentence)
            words = [w.lower() for w in words]
            # at the moment don't remove stopwords
            # TODO: more complex preprocessing
            sentences.append(words)
    # train gensim word2vec model
    embedding = gensim.models.Word2Vec(sentences=sentences, size=embedding_dim,
                                    window=5, workers=4, min_count=2)
    embedding.wv.save_word2vec_format('airline_tweets_embedding.txt', binary=False)
    return embedding

df = pd.read_csv(DATASET_PATH)
tweets = df['text'].apply(preprocess_tweet).values
x = tweets
labels = pd.get_dummies(df['airline_sentiment']).values

# random train-val-test partition (80-10-10)
indices = list(range(len(x)))
np.random.shuffle(indices)
train_indices = indices[:int(0.8*len(x))]
val_indices = indices[int(0.8*len(x)):int(0.9*len(x))]
test_indices = indices[int(0.9*len(x)):]

# max length (word_length) could be more in a real unseen tweet, but we know
# that it is limited by 140 characters.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x[train_indices])
word_index = tokenizer.word_index
max_length = max([len(s.split()) for s in x[train_indices]])
vocabulary_size = len(tokenizer.word_index) + 1
x = tokenizer.texts_to_sequences(x)
x = pad_sequences(x, maxlen=max_length, padding='post')

x_train = x[train_indices]
y_train = labels[train_indices]
x_val = x[val_indices]
y_val = labels[val_indices]
x_test = x[test_indices]
y_test = labels[test_indices]

# read pretrained embeddings into dict
train_embedding(tweets)
# embedding_path = 'airline_tweets_embedding.txt'
embedding_path = 'glove.twitter.27B/glove.twitter.27B.100d.txt'
embeddings_index = {}
f = open(embedding_path, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    if word in word_index:
        word_coefs = np.asarray(values[1:])
        embeddings_index[word] = word_coefs
f.close()

# map word indices from Tokenizer to embedding coefficients
embedding_matrix = np.zeros((vocabulary_size,embedding_dim))

for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


print('x train shape: ', x_train.shape)
print('y train shape: ', y_train.shape)
print('x val shape: ', x_val.shape)
print('y val shape: ', y_val.shape)

from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, Conv1D, MaxPooling1D, Flatten
from keras.initializers import Constant
# use EMBEDDING + GRU
nn = Sequential()
nn.add(Embedding(vocabulary_size, embedding_dim, embeddings_initializer=Constant(embedding_matrix),
                input_length=max_length, trainable=False))
#nn.add(GRU(units=32, dropout=0.1, recurrent_dropout=0.1))
#nn.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
nn.add(Conv1D(256, 10, activation='relu'))
nn.add(Conv1D(128, 10, activation='relu'))
nn.add(Flatten())
nn.add(Dense(128, activation='relu'))
nn.add(Dense(3, activation='softmax'))

nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



nn.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))

#Confusion Matrix
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np
#Compute probabilities
Y_pred = nn.predict(x_test)
#Assign most probable label
y_pred = np.argmax(Y_pred, axis=1)
#Plot statistics
print( 'Analysis of results' )
target_names = ['0','1','2']
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))
