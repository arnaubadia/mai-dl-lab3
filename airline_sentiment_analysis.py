import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


DATASET_PATH = 'Tweets.csv'

def preprocess_tweets(tweets):
    # convert emojis to text
    pass
    # remove capital letters

    # lemmatize?

    # remove @


df = pd.read_csv(DATASET_PATH)
tweets = df['text'].values
labels = pd.get_dummies(df['airline_sentiment']).values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)

# max length (word_length) could be more in a real unseen tweet, but we know
# that it is limited by 140 characters.
max_length = max([len(s.split()) for s in tweets])
vocabulary_size = len(tokenizer.word_index) + 1

x = tokenizer.texts_to_sequences(tweets)
x = pad_sequences(x, maxlen=max_length, padding='post')

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


from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU

embedding_dim = 100

# use EMBEDDING + GRU
nn = Sequential()
nn.add(Embedding(vocabulary_size, embedding_dim, input_length=max_length))
#nn.add(GRU(units=32, dropout=0.1, recurrent_dropout=0.1))
nn.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
nn.add(Dense(3,activation='softmax'))

nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nn.fit(x_train, y_train, batch_size=128, epochs=25, validation_data=(x_val, y_val))
