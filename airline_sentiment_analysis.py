import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import emoji
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


DATASET_PATH = 'Tweets.csv'
embedding_dim = 100


"""
********************** PREPROCESSING AND EMBEDDINGS ***************************
"""

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
    # separate emojis
    t2 = []
    for i in range(len(t)):
         t2.append(t[i])
         if i < len(t)-1 and t[i] != ' ' and t[i+1] in emoji.UNICODE_EMOJI:
             t2.append(' ')
    t = ''.join(t2)
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
np.random.seed(42)
np.random.shuffle(indices)
train_indices = indices[:int(0.8*len(x))]
val_indices = indices[int(0.8*len(x)):int(0.9*len(x))]
test_indices = indices[int(0.9*len(x)):]

# max length (word_length) could be more in a real unseen tweet, but we know
# that it is limited by 140 characters.

tokenizer = Tokenizer()
#tokenizer.fit_on_texts(x[train_indices])
tokenizer.fit_on_texts(x)
word_index = tokenizer.word_index
# max_length = max([len(s.split()) for s in x[train_indices]])
max_length = max([len(s.split()) for s in x])
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
# train_embedding(tweets)
# embedding_path = 'airline_tweets_embedding.txt'
embedding_path = 'glove.twitter.27B/glove.twitter.27B.100d.txt'
# embedding_path = 'glove.42B.300d.txt'
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


"""
********************** NEURAL NETWORK **************************************
"""

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
nn.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2))
#nn.add(Conv1D(256, 10, activation='relu'))
#nn.add(Conv1D(128, 10, activation='relu'))
#nn.add(Flatten())
#nn.add(Dense(128, activation='relu'))
nn.add(Dense(3, activation='softmax'))

nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('best-weights.hdf5', save_best_only=True, monitor='val_loss', mode='min')


history = nn.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_val, y_val),
                callbacks=[earlyStopping, mcp_save])

"""
********************** PLOTS AND RESULTS **************************************
"""
#Restore the model with the best weights we found during training
nn.load_weights('best-weights.hdf5')

#Accuracy plot
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('accuracy.pdf')
plt.close()
#Loss plot
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig('loss.pdf')


# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
print('Accuracy: ', sum(np.argmax(y_test,axis=1) == y_pred)/len(y_pred))
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

plot_confusion_matrix(np.argmax(y_test,axis=1), y_pred, target_names, normalize=True)
plt.savefig('confusion_matrix.pdf')
plt.close()
