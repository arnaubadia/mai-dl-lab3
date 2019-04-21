import numpy as np
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
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
    # separate emojis
    t2 = []
    for i in range(len(t)):
         t2.append(t[i])
         if i < len(t)-1 and t[i] != ' ' and t[i+1] in emoji.UNICODE_EMOJI:
             t2.append(' ')
    t = ''.join(t2)
    return t

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv(DATASET_PATH)
tweets = df['text'].apply(preprocess_tweet).values
x = tweets
labels = df['airline_sentiment'].values

# random train-val-test partition (80-10-10)
indices = list(range(len(x)))
np.random.seed(42)
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


from sklearn.metrics import classification_report,confusion_matrix
y_pred = nbc.predict(X_test_one_hot)
score = nbc.score(X_test_one_hot, y_test)
print('Accuracy: ', score)

target_names = ['negative', 'neutral', 'positive']
plot_confusion_matrix(y_test, y_pred, target_names, normalize=True)
plt.show()
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
