import numpy as np
import pandas as pd
from keras import models
from keras import layers
from collections import Counter
import matplotlib.pyplot as plt
import re
from keras.utils import to_categorical
import tensorflow as tf

#Data Read
data = pd.read_csv("C:\\Users\\Wojtek\\Desktop\\alldata.csv", sep=',', encoding='latin-1')
data2 = pd.DataFrame(data)

#Shuffle
data3 = data2.sample(frac=1, random_state=42).reset_index(drop=True)

#Division of data
text = data3.iloc[:, 1]
labels = data3.iloc[:, 0]
for i in enumerate(labels):
    if i[1] == 'positive':
        labels[i[0]] = 2
    elif i[1] == 'negative':
        labels[i[0]] = 0
    else:
        labels[i[0]] = 1
        
labels = to_categorical(labels)

#Strings transformations
text = [*map(lambda x: re.sub('[\W_]+', ' ', x), text)]
text = [*map(lambda x: x.lower(), text)]
split_text = map(lambda x: x.split(), text)
split_text2 = pd.DataFrame(split_text)
split_text3 = list(filter(None, split_text2.values.reshape(split_text2.size)))
split_text4 = list(filter(str.isalpha, split_text3))


#Creating dictionary consisting of 5000 most popular words
Counter = Counter(split_text4)
nowords = 5000
most_popular = Counter.most_common(nowords - 1)
most_popular_word = [tup[0] for tup in most_popular]
myDict = {s: i + 1 for i, s in enumerate(most_popular_word)}


#Translating data into dictionary keys
occurences = split_text2.applymap(myDict.get).fillna(0).astype(int)
occurences = np.asarray(occurences)


#Reforging data into categorical classification if the word exists
results = np.zeros((len(occurences), nowords))
for i, sequence in enumerate(occurences):
    results[i, list(map(int, sequence))] = 1.


#Division into training and test sets
tr_results = results[1:3500]
tst_results = results[3500:]
tr_labels = labels[1:3500]
tst_labels = labels[3500:]

train = np.asarray(tr_results).astype('float32')
test = np.asarray(tst_results).astype('float32')


#Division training set into training and validation set
x_val = train[:1000]
x_train = train[1000:]
y_val = tr_labels[:1000]
y_train = tr_labels[1000:]


# 64-32-16-8-3 -> 0.741
# 256-3 -> 0.748
# 256-8 -> 0.748
# 128-16-8-3 -> 0.75/0.63 -> epoch 7
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(5000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    batch_size=256,
                    validation_data=(x_val, y_val))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()

# highest accuracy index and value
print(val_acc.index(max(val_acc)))
print(max(val_acc))
# lowest loss index and value
print(val_loss.index(min(val_loss)))
print(min(val_loss))


#Creating a model with efficient hyperparameters
model2 = models.Sequential()
model2.add(layers.Dense(128, activation='relu', input_shape=(5000,)))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(8, activation='relu'))
model2.add(layers.Dense(3, activation='softmax'))
model2.compile(optimizer='rmsprop',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history = model2.fit(train,
                     tr_labels,
                     epochs=7,
                     batch_size=256)

tst_labels_model = model2.predict(tst_results)
tst_labels_model2 = np.round(tst_labels_model)
test_labels_model3 = tf.math.argmax(tst_labels_model2, axis=1).numpy()
tst_labels = tf.math.argmax(tst_labels, axis=1).numpy()

# noinspection PyTypeChecker - Printing test accuracy
print(sum(test_labels_model3 == tst_labels) / len(tst_labels))


#Saving result
# pd.DataFrame({'Neural' : test_labels_model3, 'Real':tst_labels}).to_csv(
# "C:\\Users\\Wojtek\\Desktop\\resultsCategory.csv")



#Checking single results

# change sequence to a sequence of numbers corresponding to most popular words
own_sentence = "We have a new and innovative strategy. Our company reported a raise in the amount of assets?"
own_sentence = re.sub('[\W_]+', ' ', own_sentence)
own_sentence2 = pd.DataFrame(own_sentence.lower().split())
own = own_sentence2.applymap(myDict.get).fillna(0).astype(int)

# change the representation of numbers to a vector of length=len(most_popular_words), and put ones if a word occur
own2 = np.zeros(nowords)
own2[own.values] = 1
own2 = np.asarray(own2).astype('float32')
own2 = own2.reshape(1, 5000)
lab = model2.predict(own2)[0]
print(lab)
if lab.argmax() == 0:
    print('Negative')
elif lab.argmax() == 1:
    print('Neutral')
else:
    print('Negative')
