import numpy as np
import pandas as pd
from keras import models
from keras import layers
from collections import Counter
import matplotlib.pyplot as plt
import re

# (tr_im, tr_l), (tst_im, tst_l) = mnist.load_data()

# network = models.Sequential()
# network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# network.add(layers.Dense(10, activation='softmax'))
# network.compile(optimizer='rmsprop',
#                loss='categorical_crossentropy',
#                metrics=['accuracy'])

# tr_im = tr_im.reshape((60000, 28 * 28))
# tst_im = tst_im.reshape((10000, 28 * 28))
# tr_im = tr_im.astype('float32') / 255
# tst_im = tst_im.astype('float32') / 255

# tr_l = to_categorical(tr_l)
# tst_l = to_categorical(tst_l)

# network.fit(tr_im, tr_l, epochs=5, batch_size=128)


data = pd.read_csv("C:\\Users\\Wojtek\\Desktop\\alldata.csv", sep=',', encoding='latin-1')
data2 = pd.DataFrame(data)
data3 = data2.sample(frac=1, random_state=42).reset_index(drop=True)

text = data3.iloc[:, 1]
labels = data3.iloc[:, 0]
for i in enumerate(labels):
    if i[1] == 'positive':
        labels[i[0]] = 1
    else:
        labels[i[0]] = 0

labels = np.asarray(labels).astype('float32')
#own_sentence = re.sub('[\W_]+',' ',own_sentence)
text=[*map(lambda x: re.sub('[\W_]+',' ',x),text)]


text = [*map(lambda x: x.lower(), text)]
split_text = map(lambda x: x.split(), text)
split_text2 = pd.DataFrame(split_text)
split_text3 = list(filter(None, split_text2.values.reshape(split_text2.size)))
split_text4 = list(filter(str.isalpha, split_text3))

Counter = Counter(split_text4)

nowords = 5000
most_popular = Counter.most_common(nowords - 1)
most_popular_word = [tup[0] for tup in most_popular]
myDict = {s: i + 1 for i, s in enumerate(most_popular_word)}

# occurences=split_text2.replace(myDict)
occurences = split_text2.applymap(myDict.get).fillna(0).astype(int)
# for i, word in split_text2.iterrows():
#    print(i)
#    for j in enumerate(word):
#        for k in enumerate(most_popular_word):
#           if(j[1]==k[1]):
#                occurences[i,j[0]]=k[0]

# pd.DataFrame(occurences2).to_csv("C:\\Users\\Wojtek\\Desktop\\alldata2.csv")
# pd.DataFrame(data).to_csv("C:\\Users\\Wojtek\\Desktop\\alldata3.csv")


# data=pd.read_csv("C:\\Users\\Wojtek\\Desktop\\alldata3.csv",usecols=[1,2])
# occurences2 = pd.read_csv("C:\\Users\\Wojtek\\Desktop\\alldata2.csv")

# occurences2 = pd.DataFrame(occurences2)
occurences = np.asarray(occurences)

results = np.zeros((len(occurences), nowords))
for i, sequence in enumerate(occurences):
    results[i, list(map(int, sequence))] = 1.

tr_results = results[1:3500]
tst_results = results[3500:]
tr_labels = labels[1:3500]
tst_labels = labels[3500:]

train = np.asarray(tr_results).astype('float32')
test = np.asarray(tst_results).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(5000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train[:1000]
x_train = train[1000:]
y_val = tr_labels[:1000]
y_train = tr_labels[1000:]

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

# highest accuracy index
print(val_acc.index(max(val_acc)))
# lowest loss index
print(val_loss.index(min(val_loss)))

# tst_labels_model=model.predict(tst_results)
# tst_labels_model2=np.round(tst_labels_model)
# tst_labels_model3=tst_labels_model2.reshape(1345,)

# ress=pd.concat([pd.DataFrame(data[-1345:]).reset_index(drop=True),pd.DataFrame(tst_labels_model3).reset_index(drop=True)],axis=1)
# ress.to_csv("C:\\Users\\Wojtek\\Desktop\\results.csv")

model2 = models.Sequential()
model2.add(layers.Dense(16, activation='relu', input_shape=(5000,)))
model2.add(layers.Dense(16, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['accuracy'])
history = model2.fit(train,
                     tr_labels,
                     epochs=11,
                     batch_size=256)

tst_labels_model = model2.predict(tst_results)
tst_labels_model2 = np.round(tst_labels_model)
tst_labels_model3 = tst_labels_model2.reshape(1345, )

# noinspection PyTypeChecker
print(sum(tst_labels_model3 == tst_labels) / len(tst_labels))

# ress=pd.concat([pd.DataFrame(data[-1345:]).reset_index(drop=True),pd.DataFrame(tst_labels_model3).reset_index(drop=True)],axis=1)
# ress.to_csv("C:\\Users\\Wojtek\\Desktop\\results.csv")


# change sequence to a sequence of numbers corresponding to most popular words
own_sentence = "We have a new and innovative strategy. Our company reported a raise in the amount of assets?"
own_sentence = re.sub('[\W_]+',' ',own_sentence)
own_sentence2 = pd.DataFrame(own_sentence.lower().split())
# own=np.zeros(len(own_sentence2))
own = own_sentence2.applymap(myDict.get).fillna(0).astype(int)
# for i in enumerate(own_sentence2):
#    for k in enumerate(most_popular_word):
#        if(i[1]==k[1]):
#            own[i[0]]=k[0]

# change the representation of numbers to a vector of length=len(most_popular_words), and put ones if a word occur
own2 = np.zeros(nowords)
# own2[[*map(int,own)]]=1
own2[own.values] = 1
own2 = np.asarray(own2).astype('float32')
own2 = own2.reshape(1, 5000)
lab = model2.predict(own2)[0][0]
print(lab)
if (round(lab)) == 1:
    print('Positive')
else:
    print('Neutral or negative')
