# 1、A first recurrent layer in Keras
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN

# SimpleRNN 可以以两种不同的模式运行：
# 它可以返回每个时间步的连续输出的完整序列（一个三维形状张量（批量大小、时间步、输出特征）），
# 或者它只能返回每个输入序列的最后一个输出（一个二维张量形状R（批量大小，输出特征）。
# 这两种模式由返回序列构造函数参数控制。让我们来看一个例子：
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

# 有时，为了增加网络的代表性能力，一个接一个地堆叠几个重复的层是有用的。在这种设置中，必须让所有中间层返回完整序列：
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()

# 现在，让我们尝试在IMDB电影评论分类问题上使用这样的模型。首先，让我们对数据进行预处理：
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np

old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# 让我们使用Embedding层和SimpleRNN层来训练一个简单的循环网络：
from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

# 显示训练和验证的损失和准确性：
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 在第3章中，我们对这个数据集的第一个方法使我们获得了88%的测试精度。
# 不幸的是，与此基线相比，我们的小循环网络根本没有表现得很好（仅高达85%的验证准确性）。
# 其中一个问题是我们的输入只考虑前500个字，而不是完整的序列——因此我们的RNN比我们早期的基线模型访问的信息更少。
# 还有就是simpernn不擅长处理长序列，比如文本。其他类型的循环层的性能要好得多。让我们来看一些更高级的层。

# 2、A concrete LSTM example in Keras
# 我们将使用LSTM层建立模型并在IMDB数据上进行训练。类似于我们刚刚介绍的SimpleRNN网络。
# 我们只指定LSTM层的输出维度，并将所有其他参数（有很多）留给Keras默认值。Keras具有良好的默认值，不必花时间手动调整参数。
from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
