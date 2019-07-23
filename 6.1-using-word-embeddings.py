# # 1、Learning word embeddings with the Embedding layer
# from keras.layers import Embedding
# from keras.datasets import imdb
# from keras import preprocessing
# from keras.models import Sequential
# from keras.layers import Flatten, Dense
# import numpy as np
#
# old = np.load
# np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)
# # The Embedding layer takes at least two arguments:
# # the number of possible tokens, here 1000 (1 + maximum word index),
# # and the dimensionality of the embeddings, here 64.
# embedding_layer = Embedding(1000, 64)
#
# # 当我们实例化一个嵌入层时，它的权重（它的令牌向量的内部字典）最初是随机的，就像任何其他层一样。
# # 在训练过程中，这些词向量将通过反向传播逐步调整，将空间构造成下游模型可以利用的内容。
# # 一旦完全训练了，你的嵌入空间就会显示出很多结构——一种专门针对你训练模型的特定问题的结构。
# # 将这个想法应用到我们已经熟悉的IMDB电影评论情绪预测任务中。
# # 快速准备数据，把电影评论限制在前10000个最常见的词（就像我们第一次使用这个数据集时做的那样），并将评论减少到20个词之后。
# # 我们的网络只需学习每10000个单词的8维嵌入，将输入的整数序列（2d整数张量）转换为嵌入序列（3d浮点张量），将张量展平为2d，并在顶部训练单个密集层进行分类。
#
# # Number of words to consider as features
# max_features = 10000
# # Cut texts after this number of words(among top max_features most common words)
# maxlen = 20
#
# # Load the data as lists of integers.
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#
# # This turns our lists of integers
# # into a 2D integer tensor of shape `(samples, maxlen)`
# x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
#
# model = Sequential()
# # We specify the maximum input length to our Embedding layer
# # so we can later flatten the embedded inputs
# model.add(Embedding(10000, 8, input_length=maxlen))
# # After the Embedding layer, our activations have shape `(samples, maxlen, 8)`.
#
# # We flatten the 3D tensor of embeddings into a 2D tensor of shape `(samples, maxlen * 8)`
# model.add(Flatten())
#
# # We add the classifier on top
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()
#
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_split=0.2)

# 2、Using pre-trained word embeddings
# 我们将使用类似于我们刚刚过去的模型 - 在矢量序列中嵌入句子，展平它们并在顶部训练密集层。
# 但我们将使用预先训练的字嵌入来实现，而不是使用Keras中打包的预标记化IMDB数据
# 我们将从头开始，通过下载原始文本数据。
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import matplotlib.pyplot as plt

# 1)Download the IMDB data as raw text
imdb_dir = 'Downloads/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),'r',encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

# 2）Tokenize the data
# 对我们收集的文本进行矢量化，并准备训练和验证分割。
# 因为预训练的单词嵌入对于几乎没有可用训练数据的问题特别有用，我们将添加以下内容：
# 我们将训练数据限制在其前200样本。 因此，在查看了200个例子后，我们将学习如何对电影评论进行分类......
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# 3）Download the GloVe word embeddings
# 4）Pre-process the embeddings
glove_dir = 'Downloads/glove'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# 现在让我们构建一个嵌入矩阵，我们可以将其加载到嵌入层中。
# 它必须是一个矩阵shape（max_words，embedding_dim），其中每个条目i包含我们的参考词索引（在标记化期间构建）中索引i的单词的embedding_dim维向量。
# 注意，索引0不应代表任何单词或标记 - 它是占位符。
embedding_dim = 100

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# 5）Define a model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# 6）Load the GloVe embeddings in the model
# 嵌入层具有单个权重矩阵：2D浮点矩阵，其中每个条目i是意图与索引i相关联的单词向量。
# 让我们将我们准备的GloVe矩阵加载到嵌入层中，这是我们模型中的第一层：
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
# 此外，我们冻结嵌入层（我们将其trainable属性设置为False），遵循与您在训练前的信号特征上下文中已经熟悉的相同的理由：
# 当模型的某些部分经过预先训练时（如我们的 嵌入层）和零件随机初始化（如我们的分类器），
# 训练前的部分不应更新，以免忘记他们已经知道的内容。
# 由随机初始化的层触发的大梯度更新对于已经学习的特征将是非常具有破坏性的。

# 7）Train and evaluate
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
model.save_weights('pre_trained_glove_model.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
# 该模型很快就开始过度拟合，不出所料，因为训练样本数量很少。 出于同样的原因，验证准确性具有很大的差异，但似乎达到了高达50s。
# 注意，由于我们的训练样本很少，因此性能在很大程度上取决于我们挑选的200个样本，我们随机选择它们。 如果它对你来说效果很差，那么试着选择一组200个样本，这只是为了练习（在现实生活中你不会选择你的训练数据）。
# 我们还可以尝试训练相同的模型，而无需加载预先训练的单词嵌入并且不冻结嵌入层。 在这种情况下，我们将学习一个特定于任务的嵌入我们的输入令牌，当大量数据可用时，这通常比预先训练的词嵌入更强大。
# 但是，在我们的案例中，我们只有200个培训样本。 我们来试试：
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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

# 验证准确性在50秒内停滞。 因此，在我们的案例中，预先训练的单词嵌入确实优于共同学习的嵌入。
# 如果你增加训练样本的数量，这将很快停止 - 尝试将其作为练习。
# 最后，让我们在测试数据上评估模型。 首先，我们需要标记化测试数据：
test_dir = os.path.join(imdb_dir, 'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname),encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
# 让我们加载并评估第一个模型：
model.load_weights('pre_trained_glove_model.h5')
print(model.evaluate(x_test, y_test))
