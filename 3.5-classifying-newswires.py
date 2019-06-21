from keras.datasets import reuters
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import matplotlib.pyplot as plt
import copy

# 解决Object arrays cannot be loaded when allow_pickle=False 的错误
old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# num_words=10000 将数据限制为数据中找到的10,000个最常出现的单词，包含8,982个训练示例和2,246个测试示例
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# print(len(train_data))
# print(len(test_data))
# print(train_data[10])  与IMDB评论一样，每个示例都是整数列表（单词索引）

# 将新闻专线解码回文本，
# 注意，索引偏移3，因为保留0,1和2“填充”，“序列开始”和“未知”的索引。
# 与示例关联的标签是0到45之间的整数 - 主题索引：
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])


# print(train_labels[10])

# 编码数据
# 要对标签进行矢量化，有两种方法：可以将标签列表转换为整数张量，也可以使用one-hot encoding。
# 方法一：
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)  # 矢量化的训练数据
x_test = vectorize_sequences(test_data)  # 矢量化的测试数据


# 标签的单热编码包括将每个标签嵌入为全零向量，其中1代替 标签索引
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# 也可以使用Keras内置方法
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# 模型定义
# 使用大小为46的Dense层结束网络，对于每个输入样本，网络将输出46维向量，此向量中的每个维度将编码不同的输出类
# 最后一层使用softmax激活。意味着网络将在46个不同的输出类别上输出概率分布-对于每个输入样本，网络将产生46维输出向量，其中输出[i]是样本属于类i的概率。 46分将总和为1。
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 编译模型
# 在这种情况下使用的最佳损失函数是categorical_crossentropy
# 它测量两个概率分布之间的距离：这里，在网络输出的概率分布和标签的真实分布之间。
# 通过最小化这两个分布之间的距离，可以训练网络输出尽可能接近真实标签的内容。
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 留出验证集
# 让我们在训练数据中设置1,000个样本以用作验证集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# 训练模型
# 现在，训练网络20个 epochs
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# 显示loss 和 accuracy 曲线
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 网络在九个时代之后开始过度拟合。 让我们从头开始训练一个新网络九个时期，然后在测试集上进行评估。
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)

# 该方法达到约80％的准确度。 对于平衡的二元分类问题，纯随机分类器达到的准确度将是50％。
# 但在这种情况下它接近19％，所以结果似乎相当不错，至少与random baseline 相比：
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
print(float(np.sum(np.array(test_labels) == np.array(test_labels_copy))) / len(test_labels))

# 生成对新数据的预测
predictions = model.predict(x_test)
# 预测中的每个条目都是长度为46的向量:
print(predictions[0].shape)
# 此向量中的系数总和为1：
print(np.sum(predictions[0]))
# 最大的条目是预测类，即具有最高概率的类：
print(np.argmax(predictions[0]))

# 处理标签和损失的不同方式
# 我们之前提到过，编码标签的另一种方法是将它们转换为整数张量，如下所示：
# y_train = np.array(train_labels)
# y_test = np.array(test_labels)
# 这种方法唯一会改变的是损失函数的选择。 从头开始重新训练模型中使用的损失函数,ategorical_crossentropy,期望标签遵循分类编码。
# 但是对于整数标签，应该使用sparse_categorical_crossentropy：
# model.compile(optimizer='rmsprop',
#               loss='sparse_categorical_crossentropy',
#               metrics=['acc'])

# 具有足够大的中间层的重要性
# 我们之前提到过，因为最终输出是46维的，所以应该避免使用少于46个隐藏单元的中间层。
# 现在看看当我们通过使中间层明显小于46维而引入信息瓶颈时会发生什么：以4维为例
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(4, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(partial_x_train,
#           partial_y_train,
#           epochs=20,
#           batch_size=128,
#           validation_data=(x_val, y_val))
