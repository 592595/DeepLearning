import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import backend as K

# 1、A temperature forecasting problem
# 我们来看看数据：
data_dir = 'Downloads/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# 让我们将所有这些420,551行数据转换为Numpy数组：
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# 例如，这是温度（以摄氏度为单位）随时间变化的曲线图：
temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()

# 在此图中，可以清楚地看到温度的年度周期。
# 这是前十天温度数据的更窄的图（由于数据每十分钟记录一次，我们每天得到144个数据点）：
plt.plot(range(1440), temp[:1440])
plt.show()

# 在这个图上，可以看到每日周期，特别是过去4天。 我们还可以注意到，这十天的时间必须来自一个相当寒冷的冬季。
# 如果我们试图在给定几个月的过去数据的情况下预测下个月的平均温度，由于数据的可靠年度周期性，问题将很容易。
# 但是，在几天的时间内查看数据，温度看起来更加混乱。 那么这个时间序列是否可以在日常范围内预测？ 我们来看看。

# 2、Preparing the data
# 给定的数据可以追溯到回溯时间步长（时间步长为10分钟）并且每个步骤采样时间步长，我们能否预测延迟时间步长的温度？我们将使用以下参数值：
# lookback = 720, i.e. our observations will go back 5 days.
# steps = 6, i.e. our observations will be sampled at one data point per hour.
# delay = 144, i.e. our targets will be 24 hours in the future.
# 我们需要做两件事：
# 1）将数据预处理为神经网络可以摄取的格式。这很简单：数据已经是数字的，所以我们不需要做任何矢量化。然而，数据中的每个时间序列具有不同的规模（例如，温度通常在-20和+30之间，但是以mbar测量的压力大约是1000）。因此，我们将独立地对每个时间序列进行标准化，以便它们都以相似的比例获取小值。
# 2）编写一个Python生成器，它接收我们当前的浮点数据数组，并从最近的过去产生批量数据，以及未来的目标温度。由于我们的数据集中的样本是高度冗余的（例如样本N和样本N + 1将具有共同的大部分时间步长），因此明确分配每个样本将是非常浪费的。相反，我们将使用原始数据动态生成样本。

# 我们通过减去每个时间序列的平均值并除以标准偏差来预处理数据。我们计划使用前200,000个时间步作为训练数据，因此我们仅计算这部分数据的平均值和标准差：
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std


# 数据生成器
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


# 现在让我们使用抽象生成器函数来实例化三个生成器，一个用于训练，一个用于验证，一个用于测试。 每个都将查看原始数据的不同时间段：训练生成器查看前200,000个步骤，验证生成器查看以下100,000个，并且测试生成器查看剩余部分。
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size


# 3、A common-sense, non-machine-learning baseline
# 假设温度时间序列是连续的（明天的温度可能接近今天的温度）以及每日期间的周期。
# 常识方法是始终预测从现在起24小时的温度将等于现在的温度。让我们使用平均绝对误差度量（MAE）来评估这种方法。
# 平均绝对误差简单地等于：np.mean(np.abs(preds - targets))
# 这是我们的评估循环：
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))


evaluate_naive_method()
# 它产生的MAE为0.29。 由于我们的温度数据已经标准化为以0为中心并且标准偏差为1，因此该数字不能立即解释。
# 它转换为平均绝对误差0.29 * temperature_std摄氏度，即2.57˚C。
# 这是一个相当大的平均绝对误差现在游戏是利用我们的深度学习知识做得更好。


# 4、A basic machine learning approach
# 这是一个简单的完全连接模型，我们首先展平数据，然后通过两个密集层运行。
# 注意最后一个Dense图层缺少激活功能，这是回归问题的典型特征。 我们使用MAE作为损失。
# 由于我们正在评估完全相同的数据并使用与我们的常识方法完全相同的指标，因此结果将直接具有可比性。
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
# 让我们显示验证和训练的损失曲线：
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 5、A first recurrent baseline
# 使用GRU层：
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 好多了！ 我们能够显着击败常识基线，这样就证明了机器学习的价值，以及与此类任务中的序列扁平化密集网络相比，循环网络的优越性。我们新的验证MAE约为0.265（在我们开始显着过度拟合之前）转换为去标准化后的平均绝对误差2.35˚C。 这是我们初始误差2.57˚C的稳固收益，但我们可能还有一些改进余地。
# 但是过度拟合了...

# 6、Using recurrent dropout to fight overfitting
# Keras中的每个循环层都有两个与dropout相关的参数：dropout，一个指定图层输入单位的丢失率的float，以及recurrent_dropout，指定循环单位的丢失率。让我们将丢失和重复丢失添加到我们的GRU层，看看它如何影响过度拟合。
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# 巨大的成功; 在前30个时代，我们不再过度拟合。 然而，虽然我们有更稳定的评估分数，但我们的最佳分数并不比以前低很多。

# 7、Stacking recurrent layers
# 由于我们不再过度拟合，我们似乎遇到了性能瓶颈，我们应该开始考虑增加网络的容量。
# 通常通过增加层中的单元数或添加更多层来增加网络容量。循环层堆叠是构建更强大的循环网络的经典方法
model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1,
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# 我们可以看到，添加的图层确实提高了我们的结果，虽然不是很显着。 我们可以得出两个结论：
# 1）	由于我们仍然没有过度拟合，我们可以安全地增加层的大小，以寻求一点验证损失改进。 但是，这确实具有不可忽略的计算成本。
# 2）	由于添加一个层并没有对我们产生重大影响，因此我们可能会看到此时增加网络容量的收益递减。


# 8、Using bidirectional RNNs
# 本节中的RNN层到目前为止按时间顺序处理序列（较早的时间步长）可能是一个随意的决定。如果它是按照反时间顺序处理输入序列，那么我们的RNN可能表现得足够好吗（例如，新的时间步长）？
# 我们需要做的就是编写数据生成器的变体，其中输入序列沿时间维度恢复（最后一行用yield samples[:, ::-1, :], targets替换）。训练与本节第一个实验中使用的相同的一个GRU层网络。
def reverse_order_generator(data, lookback, delay, min_index, max_index,
                            shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples[:, ::-1, :], targets


train_gen_reverse = reverse_order_generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size)
val_gen_reverse = reverse_order_generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_index=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size)

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen_reverse,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen_reverse,
                              validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 结果：逆序GRU甚至在常识基线上也表现不佳，这表明在我们的案例中按时间顺序处理对于我们方法的成功非常重要。
# 因此，层的时间顺序版本必然优于逆序版本。但是对于许多其他问题，包括自然语言，这通常是不正确的：直观地说，单词在理解句子时的重要性通常不取决于它在句子中的位置。
# 让我们在上一节的LSTM IMDB示例中尝试相同的技巧：
# Number of words to consider as features
max_features = 10000
# Cut texts after this number of words (among top max_features most common words)
maxlen = 500

# 解决Object arrays cannot be loaded when allow_pickle=False 的错误
old = np.load
np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Reverse sequences
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test]

# Pad sequences
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
# 我们得到的结果与我们在上一节中尝试的时间顺序LSTM几乎相同。
# 因此，值得注意的是，在这样的文本数据集中，逆序处理与时间顺序处理一样有效。
# 要在Keras中实例化双向RNN，可以使用双向层，其将第一个参数作为循环层实例。 双向将创建此循环层的第二个单独实例，并将使用一个实例按时间顺序处理输入序列，另一个实例以相反的顺序处理输入序列。 让我们试试IMDB情绪分析任务：
K.clear_session()

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

model = Sequential()
model.add(layers.Bidirectional(
    layers.GRU(32), input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)
