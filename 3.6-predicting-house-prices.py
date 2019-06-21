from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt

# 1、加载波士顿住房数据集
# 有404个训练样本和102个测试样本，每个样本有13个数字特征，例如人均犯罪率，每个住宅的平均房间数，高速公路的可达性等等。
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# print(train_data.shape)
# print(test_data.shape)
# 目标是自住房屋的中位数，价值数千美元：
# print(train_targets)

# 2、规范化数据
# 将神经网络输入所有采用不同范围的值都是有问题的，处理此类数据的一种广泛的最佳实践是进行特征标准化：
# 对于输入数据中的每个特征（输入数据矩阵中的一列），减去feature的平均值并除以标准偏差，以便该要素以0为中心并具有单位标准差。 这很容易在Numpy中完成。
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


# 3、模型定义
# 拥有的训练数据越少，过度拟合就越差，使用小型网络是缓解过度拟合的一种方法。
# 这里将使用一个非常小的网络，其中有两个隐藏层，每个层有64个单元。
# 网络以单个单元结束而不激活（它将是线性层）。 这是标量回归的典型设置（尝试预测单个连续值的回归）。
# 应用激活函数会限制输出可以采用的范围， 例如，如果将sigmoid激活函数应用于最后一层，网络只能学习预测0到1之间的值。
# 这里，因为最后一层是纯线性的，所以网络可以自由地学习预测任何范围内的值。
# 对于回归问题，广泛使用的损失函数：mse loss函数 - 均方误差编译网络，即预测和目标之间差异的平方。
# 在训练期间监控新指标：平均绝对误差（MAE）。 它是预测和目标之间差异的绝对值。 例如，此问题的MAE为0.5意味着您的预测平均减少500美元。
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 4、K-fold验证
# 它包括将可用数据拆分为K个分区（通常为K = 4或5），实例化K个相同模型，并在评估剩余分区时对K-1分区进行训练。
# 所用模型的验证分数是获得的K验证分数的平均值。
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据：来自分区#k的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备培训数据：来自所有其他分区的数据
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 构建Keras模型（已编译）
    model = build_model()
    # 训练模型(in silent mode, verbose=0)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=0)
    # 评估验证数据的模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

# 使用num_epochs = 100运行此结果会产生以下结果：
print(all_scores)
print(np.mean(all_scores))

# 5、为了记录模型在每个时期的表现，我们将修改训练循环以保存每个epoch验证分数日志：
# 内存清理
K.clear_session()
num_epochs = 500  # 尝试更长时间地训练网络：500个时代。
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据：来自分区#k的数据
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    # 准备培训数据：来自所有其他分区的数据
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # 构建Keras模型（已编译）
    model = build_model()
    # 训练模型 (in silent mode, verbose=0)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
# 然后，可以计算所有flod的每个epoch 的MAE分数的平均值。

# 6、建立连续平均 K-fold 验证分数的历史
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 7、绘制验证分数
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# 8、绘制验证分数，排除前10个数据点
# 由于缩放问题和相对较高的方差，可能有点难以看到该情节。 我们来做以下事情：
# 省略前10个数据点，这些数据点与曲线的其余部分的比例不同。
# 用先前点的指数移动平均值替换每个点，以获得平滑的曲线。
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

# 9、训练最终模型
# 一旦完成模型的其他参数调整（除了时期数量，还可以调整隐藏层的大小）
# 可以在所有训练数据上训练最终生产模型，并使用最佳参数 ，然后看看它在测试数据上的表现。
# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
