# 使用Python库Keras学习对手写数字进行分类的神经网络（将手写数字（28*28px）的灰度图像分为10个类别：0-9；
import keras
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

print(keras.__version__)

# 这是输入数据
# Numpy，在这里格式化为float32类型：
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# network如下
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 网络编译步骤：
# categorical_crossentropy是用作学习权重张量的反馈信号的损失函数，训练阶段将尝试最小化。
# 这种损失的减少是通过小批量随机梯度下降发生的。
# 控制梯度下降的特定使用的确切规则由作为第一个参数传递的rmsprop优化器定义。
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 最后，训练循环：
# 网络将开始以128个样本的小批量重复训练数据
# 5次以上（对所有训练数据的每次迭代称为纪元）。
# 在每次迭代时，网络将根据批次上的损失计算权重的梯度，并相应地更新权重。
# 在这5个时期之后，网络将执行2,345个梯度更新（每个时期469个），
# 并且网络的丢失将足够低，使得网络能够以高精度对手写数字进行分类。
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)
