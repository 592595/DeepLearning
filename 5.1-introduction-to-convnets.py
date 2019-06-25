# 1、实例化一个小convnet
# 下面的6行代码展示了基本convnet的外观
# 它是一堆conv2d和maxpooling2d层
# convnet采用形状的输入张量(image_height, image_width, image_channels)（不包括批处理尺寸）
# 在我们的例子中，我们将配置convnet来处理大小（28、28、1）的输入，这是mnist图像的格式
# 我们通过将参数input_shape=(28, 28, 1) 传递到第一层来实现这一点

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 让我们展示到目前为止我们的convnet的体系结构
# 可以看到每个conv2d和maxpooling2d层的输出都是一个三维形状张量（高度、宽度、通道）。当我们深入网络时，宽度和高度尺寸往往会缩小。通道数由传递给conv2d层的第一个参数控制（例如32或64）。
print(model.summary())
# 下一步是将最后一个输出张量（形状（3，3，64））输入到一个像我们已经熟悉的那样紧密连接的分类器网络中：a stack of Dense layers
# 这些分类器处理向量是1d，而我们的输出是一个3d张量
# 因此，首先，我们必须将三维输出展平为一维，然后在顶部添加一些密集层

# 2、在convnet上添加分类器
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
# 我们将进行10路分类，所以我们使用一个具有10个输出和一个SoftMax激活的最后一层。现在我们的网络是这样的：
print(model.summary())
# 我们的（3，3，64）输出在经过两个密集层之前被展平为形状向量（576，）。
# 现在，让我们用mnist数字训练我们的convnet。

# 3、在mnist图像上训练convnet

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 根据测试数据评估模型:
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
# 虽然第2章中的密集连接网络的测试精度为97.8%，但我们的基本convnet的测试精度为99.3%：我们的错误率降低了68%（相对）。
