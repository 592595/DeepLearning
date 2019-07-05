from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

# 1.特征提取
# 我们通过使用在ImageNet上训练的VGG16网络的卷积基，从猫和狗的图像中提取特征
# 然后在这些特征的基础上训练猫和狗的分类器，来实践这一点。
# 实例化vg16模型
conv_base = VGG16(weights='imagenet',  # 指定从哪个权重检查点初始化模型
                  include_top=False,
                  # 是指包括或不包括网络顶部的密接分类器。默认情况下，这个紧密连接的分类器将对应于来自ImageNet的1000个类。因为我们打算使用自己的密接分类器（只有两个类，cat和dog），所以不需要包含它。
                  input_shape=(150, 150, 3))  # 我们将提供给网络的图像张量的形状。这个参数完全是可选的：如果我们不传递它，那么网络将能够处理任何大小的输入。
conv_base.summary()  # 结构细节

# 最终的特征图具有形状（4，4，512）。这就是我们将在其上粘贴一个密接分类器的特性。
# 特征提取的两种方法：
# 1）在数据集上运行卷积基，将其输出记录到磁盘上的一个numpy数组中，然后使用这个日期作为一个独立的、紧密连接的分类器的输入，类似于本书第一章中所看到的。这个解决方案运行起来非常快速和便宜，因为它只需要对每个输入图像运行一次卷积基，而卷积基是迄今为止管道中最昂贵的部分。然而这种技术根本不允许我们利用数据扩充。
# 2）扩展我们的模型（conv-），在顶部添加密集层，并在输入数据上端到端地运行整个模型。这允许我们使用数据增强，因为每次模型看到每个输入图像都要经过卷积基。然而这种技术要比第一种技术代价更大。
# 让我们浏览一下设置第一个模型所需的代码：根据我们的日期记录conv-base的输出，并使用这些输出将输入数据保存到新模型中。
# 特征提取方法一：我们将首先简单地运行先前引入的图像数据生成器的实例，以提取图像作为numpy数组及其标签。我们将通过调用conv的预测方法从这些图像中提取特征。
base_dir = 'Downloads/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 1000)
test_features, test_labels = extract_features(test_dir, 1000)

# 提取的特征目前是形状的（samples，4，4，512）。我们将把它们送入一个紧密相连的分类器，因此首先我们必须将它们展平到（samples，8192）：
train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

# 此时，我们可以定义我们的密接分类器（注意使用dropout进行正则化），并在刚刚记录的数据和标签上对其进行训练：
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=30,
                    batch_size=20,
                    validation_data=(validation_features, validation_labels))

# 训练非常快，因为我们只需要处理两个密集的层——即使在CPU上，一个时代也不到一秒钟。让我们看看训练期间的损失和准确度曲线：
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
# 我们的验证精度达到了90%左右，比我们在前一节中从零开始训练的小模型所能达到的效果要好得多。
# 然而，我们的曲线图也表明，我们几乎从一开始就过度拟合了。
# 这是因为该技术不利用数据扩充，这对于防止对小图像数据集的过度拟合至关重要。

# 特征提取方法二：因为模型的行为与层类似，所以可以向顺序模型添加模型（如conv_库），就像添加层一样：
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
print(model.summary())

# 在我们编译和训练模型之前，要做的一件非常重要的事情是冻结卷积基。在keras中，通过将网络的可训练属性设置为false来冻结网络：
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))
# 使用此设置，将只训练我们添加的两个密集层的权重。总共有四个权重张量：每层两个（主权重矩阵和偏移向量）。
# 注意，为了使这些更改生效，我们必须首先编译模型。如果在编译之后修改了权重可训练性，那么应该重新编译模型，否则这些更改将被忽略。
# 现在，我们可以开始训练我们的模型，使用前面示例中使用的相同的数据扩充配置：
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    # This is the target directory
    train_dir,
    # All images will be resized to 150x150
    target_size=(150, 150),
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2)

# 保存：
model.save('cats_and_dogs_small_3.h5')

# 绘制：
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

# 2、微调 -- 步骤如下：
# 1）将我们自定义网络添加到已训练的基础网络之上。
# 2）冻结基础网络。
# 3）训练我们添加的部件。
# 4）解冻基础网络中的某些层。
# 5）共同训练这些层和添加的零件。
# 在进行特征提取时，我们已经完成了前3个步骤。让我们继续第四步：我们将解冻conv_基地，然后冻结里面的各个层。

# 这就是我们的卷积基的样子：
print(conv_base.summary())
# 我们将对最后3个卷积层进行微调，这意味着在Block4_池之前的所有层都应冻结，并且可以训练Block5_Conv1、Block5_Conv2和Block5_Conv3层。
# 让我们来设置它，从上一个例子中我们停止的地方开始:
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# 微调我们的网络。我们将使用rmsprop优化器，使用非常低的学习率来实现这一点。使用低学习率的原因是，我们希望将所做的修改的大小限制为我们正在微调的3个层的表示。太大的更新可能会损害这些表示。
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)
# 保存
model.save('cats_and_dogs_small_4.h5')
# 绘制
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


# 使曲线更易读的方法，我们可以用这些量的指数移动平均值来替换每一个损失和精度，从而使它们更平滑。下面是一个简单的实用程序函数：
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# 注意，损失曲线没有显示出任何真正的改善（事实上，它正在恶化）。
# 你可能想知道，如果损失不减少，精确度如何提高？
# 答案很简单：我们显示的是逐点损失值的平均值，但实际影响准确性的是损失值的分布，而不是它们的平均值，
# 因为准确性是模型预测的类概率的二元阈值化的结果。即使这没有反映在平均损失中，该模型可能仍在改进。
# 我们现在可以根据测试数据最终评估此模型：
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)
# 这里我们得到了97%的测试精度。在最初围绕这个数据集的竞争中，这将是最重要的结果之一。
# 然而，利用现代的深度学习技术，我们仅使用了一小部分可用的培训数据（约10%）就取得了这一结果。
# 与2000个样本相比，能够训练20000个样本有很大的区别！
