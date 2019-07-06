from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.applications import VGG16
from keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import cv2

# 1、可视化中间激活
# 加载5.2节中保存的模型
model = load_model('cats_and_dogs_small_2.h5')
model.summary()  # As a reminder.

# 这将是我们将使用的输入图像——猫的图像，不是网络训练的图像的一部分
img_path = 'Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'
# We preprocess the image into a 4D tensor
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# Remember that the model was trained on inputs
# that were preprocessed in the following way:
img_tensor /= 255.
# Its shape is (1, 150, 150, 3)
print(img_tensor.shape)

# 显示一下我们的图片
plt.imshow(img_tensor[0])
plt.show()

# 为了提取我们想要查看的特征图，我们将创建一个以成批图像为输入的keras模型，并输出所有卷积和池层的激活。为此，我们将使用keras类模型。
# 使用两个参数来实例化模型：输入张量（或输入张量列表）和输出张量（或输出张量列表）。
# 结果类是一个keras模型，就像我们熟悉的顺序模型一样，将指定的输入映射到指定的输出。
# 使模型类与众不同的是，它允许具有多个输出的模型，与顺序的不同。
# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:8]]
# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# 当输入图像时，此模型返回原始模型中层激活的值。
# 这是第一次在本书中遇到多输出模型：到目前为止，我们所看到的模型只有一个输入和一个输出。
# 在一般情况下，模型可以有任意数量的输入和输出。这一个有一个输入和8个输出，每层激活一个输出。
# This will return a list of 5 Numpy arrays:
# one array per layer activation
activations = activation_model.predict(img_tensor)

# 例如，这是激活CAT图像输入的第一个卷积层：
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# 这是一张带有32个频道的148x148功能图。让我们尝试可视化第三个频道：
plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
plt.show()

# 这个通道似乎编码了一个对角边缘检测器。
# 让我们试试第30个通道——但是注意，我们自己的通道可能会有所不同，因为卷积层学习的特定过滤器并不具有确定性。
plt.matshow(first_layer_activation[0, :, :, 30], cmap='viridis')
plt.show()

# 这一个看起来像一个“亮绿点”探测器，对猫眼的编码很有用。
# 在这一点上，让我们开始并绘制一个网络中所有激活的完整可视化图。
# 我们将提取并绘制8个激活图中的每个通道，并将结果堆叠成一个大图像张量，通道并排堆叠。
# These are the names of the layers, so can have them as part of our plot
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

# Now let's display our feature maps
for layer_name, layer_activation in zip(layer_names, activations):
    # This is the number of features in the feature map
    n_features = layer_activation.shape[-1]

    # The feature map has shape (1, size, size, n_features)
    size = layer_activation.shape[1]

    # We will tile the activation channels in this matrix
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                            :, :,
                            col * images_per_row + row]
            # Post-process the feature to make it visually palatable
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
            row * size: (row + 1) * size] = channel_image

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

# 2、可视化convnet筛选器
# 检查convnets学习的过滤器的另一个简单方法是显示每个过滤器要响应的可视模式。
# 这可以通过输入空间中的梯度上升来实现：将梯度下降应用于convnet输入图像的值，以便从空白输入图像开始最大化特定过滤器的响应。所产生的输入图像将是所选滤波器最大响应的图像。
# 这个过程很简单：我们将建立一个损失函数，使给定卷积层中给定滤波器的值最大化，然后我们将使用随机梯度下降来调整输入图像的值，从而使这个激活值最大化。
# 例如，在VGG16网络的“block3_conv1”层中激活过滤器0时会丢失，在ImageNet上进行了预训练。
model = VGG16(weights='imagenet',
              include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:, :, :, filter_index])

# 为了实现梯度下降，我们需要这个损失相对于模型输入的梯度。为此，我们将使用与keras后端模块打包的gradients函数：
# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads = K.gradients(loss, model.input)[0]

# 使用梯度下降过程的一个非明显的弊端是使梯度张力正常化，
# 除以它的l2范数（张量中值的平方平均值的平方根）。这样可以确保对输入图像所做的更新的幅度始终在同一范围内。
# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

# 现在我们需要一种方法来计算损失张量和梯度张量的值，给定一个输入图像。
# 我们可以定义一个keras后端函数来实现这一点：iterate是一个函数，它接受一个numpy张量（作为大小为1的张量列表），并返回两个numpy张量的列表：loss值和渐变值。
iterate = K.function([model.input], [loss, grads])
# Let's test it:
loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])

# 此时，我们可以定义一个python循环来执行随机梯度下降：
# We start from a gray image with some noise
input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.

# Run gradient ascent for 40 steps
step = 1.  # this is the magnitude of each gradient update
for i in range(40):
    # Compute the loss value and gradient value
    loss_value, grads_value = iterate([input_img_data])
    # Here we adjust the input image in the direction that maximizes the loss
    input_img_data += grads_value * step


# 生成的图像张量将是形状的浮点张量（1150、150、3），其值可能不在[0、255]之内。
# 因此，我们需要对这个张量进行后期处理，以将其转换为可显示的图像。我们通过以下简单的实用功能来实现：
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# 现在我们有了所有的片段，让我们把它们放在一起，形成一个python函数，它接受一个层名和一个过滤器索引作为输入，并返回一个有效的图像张量，它表示最大化激活指定过滤器的模式：
def generate_pattern(layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0))
plt.show()

# 在Block3_Conv1层中，过滤器0似乎对波尔卡点图案有响应。
# 现在有趣的部分：我们可以开始可视化每一层中的每一个过滤器。
# 为了简单起见，我们将只查看每层的前64个过滤器，并且只查看每个卷积块的第一层（block1_conv1、block2_conv1、block3_conv1、block4_conv1、block5_conv1）。
# 我们将在64x64过滤模式的8x8网格上排列输出，每个过滤模式之间有一些黑边。
for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']:
    size = 64
    margin = 5

    # This a empty (black) image where we will store our results.
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):  # iterate over the rows of our results grid
        for j in range(8):  # iterate over the columns of our results grid
            # Generate the pattern for filter `i + (j * 8)` in `layer_name`
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            # Put the result in the square `(i, j)` of the results grid
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    # Display the results grid
    plt.figure(figsize=(20, 20))
    plt.imshow(results)
    plt.show()

# 3、可视化类激活的热图
# 我们将再次使用预先训练的VGG16网络演示此技术：
K.clear_session()
# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model = VGG16(weights='imagenet')

# 让我们考虑以下两只非洲象的图像，可能是母亲和它的幼崽，漫步在稀树草原（根据知识共享许可证）：
# 让我们将此图像转换为VGG16模型可以读取的内容：模型在大小为224x244的图像上进行训练，根据实用函数keras.applications.vgg16.preprocess_input中打包的一些规则进行预处理。所以我们需要加载图像 ，将其大小调整为224x224，将其转换为Numpy float32张量，并应用这些预处理规则。

# The local path to our target image
img_path = '/Users/fchollet/Downloads/creative_commons_elephant.jpg'

# `img` is a PIL image of size 224x224
img = image.load_img(img_path, target_size=(224, 224))

# `x` is a float32 Numpy array of shape (224, 224, 3)
x = image.img_to_array(img)

# We add a dimension to transform our array into a "batch"
# of size (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# Finally we preprocess the batch
# (this does channel-wise color normalization)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])

# 预测此图片的前三名是：
# 非洲象（概率为92.5％）
# Tusker（概率为7％）
# 印度象（概率为0.4％）
# 因此，我们的网络已经认识到我们的形象包含一定数量的非洲大象。 最大激活的预测向量中的条目是对应于“非洲大象”类的条目，在索引386处：
print(np.argmax(preds[0]))

# 为了可视化我们图像的哪个部分是最像“非洲象”的，我们设置Grad-CAM过程：
# This is the "african elephant" entry in the prediction vector
african_elephant_output = model.output[:, 386]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([x])

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

# 出于可视化目的，我们还将热图在0和1之间标准化：
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

# 最后，我们将使用OpenCV生成一个图像，将原始图像与我们刚刚获得的热图叠加在一起：
# We use cv2 to load the original image
img = cv2.imread(img_path)

# We resize the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# We convert the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# We apply the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 here is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img

# Save the image to disk
cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)

# 此可视化技术回答了两个重要问题：
# 	1）为什么网络认为这个图像包含非洲象？
# 	2）非洲大象在哪里？
# 特别值得注意的是，大象幼崽的耳朵被强烈激活：这可能是网络如何区分非洲和印度大象的区别。
