from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time
from matplotlib import pyplot as plt
# 让我们首先定义我们考虑的两个图像的路径：样式参考图像和目标图像。
# 为了确保处理的所有图像具有相似的大小（大小不同会使样式传输更加困难），我们稍后会将它们全部调整为400px的共享高度。
# This is the path to the image you want to transform.
target_image_path = '/home/ubuntu/data/portrait.png'
# This is the path to the style image.
style_reference_image_path = '/home/ubuntu/data/popova.jpg'

# Dimensions of the generated picture.
width, height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)

# 我们需要一些辅助功能来加载、预处理和后处理将进入和退出vg19 convnet的图像：
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
# 让我们设置vg19网络。它将一批三个图像作为输入：样式引用图像、目标图像和将包含生成图像的占位符。
# 占位符只是一个符号张量，其值通过numpy数组从外部提供。
# 样式引用和目标图像是静态的，因此使用k.constant定义，而生成图像的占位符中包含的值将随时间而更改。
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))

# This placeholder will contain our generated image
combination_image = K.placeholder((1, img_height, img_width, 3))

# We combine the 3 images into a single batch
input_tensor = K.concatenate([target_image,
                              style_reference_image,
                              combination_image], axis=0)

# We build the VGG19 network with our batch of 3 images as input.
# The model will be loaded with pre-trained ImageNet weights.
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False)
print('Model loaded.')

# 让我们定义内容丢失，以确保vg19 convnet的顶层具有目标图像和生成图像的类似视图：
def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 现在，这里是样式丢失。它利用一个辅助函数来计算输入矩阵的G矩阵，即原始特征矩阵中的相关图。

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# 在这两个损失分量中，我们添加了第三个，即“总变化损失”。它旨在鼓励生成图像的空间连续性，从而避免过度像素化的结果。你可以把它理解为一种正规化的损失。
def total_variation_loss(x):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# 我们最小化的损失是这三个损失的加权平均值。为了计算内容丢失，我们只利用一个顶层，即block5_conv2层，而对于样式丢失，我们使用的是一个比跨越低级和高级层的层列表。我们在最后加上总的变化损失。
# 根据您使用的样式参考图像和内容图像，您可能需要调整内容权重系数，内容损失对总损失的贡献。较高的内容权重意味着目标内容在生成的图像中更容易识别。
# Dict mapping layer names to activation tensors
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
# Name of layer used for content loss
content_layer = 'block5_conv2'
# Name of layers used for style loss
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
# Weights in the weighted average of the loss components
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025

# Define the loss by adding all components to a `loss` variable
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
                                      combination_features)
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(style_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# 最后建立了梯度下降过程。在最初的Gatys等人本文采用L-BFGS算法进行了优化，这也是本文将要用到的算法。这是与前一节中的深梦例子的关键区别。L-BFGS算法与scipy一起提供。但是，scipy的实现有两个小小的限制：
# 1)它需要作为两个单独的函数传递损失函数的值和渐变的值。
# 2)它只能应用于平面向量，而我们有一个三维图像阵列。
# 单独计算损失函数的值和梯度的值对我们来说是非常低效的，因为这会导致在两者之间进行大量的冗余计算。通过联合计算，我们的速度几乎是原来的两倍。为了绕过这个问题，我们设置了一个名为evaluator的python类，它将同时计算loss value和gradients值，在第一次调用时返回loss值，并为下一次调用缓存渐变。
# Get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)[0]

# Function to fetch the values of the current loss and the current gradients
fetch_loss_and_grads = K.function([combination_image], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, img_height, img_width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# 最后，我们可以使用scipy的l-bfgs算法运行梯度上升过程，在算法的每次迭代中保存当前生成的图像（这里，一次迭代代表20个梯度上升步骤）：
result_prefix = 'style_transfer_result'
iterations = 20

# Run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss.
# This is our initial state: the target image.
# Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
x = preprocess_image(target_image_path)
x = x.flatten()
for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # Save current generated image
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# Content image
plt.imshow(load_img(target_image_path, target_size=(img_height, img_width)))
plt.figure()

# Style image
plt.imshow(load_img(style_reference_image_path, target_size=(img_height, img_width)))
plt.figure()

# Generate image
plt.imshow(img)
plt.show()
