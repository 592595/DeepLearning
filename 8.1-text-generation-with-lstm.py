import keras
import numpy as np
from keras import layers
import random
import sys
# 1)	Preparing the data
# 首先下载语料库并将其转换为小写：
path = keras.utils.get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('Corpus length:', len(text))
# 接下来，我们将提取部分重叠的长度maxlen序列，一个热编码，并将它们打包成一个三维numpy数组x形状(sequences, maxlen, unique_characters)。
# 同时，我们准备了一个包含相应目标的数组Y：在每个提取序列之后出现的一个热编码字符。
# Length of extracted character sequences
maxlen = 60

# We sample a new sequence every `step` characters
step = 3

# This holds our extracted sequences
sentences = []

# This holds the targets (the follow-up characters)
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))

# List of unique characters in the corpus
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
# Dictionary mapping unique characters to their index in `chars`
char_indices = dict((char, chars.index(char)) for char in chars)

# Next, one-hot encode the characters into binary arrays.
print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# 2)	Building the network
# 我们的网络是一个单一的LSTM层，然后是一个密集的分类器和所有可能的字符的SoftMax。
# 但是让我们注意到，循环神经网络并不是生成序列数据的唯一方法；1d-convnets在最近的时间里也被证明是非常成功的。
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))
# 由于我们的目标是一个热编码的，我们将使用分类交叉熵作为损失来训练模型：
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# 3)	Training the language model and sampling from it
# 给定一个经过训练的模型和一个种子文本片段，我们通过重复生成新文本：
# a）根据目前可用的文本，从模型中绘制下一个字符的概率分布。
# b）将分配重新加权到一定的“温度”
# c）根据重新加权分布随机抽取下一个字符
# d）在可用文本末尾添加新字符
# 这是我们用来重新计算出模型的原始概率分布的代码，并从中绘制一个字符索引（“采样函数”）：
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
# 最后，这是我们重复训练和生成文本的循环。我们开始在每一个时代之后使用不同的温度范围生成文本。
# 这使我们能够看到生成的文本是如何随着模型开始聚合而演变的，以及采样策略中温度的影响。
for epoch in range(1, 60):
    print('epoch', epoch)
    # Fit the model for 1 epoch on the available training data
    model.fit(x, y,
              batch_size=128,
              epochs=1)

    # Select a text seed at random
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(generated_text)

        # We generate 400 characters
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.

            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

