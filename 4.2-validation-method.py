# Hold-out validation：
num_validation_samples = 10000
# Shuffling the data is usually appropriate.
np.random.shuffle(data)

# 定义验证集
validation_data = data[:num_validation_samples]
data = data[num_validation_samples:]

# 定义训练集
training_data = data[:]

# 在训练数据上训练模型，并在验证数据上对其进行评估
model = get_model()
model.train(training_data)
validation_score = model.evaluate(validation_data)

# 一旦调整了超参数，就可以从头开始训练所有非测试数据的最终模型
# 此时，您可以调整模型，重新训练，评估，再次调整......
model = get_model()
model.train(np.concatenate([training_data,
                            validation_data]))
test_score = model.evaluate(test_data)

#  K-fold cross-validation:
k = 4
num_validation_samples = len(data) // k
np.random.shuffle(data)
validation_scores = []
# 选择validationdata分区
for fold in range(k):
    validation_data = data[num_validation_samples * fold:num_validation_samples * (fold + 1)]
    # 使用剩余的数据作为训练数据。 请注意，+运算符是列表连接，而不是求和。
    training_data = data[:num_validation_samples * fold] + data[num_validation_samples * (fold + 1):]
    # 创建一个全新的模型实例（未经过训练）
    model = get_model()
    model.train(training_data)
    validation_score = model.evaluate(validation_data)
    validation_scores.append(validation_score)
# 验证分数：k folds的验证分数的平均值
validation_score = np.average(validation_scores)
# 在所有可用的非测试数据上训练最终模型
model = get_model()
model.train(data)
test_score = model.evaluate(test_data)
