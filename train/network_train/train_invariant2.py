import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import numpy as np
import os

# review 再把dataset组合一下，以便指定更大一些的batch_size，使训练更加稳定 done
#  dataset都已经搞定了，batch_size也增加了，还是存在不为0的情况
# TODO 这里要不直接把循环不变式看成一个分类问题，再运用softmax函数来规范化0和1的预测，是不是就会没有损失？
#   好像也不是，这样会引入其他激活函数

# review 这个问题实质上就是一个简单的分类问题，在I框内的小于等于0，在I框外面的大于0就完事了
#  这个问题最后loss完全等于0了，应该是解决了
#  尝试将验证集调大一点可能会好一点？
#  最后使用的网络结构是1 20 24 1，学习率是0.001
#  最后的循环不变式是一条线，可以考虑1的点在循环不变式旁边取

# review 这里最后的循环不变式的范围非常小，只是一条线，所以可以在非不变式的区间多取一些点
#  并将C2的值调高一些，导致如果循环不变式的区间扩充了一些其他点的话损失会非常大

# TODO 定义一些跟特定网络有关的参数，比如输入的维度
FEATURE_NUM = 3
BATCH_SIZE = 2048
SHUFFLE_BUFFER_SIZE = 10000

# TODO 关于x和y数据的获取和加载等有了数据再填，其实可以不用测试数据集的，或者训练集和测试集都是一样的最后的evaluate来判断是否收敛到0
# TODO 加载数据之前看看能不能打乱，因为取点的时候是分门别类取的点，所以打乱应该会比较有效的防止不收敛
data_file_path = '../../data/points_2.txt'
raw = np.loadtxt(data_file_path)
data_size = raw.shape[0]
# review 先将最原始的数据进行打乱
np.random.shuffle(raw)
training_examples = raw[:, 0:3]
# training_examples = training_examples[:, tf.newaxis]
# TODO 这里将label变成好几列
training_labels = raw[:, 3:7]
# review 艹，终于知道为啥构建不了dataset了，training labels也要有第一个维度的值才行
# training_labels = training_labels[:, tf.newaxis]
all_dataset = tf.data.Dataset.from_tensor_slices((training_examples, training_labels))
# review 用整个数据集当做测试数据集测试效果
test_dataset = all_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# review 将数据集以6-4的方式划分为验证集和训练集
train_size = int(data_size * 0.6)
train_dataset = all_dataset.take(train_size).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_dataset = all_dataset.skip(train_size).shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)


# TODO 自定义的损失函数，对y_true为0和1两个类别分别计算损失，不过计算的损失和y_true没有直接的代数上面的相关
# review 这里提供的y_true和y_pred的维度如下：[batch_size, d0, .. dN]
def custom_loss(y_true, y_pred):
    # review 声明两个表示loss所占比例的常量值
    y_true_label = y_true[:, 0]
    # review 后面这几列是input，需要通过input来判断是否在Q中从而是否产生损失
    y_true_input = y_true[:, 1:4]
    print(y_true.shape)
    c1, c2 = tf.constant(1.0, dtype=float), tf.constant(1.0, dtype=float)
    # review 需要当y_true为1的时候，y_pred预测一个正数值，y_true为0的时候，y_pred预测一个负数值
    #  如果在这里引入像SVM这样的能不能让分类分散的更开一点？
    # review 这里用vectorization的方式编码了判断分别要求label是1，且不在Q里面且在I里面的时候才产生损失
    total_loss = c1 * (1.0 - y_true_label) * tf.math.maximum(y_pred, 0) + c2 * tf.abs(
        y_true_input[:, 0] - y_true_input[:, 1] - y_true_input[:, 2]) * y_true_label * tf.math.maximum(-y_pred, 0)
    # total_loss = c1 * (1.0 - y_true_label) * tf.math.maximum(y_pred, 0)
    return tf.math.reduce_mean(total_loss)


def model_builder(hp):
    tune_model = tf.keras.models.Sequential()
    tune_model.add(tf.keras.layers.Flatten(input_shape=(FEATURE_NUM,)))
    # TODO 先在这里定义一个两层的网络，至于网络的层数后面可以再尝试
    hp_units = hp.Int('units1', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    hp_units2 = hp.Int('units2', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units2, activation='relu'))
    hp_units3 = hp.Int('units3', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units3, activation='relu'))
    hp_units4 = hp.Int('units4', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units4, activation='relu'))
    hp_units5 = hp.Int('units5', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units5, activation='relu'))
    hp_units6 = hp.Int('units6', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units6, activation='relu'))
    tune_model.add(tf.keras.layers.Dense(units=1))
    # hp.Choice是在几个备选中选择
    # TODO optimizer的选择后面再看，这里应该不需要metrics
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=0.001,
                                                                 decay_steps=11 * 100,
                                                                 decay_rate=10,
                                                                 # 指明decay是连续的还是阶梯式的decay
                                                                 staircase=False)
    tune_model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                       loss=custom_loss)

    return tune_model


tuner = kt.Hyperband(hypermodel=model_builder, objective='val_loss',
                     max_epochs=10, factor=3, directory='../my_dir', project_name='choose_hyper2')
stop_early = tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss')

# TODO Tuner search传入的值可以假想成和fit是一样的
tuner.search(train_dataset, validation_data=validation_dataset, epochs=50, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')} and the optimal number of units in the second densely-connected layer
 is {best_hps.get('units2')} and the optimal number of units in the third densely-connected layer
 is {best_hps.get('units3')} and the optimal number of units in the fourth densely-connected layer
 is {best_hps.get('units4')} and the optimal number of units in the fourth densely-connected layer
 is {best_hps.get('units5')} and the optimal number of units in the fourth densely-connected layer
 is {best_hps.get('units6')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# review 首先挑选epoch超参数
model = tuner.hypermodel.build(best_hps)
model.load_weights('../../checkpoints/weight2')
# history = model.fit(train_dataset, validation_data=validation_dataset, epochs=20000)
# model.save_weights('../../checkpoints/weight2')
# model.load_weights('../../checkpoints/weight2')
# loss = model.evaluate(test_dataset)

n_hidden = [3, 24, 4, 4, 20, 16, 16, 1]
net_dir = "../../net/"
# review 对model的每一层进行访问并得到每一层的权值分别保存在txt中便于后续提取
#  第0层没有权值，从后面开始weight是一个长度为2的list，第一个元素装的w，第二个元素装的b，W的维度为(输入神经元个数, 输出神经元个数)，b的维度为输出神经元个数
for index, layer in enumerate(model.layers):

    # save parameters of the net
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    if index >= 1:
        layer_weight = layer.get_weights()
        weight_w = layer_weight[0]
        weight_b = layer_weight[1]
        np.savetxt(net_dir + "w" + str(index-1), weight_w)
        np.savetxt(net_dir + "b" + str(index-1), weight_b)

with open(net_dir + "structure", "w") as f:
    f.write(str(len(n_hidden)) + "\n")
    for i in range(0, len(n_hidden)):
        f.write(str(n_hidden[i]) + "\n")


# val_loss_result = history.history['loss']
# val_loss_index = val_loss_result.index(min(val_loss_result)) + 1
# print('Best epoch is {0:03d}'.format(val_loss_index))
# hyper_model = tuner.hypermodel.build(best_hps)
# hyper_model.fit(train_dataset, validation_data=validation_dataset, epochs=val_loss_index)
# loss = hyper_model.evaluate(test_dataset)
# print('Best loss is {0:5.7f}'.format(loss))
# hyper_model.save_weights('../../checkpoints/weight2')
# hyper_model.load_weights('../../checkpoints/weight2')
