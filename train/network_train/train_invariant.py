import tensorflow as tf
from tensorflow import keras
import kerastuner as kt
import numpy as np

# review 这个问题实质上就是一个简单的分类问题，在I框内的小于等于0，在I框外面的大于0就完事了
#  这个问题最后loss完全等于0了，应该是解决了
#  尝试将验证集调大一点可能会好一点？
#  最后使用的网络结构是1 12 8 1，学习率是0.01

# TODO 定义一些跟特定网络有关的参数，比如输入的维度
FEATURE_NUM = 1

# TODO 关于x和y数据的获取和加载等有了数据再填，其实可以不用测试数据集的，或者训练集和测试集都是一样的最后的evaluate来判断是否收敛到0
# TODO 加载数据之前看看能不能打乱，因为取点的时候是分门别类取的点，所以打乱应该会比较有效的防止不收敛
data_file_path = '../../data/points_1.txt'
raw = np.loadtxt(data_file_path)
data_size = raw.shape[0]
# review 先将最原始的数据进行打乱
np.random.shuffle(raw)
training_examples = raw[:, 0]
training_examples = training_examples[:, tf.newaxis]
training_labels = raw[:, 1]
all_dataset = tf.data.Dataset.from_tensor_slices((training_examples, training_labels))
# review 用整个数据集当做测试数据集测试效果
test_dataset = all_dataset
# review 将数据集以8-2的方式划分为验证集和训练集
train_size = int(data_size * 0.8)
train_dataset = all_dataset.take(train_size)
validation_dataset = all_dataset.skip(train_size)


# TODO 自定义的损失函数，对y_true为0和1两个类别分别计算损失，不过计算的损失和y_true没有直接的代数上面的相关
# review 这里提供的y_true和y_pred的维度如下：[batch_size, d0, .. dN]
def custom_loss(y_true, y_pred):
    # review 声明两个表示loss所占比例的常量值
    print(y_true.shape)
    c1, c2 = tf.constant(1.0, dtype=float), tf.constant(1.0, dtype=float)
    total_loss = c1 * (1.0 - y_true) * tf.math.maximum(y_pred, 0) + c2 * y_true * tf.math.maximum(-y_pred, 0)
    return tf.math.reduce_mean(total_loss)


def model_builder(hp):
    tune_model = tf.keras.models.Sequential()
    tune_model.add(tf.keras.layers.Flatten(input_shape=(FEATURE_NUM,)))
    # TODO 先在这里定义一个两层的网络，至于网络的层数后面可以再尝试
    hp_units = hp.Int('units1', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    hp_units2 = hp.Int('units2', min_value=4, max_value=36, step=4)
    tune_model.add(tf.keras.layers.Dense(units=hp_units2, activation='relu'))
    tune_model.add(tf.keras.layers.Dense(units=1))
    # hp.Choice是在几个备选中选择
    # TODO optimizer的选择后面再看，这里应该不需要metrics
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    tune_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                       loss=custom_loss)

    return tune_model


tuner = kt.Hyperband(hypermodel=model_builder, objective='val_loss',
                     max_epochs=10, factor=3, directory='my_dir', project_name='choose_hyper')
stop_early = tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss')

tuner.search(training_examples, training_labels, validation_split=0.2, epochs=50, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units1')} and the optimal number of units in the second densely-connected layer
 is {best_hps.get('units2')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# review 首先挑选epoch超参数
model = tuner.hypermodel.build(best_hps)
history = model.fit(training_examples, training_labels, validation_split=0.2, epochs=50)
val_loss_result = history.history['val_loss']
val_loss_index = val_loss_result.index(min(val_loss_result)) + 1
print('Best epoch is {0:03d}'.format(val_loss_index))
hyper_model = tuner.hypermodel.build(best_hps)
hyper_model.fit(training_examples, training_labels, validation_split=0.2, epochs=val_loss_index)
loss = hyper_model.evaluate(training_examples, training_labels)
print('Best loss is {0:5.5f}'.format(loss))



