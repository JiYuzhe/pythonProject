# review 这里不对w进行分开训练

import os
import numpy as np
import tensorflow as tf

# review 这个方法是直接从net文件夹下面取出所有的w和b进行返回
def load_parameter(training_dimension, load_net_dir="../../net/ex3/"):
    # review np.loadtxt直接将加载的文本变成矩阵
    # w1_txt = np.loadtxt(load_net_dir + "w1", dtype=np.float32)
    # review np.reshape里面新的shape里面可以有-1，-1的意思是根据其他维度的值来推断这个维度的值
    # # TODO 这里也要随着input_dim改变进行改变
    # w10 = tf.Variable(tf.constant(np.reshape(w1_txt[0, :], [-1, n_hidden[1]])), trainable=training_dimension[0])
    # w11 = tf.Variable(tf.constant(np.reshape(w1_txt[1, :], [-1, n_hidden[1]])), trainable=training_dimension[1])
    # w12 = tf.Variable(tf.constant(np.reshape(w1_txt[1, :], [-1, n_hidden[1]])), trainable=training_dimension[2])
    #
    # # review tf.concat进行拼接，axis=0表示拼接后改变第0维的大小，1为拼接后改变第1维的大小
    # #  这里的w3和b3也要进行改造才能让维度正确
    # w1 = tf.concat((w10, w11, w12), axis=0)
    w1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w1", dtype=np.float32)))
    w2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w2", dtype=np.float32)))
    w3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w3", dtype=np.float32)))
    # review 修改后就不需要单独处理w3和b3了
    # w3_np = np.loadtxt(load_net_dir + "w3", dtype=np.float32)
    # w3_np = w3_np[:, np.newaxis]
    # w3 = tf.Variable(w3_np)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b1", dtype=np.float32)))
    b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b2", dtype=np.float32)))
    b3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b3", dtype=np.float32)))
    # b3_np = np.loadtxt(load_net_dir + "b3", dtype=np.float32)
    # b3_np = b3_np[np.newaxis]
    # b3 = tf.Variable(tf.constant(b3_np))
    return w1, w2, w3, b1, b2, b3

# review 虽然叫n_hidden，但是包括了输入层和输出层的维度
input_dim = 3
n_hidden = [input_dim, 16, 32, 2]
net_dir = "../../net/ex3/"
data_dir = "../../data/point3/"

def random_parameter():
    # # question 这里要随着input_dim的变化而变化
    # if training_dimension[0]:
    #     # review tf.truncated_normal 截断的产生正态分布函数，正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
    #     w10 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    # else:
    #     # question 如果不能训练设置为全是0的初始参数
    #     w10 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[0])
    #
    # if training_dimension[1]:
    #     w11 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    # else:
    #     w11 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[1])
    #
    # if training_dimension[2]:
    #     w12 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    # else:
    #     w12 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[1])
    #
    # w1 = tf.concat((w10, w11, w12), axis=0)
    w1 = tf.Variable(tf.truncated_normal((n_hidden[0], n_hidden[1]), 0.1))
    b1 = tf.Variable(tf.zeros([n_hidden[1]]))
    w2 = tf.Variable(tf.truncated_normal((n_hidden[1], n_hidden[2]), 0.1))
    b2 = tf.Variable(tf.zeros([n_hidden[2]]))
    w3 = tf.Variable(tf.truncated_normal((n_hidden[2], n_hidden[3]), 0.1))
    b3 = tf.Variable(tf.zeros([n_hidden[3]]))
    # w4 = tf.Variable(tf.truncated_normal((n_hidden[3], n_hidden[4]), 0.1))
    # b4 = tf.Variable(tf.zeros([n_hidden[4]]))
    # w5 = tf.Variable(tf.truncated_normal((n_hidden[4], n_hidden[5]), 0.1))
    # b5 = tf.Variable(tf.zeros([n_hidden[5]]))
    return w1, w2, w3, b1, b2, b3


def train_barrier():
    # if load:
    #     w1, w2, w3, b1, b2, b3 = load_parameter(training_dimension, load_net_dir)
    # else:
    #     w1, w2, w3, b1, b2, b3 = random_parameter(training_dimension)
    w1, w2, w3, b1, b2, b3 = random_parameter()

    # TODO trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
    labeled_data, label = get_all_data()

    # define the network
    # review tf的占位符，在里面None表示行不定，类似于reshape里面的-1
    x_labeled = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_labeled")
    y_labeled = tf.placeholder(tf.float32, [None, 2], name="y_labeled")

    layer1_labeled = tf.nn.relu(tf.matmul(x_labeled, w1) + b1, name="layer1_labeled")
    layer2_labeled = tf.nn.relu(tf.matmul(layer1_labeled, w2) + b2, name="layer2_labeled")
    output_all = tf.matmul(layer2_labeled, w3) + b3

    # TODO 损失函数调回原来的损失函数
    real = tf.reduce_sum(y_labeled * output_all, 1)
    # review reduce_max的维度为1也是按行找最大值，行数不变，列数变为1
    #  loss0包括初始区域的损失和非安全区域的损失，一起的
    other = tf.reduce_max((1 - y_labeled) * output_all - (y_labeled * 10000), 1)
    loss0 = tf.reduce_sum(tf.maximum(0.0, other - real))
    loss = loss0

    # review 这里指示学习率随训练次数下降
    epoch_num = 1000
    batch_size = 1280
    stop_flag = 0
    current_epoch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.01, current_epoch, decay_steps=epoch_num/20, decay_rate=0.9, staircase=True)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    # review batch_num的数量是一致的，batch_num设置为大的
    for ep in range(0, epoch_num):
        current_epoch = ep
        batch_num = int(labeled_data.shape[0] / batch_size)

        # review 通过这种方式让两种类别的data数量一样多
        for i in range(0, batch_num):
            start_batch = i % batch_num
            batch_data = labeled_data[batch_size * start_batch:batch_size * (start_batch + 1), :]
            batch_label = label[batch_size * start_batch: batch_size * (start_batch + 1), :]

            if i % 1 == 0:
                epoch_train_loss = sess.run(loss, feed_dict={x_labeled: labeled_data, y_labeled: label})

                print("epoch: %d/%d, batch: %d/%d, " %
                      (ep, epoch_num-1, i, batch_num-1), end="")
                print("mixed loss: %f" % epoch_train_loss)

                if epoch_train_loss == 0:
                    stop_flag = 1
                    break
            sess.run(train_step, feed_dict={x_labeled: batch_data, y_labeled: batch_label})
        if stop_flag == 1:
            break

    # 跳出所有epoch循环后再run一次loss，看训练后的loss是否等于0，等于0返回true，否则返回false
    epoch_train_loss = sess.run(loss, feed_dict={x_labeled: labeled_data, y_labeled: label})
    if epoch_train_loss == 0:
        r = True
    else:
        r = False
    w1 = sess.run(w1)
    b1 = sess.run(b1)
    w2 = sess.run(w2)
    b2 = sess.run(b2)
    w3 = sess.run(w3)
    b3 = sess.run(b3)
    sess.close()

    # save parameters of the net
    if not os.path.exists(net_dir):
        os.mkdir(net_dir)
    np.savetxt(net_dir + "w1", w1)
    np.savetxt(net_dir + "w2", w2)
    np.savetxt(net_dir + "w3", w3)
    np.savetxt(net_dir + "b1", b1)
    np.savetxt(net_dir + "b2", b2)
    np.savetxt(net_dir + "b3", b3)
    with open(net_dir + "structure", "w") as f:
        f.write(str(len(n_hidden)) + "\n")
        for i in range(0, len(n_hidden)):
            f.write(str(n_hidden[i]) + "\n")

    return r


def abstract_network_barrier(load=False, load_net_dir=net_dir):
    r = train_barrier()


# trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
def get_all_data():
    # 加载的是trace和unsafe里面的点
    train_l1 = np.loadtxt(data_dir + "invariant.txt", dtype=np.float32)
    train_l2 = np.loadtxt(data_dir + "non-invariant.txt", dtype=np.float32)

    # review unsafe的点的数量和trace的要差的不是很多
    if train_l1.shape[0] >= 2 * train_l2.shape[0]:
        # review np.repeat 进行重复，传入的int是倍数
        train_l2 = train_l2.repeat(int(train_l1.shape[0] / train_l2.shape[0]), axis=0)
    elif train_l2.shape[0] >= 2 * train_l1.shape[0]:
        train_l1 = train_l1.repeat(int(train_l2.shape[0] / train_l1.shape[0]), axis=0)

    # review np.random.shuffle将数组的顺序打乱，打乱是along the first axis的
    np.random.shuffle(train_l1)
    np.random.shuffle(train_l2)

    # trace里面的点的label[0, 1], unsafe的label是[1, 0]
    # question 为什么这样设置label暂时还不知道
    label_l1 = np.array([0, 1] * len(train_l1)).reshape(-1, 2)
    label_l2 = np.array([1, 0] * len(train_l2)).reshape(-1, 2)

    # 把l1和l2拼起来，data和label都竖着拼起来
    train = np.concatenate((train_l1, train_l2), axis=0)
    label = np.concatenate((label_l1, label_l2), axis=0)

    # 相当于输入了一个训练样本的数量，将这个indice进行打乱
    # review np.random.permutation也相当于打乱，along the first axis的顺序
    # review 如果输入值是一个int，会randomly permute np.arange(x)
    # question 二次打乱点，让两个点融合
    shuffle_indices = np.random.permutation(train.shape[0])
    # review 这里的shuffle_indices是一个索引数组，将这个数组作为索引值
    train = train[shuffle_indices]
    label = label[shuffle_indices]

    return train, label


abstract_network_barrier(load_net_dir=net_dir)


