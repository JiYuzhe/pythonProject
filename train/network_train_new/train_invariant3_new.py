# review 之前的训练方法结果有点问题，使用老的tensorflow版本训练

import os
import numpy as np
import tensorflow as tf

# review 这个方法是直接从net文件夹下面取出所有的w和b进行返回
def load_parameter(training_dimension, load_net_dir="../../net/ex3/"):
    # review np.loadtxt直接将加载的文本变成矩阵
    w1_txt = np.loadtxt(load_net_dir + "w1", dtype=np.float32)
    # review np.reshape里面新的shape里面可以有-1，-1的意思是根据其他维度的值来推断这个维度的值
    # TODO 这里也要随着input_dim改变进行改变
    w10 = tf.Variable(tf.constant(np.reshape(w1_txt[0, :], [-1, n_hidden[1]])), trainable=training_dimension[0])
    w11 = tf.Variable(tf.constant(np.reshape(w1_txt[1, :], [-1, n_hidden[1]])), trainable=training_dimension[1])
    w12 = tf.Variable(tf.constant(np.reshape(w1_txt[1, :], [-1, n_hidden[1]])), trainable=training_dimension[2])

    # review tf.concat进行拼接，axis=0表示拼接后改变第0维的大小，1为拼接后改变第1维的大小
    #  这里的w3和b3也要进行改造才能让维度正确
    w1 = tf.concat((w10, w11, w12), axis=0)
    w2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w2", dtype=np.float32)))
    w3_np = np.loadtxt(load_net_dir + "w3", dtype=np.float32)
    w3_np = w3_np[:, np.newaxis]
    w3 = tf.Variable(w3_np)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b1", dtype=np.float32)))
    b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b2", dtype=np.float32)))
    # b3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b3", dtype=np.float32)))
    b3_np = np.loadtxt(load_net_dir + "b3", dtype=np.float32)
    b3_np = b3_np[np.newaxis]
    b3 = tf.Variable(tf.constant(b3_np))
    return w1, w2, w3, b1, b2, b3

# review 虽然叫n_hidden，但是包括了输入层和输出层的维度
input_dim = 3
n_hidden = [input_dim, 16, 32, 1]
net_dir = "../../net/ex3/"
data_dir = "../../data/point3/"

def random_parameter(training_dimension):
    # question 这里要随着input_dim的变化而变化
    #  还有一个问题，当训练次数增大和学习率降低的时候，会导致w10的值非常过拟合，而w11和w12的值全是0的情况
    if training_dimension[0]:
        # review tf.truncated_normal 截断的产生正态分布函数，正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
        w10 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    else:
        # question 如果不能训练设置为全是0的初始参数
        w10 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[0])

    if training_dimension[1]:
        w11 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    else:
        w11 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[1])

    if training_dimension[1]:
        w12 = tf.Variable(tf.truncated_normal((1, n_hidden[1]), 0.1))
    else:
        w12 = tf.Variable(tf.zeros((1, n_hidden[1])), trainable=training_dimension[1])

    w1 = tf.concat((w10, w11, w12), axis=0)
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


def train_barrier(training_dimension, load=False, load_net_dir=net_dir):
    if load:
        w1, w2, w3, b1, b2, b3 = load_parameter(training_dimension, load_net_dir)
    else:
        w1, w2, w3, b1, b2, b3 = random_parameter(training_dimension)
    # w1, w2, w3, b1, b2, b3 = random_parameter(training_dimension)

    # TODO trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
    # 不变式区间里面的点是unlabeled的
    invariant_data = get_invariant_data()
    non_invariant_data = get_non_invariant_data()

    # define the network
    # review tf的占位符，在里面None表示行不定，类似于reshape里面的-1
    x_invariant = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_invariant")
    layer1_labeled = tf.nn.relu(tf.matmul(x_invariant, w1) + b1, name="layer1_labeled")
    layer2_labeled = tf.nn.relu(tf.matmul(layer1_labeled, w2) + b2, name="layer2_labeled")
    output_invariant = tf.matmul(layer2_labeled, w3) + b3

    # review 这里修改为我定义的损失函数，这里应该是大于0产生损失
    loss0 = tf.reduce_sum(tf.abs(tf.maximum(0.0, output_invariant+0.5)))

    # review 对于non-invariant的数据而言，小于0产生损失
    x_non_invariant = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_non_invariant")
    layer1_unlabeled = tf.nn.relu(tf.matmul(x_non_invariant, w1) + b1, name="layer1_unlabeled")
    layer2_unlabeled = tf.nn.relu(tf.matmul(layer1_unlabeled, w2) + b2, name="layer2_unlabeled")
    output_non_invariant = tf.matmul(layer2_unlabeled, w3) + b3

    loss1 = tf.reduce_sum(tf.abs(tf.maximum(-(output_non_invariant-0.5), 0.0)))

    alpha = 1
    loss = loss0 + alpha * loss1

    # review 这里指示学习率随训练次数下降
    epoch_num = 300
    batch_size1 = 1280
    batch_size2 = 1280
    stop_flag = 0
    current_epoch = tf.Variable(0)
    # learning_rate = tf.train.exponential_decay(
    #     0.01, current_epoch, decay_steps=epoch_num/10, decay_rate=0.9, staircase=True)
    learning_rate = 0.01
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    # review batch_num的数量是一致的，batch_num设置为大的
    for ep in range(0, epoch_num):
        current_epoch = ep
        batch_num1 = int(invariant_data.shape[0] / batch_size1)
        batch_num2 = int(non_invariant_data.shape[0] / batch_size2)
        if batch_num1 > batch_num2:
            batch_num = batch_num1
        else:
            batch_num = batch_num2

        # review 通过这种方式让两种类别的data数量一样多
        for i in range(0, batch_num):
            start_batch1 = i % batch_num1
            batch_data1 = invariant_data[batch_size1 * start_batch1:batch_size1 * (start_batch1 + 1), :]

            start_batch2 = i % batch_num2
            batch_data2 = non_invariant_data[batch_size2 * start_batch2:batch_size2 * (start_batch2 + 1), :]

            if i % 1 == 0:
                epoch_train_loss = sess.run(loss, feed_dict={x_invariant: invariant_data,
                                                             x_non_invariant: non_invariant_data})
                epoch_loss0 = sess.run(loss0, feed_dict={x_invariant: invariant_data})
                epoch_loss1 = sess.run(loss1, feed_dict={x_non_invariant: non_invariant_data})

                print("epoch: %d/%d, batch: %d/%d, " %
                      (ep, epoch_num-1, i, batch_num-1), end="")
                print("mixed loss: %f, loss0: %f, loss1: %f" % (epoch_train_loss, epoch_loss0, epoch_loss1))

                if epoch_train_loss == 0:
                    stop_flag = 1
                    break
            sess.run(train_step, feed_dict={x_invariant: batch_data1, x_non_invariant: batch_data2})
        if stop_flag == 1:
            break

    # 跳出所有epoch循环后再run一次loss，看训练后的loss是否等于0，等于0返回true，否则返回false
    epoch_train_loss = sess.run(loss, feed_dict={x_invariant: invariant_data,
                                                 x_non_invariant: non_invariant_data})
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
    training_flag = [False] * input_dim
    # review 还是要加载之前的训练结果
    for d in range(0, input_dim):
        # 这里对多个input_dim逐层展开
        training_flag[d] = True
        # train_barrier返回布尔值
        # load参数默认是False，但是在进行一轮迭代后load就被设置为True了
        r = train_barrier(training_dimension=training_flag, load=load, load_net_dir=load_net_dir)
        if r:
            break
        load = True

    with open(net_dir + "training_flag", "w") as f:
        for i in range(0, len(training_flag)):
            f.write(str(int(training_flag[i])) + "\n")


# trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
def get_invariant_data():
    # 加载的是trace和unsafe里面的点
    train_data = np.loadtxt(data_dir + "invariant.txt", dtype=np.float32)
    np.random.shuffle(train_data)

    return train_data


# 不变式区间里面的点是unlabeled的
def get_non_invariant_data():
    train = np.loadtxt(data_dir + "non-invariant.txt", dtype=np.float32)
    np.random.shuffle(train)

    return train


abstract_network_barrier(load_net_dir=net_dir)
