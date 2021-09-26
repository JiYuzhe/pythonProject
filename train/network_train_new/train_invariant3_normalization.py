# review 使用one-take方法并采用一个输出神经元的最朴实的损失函数，同时增加一个对数的惩罚项，让网络的输出值尽量原理原点0

# review 之前的训练方法结果有点问题，使用老的tensorflow版本训练

import os
import numpy as np
import tensorflow as tf

# review 虽然叫n_hidden，但是包括了输入层和输出层的维度
input_dim = 3
n_hidden = [input_dim, 20, 30, 40, 8, 1]
net_dir = "../../net/ex5 /"
data_dir = "../../data/point3/"


def random_parameter():
    w1 = tf.Variable(tf.truncated_normal((n_hidden[0], n_hidden[1]), 0.1))
    b1 = tf.Variable(tf.zeros([n_hidden[1]]))
    w2 = tf.Variable(tf.truncated_normal((n_hidden[1], n_hidden[2]), 0.1))
    b2 = tf.Variable(tf.zeros([n_hidden[2]]))
    w3 = tf.Variable(tf.truncated_normal((n_hidden[2], n_hidden[3]), 0.1))
    b3 = tf.Variable(tf.zeros([n_hidden[3]]))
    w4 = tf.Variable(tf.truncated_normal((n_hidden[3], n_hidden[4]), 0.1))
    b4 = tf.Variable(tf.zeros([n_hidden[4]]))
    w5 = tf.Variable(tf.truncated_normal((n_hidden[4], n_hidden[5]), 0.1))
    b5 = tf.Variable(tf.zeros([n_hidden[5]]))
    return w1, w2, w3, b1, b2, b3, w4, w5, b4, b5


def train_barrier():
    w1, w2, w3, b1, b2, b3, w4, w5, b4, b5 = random_parameter()

    # TODO trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
    # 不变式区间里面的点是unlabeled的
    invariant_data = get_invariant_data()
    non_invariant_data = get_non_invariant_data()

    # define the network
    # review tf的占位符，在里面None表示行不定，类似于reshape里面的-1
    x_invariant = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_invariant")
    layer1_labeled = tf.nn.relu(tf.matmul(x_invariant, w1) + b1, name="layer1_labeled")
    layer2_labeled = tf.nn.relu(tf.matmul(layer1_labeled, w2) + b2, name="layer2_labeled")
    # TODO 修改层数的时候要修改这里
    layer3_labeled = tf.nn.relu(tf.matmul(layer2_labeled, w3) + b3, name="layer3_labeled")
    layer4_labeled = tf.nn.relu(tf.matmul(layer3_labeled, w4) + b4, name="layer4_labeled")
    output_invariant = tf.matmul(layer4_labeled, w5) + b5

    # review 这里修改为我定义的损失函数，这里应该是大于0产生损失
    loss0 = tf.reduce_sum(tf.abs(tf.maximum(0.0, output_invariant)))
    loss2 = tf.reduce_sum(tf.abs(tf.maximum(-tf.log((tf.abs(output_invariant) + 0.01)), 0)))
    # loss0 = tf.reduce_sum(tf.abs(tf.maximum(0.0, 1.0+output_invariant)))

    # review 对于non-invariant的数据而言，小于0产生损失
    x_non_invariant = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_non_invariant")
    layer1_unlabeled = tf.nn.relu(tf.matmul(x_non_invariant, w1) + b1, name="layer1_unlabeled")
    layer2_unlabeled = tf.nn.relu(tf.matmul(layer1_unlabeled, w2) + b2, name="layer2_unlabeled")
    # TODO 修改层数的时候要修改这里
    layer3_unlabeled = tf.nn.relu(tf.matmul(layer2_unlabeled, w3) + b3, name="layer3_unlabeled")
    layer4_unlabeled = tf.nn.relu(tf.matmul(layer3_unlabeled, w4) + b4, name="layer4_unlabeled")
    output_non_invariant = tf.matmul(layer4_unlabeled, w5) + b5

    loss1 = tf.reduce_sum(tf.abs(tf.maximum(0.0, -output_non_invariant)))
    loss3 = tf.reduce_sum(tf.abs(tf.maximum(-tf.log((tf.abs(output_non_invariant) + 0.01)), 0)))

    alpha = 1
    beta = 0.05
    loss4 = loss0 + alpha * loss1
    loss = 2*loss0 + alpha * loss1 + beta * (loss2 + loss3)
    # loss = loss0 + alpha * loss1

    # review 这里指示学习率随训练次数下降
    epoch_num = 300
    batch_size1 = 2560
    batch_size2 = 2560
    stop_flag = 0
    current_epoch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001, current_epoch, decay_steps=epoch_num/20, decay_rate=0.8, staircase=True)
    # learning_rate = 0.03
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
                epoch_loss2 = sess.run(loss2, feed_dict={x_invariant: invariant_data})
                epoch_loss3 = sess.run(loss3, feed_dict={x_non_invariant: non_invariant_data})
                epoch_loss4 = sess.run(loss4, feed_dict={x_invariant: invariant_data,
                                                         x_non_invariant: non_invariant_data})

                print("epoch: %d/%d, batch: %d/%d, " %
                      (ep, epoch_num - 1, i, batch_num - 1), end="")
                print("mixed loss: %f, loss0: %f, loss1: %f, loss2: %f, loss3: %f"
                      % (epoch_train_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3))

                if epoch_loss4 == 0:
                    stop_flag = 1
                    break
                if epoch_loss0 < 50.0 and epoch_loss1 < 4000.0 and (epoch_loss2+epoch_loss3) < 60000.0:
                    stop_flag = 1
                    break
            sess.run(train_step, feed_dict={x_invariant: batch_data1, x_non_invariant: batch_data2})
        if stop_flag == 1:
            break

    # 跳出所有epoch循环后再run一次loss，看训练后的loss是否等于0，等于0返回true，否则返回false
    epoch_loss4 = sess.run(loss4, feed_dict={x_invariant: invariant_data,
                                                 x_non_invariant: non_invariant_data})
    if epoch_loss4 == 0:
        r = True
    else:
        r = False
    w1 = sess.run(w1)
    b1 = sess.run(b1)
    w2 = sess.run(w2)
    b2 = sess.run(b2)
    w3 = sess.run(w3)
    b3 = sess.run(b3)
    # TODO 修改层数的时候要改这里
    w4 = sess.run(w4)
    b4 = sess.run(b4)
    w5 = sess.run(w5)
    b5 = sess.run(b5)
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
    # TODO 修改层数的时候要修改这里
    np.savetxt(net_dir + "w4", w4)
    np.savetxt(net_dir + "b4", b4)
    np.savetxt(net_dir + "w5", w5)
    np.savetxt(net_dir + "b5", b5)
    with open(net_dir + "structure", "w") as f:
        f.write(str(len(n_hidden)) + "\n")
        for i in range(0, len(n_hidden)):
            f.write(str(n_hidden[i]) + "\n")

    return r


def abstract_network_barrier():
    r = train_barrier()
    return r


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


r = abstract_network_barrier()
print(r)
