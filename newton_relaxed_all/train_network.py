# review 在这个文件对网络进行训练，理论来说训练的代码和原来没啥差别，还是用cross-entropy损失函数，或者可以试试新看的论文的损失函数
#  理论上来说这种比较弱的训练条件应该能够正确分类，如果不能正确分类先尝试更换损失函数或者增加训练时间(更换网络的参数值和层数)
#  所以将取反例的部分删去，现在的算法反例应该由验证部分提供
import numpy as np
import tensorflow as tf
import math
import os

epsilon = 0.00001
input_dim = 3
# TODO 在这里更换网络结构，刚开始将网络结构做简单一点，用一个两层的网络
n_hidden = [input_dim, 20, 8, 1]
# TODO 现在网络的命名由网络的结构以及损失函数来决定
net_dir = "./net/two_layers_entropy/"
data_dir = "./points/"


# review 可以采用load_parameter然后再训练的方式
# TODO 这里面加载的网络参数随着网络结构的变换而变换
def load_parameter(load_net_dir=net_dir):
    # review np.reshape里面新的shape里面可以有-1，-1的意思是根据其他维度的值来推断这个维度的值
    w1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w1", dtype=np.float32)))
    w2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w2", dtype=np.float32)))
    # w3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w3", dtype=np.float32)))
    # w4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w4", dtype=np.float32)))
    # # # review 修改后就不需要单独处理w3和b3了
    w3_np = np.loadtxt(load_net_dir + "w3", dtype=np.float32)
    w3_np = w3_np[:, np.newaxis]
    w3 = tf.Variable(w3_np)
    # w3_np = np.loadtxt(load_net_dir + "w3", dtype=np.float32)
    # w3_np = w3_np[:, np.newaxis]
    # w3 = tf.Variable(w3_np)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b1", dtype=np.float32)))
    b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b2", dtype=np.float32)))
    # b3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b3", dtype=np.float32)))
    # b4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b4", dtype=np.float32)))
    b3_np = np.loadtxt(load_net_dir + "b3", dtype=np.float32)
    b3_np = b3_np[np.newaxis]
    b3 = tf.Variable(tf.constant(b3_np))
    # b3_np = np.loadtxt(load_net_dir + "b3", dtype=np.float32)
    # b3_np = b3_np[np.newaxis]
    # b3 = tf.Variable(tf.constant(b3_np))
    # return w1, w2, w3, b1, b2, b3, w4, b4, w5, b5
    return w1, w2, w3, b1, b2, b3

# review 随机初始化也加上
# TODO 这里也和网络结构相关
def random_parameter():
    w1 = tf.Variable(tf.truncated_normal((n_hidden[0], n_hidden[1]), 1))
    b1 = tf.Variable(tf.zeros([n_hidden[1]]))
    w2 = tf.Variable(tf.truncated_normal((n_hidden[1], n_hidden[2]), 1))
    b2 = tf.Variable(tf.zeros([n_hidden[2]]))
    w3 = tf.Variable(tf.truncated_normal((n_hidden[2], n_hidden[3]), 0.1))
    b3 = tf.Variable(tf.zeros([n_hidden[3]]))
    # w4 = tf.Variable(tf.truncated_normal((n_hidden[3], n_hidden[4]), 0.1))
    # b4 = tf.Variable(tf.zeros([n_hidden[4]]))
    # w5 = tf.Variable(tf.truncated_normal((n_hidden[4], n_hidden[5]), 1))
    # b5 = tf.Variable(tf.zeros([n_hidden[5]]))
    # return w1, w2, w3, b1, b2, b3, w4, b4, w5, b5
    return w1, w2, w3, b1, b2, b3

def train_barrier():
    # review 这里参数的顺序一定要和load_parameter的顺序相同，参数先是把最基础的w和b的123加载，然后轮着加载新增参数wb4, wb5等等

    load = False
    if load:
        # w1, w2, w3, b1, b2, b3, w4, b4, w5, b5 = load_parameter()
        w1, w2, w3, b1, b2, b3 = load_parameter()
    else:
        # w1, w2, w3, b1, b2, b3, w4, b4, w5, b5 = random_parameter()
        w1, w2, w3, b1, b2, b3 = random_parameter()

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
    # layer3_labeled = tf.nn.relu(tf.matmul(layer2_labeled, w3) + b3, name="layer3_labeled")
    # layer4_labeled = tf.nn.relu(tf.matmul(layer3_labeled, w4) + b4, name="layer4_labeled")
    output_invariant = tf.nn.sigmoid(tf.matmul(layer2_labeled, w3) + b3, name="output_invariant")

    NEAR_0 = 1e-5
    # review 如果是改成大间隔的存在loss无法下降的情况，可以尝试一下0是大损失，1是小损失的情况
    # loss0 = tf.reduce_sum(tf.abs(tf.math.pow(1 + tf.abs(tf.maximum(0.0, output_invariant)), pow_num) - 1))
    # loss0= tf.reduce_sum(tf.math.pow(1 + tf.abs(tf.maximum(0.0, output_invariant)), pow_num) - 1) + tf.reduce_sum(tf.math.pow(1 + tf.abs(tf.maximum(0.0, output_invariant + 0.5)), pow_soft_num) - 1)
    loss0 = tf.reduce_sum(tf.abs(-tf.log(1-output_invariant+NEAR_0)))
    # loss1 = tf.reduce_sum(tf)
    # loss0 = tf.reduce_sum(tf.abs(tf.maximum(0.0, output_invariant)))
    # loss2 = tf.reduce_sum(tf.abs(tf.maximum(-tf.log((tf.abs(output_invariant) + 0.01)), 0)))
    # loss0 = tf.reduce_sum(tf.abs(tf.maximum(0.0, 1.0+output_invariant)))
    # review 修改loss2和loss3，让他们在输出值很小0.0001的时候才产生损失，否则不产生损失
    # loss2 = tf.reduce_sum(tf.maximum(0.0001-tf.abs(output_invariant), 0) * 100000)

    # review 对于non-invariant的数据而言，小于0产生损失
    x_non_invariant = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_non_invariant")
    layer1_unlabeled = tf.nn.relu(tf.matmul(x_non_invariant, w1) + b1, name="layer1_unlabeled")
    layer2_unlabeled = tf.nn.relu(tf.matmul(layer1_unlabeled, w2) + b2, name="layer2_unlabeled")
    # TODO 修改层数的时候要修改这里
    # layer3_unlabeled = tf.nn.relu(tf.matmul(layer2_unlabeled, w3) + b3, name="layer3_unlabeled")
    # layer4_unlabeled = tf.nn.relu(tf.matmul(layer3_unlabeled, w4) + b4, name="layer4_unlabeled")
    output_non_invariant = tf.nn.sigmoid(tf.matmul(layer2_unlabeled, w3) + b3, name="output_non_invariant")

    # loss1 = tf.reduce_sum(tf.abs(tf.maximum(0.0, -output_non_invariant)))
    # loss1 = tf.reduce_sum(tf.abs(tf.math.pow(1 + tf.abs(tf.maximum(0.0, -output_non_invariant)), pow_num) - 1))
    loss1 = tf.reduce_sum(tf.abs(-tf.log(output_non_invariant+NEAR_0)))
    # loss1 = tf.reduce_sum(tf.math.pow(1 + tf.abs(tf.maximum(0.0, -output_non_invariant)), pow_num) - 1) + tf.reduce_sum(tf.math.pow(1 + tf.abs(tf.maximum(0.0, -output_non_invariant+0.5)), pow_soft_num) - 1)
    # loss3 = tf.reduce_sum(tf.abs(tf.maximum(-tf.log((tf.abs(output_non_invariant) + 0.01)), 0)))
    # loss3 = tf.reduce_sum(tf.maximum(0.0001 - tf.abs(output_non_invariant), 0) * 100000)

    alpha = 1
    beta = 1
    # loss4 = loss0 + alpha * loss1
    # loss = loss0 + loss1 + (loss2 + loss3)
    loss = alpha * loss0 + beta * loss1

    # review 这里指示学习率随训练次数下降，因为Adam优化器本身自带的就是会随训练次数下降，所以不需要自己实现
    epoch_num = 100
    batch_size1 = 2560
    batch_size2 = 2560
    stop_flag = 0
    current_epoch = tf.Variable(0)
    learning_rate = 0.001
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
                # epoch_loss2 = sess.run(loss2, feed_dict={x_invariant: invariant_data})
                # epoch_loss3 = sess.run(loss3, feed_dict={x_non_invariant: non_invariant_data})
                # epoch_loss4 = sess.run(loss4, feed_dict={x_invariant: invariant_data,
                #                                          x_non_invariant: non_invariant_data})

                print("epoch: %d/%d, batch: %d/%d, " %
                      (ep, epoch_num - 1, i, batch_num - 1), end="")
                # print("mixed loss: %f, loss0: %f, loss1: %f, loss2: %f, loss3: %f"
                #       % (epoch_train_loss, epoch_loss0, epoch_loss1, epoch_loss2, epoch_loss3))
                print("mixed loss: %f, loss0: %f, loss1: %f"
                      % (epoch_train_loss, epoch_loss0, epoch_loss1))

                if epoch_train_loss <= 50:
                    stop_flag = 1
                    break
                # if epoch_loss0 < 50.0 and epoch_loss1 < 4000.0 and (epoch_loss2+epoch_loss3) < 60000.0:
                #     stop_flag = 1
                #     break
            sess.run(train_step, feed_dict={x_invariant: batch_data1, x_non_invariant: batch_data2})
        if stop_flag == 1:
            break

    # 跳出所有epoch循环后再run一次loss，看训练后的loss是否等于0，等于0返回true，否则返回false
    epoch_loss4 = sess.run(loss, feed_dict={x_invariant: invariant_data,
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
    # w4 = sess.run(w4)
    # b4 = sess.run(b4)
    # w5 = sess.run(w5)
    # b5 = sess.run(b5)
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
    # np.savetxt(net_dir + "w4", w4)
    # np.savetxt(net_dir + "b4", b4)
    # np.savetxt(net_dir + "w5", w5)
    # np.savetxt(net_dir + "b5", b5)
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
#
# sample_counter_invariant()
# sample_counter_non_invariant()
r = abstract_network_barrier()
print(r)
# sample_counter_invariant()
# sample_counter_non_invariant()

