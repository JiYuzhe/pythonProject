# review 挑一些例子查看训练结果

import tensorflow as tf
import numpy as np

# review 这个方法是直接从net文件夹下面取出所有的w和b进行返回
def load_parameter(load_net_dir="../../net/ex7/"):
    # review np.loadtxt直接将加载的文本变成矩阵
    w1_txt = np.loadtxt(load_net_dir + "w1", dtype=np.float32)
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
    w4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "w4", dtype=np.float32)))
    # # # review 修改后就不需要单独处理w3和b3了
    w5_np = np.loadtxt(load_net_dir + "w5", dtype=np.float32)
    w5_np = w5_np[:, np.newaxis]
    w5 = tf.Variable(w5_np)
    # w3_np = np.loadtxt(load_net_dir + "w3", dtype=np.float32)
    # w3_np = w3_np[:, np.newaxis]
    # w3 = tf.Variable(w3_np)
    b1 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b1", dtype=np.float32)))
    b2 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b2", dtype=np.float32)))
    b3 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b3", dtype=np.float32)))
    b4 = tf.Variable(tf.constant(np.loadtxt(load_net_dir + "b4", dtype=np.float32)))
    b5_np = np.loadtxt(load_net_dir + "b5", dtype=np.float32)
    b5_np = b5_np[np.newaxis]
    b5 = tf.Variable(tf.constant(b5_np))
    # b3_np = np.loadtxt(load_net_dir + "b3", dtype=np.float32)
    # b3_np = b3_np[np.newaxis]
    # b3 = tf.Variable(tf.constant(b3_np))
    return w1, w2, w3, b1, b2, b3, w4, b4, w5, b5

input_dim = 3
n_hidden = [input_dim, 10, 15, 20, 3, 1]
net_dir = "../../net/ex7/"
data_dir = "../../data/point3/"


def train_barrier(load_net_dir=net_dir):

    w1, w2, w3, b1, b2, b3, w4, b4, w5, b5 = load_parameter(load_net_dir)

    # TODO trace和unsafe里面的点是label的，但是关于label的设置有点不太明白
    # labeled_data, label = get_all_data()
    unlabeled_data = get_all_data_unlabeled()

    # define the network
    # review tf的占位符，在里面None表示行不定，类似于reshape里面的-1
    x_labeled = tf.placeholder(tf.float32, [None, n_hidden[0]], name="x_labeled")

    layer1_labeled = tf.nn.relu(tf.matmul(x_labeled, w1) + b1, name="layer1_labeled")
    layer2_labeled = tf.nn.relu(tf.matmul(layer1_labeled, w2) + b2, name="layer2_labeled")
    layer3_labeled = tf.nn.relu(tf.matmul(layer2_labeled, w3) + b3, name="layer3_labeled")
    layer4_labeled = tf.nn.relu(tf.matmul(layer3_labeled, w4) + b4, name="layer4_labeled")
    output_all = tf.matmul(layer4_labeled, w5) + b5

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    output_all = sess.run(output_all, feed_dict={x_labeled: unlabeled_data})
    # TODO 在这里将预测错的部分挑出来，把data和result拼在一行
    # review 在这里添加对反例属性的研究
    max_margin = 0
    x_greater_nx = 0
    nx_greater_x = 0
    # with open(data_dir + "result-non-invariant1.txt", "a+") as f:
    #     for index, result in enumerate(output_all):
    #         # review 如果是invariant的，取>0的部分；否则取小于等于0的部分
    #         if result <= 0:
    #             if abs(unlabeled_data[index][1] - unlabeled_data[index][2]) > max_margin:
    #                 max_margin = abs(unlabeled_data[index][1] - unlabeled_data[index][2])
    #             if unlabeled_data[index][1] > unlabeled_data[index][2]:
    #                 x_greater_nx += 1
    #             else:
    #                 nx_greater_x += 1
    #             f.write("{0} {1} {2}\n".format(unlabeled_data[index][0], unlabeled_data[index][1], unlabeled_data[index][2]))
    for index, result in enumerate(output_all):
        print(result)
        print("/n")

    sess.close()
    print("{0} {1} {2}".format(max_margin, x_greater_nx, nx_greater_x))


# review 这里对应的是2个output方式计算的数据加载
def get_all_data_labeled():
    # 加载的是trace和unsafe里面的点
    train_l1 = np.loadtxt(data_dir + "sample1.txt", dtype=np.float32)
    train_l2 = np.loadtxt(data_dir + "sample2.txt", dtype=np.float32)

    # trace里面的点的label[0, 1], unsafe的label是[1, 0]
    # question 为什么这样设置label暂时还不知道
    label_l1 = np.array([0, 1] * len(train_l1)).reshape(-1, 2)
    label_l2 = np.array([1, 0] * len(train_l2)).reshape(-1, 2)

    # 把l1和l2拼起来，data和label都竖着拼起来
    train = np.concatenate((train_l1, train_l2), axis=0)
    label = np.concatenate((label_l1, label_l2), axis=0)

    return train, label


def get_all_data_unlabeled():
    # 加载的是trace和unsafe里面的点
    train_l1 = np.loadtxt(data_dir + "sample1.txt", dtype=np.float32)
    return train_l1


train_barrier()
