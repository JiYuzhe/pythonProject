# 程序大意如下：
# while x < 10 and x > 1 do
#   if x < 3 and x > 1 then
#       x = x(5-x)
#   else
#       x++
# pre-condition X ~ (1, 10)
# post-condition X ~ (0, 21)
# g-condition X ~ (1, 10)

# review 首先是初始集合取点
import random
import os

data_dir = '../../data/'
# review 系统变量的数量维度
input_dim = 1
# review 这里列出各个区间的上下限
# TODO 针对开闭区间的处理，这里是开区间
initial_min = [1.0]
initial_max = [10.0]
pre_min = [1.0]
pre_max = [10.0]
post_min = [0.0]
post_max = [21.0]
g_min = [1.0]
g_max = [10.0]


def simulation(f, x):
    x = x[0]
    while 10 > x > 1:
        # review 说明满足P和G, I应该小于0
        # review I如果应该小于等于0的都标为0类
        # review 对于这个程序而言，x已经退化成了一个值，直接打印就行，不用i_dim了
        f.write(str(x) + " 0\n")

        if 1 < x < 3:
            x = x*(5-x)
        else:
            x = x+1

    # review 从程序中退出来了，说明不满足G但是还是应该满足I和P,还是0类
    f.write(str(x) + " 0\n")


# review 判断点是否非Q且非G的工具方法
def is_positive(x):
    # TODO 要判断这个点不满足G，且不满足Q
    in_g_flag, in_q_flag = True, True
    for i_dim in range(0, input_dim):
        if x[i_dim] < pre_min[i_dim] or x[i_dim] > pre_max[i_dim]:
            in_g_flag = False
        if x[i_dim] < post_min[i_dim] or x[i_dim] > post_max[i_dim]:
            in_q_flag = False
        if not in_q_flag and not in_g_flag:
            return True
    return False


# review 这个方法生成label为0的数据集
def create_init_trace_data():
    # create trace data, grid and random
    with open(data_dir + "points_1.txt", "a+") as f:
        pieces = [1000] * input_dim
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        # review 网格取点
        for i in range(pieces[0] + 1):
            # TODO 现在这里就一个维度，不需要二维取点，后面可能要更改
            x[0] = initial_min[0] + i * step_size[0]
            simulation(f, x)

        # review 随机取点
        random_points_num = 1000
        for i in range(0, random_points_num):
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(initial_min[i_dim], initial_max[i_dim])
            simulation(f, x)


# review 这个方法生成label为1的数据集
#  所取的数据是非G且非Q的，这些数据也必须是非I的
#  因为G和Q都是封闭范围内的，所以剩下的范围是无法网格取点的，所以所取的点尽量离G和Q稍微近一些
def create_non_invariant_data():
    # 写文件的方式是追加
    with open(data_dir + "points_1.txt", "a+") as f:
        # review 这个题中G是(0, 10)
        # review 设置外围轮廓的大小
        margin = 10
        pieces = [10000] * input_dim
        step_g_size = [0.] * input_dim
        step_q_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            margin_g_max = g_max[i_dim] + margin
            margin_g_min = g_min[i_dim] - margin
            margin_q_min = post_min[i_dim] - margin
            margin_q_max = post_max[i_dim] - margin
            step_g_size[i_dim] = (margin_g_max - margin_g_min) / pieces[i_dim]
            step_g_size[i_dim] = (margin_q_max - margin_q_min) / pieces[i_dim]
        x_g = [0.] * input_dim
        x_q = [0.] * input_dim
        x = [0.] * input_dim
        # review 统计满足条件的网格所取的点的数量用来平衡随机取点数量
        total_positive = 20000
        grid_count = 0
        for i in range(pieces[0] + 1):
            x_g[0] = g_min[0]-margin + i*step_g_size[0]
            x_q[0] = post_min[0]-margin + i*step_q_size[0]
            # 要判断这个点不满足G，且不满足Q, 对G和Q的margin所取的点进行相同的操作
            if is_positive(x_g):
                # review 如果都不在两个里面记录点
                for i_dim_write in range(0, input_dim):
                    f.write(str(x_g[i_dim_write]) + " ")
                f.write("1\n")
                grid_count += 1
            if is_positive(x_q):
                for i_dim_write in range(0, input_dim):
                    f.write(str(x_q[i_dim_write]) + " ")
                f.write("1\n")
                grid_count += 1
        # TODO 可能会根据实际网格采点的数量来调整margin
        print("There are {0:4d} points are sampled by the grid".format(grid_count))
        # review 接下来随机取点,随机的范围控制在(-10000, +10000)
        random_points_num = 10000
        random_min = -10000
        random_max = 10000
        while random_points_num > 0:
            for i_dim in range(0, input_dim):
                x[i_dim] = random.uniform(random_min, random_max)
                if is_positive(x):
                    for i_dim_write in range(0, input_dim):
                        f.write(str(x[i_dim_write]) + " ")
                    # review I如果应该小于等于0的都标为0类
                    f.write("1\n")
                    random_points_num -= 1


def generate_data():
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    create_non_invariant_data()
    create_init_trace_data()


generate_data()
