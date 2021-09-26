# review 现在将一个例子的所有步骤全都放到一个文件夹下面，便于文件传输
#  这个文件是用来取点的，思路是首先尝试从最开始的初始条件来取点，如果不行再变成整个不变式区间取点

# review 牛顿迭代法求平方根，计算两次之间的差距，
# review 里面有三个参数，所以虽然是一个简单的求平方根，还是需要三维的变量
#  double sqrt_newton(double n) {
#   const double eps = 1E-15;
#   double x = 1;
#   double nx = (x + n / x) / 2
#   while (abs(x - nx) > eps) {
#     x = nx;
#     nx = (x + n / x) / 2;
#   }
#   return nx;
# }
# pre-condition x=1, nx=n/2, n∈[1, 9]
# post-condition 这里将post-condition进行放松，变成nx∈[1-epsilon, 3+epsilon]之间
# G abs(x-nx) > eps
# 顺序是n x nx


# review 首先是初始集合取点
import random
import os
import math
import numpy as np

data_dir = './points/'
# review 系统变量的数量维度，因为G的条件和传入的number的值有关，所以
input_dim = 3
initial_min = [1.0, 1.0, 0.5]
initial_max = [9.0, 1.0, 4.5]
epsilon = 0.00001  # 1e-5


# review 这里面传过来的值都是满足循环判别条件G的
def simulation(f, input_para):
    number = input_para[0]
    x = input_para[1]
    nx = input_para[2]
    # TODO 不知道这里大概会运行多少步
    while math.fabs(nx - x) > epsilon:
        # review 说明满足P和G, I应该小于0
        # review I如果应该小于等于0的都标为0类
        f.write("{0} {1} {2}\n".format(number, x, nx))
        x = nx
        nx = (x + number / x) / 2

    # review 从程序中退出来了，说明不满足G但是还是应该满足I和P,还是0类
    f.write("{0} {1} {2}\n".format(number, x, nx))
    # return difference


# review 这个方法生成label为0的数据集
def create_init_trace_data():
    # create trace data, grid and random
    with open(data_dir + "invariant.txt", "a+") as f:
        pieces = [100000] * input_dim
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            # TODO 在这里加入max和min相等也就是只是一个值的判断
            if initial_max[i_dim] - initial_min[i_dim] != 0:
                step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        # review 网格取点
        for i in range(pieces[0] + 1):
            # TODO 这个例子是一个比较简单的一维例子，一个循环就行
            x[0] = initial_min[0] + i * step_size[0]
            x[1] = 1.0
            x[2] = x[0] / 2
            if math.fabs(x[1]-x[2]) > epsilon:
                simulation(f, x)
            # max_difference = np.maximum(difference, max_difference)

        # TODO 随机取点
        random_points_num = 20000
        for i in range(0, random_points_num):
            x[0] = random.uniform(initial_min[0], initial_max[0])
            x[1] = 1
            x[2] = x[0] / 2
            # review 如果x1和x2本身就很接近且不能达到后置条件，需要筛选出去
            if math.fabs(x[1]-x[2]) > epsilon:
                simulation(f, x)


# review 这个方法生成label为1的数据集
#  在非G的区间内网格取点+随机取点，将label设为1
#  G: abs(x-nx) > eps, 非G：abs(x-nx) <= eps, 非Post指nx∈[3+epsilon, 4.5]
#  这里面的点要非G且非Post
post_min = [1.0, 3.0, 3.0+epsilon]
post_max = [9.0, 4.5, 4.5]


def create_non_invariant_data():
    with open(data_dir + "non-invariant.txt", "a+") as f:
        pieces = [200] * input_dim
        pieces[2] = 120  # review 现在nx是和post-condition直接相关的类似于自变量，x是因变量，所以pieces[2]作为主导
        pieces[1] = 14
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            # TODO 在这里加入max和min相等也就是只是一个值的判断
            if post_max[i_dim] - post_min[i_dim] != 0:
                step_size[i_dim] = (post_max[i_dim] - post_min[i_dim]) / pieces[i_dim]

        x = [0.] * input_dim
        for i in range(0, pieces[0] + 1):
            x[0] = post_min[0] + step_size[0] * i
            for j in range(0, pieces[2] + 1):
                x[2] = post_min[2] + step_size[2] * j
                post_min[1] = x[2] - epsilon
                post_max[1] = x[2] + epsilon
                step_size[1] = (post_max[1] - post_min[1]) / pieces[1]
                for k in range(0, pieces[1] + 1):
                    x[1] = post_min[1] + step_size[1] * k
                    # review 这里再对点集进行判断，需要不满足Q，因为这次的Q比较简单已经在采点的时候判断过了所以这里不用再判断了
                    f.write("{0} {1} {2}\n".format(x[0], x[1], x[2]))

        # review 这里是新增的部分，对non-invariant的数据也进行random取值，只需要random第0维和第2维即可，然后算出第一维的范围再random第一维
        random_points_num = 100000
        for i in range(0, random_points_num):
            x[0] = random.uniform(post_min[0], post_max[0])
            x[2] = random.uniform(post_min[2], post_max[2])
            x[1] = random.uniform(x[2]-epsilon, x[2]+epsilon)
            f.write("{0} {1} {2}\n".format(x[0], x[1], x[2]))


# review 理论上这里面的点也应该在I里面，但是先不用其进行训练
# def generate_square_data():
#     pieces = [200] * input_dim
#     x = [0.] * input_dim
#     step_size = [0.] * input_dim
#     step_size[0] = (post_max[0] - post_min[0]) / pieces[0]
#     pieces[1] = 10
#     pieces[2] = 10
#     with open(data_dir + "invariant.txt", "a+") as f:
#         for i in range(0, pieces[0] + 1):
#             x[0] = post_min[0] + step_size[0] * i
#             root = math.sqrt(x[0])
#             x0_upper = x[0] + epsilon
#             x0_lower = x[0] - epsilon
#             root_upper = math.sqrt(x0_upper)
#             root_lower = math.sqrt(x0_lower)
#             # review step_size[1]是随着第0维的数据时刻变化的，所以在这里赋值
#             #  因为最后的条件是判断x[2]和x[0]的关系，所以这里先运行第三个维度
#             step_size[2] = (root_upper - root_lower) / pieces[2]
#             for j in range(0, pieces[2] + 1):
#                 x[2] = root_lower + step_size[2] * j
#                 x1_upper = x[2] + epsilon
#                 x1_lower = x[2] - epsilon
#                 step_size[1] = (x1_upper - x1_lower) / pieces[1]
#                 for k in range(0, pieces[1] + 1):
#                     x[1] = x1_lower + step_size[1] * k
#                     f.write("{0} {1} {2}\n".format(x[0], x[1], x[2]))


# # review 因为很多时候第一个条件没法通过，所以在第一个条件里面多选一些点
# def generate_init_data():
#     start_min = 3.0
#     start_max = 5.0
#     points = 10000
#     step = (start_max - start_min) / points
#     with open(data_dir + "variant.txt", "a+") as f:
#         x = [0.] * input_dim
#         for i in range(0, points+1):
#             x[0] = start_min + step * i
#             x[1] = 1
#             x[2]

# review 修改：即使不满足前置条件，也是在I里面的轨迹点
def generate_semi_trace_data():
    # create trace data, grid and random
    max_difference = 0
    with open(data_dir + "invariant1.txt", "a+") as f:
        pieces = [100] * input_dim
        pieces[1] = 50
        pieces[2] = 50
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            # TODO 在这里加入max和min相等也就是只是一个值的判断
            if post_max[i_dim] - post_min[i_dim] != 0:
                step_size[i_dim] = (post_max[i_dim] - post_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        # review 网格取点
        for i in range(pieces[0] + 1):
            # TODO 这个例子是一个比较简单的一维例子，一个循环就行
            x[0] = post_min[0] + i * step_size[0]
            for j in range(pieces[1] + 1):
                x[1] = post_min[1] + j * step_size[1]
                for k in range(pieces[2] + 1):
                    x[2] = post_min[2] + k * step_size[2]
                    if math.fabs(x[2] - x[1]) <= epsilon < math.fabs(x[2] * x[2] - x[0]):
                        continue
                    simulation(f, x)
                    # max_difference = max(max_difference, difference)

        # TODO 随机取点
        random_points_num = 30000
        for i in range(0, random_points_num):
            x[0] = random.uniform(post_min[0], post_max[0])
            x[1] = random.uniform(post_min[1], post_max[1])
            x[2] = random.uniform(post_min[2], post_max[2])
            if math.fabs(x[2] - x[1]) <= epsilon < math.fabs(x[2] * x[2] - x[0]):
                continue
            simulation(f, x)
            # max_difference = max(max_difference, difference)
    #
    # print(max_difference)


def generate_data():
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    create_non_invariant_data()
    # create_init_trace_data()


generate_data()
