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
# pre-condition x=1, nx=n/2, n∈[0, 10]
# post-condition abs(nx*nx - n) < 1E-5
# G abs(x-nx) > eps
# 顺序是n x nx


# review 首先是初始集合取点
import random
import os
import math

data_dir = '../../data/point3/'
# review 系统变量的数量维度，因为G的条件和传入的number的值有关，所以
input_dim = 3
initial_min = [3.0, 1.0, 1.5]
initial_max = [5.0, 1.0, 2.25]
post_min = [3.0, 1.5, 1.0]
post_max = [5.0, 2.25, 2.25]

# review 这里面传过来的值都是满足循环判别条件G的
def simulation(f, input_para):
    number = input_para[0]
    x = input_para[1]
    nx = input_para[2]
    epsilon = 0.0001
    # TODO 不知道这里大概会运行多少步
    while math.fabs(nx - x) > epsilon:
        # review 说明满足P和G, I应该小于0
        # review I如果应该小于等于0的都标为0类
        f.write("{0} {1} {2} 0 {0} {1} {2}\n".format(number, x, nx))
        x = nx
        nx = (x + number / x) / 2

    # review 从程序中退出来了，说明不满足G但是还是应该满足I和P,还是0类
    f.write("{0} {1} {2} 0 {0} {1} {2}\n".format(number, x, nx))


# review 这个方法生成label为0的数据集
def create_init_trace_data():
    # create trace data, grid and random
    with open(data_dir + "sample2.txt", "a+") as f:
        pieces = [10000] * input_dim
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
            simulation(f, x)

        # TODO 随机取点
        random_points_num = 4000
        for i in range(0, random_points_num):
            x[0] = random.uniform(initial_min[0], initial_max[0])
            x[1] = 1
            x[2] = x[0] / 2
            simulation(f, x)


# review 这个方法生成label为1的数据集
#  在非G的区间内网格取点+随机取点，将label设为1
#  G: abs(x-nx) > eps, 非G：abs(x-nx) <= eps
def create_non_invariant_data():
    with open(data_dir + "sample2.txt", "a+") as f:
        pieces = [80] * input_dim
        pieces[1] = 120
        pieces[2] = 10
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            # TODO 在这里加入max和min相等也就是只是一个值的判断
            if post_max[i_dim] - post_min[i_dim] != 0:
                step_size[i_dim] = (post_max[i_dim] - post_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        for i in range(0, pieces[0]+1):
            x[0] = post_min[0] + step_size[0] * i
            for j in range(0, pieces[1]+1):
                x[1] = post_min[1] + step_size[1] * j
                post_min[2] = x[1] - 0.0001
                post_max[2] = x[1] + 0.0001
                step_size[2] = (post_max[2]-post_min[2]) / pieces[i_dim]
                for k in range(0, pieces[2]+1):
                    x[2] = post_min[2] + step_size[2] * k
                    # review 这里再对点集进行判断，需要不满足Q
                    if math.fabs(x[2]*x[2] - x[0]) > 0.0001:
                        # print(str(math.fabs(x[2]*x[2] - x[0])))
                        f.write("{0} {1} {2} 1 {0} {1} {2}\n".format(x[0], x[1], x[2]))
                    else:
                        print(str(x[0]) + " " + str(x[2]))



def generate_data():
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    create_non_invariant_data()
    create_init_trace_data()


generate_data()
