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
invariant_min = [3.0, 1.0, 1.5]
invariant_max = [5.0, 2.5, 2.5]
post_min = [3.0, 1.0, 1.5]
post_max = [5.0, 2.5, 2.5]


# review 这个方法生成label为0的数据集
def create_init_trace_data():
    # TODO 直接区域网格取点，取密一些，然后判断是不是non—invariant的，不是invariant的加入label为0的点集合中
    non_invariant_list = []
    with open(data_dir + "invariant.txt", "a+") as f:
        pieces = [100] * input_dim
        pieces[1] = 40
        pieces[2] = 50
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            step_size[i_dim] = (invariant_max[i_dim] - invariant_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        for i in range(0, pieces[0]+1):
            x[0] = invariant_min[0] + step_size[0] * i
            for j in range(0, pieces[1]+1):
                x[1] = invariant_min[1] + step_size[1] * j
                for k in range(0, pieces[2]+1):
                    x[2] = invariant_min[2] + step_size[2] * k
                    if not is_non_invariant(x):
                        f.write("{0} {1} {2}\n".format(x[0], x[1], x[2]))
                    else:
                        y = [x[0], x[1], x[2]]
                        non_invariant_list.append(y)
    return non_invariant_list


def is_non_invariant(x):
    if math.fabs(x[2] * x[2] - x[0]) > 0.0001 and math.fabs(x[2] - x[1]) < 0.0001:
        return True

    return False


# review 这个方法生成label为1的数据集
#  在非G的区间内网格取点+随机取点，将label设为1
#  G: abs(x-nx) > eps, 非G：abs(x-nx) <= eps
def create_non_invariant_data(non_invariant_list):
    with open(data_dir + "non-invariant.txt", "a+") as f:
        for element in non_invariant_list:
            print(element)
            f.write("{0} {1} {2}\n".format(element[0], element[1], element[2]))

        pieces = [200] * input_dim
        pieces[1] = 120
        pieces[2] = 20
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
                        f.write("{0} {1} {2}\n".format(x[0], x[1], x[2]))
                    else:
                        print(str(x[0]) + " " + str(x[2]))



def generate_data():
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    non_invariant_list = create_init_trace_data()
    create_non_invariant_data(non_invariant_list)
    create_init_trace_data()


generate_data()
