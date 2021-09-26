# review 尝试一个不变式为等式的程序，这个程序的值全是int值
# 程序大意如下：int main() {
#   // variable declarations
#   int n;
#   int x;
#   int y;
#   // pre-conditions
#   assume((n >= 0));
#   (x = n);
#   (y = 0);
#   // loop body
#   while ((x > 0)) {
#     {
#     (y  = (y + 1));
#     (x  = (x - 1));
#     }
#
#   }
#   // post-condition
# assert( (n == (x + y)) );
# }

# review 首先是初始集合取点
import random
import os

data_dir = '../../data/'
# review 系统变量的数量维度
input_dim = 3
# TODO 这里是闭区间(注意处理值一直不变是条线的情况) 顺序是n, x, y
initial_min = [0, 0, 0]
# review 对正负无穷的表示，暂时是用1e4表示
initial_max = [20, 20, 0]
pre_min = [0, 0, 0]
pre_max = [20, 20, 0]


# TODO 这里因为post-condition是一条无穷的线，所以不用区间表示，且g只对x有要求，也不用专门搞个数组


# review 这里面传过来的值都是满足循环判别条件G的
def simulation(f, input_para):
    n, x, y = input_para[0], input_para[1], input_para[2]
    # TODO 修改这里的取点逻辑，为了避免取点过多，离散执行多次后再取点
    oldest = x
    while x > 0:
        # review 说明满足P和G, I应该小于0
        # review I如果应该小于等于0的都标为0类
        f.write("{0} {1} {2} 0 {0} {1} {2}\n".format(n, x, y))
        x -= 1
        y += 1

    # review 从程序中退出来了，说明不满足G但是还是应该满足I和P,还是0类
    f.write("{0} {1} {2} 0 {0} {1} {2}\n".format(n, x, y))


# review 判断点是否非Q且非G的工具方法
def is_positive(x):
    # TODO 要判断这个点不满足G，且不满足Q
    in_g_flag, in_q_flag = True, True
    if x[1] <= 0:
        in_g_flag = False
    if x[1] + x[2] - x[0] != 0:
        in_q_flag = False
    return not in_g_flag and not in_q_flag


# review 这个方法生成label为0的数据集
def create_init_trace_data():
    # create trace data, grid and random
    with open(data_dir + "points_2.txt", "a+") as f:
        # review 因为如果x和n的初始值过大会导致在循环里面会取无数的点，所以这里面不敢取太大
        pieces = [20] * input_dim
        step_size = [0.] * input_dim
        for i_dim in range(0, input_dim):
            # TODO 在这里加入max和min相等也就是只是一个值的判断
            if initial_max[i_dim] - initial_min[i_dim] != 0:
                step_size[i_dim] = (initial_max[i_dim] - initial_min[i_dim]) / pieces[i_dim]
        x = [0.] * input_dim
        # review 网格取点
        for i in range(pieces[0] + 1):
            # TODO 这个例子是三个维度，需要三个循环，但是最后一个y是一个值，且前置条件要求x和n一定相同，所以一个循环就行
            x[0] = int(initial_min[0] + i * step_size[0])
            x[1] = x[0]
            x[2] = 0
            simulation(f, x)

        # TODO 随机取点，随机取点的时候也注意x和n的数量要相等才对，因为现在是给定的封闭区间，所以在P内取点不需要随机了
        # random_points_num = 1000
        # for i in range(0, random_points_num):
        #     x[0] = int(random.uniform(initial_min[0], initial_max[0]))
        #     x[1] = x[0]
        #     x[2] = int(0)  # 前置条件要求y为0
        #     simulation(f, x)


# review 这个方法生成label为1的数据集
#  在非G的区间内网格取点+随机取点，将label设为1
#  非G设为x属于[-10， 0]
def create_non_invariant_data():
    # 写文件的方式是追加
    with open(data_dir + "points_2.txt", "a+") as f:
        # review 这个问题不需要margin取点了，直接对x<=0的整数随机取点，其他两个变量的值无所谓，然后只要不满足post就行
        # TODO 还是修改一下取点的方式吧，多取一些n=x+y附近的点
        grid_count = 0
        for x in range(0, 21):
            current_x = [0] * input_dim
            for y in range(-20, 21):
                for n in range(-20, 21):
                    f.write("{0} {1} {2} 1 {0} {1} {2}\n".format(int(n), int(-x), int(y)))


def generate_data():
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    create_non_invariant_data()
    create_init_trace_data()


generate_data()
