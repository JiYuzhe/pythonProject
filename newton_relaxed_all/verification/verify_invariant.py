import numpy as np
# from verification.comput_output import milp_output_area, milp_change_outputs, milp_final_outputs
from comput_output import milp_output_area, milp_change_outputs, milp_final_outputs
net_dir = '../net/two_layers_entropy/'


# review 更改条件后的verify对前两个条件的判定代码不需要更改，对最后一个条件的判定代码需要更改到comput_output中
def verify_invariant():
    net_structure = np.loadtxt(net_dir + "structure", dtype=np.int)
    num_layers = net_structure[0]

    # review 在加载的过程中，因为和原来代码不同，现在的代码输出只有一个神经元，所以要改造最后一个w和b
    # TODO 当网络结构发生改变后这里也要改变
    w0 = np.loadtxt(net_dir + "w1")
    w1 = np.loadtxt(net_dir + "w2")
    # w2 = np.loadtxt(net_dir + "w3")
    # w3 = np.loadtxt(net_dir + "w4")
    # w4 = np.loadtxt(net_dir + "w5")
    # w4 = w4[:, np.newaxis]
    w2 = np.loadtxt(net_dir + "w3")
    w2 = w2[:, np.newaxis]
    b0 = np.loadtxt(net_dir + "b1")
    b1 = np.loadtxt(net_dir + "b2")
    # b2 = np.loadtxt(net_dir + "b3")
    # b3 = np.loadtxt(net_dir + "b4")
    # b4 = np.loadtxt(net_dir + "b5")
    # b4 = b4[np.newaxis]
    b2 = np.loadtxt(net_dir + "b3")
    b2 = b2[np.newaxis]
    W = [w0, w1, w2]
    b = [b0, b1, b2]
    # W = [w0, w1, w2, w3, w4]
    # b = [b0, b1, b2, b3, b4]

    # review 这里和取点的initial范围一样大，其他的不需要做任何改变
    initial_min = [1.0, 1.0, 0.5]
    initial_max = [9.0, 1.0, 4.5]
    # review 分三步进行验证，这是第一步验证初始区域的
    _, _, u = milp_output_area(initial_min, initial_max, W, b)
    if u > 0:
        print('Error net for initial set and P => I.')
        # return
    else:
        print("Safe verification of initial set, means P => I.")

    # review 对于第二个条件的验证应该和第一个条件差不多，都是不变的，只变换区间
    changed_min = [1.0, 1.0, 0.5]
    changed_max = [9.0, 4.5, 4.5]
    # review 进行第二步，I and G => I'
    ok, c1, c2 = milp_change_outputs(changed_min, changed_max, W, b)
    if ok > 0:
        print("Safe verification of the second condition, means I and G => I'.")
    else:
        print('Error net for the second condition')


    # review 对于第三个条件，范围应该和第二个条件的范围一样
    final_min = [1.0, 1.0, 0.5]
    final_max = [9.0, 4.5, 4.5]
    ok, c1, c2 = milp_final_outputs(final_min, final_max, W, b)
    if ok > 0:
        print("Safe verification of final condition, means I and not G => Q.")
    else:
        print('Error net for the final condition')

verify_invariant()
