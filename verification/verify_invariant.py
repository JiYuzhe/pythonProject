import numpy as np
# from verification.comput_output import milp_output_area, milp_change_outputs, milp_final_outputs
from comput_output import milp_output_area, milp_change_outputs, milp_final_outputs
net_dir = '../net/ex7/'



def verify_invariant():
    net_structure = np.loadtxt(net_dir + "structure", dtype=np.int)
    num_layers = net_structure[0]

    # review 在加载的过程中，因为和原来代码不同，现在的代码输出只有一个神经元，所以要改造最后一个w和b
    w0 = np.loadtxt(net_dir + "w1")
    w1 = np.loadtxt(net_dir + "w2")
    w2 = np.loadtxt(net_dir + "w3")
    w3 = np.loadtxt(net_dir + "w4")
    w4 = np.loadtxt(net_dir + "w5")
    w4 = w4[:, np.newaxis]
    b0 = np.loadtxt(net_dir + "b1")
    b1 = np.loadtxt(net_dir + "b2")
    b2 = np.loadtxt(net_dir + "b3")
    b3 = np.loadtxt(net_dir + "b4")
    b4 = np.loadtxt(net_dir + "b5")
    b4 = b4[np.newaxis]
    W = [w0, w1, w2, w3, w4]
    b = [b0, b1, b2, b3, b4]

    initial_min = [3.0, 1.0, 1.5]
    initial_max = [5.0, 1.0, 2.5]
    # review 分三步进行验证，这是第一步验证初始区域的
    _, _, u = milp_output_area(initial_min, initial_max, W, b)
    if u > 0:
        print('Error net for initial set and P => I.')
        # return
    else:
        print("Safe verification of initial set, means P => I.")

    changed_min = [3.0, 1.5, 1.5]
    changed_max = [5.0, 2.2, 2.2]
    # review 进行第二步，I and G => I'
    ok, c1, c2 = milp_change_outputs(changed_min, changed_max, W, b)
    if ok > 0:
        print("Safe verification of the second condition, means I and G => I'.")
    else:
        print('Error net for the second condition')

    # review 接下来进行第三步，第三个条件的判断理论上是无穷区域的，但是我觉得我们所训练的区域就设置成大于0的，所以这里的区域也设置成大于0的
    final_min = [3.0, 1.5, 1.5]
    final_max = [3.2, 1.8, 1.8]
    ok, c1, c2 = milp_final_outputs(final_min, final_max, W, b)
    if ok > 0:
        print("Safe verification of final condition, means I and not G => Q.")
    else:
        print('Error net for the final condition')

verify_invariant()
