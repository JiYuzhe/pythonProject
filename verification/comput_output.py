import numpy as np
import gurobipy as gp
from gurobipy import GRB
from verification.interval_number import IntervalNumber, interval_max, inf, sup

epsilon_Q = 0.01
epsilon_G = 0.001
# review 这个可以直接拿来用，用来判断第一个条件
def milp_output_area(x_min, x_max, W, b):
    ok = 0
    weight_size = len(W)
    input_size = len(W[0])
    intval_x = []
    for i in range(0, input_size):
        t = IntervalNumber(x_min[i], x_max[i])
        # review 将每个维度的下界和上界变成IntervalNumber对象并放到数组中
        intval_x.append(t)
    # review 这里面做了一对操作却又什么都没做,intval_x的维度是(1，input_size)
    intval_x = np.array(intval_x).squeeze().reshape([-1, input_size])
    # review 这里调用了下面的方法，得到所有神经元的输出区间
    layer_outputs = get_layer_outputs(intval_x, W, b)
    # # review 得到最后的输出层
    # ott = layer_outputs[weight_size - 1]
    # # review 这里面所有的三维的中间第二维都是空的
    # r = ott[0][0]
    # c1 = inf(r)
    # c2 = sup(r)
    # # review 这里面下界大于0或者上届小于0 都直接ok并退出，只有区域的值夹着0才进行下一步
    # if c1 >= 0 or c2 <= 0:
    #     ok = 1
    #     return ok, c1, c2
    hidden_variable_num = 0
    for i in range(1, weight_size):
        # review len只返回第一维的长度
        hidden_variable_num = hidden_variable_num + len(W[i])
    # review variable_num的大小是对每个hidden_variable都要用relu的三大约束所引出的变量，x, y和t，同时算出输出层的所有神经元个数
    # TODO 这里减去了最后一个包括lp_Aeq_beq_t_row
    all_variable_num = input_size + 3 * hidden_variable_num + len(W[weight_size - 1][0])
    # review 整体的条件分为两种，一种是lp_A_b_t_row, 一种是lp_Aeq_beq_t_row, 分别表示relu条件的编码，和wx+b的未经过relu之前的条件编码
    lp_A_b_t_row = 3 * hidden_variable_num
    lp_Aeq_beq_t_row = hidden_variable_num + len(W[weight_size - 1][0]) + 1
    # review 将两种约束竖着叠加在一起
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1  # review lp_A_b_loc记录了第一部分条件在矩阵中的起始行和当前行
    lp_Aeq_beq_loc = lp_A_b_t_row - 1  # review lp_Aeq_beq_loc记录了第二部分条件在矩阵中的起始行和当前行

    binary = np.zeros((hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    # review lp_lb和lp_ub是每个var的下界和上界
    lp_lb = np.zeros((all_variable_num, 1)).squeeze()
    lp_ub = np.ones((all_variable_num, 1)).squeeze()

    for i in range(0, input_size):
        lp_lb[i] = inf(intval_x[0][i])
        lp_ub[i] = sup(intval_x[0][i])

    # review lp_lb_ub_loc是当前variable的处理索引，因为之前输入层的variable都已经处理好了所以将位置设在input_size-1
    lp_lb_ub_loc = input_size - 1
    process_location = input_size - 1
    # review y_head_record先赋予每个输入层神经元的位置索引
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, weight_size - 1):
        # review 先得到每层的输出
        layer_output_before_relu = layer_outputs[layer_index]
        # review 对上一层的神经元所在的具体是那一列进行索引映射
        #  y_head_record_last_layer对应的是上一层的y也就是这一层的x
        y_head_record_last_layer = y_head_record
        # review 在这里声明为输出神经元的数量，也就是这层要输出的y
        y_head_record = np.zeros((1, len(W[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(W[layer_index][0])):  # review j是遍历这一层的输出神经元的维度
            # review 这里的顺序分别是x，y和t，将l和u作为上届和下界的值传进去
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0  # review 这个是指示变量的范围
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])  # review 这边对上界是类似的处理方式
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            #  review binary是记录变量t的类型的，代码里面所有东西包括编程习惯都是先+1再使用
            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))  # review 这里面装的其实是约束的一行，填补到lp_A_Aeq里面

            # review 这里三个条件就是relu的三个条件，分别是前面项的系数
            #  process_location是和lp_lb_ub_loc一一对应的
            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])  # question t的取值应该和x是关联的，但是好像没看出来关联度？
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1  # review 一共要编写lp_A_Aeq的三行，每次更改三个元素
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            # review 这里是相等的那个条件，所有相等的条件都放到了最后 𝑧 𝑘 = 𝑊 𝑘 𝑥 𝑘−1 + 𝑏 𝑘
            #  这里将等式进行变形𝑊(𝑘)*𝑥(𝑘−1) - 𝑧(𝑘) = -𝑏(𝑘)
            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(W[layer_index])):  # review 对k进行循环其实就是对W矩阵的那一列进行循环
                # review 这里面就是模拟一个神经元的输出，在lp_Aeq_t中放W的那一列的所有weight
                # review y_head_record_last_layer记录的是上层layer的y也就是这里面的x(k-1)的索引位置
                lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]  # review 这里的k指的是这一层输入的索引，j指的是这一层输出的索引
            lp_Aeq_t[0][process_location + 1] = -1  # review 这里面的-1是给到x，也就是没有经过relu后的神经元输出
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]  # review 这个条件从索引108开始
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2  # review y_head_record记录的是上一层的y的位置，也就是经过relu之后的值
            process_location = process_location + 3  # review process_location是这次已经处理的位置
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    # review 这里出循环
    layer_index = weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    # review 用t_process_loc来接棒process_location，并且后面每次只自增1
    t_process_loc = process_location

    # review 还是模拟一个神经元的输出，主要模拟输出层神经元的输出，因为输出层神经元不需要经过relu了，在lp_Aeq_t中放W的那一列的所有weight
    for j in range(0, len(W[layer_index][0])):  # review 这里对输出层的神经元索引进行遍历，因为出了之前的循环layer_index就是最后了
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(W[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        # review 对输出层神经元进行创建variable的工作
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1

    # review 最后添加一个约束要求要求nx是n的一半
    lp_Aeq_t = np.zeros((1, all_variable_num))
    lp_Aeq_t[0][0] = 1
    lp_Aeq_t[0][2] = -2
    lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
    lp_b_beq[lp_Aeq_beq_loc + 1] = 0
    lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1


    # review 创建数据类型type列表
    vtype = list("C" * all_variable_num)  # review 先将所有的type都声明为C，再将需要改成B的进行修改
    for i_c in range(0, len(binary)):
        vtype[binary[i_c]] = 'B'
    vtype = ''.join(vtype)

    # review 创建不等式符号列表
    sense = list('<' * (lp_A_b_t_row + lp_Aeq_beq_t_row))
    for i_c in range(0, lp_Aeq_beq_t_row):
        sense[i_c + lp_A_b_t_row] = '='
    sense = ''.join(sense)

    lp_f = np.zeros((1, all_variable_num)).squeeze()
    # review 这是目标要用的，前面的variable系数都是0，最后两个输出神经元相减的目标值是1
    # review 这里虽然不改代码，但是含义变了，现在的最后一个variable就是输出variable
    lp_f[all_variable_num - 1] = 1

    m = gp.Model()
    m.setParam('OutputFlag', 0)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))
    # sense is char, like vtype, can not initialize by str
    m.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row])
    m.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])

    m.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MINIMIZE)
    m.optimize()
    print("{0} {1} {2}".format(all_vars[0], all_vars[1], all_vars[2]))
    c1 = m.objVal  # review 这里是将最小值和最大值算出来，再判断是否夹着0

    m.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MAXIMIZE)
    m.optimize()
    c2 = m.objVal
    print("{0} {1} {2}".format(all_vars[0], all_vars[1], all_vars[2]))

    if c1 < 0 < c2:
        ok = 0
    else:
        ok = 1

# review 结果看起来好像是挺promising的，范围是缩小了一点
    return ok, c1, c2


# TODO 初始区域经过函数变换后的区域，这个是和特定的程序息息相关的，有的程序也不太好提炼
def change_function(x_min, x_max):
    return x_min, x_max


# review 约束条件包含了I和I'，并适应G的条件，先传进来一个较大的输入X的范围，并通过其他的限制条件保证在想要的界内
def milp_change_outputs(x_min, x_max, W, b):
    ok = 0
    weight_size = len(W)
    input_size = len(W[0])
    intval_x = []
    for i in range(0, input_size):
        t = IntervalNumber(x_min[i], x_max[i])
        # review 将每个维度的下界和上界变成IntervalNumber对象并放到数组中
        intval_x.append(t)
    # review 这里面做了一对操作却又什么都没做,intval_x的维度是(1，input_size)
    intval_x = np.array(intval_x).squeeze().reshape([-1, input_size])

    # TODO 这里可以修改，表示经过程序变换后的X'的输入输出范围，都是实际范围的超集
    changed_min, changed_max = change_function(x_min, x_max)
    interval_changed = []
    for i in range(0, input_size):
        t = IntervalNumber(changed_min[i], changed_max[i])
        interval_changed.append(t)
    interval_changed = np.array(interval_changed).squeeze().reshape([-1, input_size])

    # review 得到所有神经元的输出区间的过逼近
    layer_outputs = get_layer_outputs(intval_x, W, b)
    # TODO 得到X'作为输入区间，经过网络后输出区间的过逼近
    changed_layer_outputs = get_layer_outputs(interval_changed, W, b)

    # review 之前判断最后输出元素的范围的没必要了
    hidden_variable_num = 0
    for i in range(1, weight_size):
        # review len只返回第一维的长度
        hidden_variable_num = hidden_variable_num + len(W[i])
    # review variable_num的大小是对每个hidden_variable都要用relu的三大约束所引出的变量，x, y和t，同时算出输出层的所有神经元个数
    #  记得分清变量个数和约束个数的区别，分别代表了矩阵的行和列数
    # TODO 这里面all_variable_num要修改，应该是两倍的原来的跟网络相关的变量个数
    all_variable_num = (input_size + 3 * hidden_variable_num + len(W[weight_size - 1][0])) * 2
    # review 小于等于的约束是原来的两倍+1，两倍是因为两个网络，加2是因为还有G的约束，也是小于约束，因为是abs的，所以大小为2，再加1，表示输出神经元的约束
    lp_A_b_t_row = (3 * hidden_variable_num) * 2 + 2 + 1
    # review 等于的约束是原来的两倍+2，两倍是因为两个网络，加2是因为还有X和X'的转换约束，其中x'=nx用一次约束表示，这里面要再加1因为有n'=n的约束
    lp_Aeq_beq_t_row = (hidden_variable_num + len(W[weight_size - 1][0])) * 2 + 2
    # review 将两种约束竖着叠加在一起
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1  # review lp_A_b_loc记录了第一部分条件在矩阵中的起始行和当前行
    lp_Aeq_beq_loc = lp_A_b_t_row - 1  # review lp_Aeq_beq_loc记录了第二部分条件在矩阵中的起始行和当前行

    # review binary的大小也得是2倍的原始大小
    binary = np.zeros((2*hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    # review lp_lb和lp_ub是每个var的下界和上界
    lp_lb = np.zeros((all_variable_num, 1)).squeeze()
    lp_ub = np.ones((all_variable_num, 1)).squeeze()

    for i in range(0, input_size):
        lp_lb[i] = inf(intval_x[0][i])
        lp_ub[i] = sup(intval_x[0][i])

    # review lp_lb_ub_loc是当前variable的处理索引，因为之前输入层的variable都已经处理好了所以将位置设在input_size-1
    lp_lb_ub_loc = input_size - 1
    process_location = input_size - 1

    # TODO 从这边开始就是对网络的约束的层层复现
    # review y_head_record先赋予每个输入层神经元的位置索引，记录的应该是上一层经过relu之后的y变量所对应的位置
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, weight_size - 1):
        # review 先得到每层的输出
        layer_output_before_relu = layer_outputs[layer_index]
        # review y_head_record_last_layer对应的是上一层的y也就是这一层的x
        y_head_record_last_layer = y_head_record
        # review 在这里声明为输出神经元的数量，也就是这层要输出的y
        y_head_record = np.zeros((1, len(W[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(W[layer_index][0])):  # review j是遍历这一层的输出神经元的维度
            # review 这里的顺序分别是x，y和t，将l和u作为上届和下界的值传进去
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0  # review 这个是指示变量的范围
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])  # review 这边对上界是类似的处理方式
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            #  review binary是记录变量t的类型的，代码里面所有东西包括编程习惯都是先+1再使用
            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))  # review 这里面装的其实是约束的一行，填补到lp_A_Aeq里面

            # review 这里三个条件就是relu的三个条件，分别是前面项的系数
            #  process_location是和lp_lb_ub_loc一一对应的
            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])  # question t的取值应该和x是关联的，但是好像没看出来关联度？
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1  # review 一共要编写lp_A_Aeq的三行，每次更改三个元素
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            # review 这里是相等的那个条件，所有相等的条件都放到了最后 𝑧 𝑘 = 𝑊 𝑘 𝑥 𝑘−1 + 𝑏 𝑘
            #  这里将等式进行变形𝑊(𝑘)*𝑥(𝑘−1) - 𝑧(𝑘) = -𝑏(𝑘)
            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(W[layer_index])):  # review 对k进行循环其实就是对W矩阵的那一列进行循环
                # review 这里面就是模拟一个神经元的输出，在lp_Aeq_t中放W的那一列的所有weight
                # review y_head_record_last_layer记录的是上层layer的y也就是这里面的x(k-1)的索引位置
                lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]  # review 这里的k指的是这一层输入的索引，j指的是这一层输出的索引
            lp_Aeq_t[0][process_location + 1] = -1  # review 这里面的-1是给到x，也就是没有经过relu后的神经元输出
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]  # review 这个条件从索引108开始
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2  # review y_head_record记录的是上一层的y的位置，也就是经过relu之后的值
            process_location = process_location + 3  # review process_location是这次已经处理的位置
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    # review 这里出循环
    layer_index = weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    # review 用t_process_loc来接棒process_location，并且后面每次只自增1
    t_process_loc = process_location

    for j in range(0, len(W[layer_index][0])):  # review 这里对输出层的神经元索引进行遍历，因为出了之前的循环layer_index就是最后了
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(W[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        # review 对输出层神经元进行创建variable的工作
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1

    # TODO 这里开始第二个网络的约束编码 这里的t_process_loc再被process_location进行接棒，不需要增加
    #  同时lp_A_Aeq_loc, lp_lb_ub_loc以及binary_loc还有lp_A_b_loc都是不用过多改写的，可以照常使用

    # TODO 别忘了因为新加入了新的input，要先对其lp_lb以及process_location
    #  如果interval用原来interval-changed的范围，可能会丧失一些变化后的值域
    #  可以将interval变化为一个比较大的范围，反正后续元素取什么都以肯定能确定的
    for i in range(0, input_size):
        lp_lb[i+1+lp_lb_ub_loc] = inf(interval_changed[0][i])
        lp_ub[i+1+lp_lb_ub_loc] = sup(interval_changed[0][i])

    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    process_location = t_process_loc
    # TODO 这里修改，第二次输入的X的位置并记录留给后面使用
    for i in range(0, input_size):
        y_head_record[i] = i + 1 + process_location
    # TODO 对齐lp_lb_ub_loc以及process_location
    lp_lb_ub_loc += 3
    # review 在这里准确记录了系统变量新的n,x和nx的位置
    output_loc = process_location
    n_loc = process_location+1
    x_loc = process_location+2
    nx_loc = process_location+3
    process_location += 3
    # TODO 等会调试的时候在这里设一个断点

    for layer_index in range(0, weight_size - 1):
        # TODO 这里修改成X'的初步过逼近结果
        layer_output_before_relu = changed_layer_outputs[layer_index]
        # review y_head_record_last_layer对应的是上一层的y也就是这一层的x
        y_head_record_last_layer = y_head_record
        # review 在这里声明为输出神经元的数量，也就是这层要输出的y
        y_head_record = np.zeros((1, len(W[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(W[layer_index][0])):  # review j是遍历这一层的输出神经元的维度
            # review 这里的顺序分别是x，y和t，将l和u作为上届和下界的值传进去
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0  # review 这个是指示变量的范围
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])  # review 这边对上界是类似的处理方式
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            # TODO 这个地方打个断点，后面看看有没有问题
            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))  # review 这里面装的其实是约束的一行，填补到lp_A_Aeq里面

            # review 这里三个条件就是relu的三个条件，分别是前面项的系数
            #  process_location是和lp_lb_ub_loc一一对应的
            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])  # question t的取值应该和x是关联的，但是好像没看出来关联度？
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1  # review 一共要编写lp_A_Aeq的三行，每次更改三个元素
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            # review 这里是相等的那个条件，所有相等的条件都放到了最后 𝑧 𝑘 = 𝑊 𝑘 𝑥 𝑘−1 + 𝑏 𝑘
            #  这里将等式进行变形𝑊(𝑘)*𝑥(𝑘−1) - 𝑧(𝑘) = -𝑏(𝑘)
            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(W[layer_index])):  # review 对k进行循环其实就是对W矩阵的那一列进行循环
                # review 这里面就是模拟一个神经元的输出，在lp_Aeq_t中放W的那一列的所有weight
                # review y_head_record_last_layer记录的是上层layer的y也就是这里面的x(k-1)的索引位置
                lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]  # review 这里的k指的是这一层输入的索引，j指的是这一层输出的索引
            lp_Aeq_t[0][process_location + 1] = -1  # review 这里面的-1是给到x，也就是没有经过relu后的神经元输出
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]  # review 这个条件从索引108开始
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2  # review y_head_record记录的是上一层的y的位置，也就是经过relu之后的值
            process_location = process_location + 3  # review process_location是这次已经处理的位置
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    # review 这里出循环
    layer_index = weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    # review 用t_process_loc来接棒process_location，并且后面每次只自增1
    t_process_loc = process_location

    # review 还是模拟一个神经元的输出，主要模拟输出层神经元的输出，因为输出层神经元不需要经过relu了，在lp_Aeq_t中放W的那一列的所有weight
    for j in range(0, len(W[layer_index][0])):  # review 这里对输出层的神经元索引进行遍历，因为出了之前的循环layer_index就是最后了
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(W[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        # review 对输出层神经元进行创建variable的工作
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1
    # TODO 到此神经网络的编码结束，在后面补上X->X'以及G的约束
    #  首先是X -> X'的约束，利用之前记录的n，x和nx的位置
    lp_Aeq_t = np.zeros((1, all_variable_num))
    lp_Aeq_t[0][x_loc] = 1
    lp_Aeq_t[0][2] = -1
    lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
    lp_b_beq[lp_Aeq_beq_loc + 1] = 0
    lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
    # TODO 等一下，沃日，这里没设置新的n等于原来的n? 在这里试着补上，即使多约束一次也不影响结果
    lp_Aeq_t = np.zeros((1, all_variable_num))
    lp_Aeq_t[0][n_loc] = 1
    lp_Aeq_t[0][0] = -1
    lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
    lp_b_beq[lp_Aeq_beq_loc + 1] = 0
    # TODO 这里之前代入错公式了，公式应该是(nx)^2 - 2*(nx)*(nx') + n = 0
    # TODO 这里又有一个坑，第二个改变是不能用一次约束表示的，为2*(nx')*(x) = x+n，所以改用二次约束表示，一次约束的条数-1
    lp_q = np.zeros((all_variable_num, all_variable_num))
    # review 应该把lp表示成上三角矩阵就行,对应的x与nx'为行列的系数为2
    lp_q[2][nx_loc] = -2
    lp_q[2][2] = 1
    c_t = np.zeros((1, all_variable_num)).squeeze()
    # review 给最初的x和n的系数附上-1，后面把这个赋给MQConstr
    c_t[0] = 1

    # review 如果所修改的参数位置对应的是同一个变量的位置，则不需要重置lp_A_t，否则需要重置lp_A_t
    # review 最后再加一个约束表示第一个网络输出的值小于等于0
    lp_A_t = np.zeros((1, all_variable_num))
    lp_A_t[0][output_loc] = 1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = 0
    lp_A_b_loc = lp_A_b_loc + 1
    # TODO 其次是对G条件的编码，是大于的形式
    # review 这里注意一定是大于大于大于，或者加个符号，变成小于
    #  吐了，这里是或者关系，不是并的关系
    lp_A_t = np.zeros((1, all_variable_num))
    lp_A_t[0][1] = -1
    lp_A_t[0][2] = 1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = -epsilon_G
    lp_A_b_loc = lp_A_b_loc + 1
    lp_A_t[0][1] = 1
    lp_A_t[0][2] = -1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = -epsilon_G
    lp_A_b_loc = lp_A_b_loc + 1

    # review 创建数据类型type列表
    vtype = list("C" * all_variable_num)  # review 先将所有的type都声明为C，再将需要改成B的进行修改
    for i_c in range(0, len(binary)):
        vtype[binary[i_c]] = 'B'
    vtype = ''.join(vtype)

    # review 创建不等式符号列表
    sense = list('<' * (lp_A_b_t_row + lp_Aeq_beq_t_row))
    for i_c in range(0, lp_Aeq_beq_t_row):
        sense[i_c + lp_A_b_t_row] = '='
    sense = ''.join(sense)

    lp_f = np.zeros((1, all_variable_num)).squeeze()
    # review 这是目标要用的，前面的variable系数都是0，最后两个输出神经元相减的目标值是1
    # review 这里虽然不改代码，但是含义变了，现在的最后一个variable就是输出variable
    lp_f[all_variable_num - 1] = 1
    debug_lp_f1 = np.zeros((1, all_variable_num)).squeeze()
    debug_lp_f1[2] = 1

    # TODO 因为满足的条件有两个，所以整两个model分别表示上半部分和下半部分
    m1 = gp.Model()
    m1.setParam('OutputFlag', 0)
    m1.setParam('NonConvex', 2)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m1.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))
    # sense is char, like vtype, can not initialize by str
    m1.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row-1], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row-1])
    # m1.addMConstrs(A=lp_A_Aeq[greater_loc_1:greater_loc_2], x=all_vars, sense='>', b=lp_b_beq[greater_loc_1:greater_loc_2])
    m1.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])
    # review 在这里添加唯一一个二次约束
    m1.addMQConstr(Q=lp_q, c=c_t, sense='=', rhs=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars)

    # review 需要I'的输出值全部小于0，所以找输出值最大的看是不是小于0
    m1.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MAXIMIZE)
    m1.optimize()
    c1 = m1.objVal
    print("{0} {1} {2} {3} {4} {5}".format(all_vars[0], all_vars[1], all_vars[2], all_vars[n_loc], all_vars[x_loc],
                                           all_vars[nx_loc]))

    # TODO 这里是第二个模型，表示第二个部分
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    m2.setParam('NonConvex', 2)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m2.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))
    # sense is char, like vtype, can not initialize by str
    m2.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row-2], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row-2])
    m2.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row-1:lp_A_b_t_row], x=all_vars, sense='<', b=lp_b_beq[lp_A_b_t_row-1:lp_A_b_t_row])
    # m2.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row-1:g], x=all_vars, sense='>', b=lp_b_beq[greater_loc_2:greater_loc_2+1])
    m2.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])
    # review 在这里添加唯一一个二次约束
    m2.addMQConstr(Q=lp_q, c=c_t, sense='=', rhs=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars)

    # review 需要I'的输出值全部小于0，所以找输出值最大的看是不是小于0
    m2.setMObjective(None, lp_f, 0.0, None, None, all_vars, GRB.MAXIMIZE)
    m2.optimize()
    c2 = m2.objVal
    print("{0} {1} {2} {3} {4} {5}".format(all_vars[0], all_vars[1], all_vars[2], all_vars[n_loc], all_vars[x_loc],
                                           all_vars[nx_loc]))

    if c1 <= 0 and c2 <= 0:
        ok = 1
    else:
        ok = 0

    return ok, c1, c2


# review 第三个条件的判断
def milp_final_outputs(x_min, x_max, W, b):
    ok = 0
    weight_size = len(W)
    input_size = len(W[0])
    intval_x = []
    for i in range(0, input_size):
        t = IntervalNumber(x_min[i], x_max[i])
        # review 将每个维度的下界和上界变成IntervalNumber对象并放到数组中
        intval_x.append(t)
    # review 这里面做了一对操作却又什么都没做,intval_x的维度是(1，input_size)
    intval_x = np.array(intval_x).squeeze().reshape([-1, input_size])

    # review 得到所有神经元的输出区间的过逼近
    layer_outputs = get_layer_outputs(intval_x, W, b)

    # review 之前判断最后输出元素的范围的没必要了
    hidden_variable_num = 0
    for i in range(1, weight_size):
        # review len只返回第一维的长度
        hidden_variable_num = hidden_variable_num + len(W[i])
    # review variable_num的大小是对每个hidden_variable都要用relu的三大约束所引出的变量，x, y和t，同时算出输出层的所有神经元个数
    #  记得分清变量个数和约束个数的区别，分别代表了矩阵的行和列数
    # review 这里面all_variable_num和第一个条件一样，只是一个网络的变量个数就行
    all_variable_num = input_size + 3 * hidden_variable_num + len(W[weight_size - 1][0])
    # review 小于等于的约束是原来的+2+1，加2是因为还有G的约束，也是小于约束，因为是abs的，所以大小为2，再加1，表示输出神经元的约束
    lp_A_b_t_row = 3 * hidden_variable_num + 2 + 1
    # review 等于的约束和原来的一样，因为不涉及变量的转换之类的
    lp_Aeq_beq_t_row = (hidden_variable_num + len(W[weight_size - 1][0])) * 2 + 1
    # review 将两种约束竖着叠加在一起
    lp_A_Aeq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, all_variable_num))
    lp_b_beq = np.zeros((lp_A_b_t_row + lp_Aeq_beq_t_row, 1)).squeeze()
    lp_A_b_loc = -1  # review lp_A_b_loc记录了第一部分条件在矩阵中的起始行和当前行
    lp_Aeq_beq_loc = lp_A_b_t_row - 1  # review lp_Aeq_beq_loc记录了第二部分条件在矩阵中的起始行和当前行

    # review binary的大小也和原来一样
    binary = np.zeros((hidden_variable_num, 1), dtype=np.int).squeeze()
    binary_loc = -1

    # review lp_lb和lp_ub是每个var的下界和上界
    lp_lb = np.zeros((all_variable_num, 1)).squeeze()
    lp_ub = np.ones((all_variable_num, 1)).squeeze()

    for i in range(0, input_size):
        lp_lb[i] = inf(intval_x[0][i])
        lp_ub[i] = sup(intval_x[0][i])

    # review lp_lb_ub_loc是当前variable的处理索引，因为之前输入层的variable都已经处理好了所以将位置设在input_size-1
    lp_lb_ub_loc = input_size - 1
    process_location = input_size - 1

    # TODO 从这边开始就是对网络的约束的层层复现
    # review y_head_record先赋予每个输入层神经元的位置索引，记录的应该是上一层经过relu之后的y变量所对应的位置
    y_head_record = np.zeros((1, input_size), dtype=np.int).squeeze()
    for i in range(0, input_size):
        y_head_record[i] = i

    for layer_index in range(0, weight_size - 1):
        # review 先得到每层的输出
        layer_output_before_relu = layer_outputs[layer_index]
        # review y_head_record_last_layer对应的是上一层的y也就是这一层的x
        y_head_record_last_layer = y_head_record
        # review 在这里声明为输出神经元的数量，也就是这层要输出的y
        y_head_record = np.zeros((1, len(W[layer_index][0])), dtype=np.int).squeeze()
        for j in range(0, len(W[layer_index][0])):  # review j是遍历这一层的输出神经元的维度
            # review 这里的顺序分别是x，y和t，将l和u作为上届和下界的值传进去
            lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
            lp_lb[lp_lb_ub_loc + 2] = max(0, inf(layer_output_before_relu[0][j]))
            lp_lb[lp_lb_ub_loc + 3] = 0  # review 这个是指示变量的范围
            lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])  # review 这边对上界是类似的处理方式
            lp_ub[lp_lb_ub_loc + 2] = max(0, sup(layer_output_before_relu[0][j]))
            lp_ub[lp_lb_ub_loc + 3] = 1
            lp_lb_ub_loc = lp_lb_ub_loc + 3

            #  review binary是记录变量t的类型的，代码里面所有东西包括编程习惯都是先+1再使用
            binary[binary_loc + 1] = process_location + 3
            binary_loc = binary_loc + 1

            lp_A_t = np.zeros((1, all_variable_num))  # review 这里面装的其实是约束的一行，填补到lp_A_Aeq里面

            # review 这里三个条件就是relu的三个条件，分别是前面项的系数
            #  process_location是和lp_lb_ub_loc一一对应的
            lp_A_t[0][process_location + 1] = -1
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -inf(layer_output_before_relu[0][j])  # question t的取值应该和x是关联的，但是好像没看出来关联度？
            lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 1] = -inf(layer_output_before_relu[0][j])

            lp_A_t[0][process_location + 1] = 1  # review 一共要编写lp_A_Aeq的三行，每次更改三个元素
            lp_A_t[0][process_location + 2] = -1
            lp_A_t[0][process_location + 3] = 0
            lp_A_Aeq[lp_A_b_loc + 2, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 2] = 0

            lp_A_t[0][process_location + 1] = 0
            lp_A_t[0][process_location + 2] = 1
            lp_A_t[0][process_location + 3] = -sup(layer_output_before_relu[0][j])
            lp_A_Aeq[lp_A_b_loc + 3, :] = lp_A_t
            lp_b_beq[lp_A_b_loc + 3] = 0
            lp_A_b_loc = lp_A_b_loc + 3

            # review 这里是相等的那个条件，所有相等的条件都放到了最后 𝑧 𝑘 = 𝑊 𝑘 𝑥 𝑘−1 + 𝑏 𝑘
            #  这里将等式进行变形𝑊(𝑘)*𝑥(𝑘−1) - 𝑧(𝑘) = -𝑏(𝑘)
            lp_Aeq_t = np.zeros((1, all_variable_num))
            for k in range(0, len(W[layer_index])):  # review 对k进行循环其实就是对W矩阵的那一列进行循环
                # review 这里面就是模拟一个神经元的输出，在lp_Aeq_t中放W的那一列的所有weight
                # review y_head_record_last_layer记录的是上层layer的y也就是这里面的x(k-1)的索引位置
                lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]  # review 这里的k指的是这一层输入的索引，j指的是这一层输出的索引
            lp_Aeq_t[0][process_location + 1] = -1  # review 这里面的-1是给到x，也就是没有经过relu后的神经元输出
            lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
            lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]  # review 这个条件从索引108开始
            lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1

            y_head_record[j] = process_location + 2  # review y_head_record记录的是上一层的y的位置，也就是经过relu之后的值
            process_location = process_location + 3  # review process_location是这次已经处理的位置
            layer_output_before_relu[0][j] = interval_max(layer_output_before_relu[0][j], 0)

    # review 这里出循环
    layer_index = weight_size - 1
    layer_output_before_relu = layer_outputs[layer_index]
    y_head_record_last_layer = y_head_record
    # review 用t_process_loc来接棒process_location，并且后面每次只自增1
    t_process_loc = process_location

    for j in range(0, len(W[layer_index][0])):  # review 这里对输出层的神经元索引进行遍历，因为出了之前的循环layer_index就是最后了
        lp_Aeq_t = np.zeros((1, all_variable_num))
        for k in range(0, len(W[layer_index])):
            lp_Aeq_t[0][y_head_record_last_layer[k]] = W[layer_index][k, j]

        lp_Aeq_t[0][t_process_loc + 1] = -1
        lp_A_Aeq[lp_Aeq_beq_loc + 1, :] = lp_Aeq_t
        lp_b_beq[lp_Aeq_beq_loc + 1] = -b[layer_index][j]
        lp_Aeq_beq_loc = lp_Aeq_beq_loc + 1
        # review 对输出层神经元进行创建variable的工作
        lp_lb[lp_lb_ub_loc + 1] = inf(layer_output_before_relu[0][j])
        lp_ub[lp_lb_ub_loc + 1] = sup(layer_output_before_relu[0][j])
        lp_lb_ub_loc = lp_lb_ub_loc + 1
        t_process_loc = t_process_loc + 1

    # review 到这里网络的编码结束
    # review 如果所修改的参数位置对应的是同一个变量的位置，则不需要重置lp_A_t，否则需要重置lp_A_t
    # review 最后再加一个约束表示第一个网络输出的值小于等于0
    output_loc = lp_lb_ub_loc
    lp_A_t = np.zeros((1, all_variable_num))
    lp_A_t[0][output_loc] = 1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = 0
    lp_A_b_loc = lp_A_b_loc + 1
    # TODO 其次是对非G条件的编码，这里第一个是小于，第二个是大于
    lp_A_t = np.zeros((1, all_variable_num))
    lp_A_t[0][1] = 1
    lp_A_t[0][2] = -1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = 0.0001
    lp_A_b_loc = lp_A_b_loc + 1
    lp_A_t[0][1] = 1
    lp_A_t[0][2] = -1
    lp_A_Aeq[lp_A_b_loc + 1, :] = lp_A_t
    lp_b_beq[lp_A_b_loc + 1] = -0.0001
    lp_A_b_loc = lp_A_b_loc + 1

    # review 创建数据类型type列表
    vtype = list("C" * all_variable_num)  # review 先将所有的type都声明为C，再将需要改成B的进行修改
    for i_c in range(0, len(binary)):
        vtype[binary[i_c]] = 'B'
    vtype = ''.join(vtype)

    # review 创建不等式符号列表
    sense = list('<' * (lp_A_b_t_row + lp_Aeq_beq_t_row))
    for i_c in range(0, lp_Aeq_beq_t_row):
        sense[i_c + lp_A_b_t_row] = '='
    sense = ''.join(sense)

    # TODO 虽然这里有abs包裹，但是因为条件是小于，所以只需要整一个model就行，
    m1 = gp.Model()
    m1.setParam('OutputFlag', 0)
    m1.setParam('NonConvex', 2)
    all_vars = []
    for i in range(0, all_variable_num):
        all_vars.append(m1.addVar(lb=lp_lb[i], ub=lp_ub[i], vtype=vtype[i]))
    # sense is char, like vtype, can not initialize by str
    m1.addMConstrs(A=lp_A_Aeq[0:lp_A_b_t_row-1], x=all_vars, sense='<', b=lp_b_beq[0:lp_A_b_t_row-1])
    m1.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row-1:lp_A_b_t_row], x=all_vars, sense='>', b=lp_b_beq[lp_A_b_t_row-1:lp_A_b_t_row])
    m1.addMConstrs(A=lp_A_Aeq[lp_A_b_t_row:], x=all_vars, sense='=', b=lp_b_beq[lp_A_b_t_row:])

    # TODO 这里的objective变了
    q_new = np.zeros((all_variable_num, all_variable_num))
    q_new[2][2] = 1
    lp_f = np.zeros((1, all_variable_num)).squeeze()
    lp_f[0] = -1
    m1.setMObjective(Q=q_new, c=lp_f, constant=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars, sense=GRB.MAXIMIZE)
    m1.optimize()
    c1 = m1.objVal

    m1.setMObjective(Q=q_new, c=lp_f, constant=0, xQ_L=all_vars, xQ_R=all_vars, xc=all_vars, sense=GRB.MINIMIZE)
    m1.optimize()
    c2 = m1.objVal

    if c1 <= 10*epsilon_Q and c2 >= -10*epsilon_Q:
        ok = 1
    else:
        ok = 0

    return ok, c1, c2


# review 这相当于是一个逐层的过逼近，先作为初步的筛选条件
def get_layer_outputs(y, W, b):
    # REVIEW W是个三维的数组，shape(0)也就是第一个维数是有几层，每一层是一个权值的二维数组
    t_num_layers = len(W)

    # TODO layer_outputs这个数组到底是个啥东西？几维的，怎么有三层的索引访问
    layer_outputs = []
    layer_outputs_after_relu = []

    t_input = y

    # review 在这里给layer_outputs添加一个第二层的所有神经元结果
    layer_outputs.append(np.dot(t_input, W[0]) + b[0])
    # review 用IntervalNumber的interval_max来模拟实现relu
    layer_outputs_after_relu.append(interval_max(np.array(layer_outputs[0]), 0))

    # review 这个循环是两层，输入的结果已经是过了一个隐层后的结果，之后每过一个隐层循环一次
    for t_layer_index in range(1, t_num_layers):
        # question 这中间的0是什么鬼？可能这个0维是必须要带的吧
        # review 这里面active_flag其实就是一个长度为len(layer_outputs[t_layer_index - 1][0])的一维数组
        active_flag = np.zeros((1, len(layer_outputs[t_layer_index - 1][0]))).squeeze()
        for tj in range(0, len(layer_outputs[t_layer_index - 1][0])):
            # REVIEW 下界大于0，标记为active
            if inf(layer_outputs[t_layer_index - 1][0][tj]) >= 0:
                active_flag[tj] = 1
        layer_outputs_active = []
        layer_outputs_inactive = []
        W_active = []
        W_inactive = []
        W_mixed = []
        b_mixed = []
        for i in range(0, len(active_flag)):
            if active_flag[i] == 1:
                # review 这里面的i应该是这一层的神经元的索引，layer_outputs_active里面的元素没用到
                layer_outputs_active.append(layer_outputs_after_relu[t_layer_index - 1][0][i])
                W_active.append(W[t_layer_index][i, :])
                # TODO 这里面的W和b的mix是干嘛的，取上一层的第i列
                W_mixed.append(W[t_layer_index - 1][:, i])
                b_mixed.append(b[t_layer_index - 1][i])
            else:
                layer_outputs_inactive.append(layer_outputs_after_relu[t_layer_index - 1][0][i])
                W_inactive.append(W[t_layer_index][i, :])

        # review 应该是全是false的意思，没有一个是被激活状态的
        if sum(active_flag == 1) == 0:
            # review 数组的维度是(1, len(layer_outputs_inactive))
            layer_outputs_inactive = np.array(layer_outputs_inactive).reshape([-1, len(layer_outputs_inactive)])
            W_inactive = np.array(W_inactive)
            output_inactive = np.dot(layer_outputs_inactive, W_inactive) + b[t_layer_index]
            layer_outputs.append(output_inactive)
        # review 全是true，没有一个false
        elif sum(active_flag == 0) == 0:
            # review 先拿两个W相乘，然后在拿这个和输入来乘
            W_mul = np.dot(np.array(W_mixed).transpose(), np.array(W_active))
            b_mul = np.dot(np.array(b_mixed).transpose(), np.array(W_active))
            if t_layer_index == 1:
                output_active = np.dot(t_input, W_mul) + b_mul
            else:
                output_active = np.dot(layer_outputs_after_relu[t_layer_index - 2], W_mul) + b_mul
            layer_outputs.append(output_active)
        else:
            # review 就是将active和非active的两个方式结合起来
            layer_outputs_inactive = np.array(layer_outputs_inactive).reshape([-1, len(layer_outputs_inactive)])
            W_inactive = np.array(W_inactive)
            output_inactive = np.dot(layer_outputs_inactive, W_inactive) + b[t_layer_index]
            W_mul = np.dot(np.array(W_mixed).transpose(), np.array(W_active))
            b_mul = np.dot(np.array(b_mixed).transpose(), np.array(W_active))
            if t_layer_index == 1:
                output_active = np.dot(t_input, W_mul) + b_mul
            else:
                output_active = np.dot(layer_outputs_after_relu[t_layer_index - 2], W_mul) + b_mul
            layer_outputs.append(output_inactive + output_active)
        layer_outputs_after_relu.append(interval_max(np.array(layer_outputs[t_layer_index]), 0))

    # review layer_outputs里面以每层为列表成员放了所有神经元的输出区间
    return layer_outputs
