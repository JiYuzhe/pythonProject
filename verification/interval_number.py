import sys


# review 装着一个interval的下界和上界，同时覆写了一些操作符
class IntervalNumber:
    def __init__(self, l=0, u=0):
        self.l = l
        self.u = u

# review python类中将对象转化为供解释器读取的形式，类似于java的toString
    def __repr__(self):
        return "[" + str(self.l) + ", " + str(self.u) + "]"

    def __neg__(self):
        return IntervalNumber(-self.u, -self.l)

# review python调用这种二元操作符的逻辑：比如a+b先找a的add方法，如果a没有add就找b的radd方法，在左边的操作数的add或者右边操作数的radd方法
    def __add__(self, other):
        if type(self).__name__ == type(other).__name__:
            return IntervalNumber(self.l + other.l, self.u + other.u)
        else:
            # review 这样可以实现类对象和比如数值之间的加法
            return IntervalNumber(self.l + other, self.u + other)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

# review 两个IntervalNumber对象之间是不能乘的，相当于对l和u都乘以特定的倍数
    def __mul__(self, other):
        if type(self).__name__ == type(other).__name__:
            print("An IntervalNumber can not multiply another IntervalNumber, please check code.")
            sys.exit(0)
        if other >= 0:
            return IntervalNumber(self.l * other, self.u * other)
        else:
            return IntervalNumber(self.u * other, self.l * other)

    def __rmul__(self, other):
        return self * other


def interval_max(x, y):
    # review 这里面的x如果是IntervalNumber对象，y就是一个基本数据类型
    if type(x).__name__ == "IntervalNumber":
        if x.u <= y:
            # review 如果x的上界都比y小，返回一个上下界都是y的IntervalNumber
            return IntervalNumber(y, y)
        elif x.l >= y:
            # review 如果下界都大于y，就返回自己本身就行
            return x
        else:
            # review 否则将将下界改成y
            return IntervalNumber(y, x.u)
    # review 如果x是一个二维数组，对里面第一行所有元素调用interval_max 第一行的所有元素应该都是IntervalNumber
    elif type(x[0][0]).__name__ == "IntervalNumber":
        for i in range(0, len(x[0])):
            x[0][i] = interval_max(x[0][i], y)
        return x
    else:
        print("Incorrect use of interval_max, please check code.")
        sys.exit(0)


# review 这两个方法其实就是换名的getter而已
def inf(x):
    return x.l


def sup(x):
    return x.u
