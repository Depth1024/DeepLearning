import numpy as np


# 根据给出的点集points，求平均误差Error
# points是一个二维数组，第一个数表示“点序号”，第二个数表示“x or y”
def compute_error_for_line_given_points(b, w, points):
    totalError = 0  #总误差totalError
    # 遍历点集points，len(points)表示点集中点的数量
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += ((w * x + b) - y) ** 2
    # 返回平均误差Error = 总误差totalError/点个数
    return totalError / float(len(points))


# 梯度下降算法求解w和b
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 计算w和b偏导的平均值，求和、除以点数
        w_gradient += (2 * (w_current * x + b_current - y) * x) / N
        b_gradient += (2 * (w_current * x + b_current - y)) / N
    # 依梯度迭代公式，求得本次迭代后新的w b的值
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]


# 循环迭代梯度信息，求出最终的w和b
def gradient_descent_runner(points, starting_b, starting_w,
                            learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    for i in range(num_iterations):   # num_iterations 即为迭代次数
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


# 运行
def run():
    # 使用numpy现成的genfromtxt导入txt文件里的点集数据
    points = np.genfromtxt("data.csv", delimiter=",")
    # 设置初始参数
    learning_rate = 0.0001  # learningrate的选取尽可能小
    initial_b = 0
    initial_w = 0
    num_iterations = 1000  # 迭代1000次

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_w,
                  compute_error_for_line_given_points(initial_b, initial_w, points)
                  )
          )
    print("Running...")
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, w,
                 compute_error_for_line_given_points(b, w, points))
          )


if __name__ == '__main__':
    run()
