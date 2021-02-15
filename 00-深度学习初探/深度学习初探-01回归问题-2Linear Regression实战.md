## 深度学习初探/01-回归问题/2-Linear Regression实战
###### 1、使用梯度下降算法实现Linear Regression的方法
Linear Regression的目的是根据样本点集Points，拟合出一条最理想的线性函数【$y=wx+b$】。

则对于任意点$P(x_i,y_i)$，需要使$(wx_i+b-y_i)$尽可能小；
对于点集Points的所有点，需要使误差之和$totalError=\Sigma(wx_i+b-y_i)=\Sigma(wx_i+b-y_i)_{min}$
$\Rightarrow$  平均误差$Avg= (totalError/点数)_{min}$

为了便于求最值（即便于求导），我们设损失函数$Loss Function$：
>$Loss = (y^*-y)^2 = (wx + b -y)^2 = \Sigma(wx+b-y)^2/N$

而后使用梯度下降算法求解即可。
###### 2、具体实现01：求平均误差Error=$\Sigma(wx_i+b-y_i)_{min} /$点数
根据给出的点集points，求平均误差Error
```python
# points是一个二维数组，第一个数表示“点序号”，第二个数表示“x or y”
def compute_error_for_line_given_points( w , b , points):
    totalError = 0  #总误差totalError
    # 遍历点集points，len(points)表示点集中点的数量
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += ((w * x + b) - y) ** 2
    # 返回平均误差Error = 总误差totalError/点个数
    return totalError / float(len(points))
```
###### 3、具体实现02：梯度下降处理
对于$Loss = \Sigma (wx+b-y)^2$这样的“凹函数”而言，各个变量($w、b$)偏导驻点的交汇处，即为整个函数的极小值点。如图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210214175243523.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1paX2xvbmdyb2Fk,size_16,color_FFFFFF,t_70#pic_center)

由梯度下降的迭代公式：
>$w^* = w - learningRate*\frac{\partial Loss}{\partial w}$,其中$\frac{\partial Loss}{\partial w}=2*(wx+b-y)*x=\Sigma[2*(wx+b-y)*x]/N$
>$b^* = b - learningRate*\frac{\partial Loss}{\partial b}$,其中$\frac{\partial Loss}{\partial b}=2*(wx+b-y)=\Sigma[2*(wx+b-y)]/N$
```python
# 梯度下降算法求解w和b
def step_gradient( b_current , w_current , points , learningRate ):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range( 0 , len(points)):
        x = points[i,0]
        y = points[i,1]
        # 计算w和b偏导的平均值，求和、除以点数
        w_gradient += (2 * (w_current * x + b_current - y) * x) / N
        b_gradient += (2 * (w_current * x + b_current - y)) / N
    # 依梯度迭代公式，求得本次迭代后新的w b的值
    new_w = w_current - (learningRate * w_gradient)
    new_b = b_current - (learningRate * b_gradient)
    return [new_b, new_w]
```
###### 4、具体实现03-循环迭代梯度信息
限定一个迭代次数（如100次），以此迭代结果作为w和b的较优解投入使用。
```python
# 循环迭代梯度信息，求出最终的w和b
def gradient_descent_runner(points, starting_w, starting_b,
                            learning_rate, num_iterations):
    w = starting_w
    b = starting_b
    for i in range(num_iterations):   # num_iterations 即为迭代次数
        w, b = step_gradient(b, w, np.array(points), learning_rate)
    return [w, b]
```
###### 5、具体实现04-调参运行
```python
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
```
###### 6、实验结果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210215111659675.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1paX2xvbmdyb2Fk,size_16,color_FFFFFF,t_70#pic_center)