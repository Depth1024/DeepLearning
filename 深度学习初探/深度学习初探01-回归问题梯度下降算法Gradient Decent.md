## 深度学习初探/01-回归问题/梯度下降算法Gradient Decent
#### 一、梯度下降算法的本质
通过迭代计算，不断调整$x$的取值，从而求得函数的“极小值”

#### 二、梯度下降算法的具体实现（类似于“二分查找”）
【$x_1$】为每次迭代后所求得的“下一步$x$值”;
【$x_0$】为“当前$x$值”;
【$f'(x_0)$】为“$f(x)$在点$x_0$处的导数值”
【LR】Learning Rate，即学习速率，用于控制每一次迭代的“步长Step”，避免每一次调整距离过大
【$x^*$】最终求得的极小值点
> 迭代公式：$x_1 = x_0 - f'(x_0)*LR$
>
> 这个公式可以保证$x$始终朝着“下坡”方向行进，且到达终点后，由于$f'(x)  \approx 0$，故$x$会在极小值点$x^*$附近“抖动”，从而求得极小值点$x^*$ 

###### $\Delta$不同Learning Rate对学习效率的影响
比较理想的Learning Rate可以快速获得最优解，如下图中设LR=0.005，有效优化了学习过程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210126160930215.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RvY01hblpMUg==,size_16,color_FFFFFF,t_70#pic_center)
如若调整LR=0.05，则会导致$x$的值在极值点左右来回摆动，使学习效率大幅降低，如图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210126161047112.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RvY01hblpMUg==,size_16,color_FFFFFF,t_70#pic_center)
##### 三、简单的回归问题
###### 1、Linear Regression 线性回归
Linear Regression的函数模型是【$y = wx+b$】，其目的是根据给出的一组已知$x_i和y_i$取值的方程式$\begin{cases}
y_1 = w*x_1 + b +\epsilon\\
y_2 = w*x_2 + b+\epsilon\\
y_3 = w*x_3 + b+\epsilon\\
......\\
y_k = w*x_k+b+\epsilon
\end{cases}$（其中$\epsilon$为微弱的高斯噪声），求出对应参数$w、b$，从而拟合出线性函数【$y = wx+b$】，使得对于任意输入的$x_N$，都能给出一个估计值$y_N$
###### 2、Logistic Regression 逻辑回归
Logistic Regression的函数模型是【$y=Logistic Function*(wx+b)$】，其中，LogisticFunction起到”压缩“作用，使得$y\in[0,1]$
应用场景：二分分类问题（抛硬币、猫图判别）
###### 3、Classification 分类问题
如”手写数字识别“问题，该类问题的特点为：$\Sigma p_i =1$