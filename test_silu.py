import torch

# # 定义 SILU 函数
def silu(x):
    return x * torch.sigmoid(x)

# # 分段近似 SILU 函数
def piecewise_silu(x):
    mask1 = x < -5.0
    mask2 = (x >= -5.0) & (x < -1.5)
    mask3 = (x >= -1.5) & (x <= 0.75)
    mask4 = (x > 0.75)

    y = torch.zeros_like(x)
    y[mask1] = -0.0135
    a = -0.06244
    b = -0.3457
    y[mask2] = a * x[mask2] + b
    # a = -0.009
    # b = -0.130
    # c = -0.448
    # y[mask2] = a * x[mask2]**2 + b*x[mask2]+c
    # 在 mask2 部分使用二次函数近似 SILU 函数
    a = 0.213  # 二次项系数
    b = 0.510  # 一次项系数
    c = 0.013   # 常数项
    y[mask3] = a * x[mask3] ** 2 + b * x[mask3] + c - 0.028
    # y[mask2] = x[mask2] * torch.sigmoid(x[mask2])
    a,b=1.05,-0.2781
    y[mask4] = a * x[mask4] + b

    return y

# 生成输入数据
x = torch.linspace(-10, 10, 1000)

# 计算 SILU 函数和分段近似 SILU 函数的输出
y_silu = silu(x)
y_piecewise_silu = piecewise_silu(x)

# 绘制 SILU 函数和分段近似 SILU 函数的图像
import matplotlib.pyplot as plt

plt.plot(x.numpy(), y_silu.numpy(), label='SILU')
plt.plot(x.numpy(), y_piecewise_silu.numpy(), label='Piecewise SILU')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('SILU Function and Piecewise Approximation')
plt.savefig('test_silu.png')

print(torch.abs(y_silu-y_piecewise_silu))
