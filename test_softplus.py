import torch

# # 定义 SILU 函数
def softplus(x):
    return torch.log1p(torch.exp(x))

# # 分段近似 SILU 函数
def piecewise_softplus(x):
    mask1 = x < -7.0
    mask2 = (x >= -7.0) & (x < -1.5)
    mask3 = (x >= -1.5) & (x <= 0.75)

    y = torch.zeros_like(x)
    y[mask1] = -0.002
    # 在 mask2 部分使用二次函数近似 SILU 函数
    a = 0.010  # 二次项系数
    b = 0.113  # 一次项系数
    c = 0.314   # 常数项
    y[mask2] = a * x[mask2] ** 2 + b * x[mask2] + c - 0.028
    # y[mask2] = x[mask2] * torch.sigmoid(x[mask2])
    a,b=1.05,-0.2781
    y[mask3] = a * x[mask3] + b

    return y

def fast_softplus(input):
    temp = torch.abs(((input*1.4426950409+126.94201519)*(2**23)).to(torch.int)).view(torch.float32)
    # return ((temp+1-0.027).view(torch.int).to(torch.float32)/(2**23)-126.94201519)/1.4426950409-0.0
    # mask1 = temp<=0
    # mask2 = temp>0
    # result = torch.zeros_like(temp)
    # result[mask1] = temp[mask1]*(1-temp[mask1]*(0.5-0.25*temp[mask1]*temp[mask1]))
    # result[mask2] = 0.103*temp[mask2] + 0.510*temp[mask2]+0.693*temp[mask2]
    # return result
    # return torch.log1p(temp)
    return temp*(1-temp*(0.5-0.25*temp*temp))
# 生成输入数据
x = torch.linspace(-10, 0, 1000)

# 计算 SILU 函数和分段近似 SILU 函数的输出
y_softplus = softplus(x)
# y_piecewise_softplus = piecewise_softplus(x)
y_piecewise_softplus = fast_softplus(x)

# 绘制 SILU 函数和分段近似 SILU 函数的图像
import matplotlib.pyplot as plt

plt.plot(x.numpy(), y_softplus.numpy(), label='y_softplus')
plt.plot(x.numpy(), y_piecewise_softplus.numpy(), label='y_piecewise_softplus')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('test_softplus.png')

print(torch.abs(y_softplus-y_piecewise_softplus))
