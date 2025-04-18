import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(0)

# 生成符合要求的数据
data = np.random.normal(loc=0.8, scale=0.3, size=4096*16)

# 绘制直方图
count, bins, ignored = plt.hist(data, bins=np.arange(0, 1.05, 0.05), density=True, alpha=0.5, color='g', edgecolor='black')

# 保存频数到文本文件
with open('frequency.txt', 'w') as f:
    for i in range(len(count)):
        f.write(f'{count[i]}\n')

# 添加标题和标签
plt.title('Histogram of Normal Distribution Data')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 在图中标注0所在位置
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)

# 显示图形
plt.show()
