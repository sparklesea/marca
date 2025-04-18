import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import Tensor

def expand_tensor(tensor: Tensor):
    """
    扩展张量的第二个维度，使其大小与第一个维度相同，通过重复元素。
    
    参数:
    tensor (torch.Tensor): 输入的二维张量，形状为 (n, m)。
    target_size (int): 扩展后第二个维度的大小。
    
    返回:
    torch.Tensor: 扩展后的张量，形状为 (n, target_size)。
    """
    if tensor.dim() != 2:
        raise ValueError("输入张量必须是二维的")
    
    n, m = tensor.size()
    print(n, m)
    if n % m != 0:
        raise ValueError("目标大小必须是第一个维度大小的整数倍，以实现简单重复")
    
    repeat_times = n // m
    expanded_tensor = tensor.repeat_interleave(repeat_times, dim=1)[:, :n]
    assert expanded_tensor.size() == (n, n)
    
    return expanded_tensor

def tensor_to_heatmap(tensor):
    """
    将张量转换为热力图。
    
    参数:
    tensor (torch.Tensor): 输入的二维张量。
    
    返回:
    PIL.Image: 热力图图片。
    """
    # 将张量转换为 numpy 数组并归一化到 0-1 范围
    tensor_np = tensor.cpu().numpy()

    fig, ax = plt.subplots()

    cax = ax.imshow(tensor_np, cmap='viridis')

    fig.colorbar(cax)

    # plt.show()
    
    # # 使用 Matplotlib 生成热力图
    # plt.imshow(tensor_np, cmap='hot', interpolation='nearest')
    # plt.axis('off')  # 关闭坐标轴
    
    # 将 Matplotlib 图像保存到缓冲区并转换为 PIL 图像
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf)
    plt.close()  # 关闭 Matplotlib 图形
    
    return image

def convert_pt_to_heatmap(input_folder, output_folder):
    """
    将文件夹中的所有 .pt 文件转换为热力图图片。
    
    参数:
    input_folder (str): 输入文件夹路径。
    output_folder (str): 输出文件夹路径。
    target_size (int): 扩展后第二个维度的大小。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.pt'):
            file_path = os.path.join(input_folder, filename)
            
            # 加载 .pt 文件
            loaded_tensor = torch.load(file_path)
            
            # 确保加载的是二维张量
            if not isinstance(loaded_tensor, torch.Tensor) or loaded_tensor.dim() != 2:
                print(f"跳过文件 {filename}，因为它不包含一个二维张量")
                continue
            
            # 扩展张量
            expanded_tensor = expand_tensor(loaded_tensor)

            # 将张量转换为热力图
            heatmap_image = tensor_to_heatmap(expanded_tensor)
            
            # 保存热力图图片
            output_file_path = os.path.join(output_folder, filename.replace('.pt', '.png'))
            heatmap_image.save(output_file_path)
            print(f"已保存 {output_file_path}")

import io  # 确保导入 io 模块

# 示例用法
# task = "piqa"
tasks = ["lambada_openai","arc_easy","hellaswag"]

for task in tasks:
    root_folder = "/root/huangshan/research/marca/3rdparty/mamba-minimal/profile_result/"

    # deltaB
    input_folder = root_folder + f"pt/deltaB/{task}"  # 输入文件夹路径
    output_folder = root_folder + f"fig/deltaB_heatmap/{task}"  # 输出文件夹路径
    convert_pt_to_heatmap(input_folder, output_folder)

    # deltaA
    input_folder = root_folder + f"pt/deltaA/{task}"  # 输入文件夹路径
    output_folder = root_folder + f"fig/deltaA_heatmap/{task}"
    convert_pt_to_heatmap(input_folder, output_folder)