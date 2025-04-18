import os
import torch

tasks = ["lambada_openai","piqa","winogrande","arc_easy","hellaswag"]
# 配置
for task in tasks:
    data_dir = f'/root/huangshan/research/marca/3rdparty/mamba-minimal/profile_result/pt/deltaB/{task}'  # 替换成你的文件夹路径
    save_path = f'/root/huangshan/research/marca/3rdparty/mamba-minimal/profile_result/pt/deltaB_mask/{task}.pt'         # 保存最终 mask 的路径
    threshold = 1e-2                    # 判断是否接近 0 的阈值

    # 存储所有 mask
    all_masks = []

    # 遍历所有 pt 文件
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.pt'):
            file_path = os.path.join(data_dir, filename)

            # 读取张量
            tensor = torch.load(file_path)  # shape: (d, n)
            assert tensor.dim() == 2, f"{filename} is not a 2D tensor."

            # 求 mean(dim=1)
            mean_vals = tensor.mean(dim=1)  # shape: (d,)

            # 接近 0 的判断：绝对值小于 threshold
            mask = (mean_vals.abs() < threshold)  # shape: (d,)

            all_masks.append(mask)

    # 拼接所有 mask： shape = (num_files, d)
    final_mask_tensor = torch.stack(all_masks)  # shape: (num_files, d)

    # 保存
    torch.save(final_mask_tensor, save_path)

    print(f"Saved final mask tensor of shape {final_mask_tensor.shape} to {save_path}")