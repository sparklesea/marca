import os
# 定义任务列表
task_list = ["piqa", "wikitext", "lambada_openai", "winogrande", "hellaswag", "arc_easy"]

# 循环处理每个任务
for task_name in task_list:
    try:
        print(f"\n===== 开始处理任务 {task_name} =====")
        os.system(f"HF_DATASETS_OFFLINE=1 python lm_harness_eval.py --model mamba --model_args pretrained=../../../sii_lijinhao/models/mamba-2.8b --tasks {task_name} --device cuda >> experiment.log")
        print(f"===== 任务 {task_name} 处理完成 =====")
    except Exception as e:
        print(f"处理任务 {task_name} 时出错: {e}")

os.system("python /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/scripts/train_fk.py")