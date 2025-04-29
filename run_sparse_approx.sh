#!/bin/bash

# 定义模型参数列表
MODELS=("2.8b" "1.4b" "790m" "370m" "130m")

# 遍历每个模型参数
for model in "${MODELS[@]}"; do
    # 删除旧的pt目录（如果存在）
    rm -rf profile_result/pt
    
    # 复制对应的模型目录
    cp -r "profile_result/pt-${model}" profile_result/pt
    
    echo "all: both + sparsedB + sparsehs\n" >> experiment-${model}.log
    # 执行评估脚本并将输出重定向到对应日志文件
    HF_DATASETS_OFFLINE=1 python evals/lm_harness_eval.py \
        --model mamba \
        --model_args "pretrained=../sii_lijinhao/models/mamba-${model}/,debug=False,sparsedB=True,sparsehs=True,fastexp=True,silu=True" \
        --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag \
        --device cuda \
        --batch_size 64 \
        >> "experiment-${model}.log"
done