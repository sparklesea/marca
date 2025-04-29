#!/bin/bash

HF_DATASETS_OFFLINE=1 python evals/lm_harness_eval.py --model mamba --model_args "pretrained=../sii_lijinhao/models/mamba-790m/,debug=True" --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment-790m

cp -r profile_result/pt profile_result/pt-790m

HF_DATASETS_OFFLINE=1 python evals/lm_harness_eval.py --model mamba --model_args "pretrained=../sii_lijinhao/models/mamba-370m/,debug=True" --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment-370m

cp -r profile_result/pt profile_result/pt-370m

HF_DATASETS_OFFLINE=1 python evals/lm_harness_eval.py --model mamba --model_args "pretrained=../sii_lijinhao/models/mamba-130m/,debug=True" --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment-130m

cp -r profile_result/pt profile_result/pt-130m