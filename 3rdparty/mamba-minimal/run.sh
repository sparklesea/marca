HF_DATASETS_OFFLINE=1 python lm_harness_eval.py --model mamba --model_args pretrained=../../../sii_lijinhao/models/mamba-2.8b --tasks arc_easy --device cuda

# batch version ?
HF_DATASETS_OFFLINE=1 python lm_harness_eval.py --model mamba --model_args "pretrained=../../../sii_lijinhao/models/mamba-2.8b" --tasks hellaswag --device cuda --batch_size 64

HF_DATASETS_OFFLINE=1 python lm_harness_eval.py --model mamba --model_args "pretrained=../../../sii_lijinhao/models/mamba-2.8b" --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log