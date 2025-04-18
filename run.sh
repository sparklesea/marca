pip uninstall mamba_ssm
pip install -e .
# python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-2.8b --tasks wikitext --device cuda --batch_size 64
# python evals/lm_harness_eval_7b.py --model mamba --model_args pretrained=/share/huangshan/mamba-7b-rw --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log

# python evals/lm_harness_eval_7b.py --model mamba --model_args pretrained=/share/huangshan/mamba-7b-rw --tasks wikitext --device cuda --batch_size 64 >> experiment.log