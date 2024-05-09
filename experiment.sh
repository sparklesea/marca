# python evals/lm_harness_eval_7b.py --model mamba --model_args pretrained=/share/huangshan/mamba-7b-rw --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 16 >> experiment.log
python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-2.8b --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log
python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-1.4b --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log
python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-790m --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log
python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-370m --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log
python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-130m --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 64 >> experiment.log

