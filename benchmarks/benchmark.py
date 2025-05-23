# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--warmup", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

warmup = args.warmup
repeats = args.repeat
device = "cuda"
# device = "cpu"
dtype = torch.float16

print(f"Loading model {args.model_name}")

is_7b = args.model_name.endswith("-rw")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
if not is_7b:
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='balanced', torch_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=dtype)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype)

model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# torch.random.manual_seed(114514)
torch.random.manual_seed(520)
if args.prompt is None:
    input_ids = torch.randint(1, 50277, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

if not is_7b:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        # cg=True,
        cg=False,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
for _ in range(warmup):
    out = fn()

if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
torch.cuda.synchronize()
print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.2f}ms")




# python benchmarks/benchmark_generation_mamba_simple.py --model-name "/share/huangshan/mamba-2.8b" --prompt "The capital of France is" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
# python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-2.8b --tasks wikitext --device cuda --batch_size 1
# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark_generation_mamba_simple.py --model-name "/share/huangshan/mamba-2.8b" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --promptlen 10 --genlen 10
# python evals/lm_harness_eval.py --model mamba --model_args pretrained=/share/huangshan/mamba-2.8b --tasks wikitext,lambada_openai,piqa,winogrande,arc_easy,hellaswag --device cuda --batch_size 1

# CUDA_VISIBLE_DEVICES=6,7 python evals/lm_harness_eval_7b.py --model mamba --model_args pretrained=/share/huangshan/mamba-7b-rw --tasks wikitext --device cuda --batch_size 64

# python benchmarks/benchmark_generation_mamba_simple.py --model-name "/share/huangshan/mamba-2.8b" --prompt "The capital of France is" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2

# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark.py --model-name "/share/huangshan/mamba-7b-rw" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --genlen 1 --promptlen 64 
# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark.py --model-name "/share/huangshan/mamba-2.8b" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --genlen 1 --promptlen 512
# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark.py --model-name "/share/huangshan/mamba-1.4b" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --genlen 1 --promptlen 512 
# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark.py --model-name "/share/huangshan/mamba-790m" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --genlen 1 --promptlen 512 
# CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark.py --model-name "/share/huangshan/mamba-130m" --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 --genlen 1 --promptlen 512 