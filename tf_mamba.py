from transformers.models.mamba.modeling_mamba import MambaForCausalLM
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F

import argparse
import time
import json

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
args = parser.parse_args()

warmup = 0
repeats = 2
# device = "cuda"
# device = "cpu"
device = "cpu:0"
dtype = torch.float16

model = MambaForCausalLM.from_pretrained(args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model.eval()

torch.random.manual_seed(114514)
if args.prompt is None:
    input_ids = torch.randint(1, 50277, (args.batch, args.promptlen), dtype=torch.long, device=device)
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
else:
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
max_length = input_ids.shape[1] + args.genlen

fn = lambda: model.generate(
    input_ids,
    max_new_tokens=args.genlen,
    min_new_tokens=args.genlen
)

for _ in range(warmup):
    out = fn()

if args.prompt is not None:
    print(tokenizer.batch_decode(out.sequences.tolist()))

# torch.cuda.synchronize()
start = time.time()
for _ in range(repeats):
    fn()
# torch.cuda.synchronize()
# print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.2f}ms")
