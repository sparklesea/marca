from html import parser
from model import Mamba, ModelArgs
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='/share/public_models/mamba-2.8b')
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=0)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--top_k", type=int, default=40)
parser.add_argument("--top_p", type=float, default=0.0)
parser.add_argument("--temperature", type=float, default=1.0)
# parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--min_p", type=float, default=0.0)
parser.add_argument("--max_length", type=int, default=100)

args = parser.parse_args()

# One of:
#     'state-spaces/mamba-2.8b-slimpj'
#     'state-spaces/mamba-2.8b'
#     'state-spaces/mamba-1.4b'
#     'state-spaces/mamba-790m'
#     'state-spaces/mamba-370m'
#     'state-spaces/mamba-130m'
# pretrained_model_name = 'state-spaces/mamba-370m'
pretrained_model_name = args.model_name

model = Mamba.from_pretrained(pretrained_model_name).cuda()
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

if args.promptlen > 0:
    input_ids = torch.randint(1, 50277, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
else:
    input_ids = tokenizer(args.prompt, return_tensors='pt').input_ids.cuda()

def generate(model,
             tokenizer,
             input_ids,
             n_tokens_to_gen: int = 50,
             sample: bool = True,
             top_k: int = 40):
    model.eval()
    
    for token_n in range(n_tokens_to_gen):
        with torch.no_grad():
            indices_to_input = input_ids
            next_token_logits = model(indices_to_input)[:, -1]
        
        probs = F.softmax(next_token_logits, dim=-1)
        (batch, vocab_size) = probs.shape
        
        if top_k is not None:
            (values, indices) = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = 0
            probs = probs / probs.sum(axis=1, keepdims=True)
        
        if sample:
            next_indices = torch.multinomial(probs, num_samples=1)
        else:
            next_indices = torch.argmax(probs, dim=-1)[:, None]
        
        input_ids = torch.cat([input_ids, next_indices], dim=1)

    output_completions = [tokenizer.decode(output.tolist()) for output in input_ids][0]
    
    return output_completions

print(model)

if args.prompt:
    print(generate(model, tokenizer, input_ids, args.top_k))
elif args.promptlen > 0:
    print(generate(model, tokenizer, input_ids, n_tokens_to_gen=args.genlen, top_k=args.top_k))
else:
    raise Exception("No prompt or promptlen specified")