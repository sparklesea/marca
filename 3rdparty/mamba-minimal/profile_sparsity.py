from html import parser
from model import Mamba, ModelArgs
from transformers import AutoTokenizer

import torch
import torch.nn.functional as F

import argparse
import os
import json
from tqdm import tqdm

ROOT_DIR = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/marca/3rdparty/mamba-minimal"

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
def prefill(model, input_ids, task):
    model.eval()

    model(input_ids, task)


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/sii_lijinhao/models/mamba-2.8b')
parser.add_argument("--promptlen", type=int, default=0)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--device", type=int, default=7)
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--task", type=str, default=None)

args = parser.parse_args()

device = torch.device(f"cuda:{args.device}") if args.device else torch.device("cpu")

model = Mamba.from_pretrained(args.model_name, debug=args.debug, device=device, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

# tasks = ["wikitext","lambada_openai","piqa","winogrande","arc_easy","hellaswag"]
if not args.task:
    tasks = ["wikitext","lambada_openai","piqa","winogrande","arc_easy","hellaswag"]
else:
    tasks = [args.task]

if args.debug:
    # delete old deltaA/deltaB
    for task in tasks:
        os.system("rm -rf " + ROOT_DIR + f"/profile_result/pt/deltaA/{task}")
        os.system("rm -rf " + ROOT_DIR + f"/profile_result/pt/deltaB/{task}")


for task in tasks:
    task_path = os.path.join(ROOT_DIR, 'lm_eval_benchmark', f'{task}.json')
    with open(task_path, 'r') as f:
        task_requests = json.load(f)
    
    print(f"start {task}!!!")

    for i, prompt in tqdm(enumerate(task_requests), total=len(task_requests)):
        # promptlen = len(tokenizer(prompt, return_tensors='pt').input_ids[0])
        # genlen = args.max_length - promptlen
        # print(f"{task}: {promptlen} {genlen}")
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
        # print(f"input type: {input_ids.dtype}")
        prefill(model, input_ids, task)

    
    # average deltaA/deltaB
    # if args.debug:
    #     for layer_id in range(model.args.n_layer):
    #         dA = torch.load(ROOT_DIR + f'/profile_result/pt/deltaA/{task}/{layer_id}.pt')
    #         dA = dA / len(task_requests)
    #         torch.save(dA, ROOT_DIR + f'/profile_result/pt/deltaA/{task}/{layer_id}.pt')

    #         dB = torch.load(ROOT_DIR + f'/profile_result/pt/deltaB/{task}/{layer_id}.pt')
    #         dB = dB / len(task_requests)
    #         torch.save(dB, ROOT_DIR + f'/profile_result/pt/deltaB/{task}/{layer_id}.pt')

    if args.debug:
        dA_path = ROOT_DIR + f'/profile_result/pt/deltaA/{task}/'
        dB_path = ROOT_DIR + f'/profile_result/pt/deltaB/{task}/'
        if not os.path.exists(dA_path):
            os.makedirs(dA_path)
        if not os.path.exists(dB_path):
            os.makedirs(dB_path)

        for layer_id in range(model.args.n_layer):

            dA = model.layers[layer_id].mixer.deltaA / len(task_requests)
            torch.save(dA, dA_path + f'{layer_id}.pt')

            dB = model.layers[layer_id].mixer.deltaB / len(task_requests)
            torch.save(dB, dB_path + f'{layer_id}.pt')

    print(f"finish {task}!!!")

# if args.promptlen > 0:
#     input_ids = torch.randint(1, 50277, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
# else:
#     input_ids = tokenizer(args.prompt, return_tensors='pt').input_ids.cuda()
