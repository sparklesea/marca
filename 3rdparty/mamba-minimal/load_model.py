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

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/sii_lijinhao/models/mamba-2.8b')
parser.add_argument("--promptlen", type=int, default=0)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--batch", type=int, default=1)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--debug", type=bool, default=True)
parser.add_argument("--task", type=str, default=None)

args = parser.parse_args()

device = torch.device(f"cuda:{args.device}") if args.device else torch.device("cpu")

model = Mamba.from_pretrained(args.model_name, debug=args.debug, device=device, dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/sii_lijinhao/models/tokenizers/gpt-neox-20b-local-cache/models--EleutherAI--gpt-neox-20b/snapshots/c292233c833e336628618a88a648727eb3dff0a7", local_files_only=True)

print(model)