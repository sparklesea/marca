from model_to_eval import Mamba
from mamba_ssm import MambaLMHeadModel
import torch

model_path = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/sii_lijinhao/models/mamba-2.8b'

model_ours = Mamba.from_pretrained(model_path, debug=False, device='cuda', dtype=torch.float16)
model_mamba = MambaLMHeadModel.from_pretrained(model_path, device='cuda', dtype=torch.float16)


print(torch.mean(model_ours.lm_head.weight), torch.std(model_ours.lm_head.weight))
print(torch.mean(model_mamba.lm_head.weight), torch.std(model_mamba.lm_head.weight))