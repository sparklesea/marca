import torch

import transformers
from transformers import AutoTokenizer

# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from model_to_eval import Mamba

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate


@register_model("mamba")
class MambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba-2.8b", max_length=2048, batch_size=1, device=torch.device("cuda"), dtype=torch.float32):
        LM.__init__(self)
        self._model = Mamba.from_pretrained(pretrained, debug=False, device=device, dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/lijinhao-240108540148/research_huangshan/sii_lijinhao/models/tokenizers/gpt-neox-20b-local-cache/models--EleutherAI--gpt-neox-20b/snapshots/c292233c833e336628618a88a648727eb3dff0a7", local_files_only=True)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = 64
        self._max_length = max_length
        self._device = device

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        raise NotImplementedError()


if __name__ == "__main__":
    cli_evaluate()
