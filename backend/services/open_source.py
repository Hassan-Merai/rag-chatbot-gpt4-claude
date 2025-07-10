# backend/services/open_source.py
# backend/services/open_source.py

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun


class LocalLLM:
    _tokenizer_cache = {}
    _model_cache = {}

    def __init__(
        self,
        model_name: str = "tiiuae/falcon-rw-1b",
        device: Optional[str] = None,
        max_tokens: int = 512,
        stream: bool = False
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self.stream = stream

        if model_name in self._tokenizer_cache:
            self.tokenizer = self._tokenizer_cache[model_name]
        else:
            #self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self._tokenizer_cache[model_name] = self.tokenizer
            

        if model_name in self._model_cache:
            self.model = self._model_cache[model_name]
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)  # Manually move to device
            self.model.eval()
            self._model_cache[model_name] = self.model

        self.streamer = TextStreamer(self.tokenizer) if self.stream else None

    def generate_batch(self, prompts: List[str]) -> List[str]:
        try:
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            generations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            return generations
        except Exception as e:
            print(f"[LocalLLM generate_batch ERROR] {e}")
            return ["[Error generating response]"] * len(prompts)

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> LLMResult:
        generations = [Generation(text=t) for t in self.generate_batch(prompts)]
        return LLMResult(generations=[generations])

