# open_source.py

from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun

class LocalLLM(LLM):
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device: Optional[str] = None,
        max_tokens: int = 512,
        stream: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self.stream = stream

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
        self.streamer = TextStreamer(self.tokenizer) if self.stream else None

    @property
    def _llm_type(self) -> str:
        return "local_hf_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                streamer=self.streamer if self.stream else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded[len(prompt):]  # Return only the generated continuation

    def generate_batch(self, prompts: List[str]) -> List[str]:
        input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generations = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return [g[len(p):] for g, p in zip(generations, prompts)]

    def generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None
    ) -> LLMResult:
        generations = [Generation(text=t) for t in self.generate_batch(prompts)]
        return LLMResult(generations=[generations])


# Example usage
if __name__ == "__main__":
    llm = LocalLLM(stream=False)
    prompt = "What is Retrieval-Augmented Generation (RAG) in simple terms?"
    print("Answer:", llm(prompt))
