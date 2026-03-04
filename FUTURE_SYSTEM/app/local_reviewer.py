import os
from threading import Lock
from typing import Any, Dict

from .config import AppConfig
from .drive_utils import find_adapter_dir


class LocalReviewerEngine:
    """Lazy-loaded local reviewer model (Unsloth + PEFT adapter)."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._loaded = False
        self._load_error = ""
        self._lock = Lock()
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._adapter_dir = ""

    @property
    def adapter_dir(self) -> str:
        return self._adapter_dir

    @property
    def load_error(self) -> str:
        return self._load_error

    def status(self) -> Dict[str, str]:
        if self._loaded:
            return {
                "state": "loaded",
                "adapter_dir": self._adapter_dir,
                "device": self._device,
            }
        if self._load_error:
            return {"state": "error", "error": self._load_error}
        return {"state": "not_loaded"}

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._load_error:
            raise RuntimeError(self._load_error)

        with self._lock:
            if self._loaded:
                return
            if self._load_error:
                raise RuntimeError(self._load_error)

            try:
                adapter_dir, files = find_adapter_dir(list(self.config.adapter_dir_candidates))
                if not adapter_dir:
                    raise FileNotFoundError(
                        "Adapter directory not found. Expected under FUTURE_SYSTEM/my_finetuned_llm."
                    )

                required = {"adapter_config.json", "adapter_model.safetensors"}
                missing = sorted(name for name in required if name not in set(files))
                if missing:
                    raise FileNotFoundError(
                        "Missing required adapter files: " + ", ".join(missing)
                    )
                self._adapter_dir = adapter_dir

                hf_token = os.getenv("HF_TOKEN", "").strip()
                if not hf_token:
                    raise RuntimeError(
                        "HF_TOKEN is not set. Provide it in launch_colab.ipynb before launching UI."
                    )

                # Import here to avoid heavy imports during module import time.
                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "No GPU detected for local reviewer (torch.cuda.is_available() is False). "
                        "Switch Colab Runtime to GPU (T4/A100) to use adapter model."
                    )

                os.environ.setdefault("UNSLOTH_DISABLE_LOG_STATS", "1")
                os.environ.setdefault("UNSLOTH_USE_MODELSCOPE", "1")
                # Import unsloth before peft/transformers to apply required patches.
                import unsloth  # noqa: F401
                from unsloth import FastLanguageModel
                from peft import PeftModel

                model, tokenizer = FastLanguageModel.from_pretrained(
                    self.config.reviewer_base_model_id,
                    max_seq_length=self.config.reviewer_max_seq_length,
                    load_in_4bit=self.config.reviewer_load_in_4bit,
                    token=hf_token,
                    disable_log_stats=True,
                )

                model = PeftModel.from_pretrained(model, self._adapter_dir)
                FastLanguageModel.for_inference(model)
                model.eval()

                self._model = model
                self._tokenizer = tokenizer
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._loaded = True
            except Exception as ex:
                self._load_error = f"{type(ex).__name__}: {ex}"
                raise RuntimeError(self._load_error) from ex

    def generate(self, prompt: str) -> str:
        self.ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        with self._lock:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": 220,
                "min_new_tokens": 0,
                "temperature": 0.0,
                "top_p": 1.0,
                "do_sample": False,
                "repetition_penalty": 1.12,
                "no_repeat_ngram_size": 5,
            }

            eos_id = getattr(self._tokenizer, "eos_token_id", None)
            if eos_id is not None:
                gen_kwargs["pad_token_id"] = eos_id

            outputs = self._model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            raw_text = self._tokenizer.decode(
                outputs[0][prompt_len:], skip_special_tokens=True
            ).strip()
            return raw_text
