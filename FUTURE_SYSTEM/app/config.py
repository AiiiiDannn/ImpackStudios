from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AppConfig:
    app_title: str = "ImpactStudio FUTURE_SYSTEM"

    # Gemini model pins.
    route_model: str = "gemini-3-flash-preview"
    summary_model: str = "gemini-3-flash-preview"
    reviewer_model: str = "gemini-3-flash-preview"
    impact_model: str = "gemini-3-flash-preview"

    # Summary gate defaults.
    summary_gate_tokens: int = 1500
    reviewer_max_prompt_tokens: int = 2048
    summary_trigger_tokens: int = 1500
    summary_chunk_chars: int = 12000
    summary_max_output_tokens: int = 900

    # Router config.
    prefer_llm_router: bool = True
    llm_router_temperature: float = 0.0

    # Generation defaults.
    default_temperature: float = 0.2
    reviewer_max_output_tokens: int = 900
    impact_max_output_tokens: int = 700

    # Local reviewer LLM (adapter) config.
    use_local_reviewer: bool = True
    local_reviewer_fallback_to_gemini: bool = True
    reviewer_base_model_id: str = "unsloth/meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
    reviewer_max_seq_length: int = 2048
    reviewer_load_in_4bit: bool = True

    # Soft limits.
    max_script_chars: int = 180000

    # Drive/project defaults.
    project_root: str = "/content/drive/MyDrive/FUTURE_SYSTEM"

    # Adapter discovery (for future local reviewer integration).
    adapter_dir_candidates: Tuple[str, ...] = field(
        default_factory=lambda: (
            "/content/drive/MyDrive/FUTURE_SYSTEM/my_finetuned_llm",
            "/content/drive/MyDrive/FUTURE_SYSTEM/my_finetined_llm",
        )
    )

    # Keep your current classroom definition in one place.
    humanity_uplifting_definition: str = (
        "Humanity-uplifting means content that increases dignity, agency, empathy, "
        "inclusion, and constructive action while reducing harm, dehumanization, and "
        "exploitation."
    )
