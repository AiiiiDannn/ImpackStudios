import os
import threading
from pathlib import Path

import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from peft import AutoPeftModelForCausalLM, PeftModel


ROOT_DIR = Path(__file__).parent


@st.cache_resource(show_spinner=True)
def load_tokenizer_and_model(backend: str, cpu_base_model: str | None = None):
    # Load local tokenizer (uses local tokenizer.json and chat_template.jinja)
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT_DIR.as_posix(),
        trust_remote_code=True,
        local_files_only=True,
    )

    # Load PEFT adapter; auto-loads base model from adapter_config.json
    # Device handling
    has_cuda = torch.cuda.is_available()

    # Backend routing
    dtype = torch.bfloat16 if has_cuda and torch.cuda.is_bf16_supported() else (
        torch.float16 if has_cuda else torch.float32
    )

    try:
        if backend == "Adapter (GPU)":
            if not has_cuda:
                raise RuntimeError("CUDA GPU required for Adapter (GPU) backend")
            # GPU path: load adapter with device_map auto
            model_kwargs = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
                "device_map": "auto",
            }
            model = AutoPeftModelForCausalLM.from_pretrained(
                ROOT_DIR.as_posix(),
                **model_kwargs,
            )
            tok = AutoTokenizer.from_pretrained(
                ROOT_DIR.as_posix(), trust_remote_code=True, local_files_only=True
            )
            tokenizer = tok
        elif backend == "Adapter + CPU base":
            if not cpu_base_model:
                raise RuntimeError(
                    "CPU base model repo id is required for 'Adapter + CPU base' backend."
                )
            # CPU path: load full‑precision base and attach adapter
            base = AutoModelForCausalLM.from_pretrained(
                cpu_base_model,
                torch_dtype=torch.float32,
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(
                base,
                ROOT_DIR.as_posix(),
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                cpu_base_model, trust_remote_code=True
            )
        elif backend == "Open demo (distilgpt2)":
            # Fully open, small CPU model for demoing the UI
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                torch_dtype=torch.float32,
            )
        elif backend == "Dummy echo":
            # No model required
            tokenizer = None  # type: ignore
            model = None  # type: ignore
        else:
            raise RuntimeError(f"Unknown backend: {backend}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

    if model is not None:
        model.eval()
    return tokenizer, model


def format_prompt(tokenizer, messages):
    # messages: list of {"role": "system"|"user"|"assistant", "content": str}
    # Use the model's chat template
    has_chat_template_api = hasattr(tokenizer, "apply_chat_template")
    template_is_set = getattr(tokenizer, "chat_template", None)
    if has_chat_template_api and template_is_set:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # Fallback: simple role-tagged concatenation
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}] {m['content']}")
        parts.append("[assistant]")
        prompt = "\n".join(parts)
    return prompt


def stream_generate(tokenizer, model, messages, max_new_tokens, temperature, top_p, repetition_penalty):
    # Dummy mode
    if model is None or tokenizer is None:
        # Stream back a simple echo with a typing effect
        text = messages[-1]["content"]
        def iterator():
            prefix = "Echo: "
            for ch in prefix + text:
                yield ch
        # Wrap iterator to look like TextIteratorStreamer
        return iterator(), None

    prompt = format_prompt(tokenizer, messages)
    inputs = tokenizer([prompt], return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer, thread


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role, content}]


def main():
    st.set_page_config(page_title="FineTuned LLM Chat", page_icon="🤖", layout="centered")
    st.title("🤖 Fine‑Tuned LLM Chat")

    has_cuda = torch.cuda.is_available()

    with st.sidebar:
        st.subheader("Generation Settings")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        top_p = st.slider("Top‑p", 0.1, 1.0, 0.95, 0.05)
        repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.1)
        max_new_tokens = st.slider("Max New Tokens", 16, 2048, 512, 16)

        backend = st.selectbox(
            "Backend",
            [
                "Adapter (GPU)",
                "Adapter + CPU base",
                "Open demo (distilgpt2)",
                "Dummy echo",
            ],
            index=2 if not has_cuda else 0,
            help="Use demo or dummy to showcase UI without gated models",
        )

        cpu_base_model = None
        if backend == "Adapter + CPU base":
            st.info(
                "Provide a full‑precision base model repo ID to run the adapter on CPU."
            )
            cpu_base_model = st.text_input(
                "CPU base model repo id",
                value="Qwen/Qwen2.5-1.5B-Instruct",
                help="Use a non‑gated, full‑precision instruct model if possible.",
            )

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    tokenizer, model = load_tokenizer_and_model(backend, cpu_base_model)
    init_session_state()

    # Render existing messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])  # content is plain text

    user_input = st.chat_input("Type your message…")
    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Placeholder for assistant stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            partial_text = ""

            streamer, thread = stream_generate(
                tokenizer,
                model,
                st.session_state.messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            for token_text in streamer:
                partial_text += token_text
                placeholder.markdown(partial_text)

            if thread is not None:
                thread.join()

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": partial_text})


if __name__ == "__main__":
    main()