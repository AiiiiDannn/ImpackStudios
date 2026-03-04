import inspect
import os
from typing import Any, Dict, List

import gradio as gr

from .config import AppConfig
from .drive_utils import find_adapter_dir, read_uploaded_text
from .impact import build_impact_prompt, format_impact_markdown
from .keys import looks_like_google_api_key, normalize_api_key
from .llm_clients import GeminiClient
from .local_reviewer import LocalReviewerEngine
from .reviewer import build_reviewer_prompt, format_reviewer_markdown, parse_reviewer_output
from .routing import choose_route
from .summary import estimate_tokens, needs_summary, summarize_long_text


def _extract_uploaded_path(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, list) and uploaded_file:
        return _extract_uploaded_path(uploaded_file[0])
    if isinstance(uploaded_file, dict):
        return uploaded_file.get("name", "") or uploaded_file.get("path", "")
    if hasattr(uploaded_file, "name"):
        return str(uploaded_file.name)
    return ""


def _combine_user_input(story_text: str, uploaded_text: str) -> str:
    story = (story_text or "").strip()
    file_body = (uploaded_text or "").strip()

    if story and file_body:
        return f"{story}\n\n[Uploaded File Content]\n{file_body}"
    return story or file_body


def _format_user_turn(story_text: str, uploaded_text: str) -> str:
    story = (story_text or "").strip()
    has_file = bool((uploaded_text or "").strip())

    if story and has_file:
        return story + "\n[Attachment included]"
    if story:
        return story
    if has_file:
        return "[Attachment only submission]"
    return ""


def _set_assistant_status(history: List[Dict[str, str]], text: str) -> List[Dict[str, str]]:
    next_history = list(history or [])
    msg = {"role": "assistant", "content": text}
    if next_history and next_history[-1].get("role") == "assistant":
        next_history[-1] = msg
    else:
        next_history.append(msg)
    return next_history


def _custom_css() -> str:
    # Keep close to v3 style shell while remaining maintainable.
    return """
:root {
    --bg-page: #cae0ee;
    --bg-shell: #eaf4fb;
    --bg-side: #d8e7f2;
    --bg-chat: linear-gradient(180deg, #0a1630 0%, #0b1f3f 100%);
    --bg-bot: rgba(14, 35, 67, 0.94);
    --bg-user: rgba(71, 87, 113, 0.9);
    --bg-composer: #f8fbfe;
    --text-main: #10233c;
    --text-muted: #4e637f;
    --text-light: #eaf2ff;
    --border-soft: rgba(136, 161, 194, 0.35);
    --brand: #0a5cad;
}

.gradio-container {
    background: radial-gradient(circle at 20% 0%, #d8ecf8 0%, var(--bg-page) 52%, #bfd8e8 100%) !important;
    color: var(--text-main) !important;
}

#main-layout {
    max-width: 1260px;
    margin: 0 auto;
    padding: 20px;
    gap: 14px;
    min-height: 100vh;
}

.app-sidebar {
    background: linear-gradient(180deg, #dcecf7 0%, var(--bg-side) 100%) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 18px !important;
    padding: 16px !important;
}

.brand-pill {
    display: inline-block;
    background: #f4f9ff;
    color: var(--brand);
    font-weight: 700;
    font-size: 26px;
    border: 1px solid #d6e6f5;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 14px;
}

.main-shell {
    background: var(--bg-shell) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 20px !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

.main-header {
    padding: 22px 24px 12px 24px;
    border-bottom: 1px solid rgba(129, 154, 185, 0.28);
}

.main-header h1 {
    margin: 0;
    color: var(--brand);
    font-size: 38px;
}

.main-header p {
    margin: 6px 0 0 0;
    color: var(--text-muted);
    font-size: 14px;
}

.chat-surface {
    padding: 14px;
}

.chat-surface > .column {
    background: var(--bg-chat) !important;
    border-radius: 14px !important;
    min-height: 56vh;
    max-height: 64vh;
    overflow: auto !important;
    padding: 14px !important;
}

.chat-stream .message {
    border-radius: 14px !important;
    padding: 12px 14px !important;
    line-height: 1.55 !important;
}

.chat-stream .message.user {
    background: var(--bg-user) !important;
    color: #eef5ff !important;
}

.chat-stream .message.bot,
.chat-stream .message.assistant {
    background: var(--bg-bot) !important;
    color: var(--text-light) !important;
}

.composer-wrap {
    border-top: 1px solid rgba(129, 154, 185, 0.25);
    padding: 14px;
    background: var(--bg-composer);
}

.composer-row {
    border: 1px solid #c7d8e8;
    border-radius: 16px;
    padding: 8px;
    background: #ffffff;
    align-items: center !important;
    gap: 8px !important;
}

.file-status,
.file-status p,
.file-status span {
    margin: 6px 0 0 2px !important;
    color: #2f4765 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}
"""


def build_interface(config: AppConfig) -> gr.Blocks:
    # Key comes from launch_colab.ipynb Step 4; UI should never ask for key again.
    env_raw_key = os.getenv("GOOGLE_API_KEY", "")
    env_key = normalize_api_key(env_raw_key)
    key_ready = looks_like_google_api_key(env_key)
    key_status = (
        f"GOOGLE_API_KEY detected (`{env_key[:8]}...`, len={len(env_key)})"
        if key_ready
        else "GOOGLE_API_KEY missing/invalid in environment. Run Step 4 in launch_colab.ipynb."
    )
    hf_token = os.getenv("HF_TOKEN", "").strip()
    hf_status = (
        f"HF_TOKEN detected (`{hf_token[:5]}...`, len={len(hf_token)})"
        if hf_token
        else "HF_TOKEN missing. Required for local reviewer base model loading."
    )
    try:
        import torch  # type: ignore
        gpu_ok = bool(torch.cuda.is_available())
    except Exception:
        gpu_ok = False
    gpu_status = (
        "GPU detected (`torch.cuda.is_available()=True`)."
        if gpu_ok
        else "GPU not detected (`torch.cuda.is_available()=False`). Local reviewer will fallback to Gemini."
    )
    adapter_dir_preview, _ = find_adapter_dir(list(config.adapter_dir_candidates))
    adapter_status = (
        f"Adapter detected at `{adapter_dir_preview}`"
        if adapter_dir_preview
        else "Adapter not found under FUTURE_SYSTEM/my_finetuned_llm."
    )

    client_holder: Dict[str, Any] = {"client": None}
    local_reviewer_holder: Dict[str, Any] = {"engine": None}

    def get_client() -> GeminiClient:
        if not key_ready:
            raise RuntimeError("GOOGLE_API_KEY not ready. Run Step 4 in launch_colab.ipynb first.")
        if client_holder["client"] is None:
            client_holder["client"] = GeminiClient(api_key=env_key)
        return client_holder["client"]

    def get_local_reviewer() -> LocalReviewerEngine:
        if local_reviewer_holder["engine"] is None:
            local_reviewer_holder["engine"] = LocalReviewerEngine(config)
        return local_reviewer_holder["engine"]

    def on_upload(uploaded_file):
        file_path = _extract_uploaded_path(uploaded_file)
        if not file_path:
            print("[STEP 10.1] Upload cleared or empty.")
            return "", "No file attached.", ""

        file_text = read_uploaded_text(file_path)
        filename = os.path.basename(file_path)
        print(f"[STEP 10.2] Uploaded file detected: {filename}")
        if not file_text.strip():
            print("[STEP 10.3 WARN] Uploaded file text extraction returned empty.")
            return "", f"Attached: {filename} (text extraction empty)", file_path
        print(f"[STEP 10.4] Uploaded text chars={len(file_text)}")
        return file_text, f"Attached: {filename}", file_path

    def clear_uploaded_state():
        print("[STEP 10.5] Attached file state cleared.")
        return "", "No file attached.", ""

    def stage_user_message(story_text: str, mission_text: str, uploaded_text: str, uploaded_path: str, chat_history):
        merged_submission = _combine_user_input(story_text, uploaded_text)
        if not merged_submission.strip():
            print("[STEP 11.0 WARN] Empty submission at stage_user_message.")
            history = list(chat_history or [])
            return history, "", history, {}

        print("[STEP 11.1] User submission staged for processing.")
        history = list(chat_history or [])
        history.append({"role": "user", "content": _format_user_turn(story_text, uploaded_text)})
        history.append({"role": "assistant", "content": "Analyzing your submission..."})

        payload = {
            "story_text": story_text or "",
            "mission_text": mission_text or "",
            "uploaded_text": uploaded_text or "",
            "uploaded_path": uploaded_path or "",
        }

        return history, "", history, payload

    def resolve_assistant_message(chat_history, pending_payload):
        history = list(chat_history or [])
        payload = pending_payload or {}

        if not payload:
            print("[STEP 11.1] No pending payload.")
            yield history, history, {}
            return

        story_text = payload.get("story_text", "")
        mission_text = payload.get("mission_text", "")
        uploaded_text = payload.get("uploaded_text", "")
        uploaded_path = payload.get("uploaded_path", "")

        merged_submission = _combine_user_input(story_text, uploaded_text)
        route_basis = (story_text or "").strip() or merged_submission[:1200]

        if not route_basis:
            msg = "[STEP 11.2] Please enter text or upload a file before submitting."
            print(msg)
            history = _set_assistant_status(history, msg)
            yield history, history, {}
            return

        msg = "[STEP 11.3] Routing request to the best agent..."
        print(msg)
        history = _set_assistant_status(history, msg)
        yield history, history, payload

        client = None
        if key_ready:
            try:
                client = get_client()
            except Exception:
                client = None

        routed_agent, route_mode = choose_route(
            route_basis,
            prefer_llm=config.prefer_llm_router and client is not None,
            client=client,
            model=config.route_model,
        )
        msg = f"[STEP 11.4] Routed to `{routed_agent}` via `{route_mode}`."
        print(msg)
        history = _set_assistant_status(history, msg)
        yield history, history, payload

        if routed_agent == "impact agent":
            msg = "[STEP 11.5] Running Impact Agent..."
            print(msg)
            history = _set_assistant_status(history, msg)
            yield history, history, payload

            if client is None:
                msg = "[STEP 11.5 WARN] Impact Agent needs valid GOOGLE_API_KEY in environment."
                print(msg)
                history = _set_assistant_status(history, msg)
                yield history, history, {}
                return

            impact_prompt = build_impact_prompt(
                user_prompt=(story_text or "").strip() or "[No user prompt provided]",
                script_text=merged_submission,
                humanity_def=config.humanity_uplifting_definition,
            )
            try:
                raw_impact = client.generate(
                    prompt=impact_prompt,
                    model=config.impact_model,
                    temperature=config.default_temperature,
                    max_output_tokens=config.impact_max_output_tokens,
                )
                final_answer = format_impact_markdown(raw_impact, route_mode)
            except Exception as ex:
                final_answer = f"Impact call failed: {type(ex).__name__}: {ex}"

            print("[STEP 11.6] Impact Agent completed.")
            history = _set_assistant_status(history, final_answer)
            yield history, history, {}
            return

        # Script Reviewer path with summary gate
        msg = "[STEP 11.5] Preparing Script Reviewer input..."
        print(msg)
        history = _set_assistant_status(history, msg)
        yield history, history, payload

        candidate_story = merged_submission
        initial_prompt_tokens = estimate_tokens(
            build_reviewer_prompt(candidate_story, config.humanity_uplifting_definition, mission_text or "None")
        )
        msg = f"[STEP 11.6] Input length check: ~{initial_prompt_tokens} tokens."
        print(msg)
        history = _set_assistant_status(history, msg)
        yield history, history, payload

        summary_status = f"`not applied` | prompt_tokens≈`{initial_prompt_tokens}`"
        summary_needed = (
            initial_prompt_tokens > config.reviewer_max_prompt_tokens
            or needs_summary(candidate_story, config.summary_trigger_tokens)
        )

        if summary_needed:
            msg = "[STEP 11.7] Long script detected. Running Gemini summary gate..."
            print(msg)
            history = _set_assistant_status(history, msg)
            yield history, history, payload

            if client is None:
                summary_status = (
                    f"`skipped` | prompt_tokens≈`{initial_prompt_tokens}`"
                    "\n\n**Summary Note:** GOOGLE_API_KEY missing; summary gate not available."
                )
            else:
                try:
                    filename_hint = os.path.basename(uploaded_path) if uploaded_path else "submission"
                    summary_text = summarize_long_text(
                        text=candidate_story,
                        client=client,
                        model=config.summary_model,
                        chunk_chars=config.summary_chunk_chars,
                        progress_cb=lambda m: None,
                    )
                    candidate_story = summary_text
                    post_tokens = estimate_tokens(
                        build_reviewer_prompt(
                            candidate_story,
                            config.humanity_uplifting_definition,
                            mission_text or "None",
                        )
                    )
                    summary_status = (
                        f"`applied` | model=`{config.summary_model}` | prompt_tokens≈`{post_tokens}`"
                    )
                    print(f"[STEP 11.8] Summary gate applied. prompt_tokens≈{post_tokens}")
                except Exception as ex:
                    summary_status = (
                        f"`failed` | model=`{config.summary_model}` | prompt_tokens≈`{initial_prompt_tokens}`"
                        f"\n\n**Summary Note:** {type(ex).__name__}: {ex}"
                    )
                    print(f"[STEP 11.8 WARN] Summary gate failed: {type(ex).__name__}: {ex}")

        else:
            msg = "[STEP 11.7] Summary gate not needed. Proceeding with original script."
            print(msg)
            history = _set_assistant_status(history, msg)
            yield history, history, payload

        msg = "[STEP 11.9] Running Script Reviewer model inference..."
        print(msg)
        history = _set_assistant_status(history, msg)
        yield history, history, payload

        local_engine = None
        if config.use_local_reviewer:
            msg = "[STEP 11.10] Initializing local Script Reviewer model..."
            print(msg)
            history = _set_assistant_status(history, msg)
            yield history, history, payload
            try:
                local_engine = get_local_reviewer()
                local_engine.ensure_loaded()
                st = local_engine.status()
                history = _set_assistant_status(
                    history,
                    "Local Script Reviewer ready "
                    f"(device=`{st.get('device', 'unknown')}`, adapter=`{st.get('adapter_dir', '')}`).",
                )
                yield history, history, payload
            except Exception as ex:
                if config.local_reviewer_fallback_to_gemini and client is not None:
                    print(f"[STEP 11.10 WARN] Local reviewer load failed, fallback to Gemini: {type(ex).__name__}: {ex}")
                    history = _set_assistant_status(
                        history,
                        "Local reviewer load failed. Falling back to Gemini reviewer.\n"
                        f"Reason: {type(ex).__name__}: {ex}",
                    )
                    yield history, history, payload
                    local_engine = None
                else:
                    print(f"[STEP 11.10 ERROR] Local reviewer load failed: {type(ex).__name__}: {ex}")
                    history = _set_assistant_status(
                        history,
                        "Local Script Reviewer failed to load.\n"
                        f"Reason: {type(ex).__name__}: {ex}\n\n"
                        "Check adapter files and HF_TOKEN in launch_colab.ipynb, then relaunch UI.",
                    )
                    yield history, history, {}
                    return

        reviewer_prompt = build_reviewer_prompt(
            candidate_story,
            config.humanity_uplifting_definition,
            mission_text or "None",
        )
        try:
            if local_engine is not None:
                raw_review = local_engine.generate(reviewer_prompt)
            elif client is not None:
                raw_review = client.generate(
                    prompt=reviewer_prompt,
                    model=config.reviewer_model,
                    temperature=config.default_temperature,
                    max_output_tokens=config.reviewer_max_output_tokens,
                )
            else:
                raise RuntimeError(
                    "No reviewer backend available. "
                    "Set HF_TOKEN for local reviewer (preferred) or GOOGLE_API_KEY for fallback."
                )
            parsed = parse_reviewer_output(raw_review)
            final_answer = format_reviewer_markdown(parsed, route_mode, summary_status)

            quality = int(parsed.get("quality", 0))
            if quality < 8:
                debug_raw = str(parsed.get("raw") or raw_review)[:2400]
                final_answer += f"\n\n### Raw Model Output (debug)\n```text\n{debug_raw}\n```"
        except Exception as ex:
            final_answer = f"Reviewer call failed: {type(ex).__name__}: {ex}"

        print("[STEP 11.11] Script Reviewer completed.")
        history = _set_assistant_status(history, final_answer)
        yield history, history, {}

    def clear_chat_state():
        return [], "", [], {}, "", "", "No file attached."

    with gr.Blocks(css=_custom_css(), title=config.app_title, fill_height=True) as demo:
        uploaded_text_state = gr.State("")
        uploaded_path_state = gr.State("")
        chat_state = gr.State([])
        pending_payload_state = gr.State({})

        with gr.Row(elem_id="main-layout"):
            with gr.Column(scale=0, min_width=260, elem_classes="app-sidebar"):
                gr.HTML('<div class="brand-pill">Impact Studios</div>')
                new_chat_btn = gr.Button("New chat")
                search_btn = gr.Button("Search chat")
                add_agent_btn = gr.Button("Add agent")
                gr.Markdown(f"**Runtime Key Status:** {key_status}")
                gr.Markdown(f"**HF Status:** {hf_status}")
                gr.Markdown(f"**GPU Status:** {gpu_status}")
                gr.Markdown(f"**Adapter Status:** {adapter_status}")

                mission_box = gr.Textbox(
                    label="Studio Mission (optional)",
                    placeholder="Add mission context for this run...",
                    lines=6,
                )

            with gr.Column(scale=1, elem_classes="main-shell"):
                gr.HTML(
                    """
                    <div class=\"main-header\">
                        <h1>Impact Studios</h1>
                        <p>Unified: Orchestrator + Script Reviewer + Impact Agent</p>
                    </div>
                    """
                )

                with gr.Column(elem_classes="chat-surface"):
                    chatbot_sig = set(inspect.signature(gr.Chatbot.__init__).parameters.keys())
                    chatbot_kwargs = {
                        "value": [],
                        "show_label": False,
                        "container": False,
                        "bubble_full_width": False,
                        "elem_classes": "chat-stream",
                    }
                    if "type" in chatbot_sig:
                        chatbot_kwargs["type"] = "messages"
                    elif "message_type" in chatbot_sig:
                        chatbot_kwargs["message_type"] = "messages"
                    chatbot = gr.Chatbot(**{k: v for k, v in chatbot_kwargs.items() if k in chatbot_sig})

                with gr.Column(elem_classes="composer-wrap"):
                    with gr.Row(elem_classes="composer-row", equal_height=True):
                        upload_btn = gr.UploadButton(
                            "Attach",
                            file_types=[".txt", ".pdf", ".docx", ".doc", ".md"],
                            file_count="single",
                            scale=0,
                        )
                        story = gr.Textbox(
                            label="",
                            placeholder="Ask anything",
                            lines=1,
                            max_lines=5,
                            show_label=False,
                            container=False,
                            scale=1,
                        )
                        submit_btn = gr.Button("Send", scale=0)

                    file_status = gr.Markdown("No file attached.", elem_classes="file-status")
                    clear_file_btn = gr.Button("Clear attached file")

        upload_btn.upload(
            fn=on_upload,
            inputs=upload_btn,
            outputs=[uploaded_text_state, file_status, uploaded_path_state],
            show_progress="hidden",
        )

        clear_file_btn.click(
            fn=clear_uploaded_state,
            inputs=None,
            outputs=[uploaded_text_state, file_status, uploaded_path_state],
            show_progress="hidden",
        )

        submit_event = submit_btn.click(
            fn=stage_user_message,
            inputs=[story, mission_box, uploaded_text_state, uploaded_path_state, chat_state],
            outputs=[chatbot, story, chat_state, pending_payload_state],
            show_progress="hidden",
        )
        submit_event.then(
            fn=resolve_assistant_message,
            inputs=[chat_state, pending_payload_state],
            outputs=[chatbot, chat_state, pending_payload_state],
            show_progress="hidden",
        )

        enter_event = story.submit(
            fn=stage_user_message,
            inputs=[story, mission_box, uploaded_text_state, uploaded_path_state, chat_state],
            outputs=[chatbot, story, chat_state, pending_payload_state],
            show_progress="hidden",
        )
        enter_event.then(
            fn=resolve_assistant_message,
            inputs=[chat_state, pending_payload_state],
            outputs=[chatbot, chat_state, pending_payload_state],
            show_progress="hidden",
        )

        new_chat_btn.click(
            fn=clear_chat_state,
            inputs=None,
            outputs=[
                chatbot,
                story,
                chat_state,
                pending_payload_state,
                uploaded_text_state,
                uploaded_path_state,
                file_status,
            ],
            show_progress="hidden",
        )

        # Keep layout parity with v3; these sidebar buttons are intentionally visual-only here.
        _ = search_btn
        _ = add_agent_btn

    return demo
