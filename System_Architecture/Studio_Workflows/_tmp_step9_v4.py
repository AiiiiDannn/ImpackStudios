# ============================================================
# STEP 9: Inference Function (copied from usemodelkaggle, non-Gradio)
# ============================================================
print("[STEP 9] Defining evaluator...")

import io

tok = tokenizer
mdl = model
mdl.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
gen_lock = Lock()
history = []

try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass

# Reviewer output cleanup switch.
# True: trim trailing dash-noise/artifacts in parsed text.
# False: keep full untrimmed text for inspection.
REVIEW_TRUNCATE_NOISE = True

_DRIVE_API = None


def _init_drive_api():
    global _DRIVE_API
    if _DRIVE_API is not None:
        return _DRIVE_API

    try:
        from google.colab import auth
        auth.authenticate_user()
    except Exception:
        # Non-Colab env may still have ADC credentials.
        pass

    try:
        import google.auth
        from googleapiclient.discovery import build
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
        _DRIVE_API = build("drive", "v3", credentials=creds, cache_discovery=False)
        return _DRIVE_API
    except Exception as e:
        print(f"[STEP 9 WARN] Drive API init failed: {e}")
        return None


def _extract_drive_id_from_url(url: str) -> str:
    if not url:
        return ""
    patterns = [
        r"/d/e/([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return ""


def _read_google_shortcut_meta(path: str):
    """
    Read .gdoc/.gsheet/.gslides shortcut metadata.
    Supports JSON shortcut files and INI-style URL shortcuts.
    Returns (file_id, url).
    """
    url = ""
    file_id = ""

    raw = ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except Exception:
        return "", ""

    # Try JSON first
    try:
        meta = json.loads(raw)
        if isinstance(meta, dict):
            url = str(meta.get("url", "") or "")
            file_id = str(meta.get("doc_id", "") or "")

            resource_id = str(meta.get("resource_id", "") or "")
            if not file_id and resource_id:
                # examples: "document/FILE_ID" or "document:FILE_ID"
                if "/" in resource_id:
                    file_id = resource_id.split("/")[-1]
                elif ":" in resource_id:
                    file_id = resource_id.split(":")[-1]

            if not file_id:
                file_id = _extract_drive_id_from_url(url)

            if file_id:
                return file_id, url
    except Exception:
        pass

    # Fallback for non-JSON shortcut content
    # e.g. [InternetShortcut] URL=https://docs.google.com/document/d/...
    m = re.search(r"(?im)^URL\s*=\s*(https?://\S+)", raw)
    if m:
        url = m.group(1).strip()
    else:
        m2 = re.search(r"https?://\S+", raw)
        if m2:
            url = m2.group(0).strip()

    if not file_id and url:
        file_id = _extract_drive_id_from_url(url)

    return file_id, url


def _google_mime_for_shortcut_ext(ext: str) -> str:
    ext = ext.lower()
    mapping = {
        ".gdoc": "application/vnd.google-apps.document",
        ".gsheet": "application/vnd.google-apps.spreadsheet",
        ".gslides": "application/vnd.google-apps.presentation",
        ".gdraw": "application/vnd.google-apps.drawing",
        ".gform": "application/vnd.google-apps.form",
        ".gsite": "application/vnd.google-apps.site",
    }
    return mapping.get(ext, "")


def _lookup_google_file_id_by_name(local_path: str):
    """
    Fallback when .g* shortcut metadata has no file id.
    Lookup by base filename in user's Drive.
    Returns (file_id, webViewLink).
    """
    svc = _init_drive_api()
    if svc is None:
        return "", ""

    stem = os.path.splitext(os.path.basename(local_path))[0]
    ext = os.path.splitext(local_path)[1].lower()
    mime = _google_mime_for_shortcut_ext(ext)

    safe_name = stem.replace("'", "\'")
    q = f"name = '{safe_name}' and trashed = false"
    if mime:
        q += f" and mimeType = '{mime}'"

    try:
        resp = svc.files().list(
            q=q,
            corpora="user",
            pageSize=10,
            fields="files(id,name,mimeType,webViewLink)",
        ).execute()
        files = resp.get("files", [])
        if files:
            return files[0].get("id", ""), files[0].get("webViewLink", "")
    except Exception:
        pass

    # broad fallback: drop mime filter
    try:
        resp = svc.files().list(
            q=f"name = '{safe_name}' and trashed = false",
            corpora="user",
            pageSize=10,
            fields="files(id,name,mimeType,webViewLink)",
        ).execute()
        files = resp.get("files", [])
        if files:
            return files[0].get("id", ""), files[0].get("webViewLink", "")
    except Exception:
        pass

    return "", ""


def _preferred_export_mimes(ext: str):
    ext = ext.lower()
    if ext == ".gsheet":
        return ["text/csv", "application/pdf", "text/plain"]
    if ext == ".gslides":
        return ["text/plain", "application/pdf"]
    if ext == ".gdraw":
        return ["application/pdf", "image/svg+xml", "text/plain"]
    if ext == ".gdoc":
        return ["text/plain", "application/pdf"]
    # generic .g* fallback
    return ["text/plain", "application/pdf", "text/csv"]


def _decode_export_bytes(data, mime_type: str):
    if isinstance(data, str):
        return data

    if mime_type == "application/pdf":
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n".join([page.extract_text() or "" for page in reader.pages])

    for enc in ("utf-8", "latin-1"):
        try:
            return data.decode(enc, errors="ignore")
        except Exception:
            pass
    return str(data)


def read_google_workspace_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if not ext.startswith('.g'):
        return ""

    file_id, web_url = _read_google_shortcut_meta(path)

    # fallback by Drive lookup if shortcut meta didn't include file id
    if not file_id:
        looked_id, looked_url = _lookup_google_file_id_by_name(path)
        if looked_id:
            file_id = looked_id
            if not web_url:
                web_url = looked_url

    if not file_id:
        return f"(Google shortcut found but file id not found: {path})"

    svc = _init_drive_api()
    if svc is None:
        return f"(Drive API not available for reading {path})"

    last_err = None
    for mime in _preferred_export_mimes(ext):
        try:
            payload = svc.files().export(fileId=file_id, mimeType=mime).execute()
            text = _decode_export_bytes(payload, mime)
            if text and text.strip():
                return text
        except Exception as e:
            last_err = e

    # metadata fallback
    try:
        meta = svc.files().get(fileId=file_id, fields="name,mimeType,webViewLink").execute()
        return (
            f"(Could not export content. name={meta.get('name')} mime={meta.get('mimeType')} "
            f"link={meta.get('webViewLink', web_url)})"
        )
    except Exception:
        return f"(Could not export Google file content. Last error: {last_err})"


def _clean_model_output(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"(?is)^.*?###\s*Response:\s*", "", text).strip()

    # Drop obvious repeated consecutive lines produced by sampling loops.
    compact_lines = []
    prev_norm = ""
    repeat_count = 0
    for ln in text.split("\n"):
        norm = re.sub(r"\s+", " ", ln).strip().lower()
        if not norm:
            if compact_lines and compact_lines[-1] != "":
                compact_lines.append("")
            prev_norm = ""
            repeat_count = 0
            continue

        if norm == prev_norm:
            repeat_count += 1
            if repeat_count >= 1:
                continue
        else:
            repeat_count = 0

        compact_lines.append(ln.rstrip())
        prev_norm = norm

    cleaned = "\n".join(compact_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _truncate_reviewer_noise(text: str) -> str:
    src = (text or "").strip()
    if not src or not REVIEW_TRUNCATE_NOISE:
        return src

    # Pattern seen in model tails: repeated long dash groups like "--- --- ---".
    m = re.search(r"(?:\s[-]{2,}\s*){3,}", src)
    if m:
        return src[:m.start()].rstrip()
    return src


def _extract_first_review_block(text: str) -> str:
    data = (text or "").strip()
    if not data:
        return ""

    tagged = re.search(r"(?is)\[BEGIN_REVIEW\](.*?)\[END_REVIEW\]", data)
    if tagged:
        data = tagged.group(1).strip()
    else:
        contamination_markers = [
            r"\nWait\s+for\s+further\s+instructions",
            r"\n###\s*Input\s*:",
            r"\n###\s*Instruction\s*:",
            r"\nIs\s+this\s+uplifting\?",
            r"\nInput\s*:",
        ]
        cut = len(data)
        for pat in contamination_markers:
            m = re.search(pat, data, re.I)
            if m:
                cut = min(cut, m.start())
        data = data[:cut].strip()

    data = _truncate_reviewer_noise(data)

    # Keep only up to the first complete overall rationale paragraph if possible.
    m_overall = re.search(r"(?is)(?:^|\n)\s*(?:Overall\s*rationale|Overall|Rationale)\s*[:\-]\s*(.+)", data)
    if m_overall:
        prefix = data[:m_overall.start()]
        tail = _truncate_reviewer_noise(m_overall.group(1).strip())
        stop = re.search(r"\n\s*\n", tail)
        first_para = tail[:stop.start()].strip() if stop else tail.strip()
        data = (prefix + "\nOverall rationale: " + first_para).strip()

    return data


def _extract_section(text: str, labels, next_labels) -> str:
    label_pat = "|".join(re.escape(x) for x in labels)
    next_pat = "|".join(re.escape(x) for x in next_labels) if next_labels else ""

    if next_pat:
        pattern = (
            rf"(?ims)(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{label_pat})\s*[:\-]\s*"
            rf"(.+?)(?=\n\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{next_pat})\s*[:\-]|\Z)"
        )
    else:
        pattern = (
            rf"(?ims)(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{label_pat})\s*[:\-]\s*(.+)$"
        )

    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _dedupe_section_lines(block: str) -> str:
    if not block:
        return ""
    out = []
    seen = set()
    for ln in block.splitlines():
        norm = re.sub(r"[^a-z0-9]+", " ", ln.lower()).strip()
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        out.append(ln.rstrip())
    return "\n".join(out).strip()


def _parse_score_to_int(score_text: str, fallback_text: str) -> int:
    cand = (score_text or "").strip()

    m = re.search(r"([1-5](?:\.\d+)?)", cand)
    if not m:
        m = re.search(r"([1-5](?:\.\d+)?)\s*(?:/|out of)\s*5", fallback_text or "", re.I)
    if not m:
        return 1

    try:
        val = float(m.group(1))
        return max(1, min(5, int(round(val))))
    except Exception:
        return 1


def _normalize_verdict(verdict_text: str) -> str:
    text = (verdict_text or "").strip()
    m = re.search(r"\b(yes|no)\b", text, re.I)
    return m.group(1).capitalize() if m else ""


def _parsed_quality(parsed: dict) -> int:
    quality = 0
    for key in ["verdict", "score", "benefits", "risks", "overall"]:
        val = (parsed.get(key) or "").strip()
        if val:
            quality += 1
            if len(val) >= 40:
                quality += 1
    return quality


def format_output(raw):
    text = _extract_first_review_block(_clean_model_output(raw))

    verdict = _extract_section(
        text,
        ["Verdict", "Final Verdict"],
        ["Score", "Rating", "Benefits", "Risks", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    if not verdict:
        m_v = re.search(r"(?im)\bIs\s+this\s+uplifting\?\s*(yes|no)\b", text)
        if not m_v:
            m_v = re.search(r"(?im)^\s*(yes|no)\s*$", text)
        if m_v:
            verdict = m_v.group(1).capitalize()

    score = _extract_section(
        text,
        ["Score", "Rating"],
        ["Benefits", "Risks", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    benefits = _extract_section(
        text,
        ["Benefits", "Strengths", "Upsides", "Positive impacts"],
        ["Risks", "Concerns", "Downsides", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    risks = _extract_section(
        text,
        ["Risks", "Concerns", "Downsides", "Negative impacts"],
        ["Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    overall = _extract_section(
        text,
        ["Overall rationale", "Overall", "Rationale", "Analysis", "Final analysis"],
        [],
    )

    # Fallback if label extraction fails but text exists.
    if not overall and text:
        overall = text

    verdict = verdict.replace("**", "").strip()
    score = score.replace("**", "").strip()
    benefits = _dedupe_section_lines(benefits)
    risks = _dedupe_section_lines(risks)
    overall = _dedupe_section_lines(overall)

    return {
        "verdict": verdict,
        "score": score,
        "benefits": benefits,
        "risks": risks,
        "overall": overall,
        "raw": text,
    }






# ===== Script Reviewer summary gate (long-input protection) =====
SCRIPT_SUMMARY_TRIGGER_TOKENS = 2048
SCRIPT_REVIEWER_MAX_PROMPT_TOKENS = 2048
SCRIPT_SUMMARY_TARGETS = ["900-1200", "500-700", "300-450"]
SCRIPT_SUMMARY_MODEL = os.getenv("SCRIPT_SUMMARY_MODEL", "gemini-3-flash-preview")
SCRIPT_SUMMARY_MODEL_FALLBACK = os.getenv("SCRIPT_SUMMARY_MODEL_FALLBACK", "gemini-2.5-flash-lite")

# Cache rejected key to avoid repeated API_KEY_INVALID spam in a single runtime.
SUMMARY_GATE_REJECTED_KEY = None


def _current_google_api_key() -> str:
    return (globals().get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def _is_google_api_key_invalid_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "api_key_invalid" in msg
        or "api key not valid" in msg
        or "invalid api key" in msg
        or ("invalid_argument" in msg and "api key" in msg)
    )


def _count_local_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0

    try:
        enc = tok(text, add_special_tokens=False)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)
    except Exception:
        try:
            return len(tok.encode(text))
        except Exception:
            return max(1, len(text) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    try:
        enc = tok(text, add_special_tokens=False)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if len(ids) <= max_tokens:
            return text
        return tok.decode(ids[:max_tokens], skip_special_tokens=True)
    except Exception:
        return text[: max_tokens * 4]


def _summary_system_instruction(target_token_range: str) -> str:
    return f"""
Role: You are a Structural Script Analyst and Narrative Architect.
Task: Compress the provided raw script treatment into a "Uplift-Ready Structural Summary."
Constraints:

Length: Target {target_token_range} tokens.
Preserve: Core conflict, protagonist's internal and external arcs, emotional peaks, and moral/thematic undercurrents.
Remove: Detailed dialogue, minor characters with no plot impact, specific scene descriptions, and fluff.
Specific Focus (The "Mission Keeper" Lens):

Highlight the Protagonist's Agency: How they drive the plot through choices.
Identify Heroine's Journey markers: Internal awakening and reclaiming of power.
Emphasize Moral Dilemmas: Points where the "Humanity Uplifting" principles are tested.
Output Structure (Markdown Headers):

## Logline: A one-sentence summary.
## Thematic Core: The central moral question or "Uplift" theme.
## Character Profile: The protagonist's initial state vs. final state (Arc).
## Structural Breakdown: Key beats (Inciting Incident, Midpoint, Climax, Resolution) with emphasis on emotional transitions.
## Social Context: Any racial, cultural, or gender-specific dynamics relevant to "Dignity & Inclusion."
""".strip()


def _build_reviewer_prompt(story_body: str, ctx: str) -> str:
    template = """Evaluate the submission for humanity-uplifting value.

Return only this format:
Verdict: <Yes or No>
Score: <1 (out of 5) ... 5 (out of 5)>
Benefits:
- <2 to 5 concise bullets, not strictly required to reach 5 bullets>
Risks:
- <2 to 5 concise bullets, not strictly required to reach 5 bullets>
Overall rationale: <2 to 4 sentences>

Consistency rule:
- Score 1-2 => Verdict No
- Score 3-5 => Verdict Yes

Submission:
{story}

Additional Context:
{context}
"""
    return template.format(
        story=(story_body or "").strip(),
        context=(ctx or "").strip() if ctx else "None",
    )


def _summarize_script_for_reviewer(raw_script: str, filename: str, target_token_range: str):
    global SUMMARY_GATE_REJECTED_KEY

    key = _current_google_api_key()
    if not key:
        return "", "", "Missing GOOGLE_API_KEY for summary gate"

    if SUMMARY_GATE_REJECTED_KEY and key == SUMMARY_GATE_REJECTED_KEY:
        return "", "", "GOOGLE_API_KEY previously rejected (API_KEY_INVALID). Summary gate skipped until key changes."

    prompt = f"Here is the raw script treatment for '{filename}'. Please summarize it:\n\n{raw_script}"

    candidates = []
    for model_name in [
        SCRIPT_SUMMARY_MODEL,
        SCRIPT_SUMMARY_MODEL_FALLBACK,
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
    ]:
        if model_name and model_name not in candidates:
            candidates.append(model_name)

    client = genai.Client(api_key=key)
    last_err = None

    for model_name in candidates:
        try:
            full_prompt = _summary_system_instruction(target_token_range) + "\n\n---\n\n" + prompt
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
            )
            text = (getattr(response, "text", "") or "").strip()
            if text:
                return text, model_name, ""
        except Exception as e:
            last_err = e
            if _is_google_api_key_invalid_error(e):
                SUMMARY_GATE_REJECTED_KEY = key
                msg = "GOOGLE_API_KEY rejected by Gemini (API_KEY_INVALID). Summary gate disabled until key is updated."
                print(f"[STEP 9 WARN] {msg}")
                return "", "", msg
            print(f"[STEP 9 WARN] Summary call failed on {model_name}: {e}")

    return "", "", str(last_err) if last_err else "Unknown summary error"




def _fit_story_to_reviewer_budget(story_text: str, context_text: str, filename: str):
    original_tokens = _count_local_tokens(story_text)
    prompt_now = _build_reviewer_prompt(story_text, context_text)
    prompt_tokens_now = _count_local_tokens(prompt_now)

    result = {
        "story": story_text,
        "summary_applied": False,
        "summary_model": "",
        "summary_target": "",
        "original_tokens": original_tokens,
        "prompt_tokens": prompt_tokens_now,
        "error": "",
    }

    if prompt_tokens_now <= SCRIPT_REVIEWER_MAX_PROMPT_TOKENS and original_tokens <= SCRIPT_SUMMARY_TRIGGER_TOKENS:
        return result

    current_story = story_text
    for target in SCRIPT_SUMMARY_TARGETS:
        summarized, model_used, err = _summarize_script_for_reviewer(current_story, filename, target)
        if not summarized:
            result["error"] = err or result["error"]
            if err and ("API_KEY_INVALID" in err or "rejected by Gemini" in err):
                break
            continue

        current_story = summarized
        prompt_now = _build_reviewer_prompt(current_story, context_text)
        prompt_tokens_now = _count_local_tokens(prompt_now)

        result.update({
            "story": current_story,
            "summary_applied": True,
            "summary_model": model_used,
            "summary_target": target,
            "prompt_tokens": prompt_tokens_now,
            "error": "",
        })

        if prompt_tokens_now <= SCRIPT_REVIEWER_MAX_PROMPT_TOKENS:
            return result

    # Last-resort hard clip if still too long
    base_prompt_tokens = _count_local_tokens(_build_reviewer_prompt("", context_text))
    max_story_tokens = max(256, SCRIPT_REVIEWER_MAX_PROMPT_TOKENS - base_prompt_tokens - 64)
    clipped_story = _truncate_to_tokens(current_story, max_story_tokens)
    final_prompt_tokens = _count_local_tokens(_build_reviewer_prompt(clipped_story, context_text))

    result.update({
        "story": clipped_story,
        "summary_applied": True,
        "prompt_tokens": final_prompt_tokens,
    })
    return result


def _print_raw_model_output(raw_text: str, tag: str = ""):
    header = f"[MODEL RAW OUTPUT] {tag}".strip()
    print("\n" + "=" * 100)
    print(header)
    print("=" * 100)
    print(raw_text if (raw_text or "").strip() else "(empty)")
    print("=" * 100 + "\n")


def _generate_reviewer_output(
    prompt: str,
    *,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    inputs = tok(prompt, return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 5,
    }

    eos_id = getattr(tok, "eos_token_id", None)
    if eos_id is not None:
        gen_kwargs["pad_token_id"] = eos_id

    outputs = mdl.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    raw_text = tok.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()
    _print_raw_model_output(raw_text, "script-reviewer")
    cleaned = _clean_model_output(raw_text)
    return cleaned, raw_text


def evaluate_story(story, context, uploaded_file):
    with gen_lock:
        if (not story or story.strip() == "") and not uploaded_file:
            return "Please enter a story or upload a file.", 1, history

        file_text = ""
        if uploaded_file is not None:
            path = uploaded_file.name
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".txt":
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        file_text = f.read()
                elif ext in (".docx", ".doc"):
                    import docx
                    doc = docx.Document(path)
                    file_text = "\n".join([p.text for p in doc.paragraphs])
                elif ext == ".pdf":
                    import PyPDF2
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        file_text = "\n".join([page.extract_text() or "" for page in reader.pages])
                elif ext.startswith('.g'):
                    file_text = read_google_workspace_file(path)
                else:
                    file_text = "(File type not supported)"
            except Exception as e:
                file_text = f"(Error reading file: {str(e)})"

        final_story = (story or "") + "\n\n" + file_text
        if final_story.strip() == "":
            return "Input is empty.", 1, history

        prompt = _build_reviewer_prompt(
            final_story.strip(),
            context.strip() if context else "None",
        )

        cleaned_text, raw_text = _generate_reviewer_output(
            prompt,
            max_new_tokens=220,
            min_new_tokens=0,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        parsed = format_output(cleaned_text)
        quality = _parsed_quality(parsed)

        # Auto-retry disabled for stability (kept code for quick re-enable).
        if False and (quality < 7 or len(cleaned_text) < 180):
            repair_prompt = prompt + "\n\nIMPORTANT: Produce exactly one [BEGIN_REVIEW]...[END_REVIEW] block with no extra text. Keep overall rationale at 2-4 sentences."
            retry_cleaned, retry_raw = _generate_reviewer_output(
                repair_prompt,
                max_new_tokens=560,
                min_new_tokens=180,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
            )
            retry_parsed = format_output(retry_cleaned)
            retry_quality = _parsed_quality(retry_parsed)

            if retry_quality >= quality or len(retry_cleaned) > len(cleaned_text):
                cleaned_text = retry_cleaned
                raw_text = retry_raw
                parsed = retry_parsed
                quality = retry_quality

        final_score = _parse_score_to_int(parsed.get("score", ""), cleaned_text)

        score_based_verdict = "Yes" if final_score >= 3 else "No"
        parsed_verdict = _normalize_verdict(parsed.get("verdict", ""))
        if parsed_verdict and parsed_verdict != score_based_verdict:
            print(f"[STEP 9 WARN] Verdict/Score mismatch from model: verdict={parsed_verdict}, score={final_score}. Using score-based verdict.")
        verdict_text = score_based_verdict
        score_text = f"{final_score} (out of 5)"
        benefits_text = (parsed.get("benefits") or "").strip() or "- Not clearly provided by the model."
        risks_text = (parsed.get("risks") or "").strip() or "- Not clearly provided by the model."
        overall_text = (parsed.get("overall") or "").strip() or cleaned_text

        history.append({
            "story": (final_story[:200] + "...") if len(final_story) > 200 else final_story,
            "score": final_score,
            "verdict": verdict_text,
            "output": (cleaned_text[:300] + "...") if len(cleaned_text) > 300 else cleaned_text,
        })

        display_text = f"""
### Verdict
{verdict_text}

### Score
{score_text}

### Benefits
{benefits_text}

### Risks
{risks_text}

### Overall Rationale
{overall_text}
""".strip()

        # Show raw model text only when parsed structure is weak.
        if quality < 8:
            debug_raw = (parsed.get("raw") or cleaned_text)[:2400]
            display_text += f"\n\n### Raw Model Output (debug)\n```text\n{debug_raw}\n```"

        return display_text, final_score, history




print("[STEP 9.1] Evaluator ready.")
