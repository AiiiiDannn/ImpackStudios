import re
from typing import Dict, List


def build_reviewer_prompt(story_text: str, humanity_def: str, mission_text: str = "") -> str:
    mission_block = mission_text.strip() if mission_text else "None"
    return f"""Evaluate the submission for humanity-uplifting value.

Humanity-uplifting definition:
{humanity_def}

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

No extra sections. No role-play text. No "wait for further instructions".

Submission:
{story_text}

Additional Context:
{mission_block}
"""


def _clean_model_output(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"(?is)^.*?###\s*Response:\s*", "", text).strip()

    compact_lines: List[str] = []
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
    if not src:
        return src
    m = re.search(r"(?:\s[-]{2,}\s*){3,}", src)
    if m:
        return src[: m.start()].rstrip()
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

    m_overall = re.search(r"(?is)(?:^|\n)\s*(?:Overall\s*rationale|Overall|Rationale)\s*[:\-]\s*(.+)", data)
    if m_overall:
        prefix = data[: m_overall.start()]
        tail = _truncate_reviewer_noise(m_overall.group(1).strip())
        stop = re.search(r"\n\s*\n", tail)
        first_para = tail[: stop.start()].strip() if stop else tail.strip()
        data = (prefix + "\nOverall rationale: " + first_para).strip()

    return data


def _extract_section(text: str, labels: List[str], next_labels: List[str]) -> str:
    label_pat = "|".join(re.escape(x) for x in labels)
    next_pat = "|".join(re.escape(x) for x in next_labels) if next_labels else ""

    if next_pat:
        pattern = (
            rf"(?ims)(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{label_pat})\s*[:\-]\s*"
            rf"(.+?)(?=\n\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{next_pat})\s*[:\-]|\Z)"
        )
    else:
        pattern = rf"(?ims)(?:^|\n)\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:{label_pat})\s*[:\-]\s*(.+)$"

    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _dedupe_section_lines(block: str) -> str:
    if not block:
        return ""
    out: List[str] = []
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


def _parse_bullets(section_text: str) -> List[str]:
    bullets: List[str] = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\*\u2022\u2013\u2014]+\s*", "", line)
        if line:
            bullets.append(line)
    return bullets[:5]


def _parsed_quality(parsed: Dict[str, object]) -> int:
    quality = 0
    for key in ["verdict", "score", "benefits", "risks", "rationale"]:
        val = str(parsed.get(key) or "").strip()
        if val:
            quality += 1
            if len(val) >= 40:
                quality += 1
    return quality


def parse_reviewer_output(raw: str) -> Dict[str, object]:
    text = _extract_first_review_block(_clean_model_output(raw))

    verdict_raw = _extract_section(
        text,
        ["Verdict", "Final Verdict"],
        ["Score", "Rating", "Benefits", "Risks", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    if not verdict_raw:
        m_v = re.search(r"(?im)\bIs\s+this\s+uplifting\?\s*(yes|no)\b", text)
        if not m_v:
            m_v = re.search(r"(?im)^\s*(yes|no)\s*$", text)
        if m_v:
            verdict_raw = m_v.group(1).capitalize()

    score_raw = _extract_section(
        text,
        ["Score", "Rating"],
        ["Benefits", "Risks", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    benefits_raw = _extract_section(
        text,
        ["Benefits", "Strengths", "Upsides", "Positive impacts"],
        ["Risks", "Concerns", "Downsides", "Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    risks_raw = _extract_section(
        text,
        ["Risks", "Concerns", "Downsides", "Negative impacts"],
        ["Overall rationale", "Overall", "Rationale", "Analysis"],
    )
    rationale = _extract_section(
        text,
        ["Overall rationale", "Overall", "Rationale", "Analysis", "Final analysis"],
        [],
    )

    if not rationale and text:
        rationale = text

    score_int = _parse_score_to_int(score_raw, text)
    verdict_from_score = "Yes" if score_int >= 3 else "No"
    parsed_verdict = _normalize_verdict(verdict_raw)
    verdict = verdict_from_score if parsed_verdict != verdict_from_score else parsed_verdict or verdict_from_score

    benefits = _parse_bullets(_dedupe_section_lines(benefits_raw))
    risks = _parse_bullets(_dedupe_section_lines(risks_raw))

    if not benefits:
        benefits = ["Not clearly provided by the model."]
    if not risks:
        risks = ["Not clearly provided by the model."]
    rationale = _dedupe_section_lines(rationale).strip() or "Model output did not provide a stable rationale block."

    parsed: Dict[str, object] = {
        "verdict": verdict,
        "score": score_int,
        "benefits": benefits,
        "risks": risks,
        "rationale": rationale,
        "raw": text,
    }
    parsed["quality"] = _parsed_quality(parsed)
    return parsed


def format_reviewer_markdown(parsed: Dict[str, object], route_mode: str, summary_status: str) -> str:
    benefits = "\n".join(f"- {item}" for item in parsed["benefits"])  # type: ignore[index]
    risks = "\n".join(f"- {item}" for item in parsed["risks"])  # type: ignore[index]
    return (
        f"**Routed Agent:** `script reviewer`\n\n"
        f"**Route Mode:** `{route_mode}`\n\n"
        f"**Summary Gate:** {summary_status}\n\n"
        f"### Verdict\n{parsed['verdict']}\n\n"
        f"### Score\n{parsed['score']} (out of 5)\n\n"
        f"### Benefits\n{benefits}\n\n"
        f"### Risks\n{risks}\n\n"
        f"### Overall Rationale\n{parsed['rationale']}"
    )
