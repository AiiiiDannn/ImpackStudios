from typing import Callable, List


def estimate_tokens(text: str) -> int:
    # Simple approximation; keep consistent with notebook behavior.
    return max(1, len(text) // 4)


def needs_summary(text: str, gate_tokens: int) -> bool:
    return estimate_tokens(text) > gate_tokens


def chunk_text(text: str, chunk_chars: int) -> List[str]:
    if len(text) <= chunk_chars:
        return [text]
    return [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]


def summarize_long_text(
    text: str,
    client,
    model: str,
    chunk_chars: int,
    progress_cb: Callable[[str], None] | None = None,
) -> str:
    if progress_cb:
        progress_cb("STEP S1 | Summary gate triggered.")

    chunks = chunk_text(text, chunk_chars)
    partials: List[str] = []
    total = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        if progress_cb:
            progress_cb(f"STEP S2.{idx} | Summarizing chunk {idx}/{total}...")
        prompt = f"""Summarize the following script chunk for downstream evaluation.
Keep key plot points, themes, tone, social context, and potential risks/benefits.
Use concise plain English.

Chunk:
{chunk}
"""
        partial = client.generate(
            prompt=prompt,
            model=model,
            temperature=0.2,
            max_output_tokens=700,
        )
        partials.append(partial.strip())

    if len(partials) == 1:
        if progress_cb:
            progress_cb("STEP S3 | Single-chunk summary complete.")
        return partials[0]

    if progress_cb:
        progress_cb("STEP S3 | Merging partial summaries...")

    merge_prompt = """Merge the partial summaries into one final structured summary.
Keep only essential information needed for script uplift evaluation.
Return clean text, no JSON.

Partials:
"""
    merge_prompt += "\n\n".join(f"[PART {i+1}]\n{p}" for i, p in enumerate(partials))

    merged = client.generate(
        prompt=merge_prompt,
        model=model,
        temperature=0.1,
        max_output_tokens=900,
    )
    if progress_cb:
        progress_cb("STEP S4 | Final merged summary complete.")
    return merged.strip()

