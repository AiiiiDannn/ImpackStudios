import re


def normalize_api_key(raw: str) -> str:
    """Clean common paste artifacts and return a single API key token."""
    if not raw:
        return ""

    key = raw.strip().strip("`").strip().strip("\"'").strip()
    if not key:
        return ""

    # Keep first non-empty line.
    if "\n" in key:
        for line in key.splitlines():
            line = line.strip()
            if line:
                key = line
                break

    # Handle "API_KEY=..." style.
    key = re.sub(r"^(api[_-]?key)\s*[:=]\s*", "", key, flags=re.IGNORECASE)

    # Clip trailing comments or separators.
    for sep in ("#", ";", ",", " ", "\t", "\r"):
        if sep in key:
            key = key.split(sep, 1)[0].strip()

    # Strong extraction: if any valid AIza token exists in pasted text, use it.
    m = re.search(r"AIza[0-9A-Za-z_-]{20,}", key)
    if m:
        return m.group(0).strip()

    # If user pasted extra prefix text, keep from AIza onward.
    idx = key.find("AIza")
    if idx > 0:
        key = key[idx:]

    return key.strip()


def looks_like_google_api_key(key: str) -> bool:
    return bool(key) and key.startswith("AIza") and len(key) >= 20


def explain_google_api_key_check(raw_value: str, normalized_value: str) -> str:
    raw_len = len(raw_value or "")
    norm = normalized_value or ""
    norm_len = len(norm)
    norm_prefix = (norm[:8] + "...") if norm else "(empty)"

    if not norm:
        reason = "normalized key is empty"
    elif not norm.startswith("AIza"):
        reason = "normalized key does not start with 'AIza'"
    elif len(norm) < 20:
        reason = "normalized key is shorter than 20 chars"
    else:
        reason = "passed"

    return (
        f"reason={reason}; "
        f"raw_len={raw_len}; "
        f"normalized_prefix={norm_prefix}; "
        f"normalized_len={norm_len}"
    )
