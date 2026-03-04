from typing import Any, Optional


def extract_response_text(response: Any) -> str:
    # New SDK usually exposes .text
    txt = getattr(response, "text", None)
    if txt:
        return str(txt).strip()

    # Fallback extraction path.
    try:
        candidates = getattr(response, "candidates", None) or []
        parts = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                piece = getattr(part, "text", None)
                if piece:
                    parts.append(str(piece))
        return "\n".join(parts).strip()
    except Exception:
        return ""


class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.2,
        max_output_tokens: int = 900,
    ) -> str:
        client = self._get_client()

        try:
            from google.genai import types  # type: ignore

            cfg = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
        except Exception:
            response = client.models.generate_content(model=model, contents=prompt)

        text = extract_response_text(response)
        if not text:
            raise RuntimeError("Gemini returned empty text.")
        return text

    def test_key(self, model: str) -> str:
        return self.generate(
            prompt="Reply with exactly: GEMINI_KEY_OK",
            model=model,
            temperature=0.0,
            max_output_tokens=16,
        )

