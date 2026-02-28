from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from tqdm import tqdm


class LocalOpenAIClient:
    """Minimal OpenAI-compatible chat client for local/self-hosted inference servers."""

    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8000/v1",
        api_key: str = "local",
        max_retries: int = 3,
        base_delay: float = 2.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        url = f"{self.api_base}/chat/completions"
        body = json.dumps(payload).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url=url,
                data=body,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise RuntimeError(
                        "Local OpenAI-compatible request failed after retries. "
                        f"endpoint={url}"
                    ) from exc
                delay = self.base_delay * (2**attempt)
                tqdm.write(
                    f"Local API error ({type(exc).__name__}), retrying in {delay:.0f}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(delay)

        raise RuntimeError("Unexpected local client retry loop state") from last_error

