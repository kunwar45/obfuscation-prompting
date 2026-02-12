import time

from tqdm import tqdm
from together import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    Together,
)

# Transient/capacity errors worth retrying (503, 429, timeouts, connection)
RETRYABLE_EXCEPTIONS = (
    InternalServerError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


class TogetherClient:
    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 2.0):
        self.client = Together(api_key=api_key)
        self.max_retries = max_retries
        self.base_delay = base_delay

    def chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> str:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except RETRYABLE_EXCEPTIONS as e:
                last_error = e
                if attempt == self.max_retries:
                    raise
                delay = self.base_delay * (2**attempt)
                tqdm.write(f"API {type(e).__name__}, retrying in {delay:.0f}s (attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(delay)
        raise last_error
