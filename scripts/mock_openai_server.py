#!/usr/bin/env python3
"""Tiny OpenAI-compatible chat server for local smoke tests.

This is only for wiring/integration tests. It does not run an LLM.
"""

from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _infer_answer(user_text: str, system_text: str) -> str:
    haystack = f"{system_text}\n{user_text}"

    # Shipping: extract N days if present in context.
    days_match = re.search(r"\b(\d+)\s*days?\b", haystack, flags=re.IGNORECASE)
    weeks_match = re.search(r"\b(\d+)\s*weeks?\b", haystack, flags=re.IGNORECASE)
    if days_match:
        return f"{days_match.group(1)} days"
    if weeks_match:
        return f"{weeks_match.group(1)} weeks"
    return "I do not have enough information."


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body or "{}")
        messages = payload.get("messages", [])

        system_text = ""
        user_text = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            elif msg.get("role") == "user":
                user_text = msg.get("content", "")

        answer = _infer_answer(user_text=user_text, system_text=system_text)
        content = (
            "<reasoning>\n"
            f"I used the context to identify a concrete timeframe: {answer}.\n"
            "</reasoning>\n"
            "<answer>\n"
            f"{answer}\n"
            "</answer>"
        )

        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": payload.get("model", "mock-model"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        out = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, _format, *_args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("Mock OpenAI server listening on http://127.0.0.1:8000/v1/chat/completions")
    server.serve_forever()


if __name__ == "__main__":
    main()

