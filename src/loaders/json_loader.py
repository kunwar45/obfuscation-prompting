import json

from .base_loader import BasePromptLoader


class JSONFilePromptLoader(BasePromptLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[dict]:
        with open(self.file_path, "r") as f:
            prompts = json.load(f)
        return prompts
