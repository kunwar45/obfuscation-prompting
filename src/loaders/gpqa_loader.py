import random

from datasets import load_dataset

from .base_loader import BasePromptLoader

GPQA_DATASET = "Idavidrein/gpqa"
LETTERS = "ABCD"


class GPQALoader(BasePromptLoader):
    """Load questions from the GPQA benchmark (Idavidrein/gpqa on HuggingFace).

    Requires HF_TOKEN in the environment (the dataset is gated).

    Args:
        subset:  "gpqa_main" (448 q), "gpqa_diamond" (198 q), or "gpqa_extended" (546 q)
        split:   HuggingFace split name — GPQA only has "train"
        limit:   Cap the number of questions (None = all)
        shuffle: Shuffle dataset order before applying limit
        seed:    RNG seed for reproducibility (controls both dataset shuffle and
                 per-question answer-choice ordering)
    """

    def __init__(
        self,
        subset: str = "gpqa_main",
        split: str = "train",
        limit: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.subset = subset
        self.split = split
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed

    def load(self) -> list[dict]:
        ds = load_dataset(GPQA_DATASET, self.subset, split=self.split)
        if self.shuffle:
            ds = ds.shuffle(seed=self.seed)
        if self.limit:
            ds = ds.select(range(min(self.limit, len(ds))))

        prompts = []
        for i, row in enumerate(ds):
            text, correct_letter = self._format_mcq(row, answer_seed=self.seed + i)
            prompts.append({
                "id": f"gpqa_{self.subset}_{i:04d}",
                "text": text,
                "metadata": {
                    "correct_letter": correct_letter,
                    "correct_answer_text": row["Correct Answer"],
                    "domain": row.get("High-level domain", ""),
                    "subdomain": row.get("Subdomain", ""),
                    "gpqa_subset": self.subset,
                },
            })
        return prompts

    @staticmethod
    def _format_mcq(row: dict, answer_seed: int = 0) -> tuple[str, str]:
        """Shuffle the four answer choices and return (formatted question, correct letter)."""
        answers = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng = random.Random(answer_seed)
        rng.shuffle(answers)

        correct_letter = LETTERS[answers.index(row["Correct Answer"])]
        choices = "\n".join(f"{letter}. {ans}" for letter, ans in zip(LETTERS, answers))
        text = f"{row['Question']}\n\n{choices}"
        return text, correct_letter
