import json
import os
import uuid
from datetime import datetime, timezone

from src.config import Config
from src.pipeline.result import PromptResult


class ResultStorage:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)

    def save(self, results: list[PromptResult]) -> str:
        run_id = str(uuid.uuid4())
        short_id = run_id.split("-")[0]
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp_str}_{short_id}.json"
        path = os.path.join(self.config.output_dir, filename)

        payload = {
            "run_id": run_id,
            "run_timestamp": now.isoformat(),
            "config": {
                "base_model": self.config.base_model,
                "monitor_model": self.config.monitor_model,
            },
            "results": [r.to_dict() for r in results],
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        return path
