import json
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


def log_query(event: dict, log_path: str = "data/processed/query_logs.jsonl") -> None:
    if not settings.query_log_enabled:
        return

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    with open(log_path, "a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")
