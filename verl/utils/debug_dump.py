import json
import os
import threading
from pathlib import Path
from typing import Any, Optional


_TRUE_VALUES = {"1", "true", "yes", "on"}


class DebugDumpWriter:
    def __init__(self):
        enabled_value = os.getenv("SDPO_DEBUG_DUMP", "0")
        self.enabled = enabled_value.strip().lower() in _TRUE_VALUES
        self.path = os.getenv("SDPO_DEBUG_DUMP_PATH", "./sdpo_debug_dump.jsonl")
        self.max_samples = int(os.getenv("SDPO_DEBUG_MAX_SAMPLES", "50"))
        self._written = 0
        self._lock = threading.Lock()
        self._is_rank0 = self._detect_rank0()

    def _detect_rank0(self) -> bool:
        for key in ("RANK", "LOCAL_RANK"):
            value = os.getenv(key)
            if value is not None:
                try:
                    return int(value) == 0
                except ValueError:
                    return False
        return True

    def should_dump(self) -> bool:
        return self.enabled and self.max_samples > 0 and self._is_rank0

    def append(self, record: dict[str, Any], path: Optional[str] = None) -> bool:
        if not self.should_dump():
            return False

        with self._lock:
            if self._written >= self.max_samples:
                return False

            dump_path = Path(path or self.path)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            with dump_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._written += 1
            return True


DEBUG_DUMP_WRITER = DebugDumpWriter()
