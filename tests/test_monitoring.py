from pathlib import Path
import json
def test_inference_log_is_valid_jsonl():
    p=Path("logs/inference/inference_log.jsonl")
    if not p.exists(): return
    for line in p.read_text(encoding="utf-8").splitlines():
        r=json.loads(line); assert "prediction" in r and "elapsed_ms" in r
