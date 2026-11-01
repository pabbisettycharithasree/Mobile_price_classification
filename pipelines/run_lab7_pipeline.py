import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "src" / "mlops_pipeline" / "mobile_price_pipeline.py"

if __name__ == "__main__":
    if not PIPELINE.exists():
        print(f"[ERROR] Lab 7 pipeline not found: {PIPELINE}")
        sys.exit(1)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, str(PIPELINE)], cwd=str(PROJECT_ROOT), env=env)
    sys.exit(result.returncode)
