import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "src" / "mlops_pipeline" / "mobile_price_pipeline.py"
env = os.environ.copy()
env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
result = subprocess.run([sys.executable, str(PIPELINE), "--with-quality-gate"], cwd=str(PROJECT_ROOT), env=env)
sys.exit(result.returncode)
