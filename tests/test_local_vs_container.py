from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_container_files_exist():
    assert (ROOT/"Dockerfile").exists()
    assert (ROOT/"api"/"main.py").exists()
