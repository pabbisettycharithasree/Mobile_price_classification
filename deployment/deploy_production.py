from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
    src=ROOT/"deployment"; required=[src/"model.pkl",src/"preprocessor.pkl"]
    if not all(p.exists() for p in required):
        from subprocess import run
        import sys
        run([sys.executable,str(ROOT/"src/mlops_pipeline/11_prepare_deployment.py")],check=True,cwd=ROOT)
    meta=json.loads((src/"model_metadata.json").read_text())
    print("Deployment package ready:",meta)
if __name__=="__main__": main()
