from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
STEPS=[[sys.executable,"src/mlops_pipeline/11_prepare_deployment.py"],
       [sys.executable,"-m","pytest","tests","-q"],
       [sys.executable,"src/mlops_pipeline/12_monitor_mobile_price.py"]]
def main():
    for cmd in STEPS:
        print("[RUN]"," ".join(cmd),flush=True); subprocess.run(cmd,cwd=ROOT,check=True)
    print("[SUCCESS] Lab 13 local integration checks completed.")
if __name__=="__main__": main()
