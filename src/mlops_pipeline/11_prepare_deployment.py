from pathlib import Path
import json, shutil
ROOT=Path(__file__).resolve().parents[2]
MODELS=ROOT/"models"; DEPLOY=ROOT/"deployment"
FILES={"model": MODELS/"mobile_price_random_forest.pkl", "preprocessor": MODELS/"mobile_price_preprocessor.pkl"}
def main():
    DEPLOY.mkdir(parents=True,exist_ok=True)
    missing=[str(p) for p in FILES.values() if not p.exists()]
    if missing: raise FileNotFoundError("Required Lab 9 artifacts missing: "+", ".join(missing))
    for name,src in FILES.items(): shutil.copy2(src, DEPLOY/("model.pkl" if name=="model" else "preprocessor.pkl"))
    meta={"project":"Mobile Price Classification","model_file":"model.pkl","preprocessor_file":"preprocessor.pkl","features":20,"target":"price_range","source":"existing Lab 9 artifacts"}
    (DEPLOY/"model_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print("Deployment artifacts prepared from existing model files.")
if __name__=="__main__": main()
