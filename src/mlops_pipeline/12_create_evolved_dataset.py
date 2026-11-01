from pathlib import Path
import pandas as pd
from mobile_price_monitoring import ROOT, DRIFT, read_baseline
def main():
    df=read_baseline().drop(columns=["price_range"],errors="ignore").copy()
    # Deterministic covariate shift for monitoring exercises; original data is never modified.
    for c in ["battery_power","ram","px_width","px_height"]: df[c]=(df[c]*1.08).round()
    DRIFT.mkdir(parents=True,exist_ok=True); df.to_csv(DRIFT/"evolved_data.csv",index=False); print("Created evolved monitoring sample")
if __name__=="__main__": main()