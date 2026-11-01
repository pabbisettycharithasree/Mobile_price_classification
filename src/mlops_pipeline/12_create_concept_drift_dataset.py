from pathlib import Path
import pandas as pd
from mobile_price_monitoring import DRIFT, read_baseline
def main():
    df=read_baseline().copy(); df["ram"]=(df["ram"]*0.85).round().astype(int); DRIFT.mkdir(parents=True,exist_ok=True); df.to_csv(DRIFT/"evolved_labeled_data.csv",index=False); print("Created labeled drift sample")
if __name__=="__main__": main()