from pathlib import Path
import pandas as pd
from mobile_price_monitoring import ROOT, write_report
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"
 if not p.exists(): write_report("prediction_drift_report.json",{"status":"no inference log yet"}); return
 df=pd.read_json(p,lines=True); counts=df.prediction.value_counts(normalize=True).to_dict(); write_report("prediction_drift_report.json",{"prediction_distribution":counts,"rows":len(df)}); print(counts)
if __name__=="__main__": main()