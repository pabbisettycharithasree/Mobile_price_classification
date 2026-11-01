from pathlib import Path
import pandas as pd
from mobile_price_monitoring import ROOT, write_report
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"; df=pd.read_json(p,lines=True) if p.exists() else pd.DataFrame(); write_report("prediction_monitoring_report.json",{"prediction_count":len(df),"unique_classes":sorted(df.prediction.unique().tolist()) if len(df) else []}); print("Prediction monitoring complete")
if __name__=="__main__": main()