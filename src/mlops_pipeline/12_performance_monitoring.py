from mobile_price_monitoring import ROOT, write_report
import pandas as pd
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"; df=pd.read_json(p,lines=True) if p.exists() else pd.DataFrame(); write_report("performance_report.json",{"requests":len(df),"status":"log available" if len(df) else "no requests yet"}); print("Performance report written")
if __name__=="__main__": main()