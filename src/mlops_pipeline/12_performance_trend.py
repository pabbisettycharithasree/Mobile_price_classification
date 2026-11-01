from mobile_price_monitoring import ROOT, write_report
import pandas as pd
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"; df=pd.read_json(p,lines=True) if p.exists() else pd.DataFrame(); write_report("performance_trend_report.json",{"observed_requests":len(df),"trend":"insufficient history" if len(df)<2 else "request history available"}); print("Trend summary written")
if __name__=="__main__": main()