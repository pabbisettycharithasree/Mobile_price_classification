from mobile_price_monitoring import ROOT, write_report
import pandas as pd
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"; df=pd.read_json(p,lines=True) if p.exists() else pd.DataFrame(); write_report("api_performance_report.json",{"request_count":len(df),"note":"Latency requires client-side test measurements; see tests/test_performance.py"}); print("API performance summary written")
if __name__=="__main__": main()