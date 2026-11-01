import json
from pathlib import Path
from mobile_price_monitoring import ROOT, write_report
def main():
 p=ROOT/"reports"/"data_drift_report.json"; data=json.loads(p.read_text()) if p.exists() else {"status":"run 12_data_drift.py first"}; write_report("drift_analysis_report.json",data); print(data)
if __name__=="__main__": main()