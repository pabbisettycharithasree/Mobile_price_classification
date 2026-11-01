import json
from pathlib import Path
from mobile_price_monitoring import ROOT, write_report
def main():
 p=ROOT/"reports"/"data_drift_report.json"; d=json.loads(p.read_text()) if p.exists() else {}; drift=d.get("drifted_features",[]); out={"retraining_recommended":bool(drift),"reason":"PSI threshold exceeded" if drift else "No measured drift or drift report not yet generated","drifted_features":drift}; write_report("monitor_trigger_report.json",out); print(out)
if __name__=="__main__": main()