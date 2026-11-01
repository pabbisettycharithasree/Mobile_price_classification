from mobile_price_monitoring import DRIFT, read_baseline, FEATURES, psi, write_report
import pandas as pd
def main():
    base=read_baseline(); cur=pd.read_csv(DRIFT/"evolved_data.csv"); scores={c:psi(base[c].dropna(),cur[c].dropna()) for c in FEATURES}; write_report("data_drift_report.json",{"psi_by_feature":scores,"drifted_features":[k for k,v in scores.items() if v>=0.2],"threshold":0.2}); print(scores)
if __name__=="__main__": main()