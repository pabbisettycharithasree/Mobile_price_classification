from pathlib import Path
from mobile_price_monitoring import ROOT, write_report
def main():
 candidates=[ROOT/"models"/"mobile_price_random_forest.pkl"]; existing=[p.name for p in candidates if p.exists()]; write_report("model_comparison_report.json",{"available_mobile_price_models":existing,"note":"Compare only models trained on the same feature schema and split."}); print(existing)
if __name__=="__main__": main()