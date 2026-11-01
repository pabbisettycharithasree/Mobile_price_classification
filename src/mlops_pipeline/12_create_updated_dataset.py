from mobile_price_monitoring import DRIFT, read_baseline
def main():
    DRIFT.mkdir(parents=True,exist_ok=True); read_baseline().to_csv(DRIFT/"post_drift_training_data.csv",index=False); print("Wrote candidate updated training data; original train.csv unchanged")
if __name__=="__main__": main()