from pathlib import Path
from mobile_price_monitoring import ROOT
def main():
 p=ROOT/"logs"/"inference"/"inference_log.jsonl"; print(p.read_text() if p.exists() else "No API requests logged yet")
if __name__=="__main__": main()