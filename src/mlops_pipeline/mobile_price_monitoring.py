from pathlib import Path
import json
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[2]
FEATURES=["battery_power","blue","clock_speed","dual_sim","fc","four_g","int_memory","m_dep","mobile_wt","n_cores","pc","px_height","px_width","ram","sc_h","sc_w","talk_time","three_g","touch_screen","wifi"]
DRIFT=ROOT/"data"/"monitoring"/"drift"; REPORTS=ROOT/"reports"; ARTIFACTS=ROOT/"artifacts"
def read_baseline():
    p=ROOT/"data"/"raw"/"train.csv"
    if not p.exists(): raise FileNotFoundError(f"Expected existing dataset at {p}")
    return pd.read_csv(p)
def write_report(name, data):
    REPORTS.mkdir(parents=True,exist_ok=True); ARTIFACTS.mkdir(parents=True,exist_ok=True)
    out=REPORTS/name; out.write_text(json.dumps(data,indent=2,default=str),encoding="utf-8"); return out
def psi(expected, actual, bins=10):
    expected=np.asarray(expected,dtype=float); actual=np.asarray(actual,dtype=float)
    edges=np.unique(np.quantile(expected,np.linspace(0,1,bins+1)))
    if len(edges)<3: return 0.0
    edges[0],edges[-1]=-np.inf,np.inf
    a=np.histogram(expected,bins=edges)[0]/len(expected); b=np.histogram(actual,bins=edges)[0]/len(actual)
    a=np.clip(a,1e-6,None); b=np.clip(b,1e-6,None)
    return float(np.sum((b-a)*np.log(b/a)))
