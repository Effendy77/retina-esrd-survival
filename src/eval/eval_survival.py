import argparse, json, numpy as np, pandas as pd
from lifelines.utils import concordance_index
# Placeholder: expects CSV with columns: time, event, pred_risk
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred_csv', required=True)
    ap.add_argument('--out', default='results/eval.json')
    args = ap.parse_args()
    df = pd.read_csv(args.pred_csv)
    c = concordance_index(df['time'], -df['pred_risk'], df['event'])
    res = {'c_index': float(c)}
    print(res)
    os.makedirs('results', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
if __name__=='__main__':
    main()
