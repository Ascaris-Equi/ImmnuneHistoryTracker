# calibrate.py
import argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_csv", required=True, help="CSV with columns: score,label")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(args.scores_csv)
    x = df["score"].values.reshape(-1,1)
    y = df["label"].values.astype(int)
    # Platt scaling: logistic regression
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(x, y)
    p = clf.predict_proba(x)[:,1]
    print(f"Calibration NLL={log_loss(y,p):.4f}")
    pars = {"coef": float(clf.coef_[0,0]), "intercept": float(clf.intercept_[0])}
    with open(args.out_json, "w") as f:
        json.dump(pars, f)
    print(f"Saved calibration params to {args.out_json}")

if __name__ == "__main__":
    main()