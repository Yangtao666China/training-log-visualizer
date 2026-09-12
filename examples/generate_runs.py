"""Actual NumPy logistic-regression runs; same data and initialization, three optimizers.

The optimizer learning rates are illustrative settings, not a tuned benchmark.
"""

import csv
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(800,6))
    true_w = np.array([1.4,-.8,.6,1.1,-1.2,.4])
    probability = np.exp(-np.logaddexp(0,-(x@true_w)))
    y = (rng.uniform(size=800)<probability).astype(float)
    x = np.column_stack((x,np.ones(len(x))))
    train_x,val_x = x[:600],x[600:]
    train_y,val_y = y[:600],y[600:]
    output = Path(__file__).resolve().parents[1]/"sample_runs"
    output.mkdir(exist_ok=True)
    for optimizer,lr in [("sgd",.03),("momentum",.03),("adam",.03)]:
        w,m,v = np.zeros(7),np.zeros(7),np.zeros(7)
        rows = []
        for epoch in range(1,121):
            pred = np.exp(-np.logaddexp(0,-(train_x@w)))
            grad = train_x.T@(pred-train_y)/len(train_y)
            if optimizer == "sgd":
                w -= lr*grad
            elif optimizer == "momentum":
                m = .9*m+grad
                w -= lr*m
            else:
                m,v = .9*m+.1*grad,.999*v+.001*grad**2
                w -= lr*(m/(1-.9**epoch))/(np.sqrt(v/(1-.999**epoch))+1e-8)
            row = {"epoch":epoch}
            for prefix,xx,yy in [("train",train_x,train_y),("val",val_x,val_y)]:
                logits = xx@w
                row[prefix+"_loss"] = float(np.mean(np.logaddexp(0,logits)-yy*logits))
                row[prefix+"_acc"] = float(np.mean((logits>0)==yy))
            rows.append(row)
        with (output/f"{optimizer}.csv").open("w",encoding="utf-8",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=["epoch","train_loss","val_loss","train_acc","val_acc"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{optimizer}: final validation loss = {rows[-1]['val_loss']:.6f}")


if __name__ == "__main__":
    main()
