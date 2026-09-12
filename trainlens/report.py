"""Strict log parsing and self-contained HTML reports; no network resources."""

import csv
import html
import json
import math
from pathlib import Path


METRICS = ("train_loss", "val_loss", "train_acc", "val_acc")


def load_run(path):
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if len(set(headers)) != len(headers):
            raise ValueError(f"{path.name}: duplicate columns")
        if not {"epoch", "train_loss", "val_loss"}.issubset(headers):
            raise ValueError(f"{path.name}: requires epoch, train_loss, val_loss")
        available = [name for name in METRICS if name in headers]
        for line, raw in enumerate(reader, 2):
            try:
                if None in raw:
                    raise ValueError("too many columns")
                epoch = int(raw["epoch"])
                if epoch < 0 or (rows and epoch <= rows[-1]["epoch"]):
                    raise ValueError("epochs must be nonnegative and strictly increasing")
                row = {"epoch":epoch}
                for name in available:
                    value = float(raw[name])
                    if not math.isfinite(value):
                        raise ValueError(f"{name} must be finite")
                    if name.endswith("_acc") and not 0 <= value <= 1:
                        raise ValueError(f"{name} must be in [0, 1]")
                    row[name] = value
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}, line {line}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path.name}: no data rows")
    return {"name":path.stem, "metrics":available, "rows":rows}


def summarize(run, metric="val_loss", mode="min"):
    if metric not in METRICS or mode not in ("min","max"):
        raise ValueError("choose a supported metric and min/max mode")
    if metric not in run["metrics"]:
        raise ValueError(f"{run['name']}: missing metric {metric}")
    choose = min if mode == "min" else max
    best = choose(run["rows"], key=lambda row:row[metric])
    return {"name":run["name"], "epochs_logged":len(run["rows"]),
            "best_epoch":best["epoch"], "selection_metric":metric,
            "selection_mode":mode, "best_value":best[metric],
            "final_val_loss":run["rows"][-1]["val_loss"],
            "gap_at_best":best["val_loss"]-best["train_loss"]}


def render_report(runs, output, metric="val_loss", mode="min", title="Training, in focus."):
    if not runs:
        raise ValueError("at least one run is required")
    names = [run["name"] for run in runs]
    if len(set(names)) != len(names):
        raise ValueError("run file stems must be unique; rename duplicate CSV files")
    summaries = [summarize(run,metric,mode) for run in runs]
    payload = json.dumps({"runs":runs, "summaries":summaries, "metric":metric,
                          "mode":mode}, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("template.html").read_text(encoding="utf-8")
    page = template.replace("__REPORT_TITLE__",html.escape(title)).replace("__REPORT_DATA__",payload)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(page,encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summaries
