import argparse
import glob
from pathlib import Path

from .report import METRICS, load_run, render_report


def main():
    parser = argparse.ArgumentParser(description="Turn training CSV logs into an offline report.")
    parser.add_argument("inputs", nargs="+", help='CSV paths or quoted patterns, e.g. "sample_runs/*.csv"')
    parser.add_argument("--output",default="report.html")
    parser.add_argument("--metric",choices=METRICS,default="val_loss")
    parser.add_argument("--mode",choices=("min","max"),default="min")
    parser.add_argument("--title",default="Training, in focus.")
    args = parser.parse_args()
    paths = []
    for pattern in args.inputs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            parser.error(f"no files match: {pattern}")
        paths.extend(Path(p).resolve() for p in matches)
    paths = list(dict.fromkeys(paths))
    try:
        summary = render_report([load_run(path) for path in paths],args.output,args.metric,args.mode,args.title)
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    print(f"Created {args.output} ({len(summary)} runs) and its JSON summary.")


if __name__ == "__main__":
    main()
