"""Fixed-parameter LightGBM baselines on Alpha20, Alpha158, and Alpha360."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from jinja2 import Template
import yaml

from examples.qtrade_fin_quant.run import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--features", nargs="+", choices=("alpha20", "alpha158", "alpha360"),
                        default=["alpha20", "alpha158", "alpha360"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("run-id must be a single directory name")
    root = Path(__file__).resolve().parents[2]
    output = root / "git_ignore_folder/reproduction/runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    os.environ["WORKSPACE_PATH"] = str(output / "workspaces")
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_MAX_THREADS"] = "8"
    from qlib.contrib.data.loader import Alpha360DL
    from rdagent.utils.qlib import ALPHA20, ALPHA158
    import pandas as pd

    expressions, names = Alpha360DL.get_feature_config()
    features = {"alpha20": ALPHA20, "alpha158": ALPHA158, "alpha360": dict(zip(names, expressions))}
    template_dir = root / "rdagent/scenarios/qlib/experiment/factor_template"
    template = Template((template_dir / "conf_baseline.yaml").read_text())
    write_json(output / "manifest.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "features": args.features, "seeds": args.seeds,
        "hyperparameters": "Fixed upstream LightGBM configuration; no paper HPO replication",
        "metric": "Qlib excess return relative to CSI300, after costs",
    })
    results = []
    for feature_name in args.features:
        for seed in args.seeds:
            run = output / f"{feature_name}-seed{seed}"
            run.mkdir()
            feature_map = features[feature_name]
            config = yaml.safe_load(template.render(
                feature_names=repr(list(feature_map)), feature_expressions=repr(list(feature_map.values())),
                train_start="2008-01-01", train_end="2014-12-31", valid_start="2015-01-01",
                valid_end="2016-12-31", test_start="2017-01-01", test_end="2020-08-01",
            ))
            config["qlib_init"]["kernels"] = 8
            config["task"]["model"]["kwargs"].update({"seed": seed, "num_threads": 8})
            (run / "conf.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            shutil.copyfile(template_dir / "read_exp_res.py", run / "read_exp_res.py")
            print(f"Running {run.name}", flush=True)
            with (run / "console.log").open("w") as log:
                subprocess.run([str(Path(sys.executable).parent / "qrun"), "conf.yaml"],
                               cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
                subprocess.run([sys.executable, "read_exp_res.py"], cwd=run, stdout=log,
                               stderr=subprocess.STDOUT, timeout=120, check=True)
            metrics = pd.read_csv(run / "qlib_res.csv", index_col=0).iloc[:, 0].to_dict()
            results.append({"features": feature_name, "seed": seed, "metrics": metrics})
            write_json(output / "results.json", results)
    median = {}
    for feature_name in args.features:
        records = [item["metrics"] for item in results if item["features"] == feature_name]
        median[feature_name] = pd.DataFrame(records).median().to_dict()
    write_json(output / "median-metrics.json", median)
    print(f"Completed {len(results)} backtests: {output}", flush=True)


if __name__ == "__main__":
    main()
