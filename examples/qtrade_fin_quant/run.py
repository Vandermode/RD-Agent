"""Run factor-only, model-only, or joint research with explicit time budgets."""

import argparse
import asyncio
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from examples.qtrade_fin_factor.run import configure


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def summarize(loop, mode: str) -> list[dict]:
    result = []
    for index, (experiment, feedback) in enumerate(loop.trace.hist):
        references = [base for base in experiment.based_experiments if base.result is not None]
        result.append({
            "iteration": index,
            "action": getattr(experiment.hypothesis, "action", mode),
            "tasks": [task.name for task in experiment.sub_tasks],
            "metrics": experiment.result.to_dict() if experiment.result is not None else None,
            "accepted": bool(feedback.decision),
            "reference_metrics": references[-1].result.to_dict() if references else None,
            "feedback": str(feedback),
            "workspace": str(experiment.experiment_workspace.workspace_path),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("factor", "model", "quant"), default="quant")
    parser.add_argument("--qtrade-env", type=Path, default=Path.home() / ".config/qtrade/assistant.env")
    parser.add_argument("--api-base", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--model")
    parser.add_argument("--conda-env", default="rdagent")
    parser.add_argument("--loops", type=int, help="Optional iteration cap; omitted for timed experiments")
    parser.add_argument("--coding-attempts", type=int, default=10)
    parser.add_argument("--duration", help="Defaults to 6h for factor/model or 12h for joint research")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu", default="0", help="Physical GPU exposed to all model subprocesses")
    parser.add_argument("--scheduler-seed", type=int, default=0)
    args = parser.parse_args()
    if args.loops is not None and args.loops < 1:
        parser.error("loops must be positive")
    if args.coding_attempts < 1:
        parser.error("coding-attempts must be positive")
    args.duration = args.duration or ("12h" if args.mode == "quant" else "6h")
    args.check = False
    output = configure(args)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "MODEL_COSTEER_MAX_LOOP": str(args.coding_attempts),
        "MODEL_COSTEER_NEW_KNOWLEDGE_BASE_PATH": str(output / "model-knowledge.pkl"),
        "FACTOR_COSTEER_NEW_KNOWLEDGE_BASE_PATH": str(output / "factor-knowledge.pkl"),
        "FACTOR_COSTEER_FILE_BASED_EXECUTION_TIMEOUT": "600",
        "RUNNING_TIMEOUT_PERIOD": "3600",
        "ENABLE_CACHE": "True",
        "QLIB_QUANT_ACTION_SELECTION": "bandit",
        "CHAT_TEMPERATURE": "1.0",
        "LITELLM_CHAT_TEMPERATURE": "1.0",
        "CHAT_MAX_TOKENS": "10000",
        "LITELLM_CHAT_MAX_TOKENS": "10000",
    })

    # Upstream settings are instantiated on import; configuration must precede it.
    import litellm
    import numpy as np
    import torch
    from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING, MODEL_PROP_SETTING, QUANT_PROP_SETTING
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop
    from rdagent.app.qlib_rd_loop.model import ModelRDLoop
    from rdagent.app.qlib_rd_loop.quant import QuantRDLoop

    litellm.request_timeout = 180
    random.seed(args.scheduler_seed)
    np.random.seed(args.scheduler_seed)
    if args.mode != "factor" and not torch.cuda.is_available():
        raise RuntimeError("Model research requires a working CUDA PyTorch installation.")

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.update({
        "mode": args.mode,
        "scheduler_seed": args.scheduler_seed,
        "training_seeds": "Upstream defaults; scheduler seed does not seed model subprocesses",
        "gpu": args.gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "temperature": 1.0,
        "max_output_tokens": 10000,
        "protocol": "public-Qlib method reproduction; not exact paper results",
        "paper": "https://arxiv.org/html/2505.15155v2#A3.SS1",
        "differences": [
            "Current gateway chat model and local BGE embeddings replace original backends",
            "Public price/volume data only; no Wind fundamentals",
            "Current pinned upstream code with corrected bandit annual-return key",
            "Upstream costed excess-return metrics; not assumed equivalent to paper ARR/Sharpe",
            "Research feedback includes backtest metrics; this is not a sealed holdout",
        ],
    })
    for path in [*Path(__file__).parent.glob("*.py"), root / "rdagent/scenarios/qlib/proposal/bandit.py"]:
        manifest["launcher_sources"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)

    loop_types = {
        "factor": (FactorRDLoop, FACTOR_PROP_SETTING),
        "model": (ModelRDLoop, MODEL_PROP_SETTING),
        "quant": (QuantRDLoop, QUANT_PROP_SETTING),
    }
    loop_class, settings = loop_types[args.mode]
    loop = loop_class(settings)
    loop._init_base_features(None)
    write_json(output / "status.json", {"state": "running", "pid": os.getpid()})
    state = "failed"
    try:
        asyncio.run(loop.run(loop_n=args.loops, all_duration=args.duration))
        summaries = summarize(loop, args.mode)
        if not any(item["metrics"] for item in summaries):
            raise RuntimeError("No completed backtest; inspect the trace and workspaces.")
        state = "finished"
    finally:
        summaries = summarize(loop, args.mode)
        write_json(output / "results.json", summaries)
        write_json(output / "status.json", {
            "state": state,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "recorded_iterations": len(summaries),
            "successful_backtests": sum(item["metrics"] is not None for item in summaries),
            "accepted_iterations": sum(item["accepted"] for item in summaries),
            "actions": [item["action"] for item in summaries],
        })
    print(f"Results: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
