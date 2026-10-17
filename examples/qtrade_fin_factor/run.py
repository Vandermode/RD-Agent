"""Run a bounded upstream FactorRDLoop using QTrade's chat gateway credentials.

Run from the repository root inside the rdagent Conda environment:
    python -m examples.qtrade_fin_factor.run --check
    python -m examples.qtrade_fin_factor.run --loops 1
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import dotenv_values


def configure(args: argparse.Namespace) -> Path:
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    values = dotenv_values(args.qtrade_env)
    if values.get("QTRADE_OPENAI_ADVISOR_PROVIDER") != "cliproxyapi":
        raise ValueError("This example expects QTrade's cliproxyapi provider.")
    key = values.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("The selected QTrade environment has no OPENAI_API_KEY.")

    # Read only the needed credential; never export the full trading environment.
    # The upstream LLM settings are logged, so keep the key in LiteLLM's provider
    # variable, which is not a field of those settings.
    for name in (
        "OPENAI_API_KEY", "CHAT_OPENAI_API_KEY", "EMBEDDING_OPENAI_API_KEY",
        "LITELLM_OPENAI_API_KEY", "LITELLM_CHAT_OPENAI_API_KEY",
        "LITELLM_EMBEDDING_OPENAI_API_KEY",
    ):
        os.environ.pop(name, None)
    os.environ["LITELLM_PROXY_API_KEY"] = key
    os.environ["LITELLM_PROXY_API_BASE"] = args.api_base
    model = args.model or values.get("QTRADE_OPENAI_ADVISOR_MODEL")
    if not model:
        raise ValueError("Select a chat model with --model.")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run-id must be a single directory name.")
    output = root / "git_ignore_folder/reproduction/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update({
        "BACKEND": "examples.qtrade_fin_factor.backend.LocalEmbeddingBackend",
        "CHAT_MODEL": f"litellm_proxy/{model}",
        "LITELLM_CHAT_MODEL": f"litellm_proxy/{model}",
        "EMBEDDING_MODEL": "local/BAAI/bge-small-en-v1.5",
        "LITELLM_EMBEDDING_MODEL": "local/BAAI/bge-small-en-v1.5",
        "CHAT_STREAM": "False",
        "CHAT_MAX_TOKENS": "8192",
        "MAX_RETRY": "2",
        "DUMP_CHAT_CACHE": "True",
        "USE_CHAT_CACHE": "True",
        "DUMP_EMBEDDING_CACHE": "True",
        "USE_EMBEDDING_CACHE": "True",
        "PROMPT_CACHE_PATH": str(output / "prompt-cache.db"),
        "LOG_TRACE_PATH": str(output / "trace"),
        "WORKSPACE_PATH": str(output / "workspaces"),
        "PICKLE_CACHE_FOLDER_PATH_STR": str(output / "pickle-cache"),
        "CONDA_DEFAULT_ENV": args.conda_env,
        "CONDA_ENV_NAME": args.conda_env,
        "MODEL_COSTEER_ENV_TYPE": "conda",
        "FACTOR_COSTEER_PYTHON_BIN": sys.executable,
        "FACTOR_COSTEER_MAX_LOOP": str(args.coding_attempts),
        "FACTOR_COSTEER_NEW_KNOWLEDGE_BASE_PATH": str(output / "knowledge.pkl"),
        "STEP_SEMAPHORE": "1",
        "MULTI_PROC_N": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "NUMEXPR_MAX_THREADS": "8",
        "TOKENIZERS_PARALLELISM": "false",
    })
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "chat_model": model,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "loops": args.loops,
        "coding_attempts": args.coding_attempts,
        "duration": args.duration,
        "check_only": args.check,
        "conda_env": args.conda_env,
        "launcher_sources": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Run directory: {output}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qtrade-env", type=Path, default=Path.home() / ".config/qtrade/assistant.env")
    parser.add_argument("--api-base", default="http://127.0.0.1:8317/v1")
    parser.add_argument("--model")
    parser.add_argument("--conda-env", default="rdagent")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--coding-attempts", type=int, default=3)
    parser.add_argument("--duration", default="30m")
    parser.add_argument("--run-id")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.loops < 1 or args.coding_attempts < 1:
        parser.error("loops and coding-attempts must be positive")
    output = configure(args)

    # Import only after configuration: upstream settings are initialized at import.
    import litellm
    litellm.request_timeout = 180
    from rdagent.oai.llm_utils import APIBackend

    if args.check:
        backend = APIBackend()
        response = backend.build_messages_and_create_chat_completion(
            system_prompt="Return the requested JSON object only.",
            user_prompt='Return {"ok": true}.',
            json_mode=True,
        )
        if json.loads(response).get("ok") is not True:
            raise RuntimeError("Chat connectivity check returned unexpected JSON.")
        embeddings = backend.create_embedding(["price momentum", "trading volume"])
        if len(embeddings) != 2 or len(embeddings[0]) != 384:
            raise RuntimeError("Local embedding check returned unexpected dimensions.")
        result = {"chat_json": True, "embeddings": {"count": 2, "dimensions": 384}}
        (output / "check.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return

    from rdagent.app.qlib_rd_loop.conf import FACTOR_PROP_SETTING
    from rdagent.app.qlib_rd_loop.factor import FactorRDLoop

    loop = FactorRDLoop(FACTOR_PROP_SETTING)
    loop._init_base_features(None)
    asyncio.run(loop.run(loop_n=args.loops, all_duration=args.duration))
    summaries = []
    for experiment, feedback in loop.trace.hist:
        summaries.append({
            "tasks": [task.name for task in experiment.sub_tasks],
            "metrics": experiment.result.to_dict() if experiment.result is not None else None,
            "feedback": str(feedback),
            "workspace": str(experiment.experiment_workspace.workspace_path),
        })
    (output / "results.json").write_text(json.dumps(summaries, indent=2, default=str) + "\n")
    if not summaries or not any(item["metrics"] for item in summaries):
        raise RuntimeError(f"No completed backtest; inspect {output / 'trace'}")
    print(f"Completed {len(summaries)} research iteration(s): {output / 'results.json'}")


if __name__ == "__main__":
    main()
