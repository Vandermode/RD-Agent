import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import QlibCondaConf, QlibCondaEnv, QlibDockerConf, QTDockerEnv


def _extract_training_log(output: str) -> str:
    number = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    pattern = rf"(Epoch\d+: train {number}, valid {number}|best score: {number} @ \d+ epoch)"
    return "\n".join(re.findall(pattern, output))


def _input_snapshot(workspace: Path, config_name: str) -> dict:
    inputs = {}
    for name in (config_name, "model.py", "combined_factors_df.parquet"):
        path = workspace / name
        if path.is_file():
            with path.open("rb") as file:
                digest = hashlib.sha256()
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
                inputs[name] = {"sha256": digest.hexdigest()}
            if path.suffix == ".parquet":
                parquet = pq.ParquetFile(path)
                inputs[name].update({"rows": parquet.metadata.num_rows, "columns": parquet.schema.names})
    return inputs


class QlibFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)

    def execute(self, qlib_config_name: str = "conf.yaml", run_env: dict | None = None, *args, **kwargs):
        run_env = dict(run_env or {})
        if MODEL_COSTEER_SETTINGS.env_type == "docker":
            qtde = QTDockerEnv(conf=QlibDockerConf(enable_cache=False))
        elif MODEL_COSTEER_SETTINGS.env_type == "conda":
            qtde = QlibCondaEnv(conf=QlibCondaConf(enable_cache=False))
        else:
            logger.error(f"Unknown env_type: {MODEL_COSTEER_SETTINGS.env_type}")
            return None, "Unknown environment type"
        qtde.prepare()

        # Env's generic command cache does not identify parquet inputs or the
        # complete training environment. A backtest must always execute afresh,
        # even if a global ENABLE_CACHE setting is enabled by another workflow.
        before = _input_snapshot(self.workspace_path, qlib_config_name)
        existing_recorders = {p.parent.name for p in self.workspace_path.glob("mlruns/*/*/meta.yaml")}
        receipt = {
            "execution_id": uuid.uuid4().hex,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command_cache_enabled": qtde.conf.enable_cache,
            "config": qlib_config_name,
            "run_env": dict(run_env),
            "inputs_before": before,
            "state": "running",
        }

        def finish(state: str, error: str | None = None):
            receipt["state"] = state
            if state != "running":
                receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
            if error is not None:
                receipt["error"] = error
            (self.workspace_path / "evaluation.json").write_text(json.dumps(receipt, indent=2) + "\n")

        finish("running")

        # Run the Qlib backtest
        execution = qtde.run(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env,
        )
        execute_qlib_log = execution.stdout
        receipt["qrun_exit_code"] = execution.exit_code
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")
        if execution.exit_code != 0:
            finish("failed", f"qrun exited with code {execution.exit_code}")
            return None, execute_qlib_log

        extraction = qtde.run(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env,
        )
        receipt["extraction_exit_code"] = extraction.exit_code
        after = _input_snapshot(self.workspace_path, qlib_config_name)
        receipt["inputs_after"] = after
        if before != after:
            finish("failed", "Backtest inputs changed during execution")
            return None, "Backtest inputs changed during execution; result rejected."
        recorder_path = self.workspace_path / "recorder.json"
        if extraction.exit_code != 0 or not recorder_path.is_file():
            finish("failed", "Result extraction failed or recorder identity is missing")
            return None, extraction.stdout
        recorder_id = json.loads(recorder_path.read_text())["recorder_id"]
        receipt["recorder_id"] = recorder_id
        current_recorders = {p.parent.name for p in self.workspace_path.glob("mlruns/*/*/meta.yaml")}
        if recorder_id in existing_recorders or recorder_id not in current_recorders:
            finish("failed", "Result does not belong to a fresh MLflow recorder")
            return None, "Result does not belong to a fresh MLflow recorder; result rejected."

        quantitative_backtesting_chart_path = self.workspace_path / "ret.parquet"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_parquet(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            finish("failed", "No return series found")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            execute_qlib_log = _extract_training_log(execute_qlib_log)
            finish("finished")
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            finish("failed", "No metrics file found")
            return None, execute_qlib_log
