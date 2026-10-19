"""Process ownership and isolated environments for the parallel study."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def identity(pid: int) -> dict | None:
    directory = Path("/proc") / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgid": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": int(fields[19]),
            "uid": directory.stat().st_uid,
            "wait_status": int(fields[49]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected: dict, current: dict | None) -> bool:
    keys = ["pid", "start_ticks", "uid", "pgid", "session"]
    if "boot_id" in expected:
        keys.append("boot_id")
    return current is not None and all(expected[key] == current[key] for key in keys)


def isolated_env(root: Path, runtime: Path, gpu: str, inherited=None) -> dict:
    """A fresh child interpreter must receive this before importing RD settings.

    This isolates application state, not the Unix user's filesystem permissions.
    Shared input data and immutable embedding weights are staged separately.
    """
    runtime.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ if inherited is None else inherited)
    for name in list(env):
        upper = name.upper()
        if (
            "KNOWLEDGE_BASE" in upper
            or upper.endswith(
                (
                    "PROMPT_CACHE_PATH",
                    "WORKSPACE_PATH",
                    "LOG_TRACE_PATH",
                    "PICKLE_CACHE_FOLDER_PATH_STR",
                    "FILELOCK_PATH",
                )
            )
            or upper.startswith("MLFLOW_")
            or upper in {"ARTIFACT_SIGNING_KEY", "ARTIFACT_SIGNING_KEY_PATH"}
        ):
            del env[name]
    # multiprocessing creates AF_UNIX sockets below TMPDIR. Linux socket
    # addresses must fit in 108 bytes, including its own pymp/listener suffix.
    # The persistent study path is deliberately not used for temporary IPC.
    short_tmp = Path(tempfile.mkdtemp(prefix=f"rdq-{os.getuid()}-", dir="/tmp"))
    paths = {
        "TMPDIR": short_tmp,
        "RDAGENT_LOCAL_CACHE_ROOT": runtime / "local-cache",
        "XDG_CACHE_HOME": runtime / "cache",
        "CUDA_CACHE_PATH": runtime / "cuda-cache",
    }
    for name, path in paths.items():
        if name != "TMPDIR":
            path.mkdir()
        env[name] = str(path)
    env.update(
        {
            "PYTHONPATH": str(root),
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": gpu,
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_MAX_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "ENABLE_CACHE": "False",
            "ARTIFACT_SIGNING_KEY_PATH": str(runtime / "artifact_signing.key"),
        }
    )
    write_json(
        runtime / "isolation.json",
        {
            "created_at": now(),
            "root": str(root),
            "runtime": str(runtime),
            "private_paths": {key: env[key] for key in [*paths, "ARTIFACT_SIGNING_KEY_PATH"]},
            "gpu": gpu,
            "initial_knowledge_paths": [],
            "mlflow_tracking": "workspace-local default (inherited overrides removed)",
            "research_state": "Launcher creates new prompt/pickle caches, knowledge stores and workspaces under each run directory",
            "boundary": "Application state paths, not Unix filesystem permissions",
        },
    )
    return env


def verify_ipc_environment(env: dict) -> None:
    """Exercise the same Unix socket allocation used by tensor sharing."""
    result = subprocess.run(
        [sys.executable, "-c", "from multiprocessing.connection import Listener; Listener(family='AF_UNIX').close()"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Private temporary directory failed the multiprocessing IPC check: {result.stderr[-1500:]}")


def group_members(pgid: int) -> list[dict]:
    result = []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            process = identity(int(path.name))
            if process and process["pgid"] == pgid and process["state"] != "Z":
                result.append(process)
    return result


def stop_owned_group(child: subprocess.Popen, expected: dict) -> None:
    # All child groups are created with start_new_session=True. Verify ownership
    # before signaling, including when a timed-out leader has already exited.
    members = group_members(expected["pgid"])
    if not members:
        child.wait()
        return
    if any(
        p["uid"] != expected["uid"] or p["session"] != expected["session"] or p["start_ticks"] < expected["start_ticks"]
        for p in members
    ):
        raise RuntimeError("Refusing to signal a process group with unexpected members")
    leader = identity(expected["pid"])
    if leader and not same_process(expected, leader):
        raise RuntimeError("Refusing to signal a reused process ID")
    os.killpg(expected["pgid"], signal.SIGTERM)
    deadline = time.monotonic() + 10
    while group_members(expected["pgid"]) and time.monotonic() < deadline:
        time.sleep(0.1)
    if group_members(expected["pgid"]):
        os.killpg(expected["pgid"], signal.SIGKILL)
    child.wait()


def run_pool(jobs: list[dict], concurrency: int, launch, validate, tick, poll_seconds=1.0) -> None:
    """Bounded pool; persist each transition and clean up only owned sessions."""
    if concurrency < 1:
        raise ValueError("Concurrency must be positive")
    waiting = iter(jobs)
    active = {}
    exhausted = False
    try:
        while active or not exhausted:
            tick()
            while len(active) < concurrency and not exhausted:
                job = next(waiting, None)
                if job is None:
                    exhausted = True
                    break
                child = launch(job)
                process = identity(child.pid)
                if process is None:
                    raise RuntimeError("Child process identity was unavailable")
                job.update(state="running", started_at=now(), process=process)
                active[child.pid] = (child, job, time.monotonic())
                tick(force=True)
            for pid, (child, job, started) in list(active.items()):
                exit_code = child.poll()
                if exit_code is None:
                    continue
                # A timeout must not leave worker descendants competing with
                # the next cohort or writing after a failed result is recorded.
                stop_owned_group(child, job["process"])
                job.update(exit_code=exit_code, finished_at=now(), elapsed_seconds=time.monotonic() - started)
                job["state"] = "finished" if validate(job) else "failed"
                del active[pid]
                tick(force=True)
            if active:
                time.sleep(poll_seconds)
    finally:
        for child, job, started in active.values():
            stop_owned_group(child, job["process"])
            job.update(
                state="interrupted",
                exit_code=child.returncode,
                finished_at=now(),
                elapsed_seconds=time.monotonic() - started,
            )
        tick(force=True)
