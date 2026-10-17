# Finance factor example with QTrade's model gateway

This runs the upstream `FactorRDLoop` from RD-Agent 1.0.0, commit
`484776c211e4fbbeef03e0ec00d6bbee7362a4f4`. It uses the official Qlib CN daily
sample data, CSI300, LightGBM, and the upstream train/validation/backtest dates:
2008–2014 / 2015–2016 / 2017–2020-08-01. The current code starts with its
`ALPHA20` feature dictionary; the website's older Alpha158 description does not
match this checkout's default feature list.

The shared QTrade configuration supplies the chat model and gateway credential.
Only that credential is loaded into LiteLLM's proxy variable; the trading
configuration is not copied into this repository. Python dependencies live in a
separate Conda environment because QTrade's deployed environment has different
NumPy/Pandas versions and does not contain RD-Agent's dependencies.

The gateway has no embeddings endpoint. `backend.py` therefore keeps RD-Agent's
LiteLLM chat implementation and uses FastEmbed's local
`BAAI/bge-small-en-v1.5` embeddings for knowledge retrieval. This is a documented
backend substitution, not an exact reproduction of results from a different LLM
or embedding model.

## Install

Run these commands from the repository root on Linux x86_64:

```bash
conda create -n rdagent python=3.11 pip -y
conda activate rdagent
uv pip install -r examples/qtrade_fin_factor/requirements.lock.txt
uv pip install --no-deps -e .
uv pip check
```

The lock records the environment used here, including the Qlib commit pinned by
RD-Agent's upstream Dockerfile and CPU-only PyTorch. No Docker daemon is required.

## Prepare the official data

```bash
python - <<'PY'
from pathlib import Path
from qlib.tests.data import GetData
target = Path.home() / ".qlib/qlib_data/cn_data"
if not (target / "calendars/day.txt").exists():
    GetData().download_data(
        file_name="v2/qlib_data_cn_1d_latest.zip",
        target_dir=target,
        delete_old=False,
    )
PY
python -m examples.qtrade_fin_factor.prepare
```

The export uses the same six fields, full-data start date, and 100 debugging
instruments as upstream `factor_data_template/generate.py`, with eight Qlib
workers. Existing exports are not overwritten. The downloaded snapshot used
here has calendar dates 1999-11-10 through 2020-09-25 and archive SHA-256
`8a323fec32e1bd0c1662c1fc753ad033f190523e450936e63c9509796ea32e41`.
The full factor export contains 7,584,444 rows across 3,875 instruments; the
debug export contains 48,700 rows across 100 instruments.

## Check and run

The default credential source is `~/.config/qtrade/assistant.env`, and the
default chat endpoint is the existing local gateway at `127.0.0.1:8317/v1`.
Override these with `--qtrade-env` and `--api-base` if needed.

```bash
python -m examples.qtrade_fin_factor.run --check
python -m examples.qtrade_fin_factor.run --loops 1
```

The first command checks JSON chat responses and real 384-dimensional local
embeddings. The second runs the upstream research loop, including candidate
generation, coding, Qlib backtesting, feedback and session checkpoints. The
defaults limit the run to one research iteration, three coding attempts and a
30-minute cooperative deadline. For a hard process deadline, use:

```bash
timeout --signal=TERM --kill-after=30s 35m \
  python -m examples.qtrade_fin_factor.run --loops 1
```

Outputs are ignored by Git and stored under
`git_ignore_folder/reproduction/runs/<run-id>/`: `manifest.json`, `results.json`,
`trace/`, `workspaces/`, prompt caches, and the knowledge base when generated.
`--run-id` selects a new directory; existing run directories are rejected.
The launcher exits with an error if no completed backtest is recorded, even if
upstream handled an experiment failure internally.

This establishes an executable upstream example. Its backtest results are fed
into the agent's research loop, so this is not a sealed final holdout evaluation
or a reproduction of AQuA's regularized RSI protocol. No QTrade market data or
existing lab state is used.

## First completed run

On 2026-10-17, `fin-factor-02` completed all five workflow stages using the
shared gateway's `gpt-6-luna`. The agent implemented five-day momentum,
20-day realized volatility, and 20-day volume surprise, including its own
repairs after an initial input-column error. Both baseline and candidate
backtests completed.

| Metric | Baseline | Baseline + 3 factors |
| --- | ---: | ---: |
| IC | 0.031071 | 0.032001 |
| Rank IC | 0.033512 | 0.034624 |
| Annualized excess return after costs | 4.3083% | 3.5878% |
| Excess-return information ratio after costs | 0.486969 | 0.361512 |

The feedback retained the baseline (`Decision: False`) and proposed a follow-up
hypothesis. Five session checkpoints and the coding knowledge base were saved.
Reloading the final checkpoint recovered one history entry with all five stages
complete; reloading the knowledge base recovered three successful task entries.
These are sample-dataset observations from one run, not an out-of-sample alpha
claim. Local evidence is in `fin-factor-02/results.json`, `verification.json`,
the trace directory, and each experiment workspace's `qlib_res.csv`.
