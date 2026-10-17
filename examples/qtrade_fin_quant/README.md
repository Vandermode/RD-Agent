# RD-Agent(Q) main-workflow reproduction

This targets the CSI300 study in [R&D-Agent-Quant, v2](https://arxiv.org/html/2505.15155v2),
using the public RD-Agent implementation. It is a method reproduction with the
available gateway and public Qlib data. It does not claim to reproduce the
paper's numerical results or its complete baseline table.

The paper specifies Alpha20 as the starting feature set; training in 2008–2014,
validation in 2015–2016, and testing in 2017–2020-08-01; six hours each for
factor-only and model-only research, and twelve hours for joint optimization.
It allows ten coding attempts per task and reports baseline results from five
seeds after hyperparameter tuning. The original chat backends include GPT-4o
and o3-mini, with Ada embeddings. Its data description also includes Wind
fundamentals. See [Appendix C](https://arxiv.org/html/2505.15155v2#A3).

## Available reproduction

| Component | This setup |
| --- | --- |
| Research modes | Upstream `FactorRDLoop`, `ModelRDLoop`, and `QuantRDLoop` |
| Main joint scheduler | Upstream Thompson sampling, with the metric-key fix below |
| Market and splits | CSI300, original chronological date ranges |
| Starting factors | Upstream Alpha20; formulas match paper Table 5 |
| Market data | Official Qlib public daily snapshot; prepared by the previous factor example |
| Chat and retrieval | QTrade gateway's configured model; local BGE embeddings |
| Execution | Isolated `rdagent` Conda environment; CUDA PyTorch on one selected GPU |
| Baselines | LightGBM with Alpha20/158/360; five seeds and fixed upstream parameters |
| Budget | 6h / 6h / 12h, plus a hard external deadline |

The installed dataset has price/volume fields, not the paper's Wind fundamentals.
The factor export follows the upstream script's 2008-12-29 start date; Qlib's
baseline features still use its full training interval. Each research manifest
records the export coverage and SHA-256 hashes of the input HDF files.
The gateway currently lacks GPT-4o, o3-mini, and an embeddings endpoint.
The fixed-parameter baselines do not reproduce the paper's unspecified tuning
search. Changing the model, embeddings, data fields, code revision, and hardware
can change both the research path and the number of iterations per time budget.

The checked-out templates use Qlib `TopkDropoutStrategy` (top 50, drop 5),
closing-price execution, 0.05% buy costs, 0.15% sell costs, and a 5 CNY minimum.
Their portfolio metrics are **excess returns over CSI300 after costs**. Do not
equate these mechanically with the paper's prose definitions of ARR and Sharpe:
the paper describes compound ARR and a risk-free benchmark, while the released
code uses Qlib's excess-return statistics. Raw metrics and return series are
retained for inspection.

The upstream research loop feeds backtest metrics back into candidate selection.
The nominal test interval therefore participates in research. These results are
research observations, not evidence from a sealed final holdout.

## Install and validate

Reuse the data and gateway setup in [the factor example](../qtrade_fin_factor/README.md).
The GPU lock is for Linux x86_64 and CPython 3.11:

```bash
conda activate rdagent
uv pip install -r examples/qtrade_fin_quant/requirements.gpu.lock.txt
uv pip install --no-deps -e .
uv pip check
CUDA_VISIBLE_DEVICES=0 python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
python -m pytest -q test/qlib/test_bandit_metrics.py \
  test/qlib/test_reproduction_report.py test/qlib/test_model_factor_proposal.py \
  test/qlib/test_model_feedback_contract.py test/qlib/test_model_factor_handoff.py
```

Only the model credential is read from QTrade. The trading environment and live
repository are not modified. The GPU launcher exposes physical GPU 0 by default;
`--gpu` selects another device. CPU factor calculations and local embeddings
remain on CPU.

## Run

Run from the repository root. Each `--run-id` must be new:

```bash
# Full-data, fixed-parameter reference baselines: 3 libraries x 5 seeds.
python -m examples.qtrade_fin_quant.baselines --run-id fin-baselines-01

# Verify the model-development path before a long experiment.
timeout --signal=TERM --kill-after=30s 65m \
  python -m examples.qtrade_fin_quant.run --mode model --loops 1 \
  --duration 1h --run-id fin-model-pilot-01

# Main joint research budget, with a grace period for the current stage.
timeout --signal=TERM --kill-after=30s 13h \
  python -m examples.qtrade_fin_quant.run --mode quant --run-id fin-quant-main-01

# Component comparisons, run separately with independent memory and caches.
timeout --signal=TERM --kill-after=30s 7h \
  python -m examples.qtrade_fin_quant.run --mode factor --run-id fin-factor-main-01
timeout --signal=TERM --kill-after=30s 7h \
  python -m examples.qtrade_fin_quant.run --mode model --run-id fin-model-main-01
```

Run research modes sequentially: upstream execution uses temporary paths shared
across processes. The time budgets bound wall-clock work, not a dollar charge.
Gateway pricing is unavailable to this launcher; the paper's reported API cost
must not be treated as a spending estimate for this backend.

The supervisor runs the joint, factor, and model phases sequentially, checks each
exit status, and stops if a phase fails:

```bash
python -m examples.qtrade_fin_quant.suite --suite-id fin-main-01 --gpu 0 \
  --baseline-run-id fin-baselines-01
```

It requires committed source and writes phase logs plus `status.json` under
`git_ignore_folder/reproduction/suites/<suite-id>/`. The nominal total is 24
hours, with up to three additional hours of stage-completion grace. The recorded
supervisor PID can be sent SIGTERM to stop it and its active child process group.
It also stops if free disk space falls below 50 GiB.
After each phase it writes `comparison.csv` and `comparison.json`, selecting the
last candidate accepted by the agent, rather than selecting a trial afterward
by its backtest return. Baseline rows are the per-metric medians over their five
configured seeds. These different replication counts are recorded in the report.

Outputs live under `git_ignore_folder/reproduction/runs/<run-id>/`, including
the manifest, checkpoints, generated code, Qlib results, return series, and
separate factor/model knowledge stores. `results.json` and `status.json` are
written when a research run exits normally or raises a Python exception.
An external hard kill can leave the last status as `running`; check process
liveness and the outer supervisor's exit status in that case. Checkpoints are
written after every workflow stage. Scheduler seed 0 seeds the main Python and
NumPy process; it does not make remote LLM calls or model-training subprocesses
deterministic. Baseline model seeds are explicitly configured.

## Upstream corrections

At upstream commit `484776c211e4fbbeef03e0ec00d6bbee7362a4f4`, the bandit reads
`1day.excess_return_with_cost.annualized_return ` with a trailing space. Real
Qlib output has no trailing space. This silently zeros both the annual-return
input and the derived return/drawdown input. The fork removes that space; the
regression test fails before the correction and passes afterward. This is an
explicit implementation difference from the pinned upstream revision.

The completed GPU pilot exposed two model-feedback issues. The model-only
workflow selected the factor feedback prompt, which requests `Replace Best
Result`, while its parser expects `Decision`. A response approving the initial
model was therefore recorded as a rejection. The fork selects the model prompt.
The workspace also filtered training logs with a negative-scores-only regular
expression, dropping the positive losses emitted by the pinned Qlib version.
The extractor now preserves positive and negative scores. Regression tests
reproduce both failures and verify the corrections.

The first joint run exposed a factor-to-model handoff issue: the model runner
merged accepted factors only when it found more than one factor experiment.
The model proposal's history contains no empty baseline factor experiment, so
the first accepted factor was skipped and the GRU still received only Alpha20.
The fork merges any nonempty list of accepted factor experiments. Regression
tests cover one factor, multiple factors, and a model-only run without factors.
The affected run was stopped and retained as diagnostic evidence; the timed
suite starts afresh after this correction.

## Chat-backend ablation

The ablation compares `gpt-6-astra`, `gpt-6.1-sol`, and `gpt-6-luna` through the
same gateway. Each backend gets three independent four-hour joint research
runs, using controller seeds 0, 1, and 2: twelve hours per backend, thirty-six
hours in total. The model order rotates across seeds so each backend occupies
each execution position once. Runs execute sequentially on the same selected
GPU. Training subprocesses retain upstream seeding defaults, and remote model
responses remain stochastic; the controller seeds alone do not ensure exact
replay.

All arms use the same frozen public dataset, Alpha20 starting factors, splits,
costs, local embeddings, temperature 1.0, maximum output of 10,000 tokens,
`medium` reasoning effort, and ten coding attempts. Each run starts with empty
research memory and its own prompt, execution, and knowledge caches. The
five-seed LightGBM baselines are shared reference results. This repeated
comparison replaces the single 12-hour trajectory in the main-paper protocol;
it does not claim to duplicate that experimental table.

```bash
python -m pytest -q test/qlib/test_backend_ablation.py
python -m examples.qtrade_fin_quant.ablation --study-id fin-llm-ablation-01 \
  --gpu 0 --baseline-run-id fin-baselines-01
```

The supervisor first completes a one-iteration joint preflight for each backend
with separate disposable memory. Research starts only after all three pass.
Each four-hour run has a five-hour hard deadline. A failed research replicate
is recorded, and the remaining replicates continue. The study stops if disk
space falls below 50 GiB or the committed source changes; use a separate checkout
for code edits while it runs. SIGTERM to the recorded supervisor PID stops its
active child process group.

Outputs are under `git_ignore_folder/reproduction/ablations/<study-id>/`:
`status.json`, individual console logs, `runs.csv`, `summary.csv`, and
`comparison.json`. Reports include the last accepted candidate in every run,
finished/failed counts, means, medians, sample standard deviations, and ranges.
They never select a replicate or candidate afterward by its largest backtest
return. Failure and missing-result counts remain visible. Logged completion
counts measure calls observed in the logs, not token usage or billed cost.
