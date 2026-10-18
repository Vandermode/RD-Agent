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
  test/qlib/test_model_feedback_contract.py test/qlib/test_model_factor_handoff.py \
  test/qlib/test_response_schema.py test/qlib/test_execution_integrity.py \
  test/qlib/test_ablation_integrity.py
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

Run these ordinary launchers sequentially. For simultaneous runs, use the
parallel controller below, which isolates temporary paths as well as research
memory. The time budgets bound wall-clock work, not a dollar charge.
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
updated after every recorded joint-research iteration and again when the run exits.
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
research memory and its own prompt and knowledge caches. Qlib training and
backtest-result caching are disabled. The
five-seed LightGBM baselines are shared reference results. This repeated
comparison replaces the single 12-hour trajectory in the main-paper protocol;
it does not claim to duplicate that experimental table.

```bash
python -m pytest -q test/qlib/test_backend_ablation.py
python -m examples.qtrade_fin_quant.ablation --study-id fin-llm-ablation-02 \
  --gpu 0 --baseline-run-id fin-baselines-01
```

The supervisor first completes a five-iteration joint preflight for each backend
with separate disposable memory and a forced action sequence: factor, factor,
model, factor, model. Each must complete at least two successful factor and two
successful model evaluations, use distinct factor data and fresh MLflow recorder
IDs, and reload its final checkpoint and both coding knowledge stores. Preflights
have a two-hour budget and are excluded from the measured comparison. Research
starts only after all three pass; its scheduler uses the original bandit policy.
Each four-hour run includes API retry time and has a five-hour hard deadline.
An in-progress stage may finish after four hours, but subsequent stages do not
start and unfinished candidates do not enter the retained-result comparison.
A failed research replicate
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

### Throughput-selected continuation

`parallel_ablation` continues the corrected three-backend study after all three
serial seed-0 trajectories finish. Run it from a separate, committed checkout
with the same factor HDF inputs, factor-data manifest, public Qlib dataset, and
pretrained BGE weights. Do not copy research knowledge stores, prompt caches,
workspaces, results, or checkpoints into the new run directories.

The handoff uses `parallel-handoff.json` in the legacy study directory. It records
the frozen serial supervisor's PID, UID, process group, session, and Linux start
ticks, plus the active Sol timeout process's identity. Only the supervisor parent
is paused; its Sol worker continues. The controller checks those identities and
requires Sol's successful exit, a finished run status, and no surviving worker
descendants before removing the paused parent and transferring the six queued
seed-1/2 jobs. A lock prevents two controllers from owning the same handoff.

```bash
python -m examples.qtrade_fin_quant.parallel_ablation \
  --study-id fin-llm-ablation-03-parallel \
  --legacy-study /path/to/serial-checkout/git_ignore_folder/reproduction/ablations/fin-llm-ablation-02 \
  --gpu 0
```

After the handoff, six frozen evaluation fixtures are copied from the last
accepted serial candidate for each backend (two training seeds each). Only code,
configuration, required factor data, and the extraction script are copied. Each
fixture uses the full original data splits and portfolio backtest; neural
training is capped at five epochs for this benchmark. The benchmark runs six
batches in the order **1, 3, 6, 6, 3, 1** concurrent processes, reversing fixture
order for the second pass. Every evaluation starts in a new workspace and must
produce a fresh completed MLflow recorder, unchanged input hashes, finite
metrics, and valid daily returns. Each has a 45-minute hard limit.

The controller chooses the highest pooled count of valid evaluations per elapsed
hour, including process startup, training, prediction and result extraction.
Only configurations passing both complete batches are eligible. GPU utilization
and memory are diagnostic measurements; telemetry includes unrelated GPU users.
The benchmark measures candidate evaluation throughput, not remote LLM API
throughput or the eventual number of research iterations.

The six remaining research runs start with fresh memory and retain the original
four-hour wall-clock budget, medium reasoning, and five-hour hard timeout. Their
training schedules are unchanged by the benchmark's epoch cap. At concurrency
three, each wave has one run per backend and the next wave waits for all three;
at six, all replicas start together; at one, the original balanced order remains.
Every run has independent prompt/pickle caches, factor/model knowledge stores,
workspaces, temporary directories, local execution cache, and artifact signing
key. This is application-state isolation, not a Unix filesystem permission
sandbox. The ordinary local executor honors `RDAGENT_LOCAL_CACHE_ROOT` for this
purpose and retains its original default for existing serial runs.

The new study directory contains `status.json`, `comparison.json`, separate
`serial-runs.csv` / `serial-summary.csv` and `parallel-runs.csv` /
`parallel-summary.csv`, plus `benchmark/throughput.csv`, frozen fixture manifests,
all benchmark evaluation receipts, and `resources.csv`. Serial seed 0 is never
pooled with seeds 1/2. Outcome selection is still the last accepted candidate.
The additional portfolio Sharpe uses net daily returns, zero risk-free rate,
238 trading days and sample standard deviation; its return series must reproduce
the selected recorder's Qlib information ratio. Missing verified series remain
explicitly missing. Source changes or less than 50 GiB free disk stop owned
workers; SIGTERM to this controller also stops only its owned worker sessions.
It never resumes or interrupts the serial Sol worker while waiting for handoff.

```bash
python -m pytest -q test/qlib/test_parallel_ablation.py \
  test/qlib/test_backend_ablation.py test/qlib/test_ablation_integrity.py \
  test/qlib/test_execution_integrity.py
```

### Response and execution safeguards

The first ablation (`fin-llm-ablation-01`) is retained as diagnostic evidence.
Eight trajectories exited early on response-format errors, and a command cache
reused factor backtests because its key omitted the parquet data. It is not a
valid equal-budget comparison. The corrected study starts with fresh memory and
output directories for every arm.

Qlib hypotheses, proposals, and feedback now use complete typed contracts.
Nested narrative fields are preserved as JSON text; an unambiguous `model_type`
inside `hyperparameters` is moved to its required top level. Missing or conflicting
fields receive up to three validation attempts with explicit repair feedback.
Exhausted repairs become explicitly rejected iterations, never promoted results.
The model-feedback path also makes one evaluation request instead of two.

Qlib runners no longer reuse results by task description, and their execution
environment explicitly disables command caching even if `ENABLE_CACHE=True` is
set elsewhere. Every backtest records `evaluation.json`: input file hashes and
factor columns before and after execution, training settings, exit codes, and
the new recorder ID. Failed commands, changed inputs, or old recorder IDs cannot
produce an accepted evaluation. The joint loop rejects duplicate recorder IDs
before adding results to research history. Reports expose verified backtests,
schema repair attempts, and iterations rejected for schema errors.
