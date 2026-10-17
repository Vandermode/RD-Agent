from types import SimpleNamespace

import pandas as pd
import pytest

from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment


def test_bandit_reads_qlib_costed_annual_return():
    # Qlib's actual metric names have no trailing whitespace. Losing this value
    # also silently removes the return/drawdown component of the bandit state.
    experiment = SimpleNamespace(result=pd.Series({
        "IC": 0.031,
        "1day.excess_return_with_cost.annualized_return": 0.12,
        "1day.excess_return_with_cost.information_ratio": 1.5,
        "1day.excess_return_with_cost.max_drawdown": -0.08,
    }))
    metrics = extract_metrics_from_experiment(experiment)
    assert metrics.arr == pytest.approx(0.12)
    assert metrics.sharpe == pytest.approx(1.5)
    assert metrics.as_vector()[4] == pytest.approx(0.12)
