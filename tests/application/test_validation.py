"""Public entry points reject ambiguous inputs before executing a strategy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from financeai.application.strategy import (
    StrategyConfig,
    load_market,
    model_from_json,
    model_to_json,
    run_strategy,
)
from financeai.cli import main
from financeai.core.errors import IncompatibleSchemaError, InvalidConfigError
from financeai.domains.market.engine import PreparedMarket
from financeai.domains.market.scenario import demo_scenarios
from financeai.domains.market.synthetic import SyntheticMarketSpec, synthetic_market
from financeai.models.ridge import RidgeModel

from .test_strategy_cli import CONFIG, _caches

MODEL = {"kind": "ridge", "alpha": 1.0, "features": ["a", "b"], "coef": [2.0, -1.0], "intercept": 0.5, "n_train": 10}


@pytest.mark.parametrize("version", [1, 1.0, "1", "1.x", "1.0.0", "01.0", None])
def test_strategy_requires_a_major_minor_version_string(version: object) -> None:
    with pytest.raises(IncompatibleSchemaError):
        StrategyConfig.from_mapping({**CONFIG, "schema_version": version})


@pytest.mark.parametrize("name", [None, 7, True, [], "", "   "])
def test_strategy_name_is_a_nonempty_string(name: object) -> None:
    with pytest.raises(InvalidConfigError):
        StrategyConfig.from_mapping({**CONFIG, "name": name})


@pytest.mark.parametrize("seconds", [10**400, 1e308, -1, True, "10", float("nan")])
def test_invalid_session_offsets_raise_typed_errors(seconds: object) -> None:
    with pytest.raises(InvalidConfigError):
        StrategyConfig.from_mapping({**CONFIG, "session": {**CONFIG["session"], "warmup_s": seconds}})


@pytest.mark.parametrize("document", ["[]", "null", "42", '["kind"]'])
def test_model_document_must_be_an_object(document: str) -> None:
    with pytest.raises(InvalidConfigError):
        model_from_json(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alpha", True),
        ("alpha", "1"),
        ("alpha", -1),
        ("alpha", 10**400),
        ("intercept", "0"),
        ("intercept", float("inf")),
        ("intercept", True),
        ("n_train", 2.9),
        ("n_train", True),
        ("n_train", -1),
        ("n_train", "10"),
        ("features", "ab"),
        ("features", None),
        ("features", ["a", "a"]),
        ("features", ["a", 2]),
        ("features", ["a", ""]),
        ("coef", None),
        ("coef", "12"),
        ("coef", [[1.0], [2.0]]),
        ("coef", [1.0]),
        ("coef", [True, 2.0]),
        ("coef", [float("nan"), 2.0]),
    ],
)
def test_malformed_model_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(InvalidConfigError):
        model_from_json({**MODEL, field: value})


def test_loaded_model_predicts_in_original_feature_units() -> None:
    model, names = model_from_json(json.dumps(MODEL))
    assert names == ["a", "b"]
    np.testing.assert_array_equal(model.predict(np.array([[3.0, 4.0], [0.0, 2.0]])), [2.5, -1.5])


def test_model_export_checks_feature_alignment() -> None:
    model = RidgeModel(alpha=1.0).fit(np.array([[1.0, 2.0], [2.0, 4.0]]), np.array([1.0, 2.0]))
    with pytest.raises(InvalidConfigError):
        model_to_json(model, ["only_one_feature"])


def test_inline_model_is_validated_when_loading_strategy() -> None:
    with pytest.raises(InvalidConfigError):
        StrategyConfig.from_mapping({**CONFIG, "model": {**MODEL, "coef": [[1], [2]]}})


def test_duplicate_venue_caches_are_ambiguous(tmp_path: Path) -> None:
    caches = _caches(tmp_path)
    with pytest.raises(InvalidConfigError, match="duplicate"):
        load_market([*caches, caches[0]], CONFIG["instruments"])


def test_replay_rejects_a_different_market_instrument_order() -> None:
    cfg = StrategyConfig.from_mapping(CONFIG)
    market = synthetic_market(SyntheticMarketSpec(seed=7), order=tuple(reversed(cfg.instruments)))
    with pytest.raises(InvalidConfigError, match="instrument order"):
        run_strategy(PreparedMarket.build(market), cfg, demo_scenarios()["base"])


def test_cli_reports_missing_files_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["run", str(tmp_path / "missing.json"), "--cache", str(tmp_path), "--out", str(tmp_path / "out")])
    assert code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert json.loads(output.err)["error"] == "io_error"


def test_invalid_utf8_strategy_has_a_typed_error(tmp_path: Path) -> None:
    path = tmp_path / "strategy.json"
    path.write_bytes(b'{"name": "\xff"}')
    with pytest.raises(InvalidConfigError):
        StrategyConfig.load(path)


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    from financeai import __version__

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"financeai {__version__}"
