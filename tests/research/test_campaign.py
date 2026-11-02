"""Campaign evidence records (portable schemas): the append-only artifact registry,
exposure and trial ledgers, stage records, the freeze seal and its opening/evaluation
records. Values are invented; no market data is read."""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from financeai.core.errors import FinanceAIError, InvalidConfigError, InvalidDataError
from financeai.core.hashing import sha256_bytes, sha256_json
from financeai.core.serialization import canonical_json, loads_strict
from financeai.research.campaign import (
    ARTIFACT_RECORD_FIELDS,
    EVALUATION_RECORD_FIELDS,
    EXPOSURE_LEDGER_HEADER,
    FREEZE_FIELDS,
    REQUIRED_SEAL_HASHES,
    STAGE_RECORD_FIELDS,
    TRIAL_LEDGER_HEADER,
    ArtifactRef,
    Campaign,
    DataRole,
    ExposureKind,
    ExposureRecord,
    ImmutableRecordError,
    OpenedDataset,
    content_sha256,
    evaluation_record_template,
    freeze_template,
    stage_record_template,
)

T0 = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


class Clock:
    """Deterministic UTC clock advancing one second per reading."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        self.now += timedelta(seconds=1)
        return self.now


def make_campaign(root: Path, campaign_id: str = "c-001") -> Campaign:
    return Campaign(root, campaign_id, clock=Clock())


def write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_field_lists_are_the_template_keys() -> None:
    assert tuple(stage_record_template()) == STAGE_RECORD_FIELDS
    assert tuple(freeze_template()) == FREEZE_FIELDS
    assert tuple(evaluation_record_template()) == EVALUATION_RECORD_FIELDS
    assert set(REQUIRED_SEAL_HASHES) == {
        "protocol_hash",
        "code_hash",
        "environment_hash",
        "configuration_hash",
        "selected_components_hash",
        "package_release_hash",
        "performance_implementation_hash",
    }
    # Templates are handed out as fresh copies: mutating one never leaks into the next.
    stage_record_template()["evidence_scope"].append("SYNTHETIC")
    assert stage_record_template()["evidence_scope"] == ["NONE"]


def test_campaign_ids_must_be_safe_path_components(tmp_path: Path) -> None:
    for bad in ("", "../escape", "a/b", ".hidden", "x y", "line\n"):
        with pytest.raises(InvalidConfigError):
            Campaign(tmp_path, bad)
    campaign = make_campaign(tmp_path)
    assert campaign.directory == tmp_path.resolve() / "runs" / "c-001"


# ---------------------------------------------------------------------------
# Artifact registry
# ---------------------------------------------------------------------------


def test_registering_a_file_records_its_hash_and_relative_path(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "out" / "report.json", b'{"x": 1}\n')
    record = campaign.register_artifact(
        "baseline_results",
        report,
        producer_attempt="a1",
        producer_stage="06",
        evidence_scope=["SYNTHETIC"],
        dependencies=(),
        description="replay summary",
    )
    assert record.sha256 == hashlib.sha256(b'{"x": 1}\n').hexdigest()
    assert (record.version, record.kind, record.path, record.size_bytes) == (1, "file", "out/report.json", 9)
    assert record.evidence_scope == ("SYNTHETIC",)
    assert record.registered_at == "2026-10-21T12:00:01.000000+00:00"
    assert campaign.artifacts() == (record,)
    assert campaign.artifact("baseline_results") == record
    assert campaign.verify_artifact("baseline_results") == record


def test_relative_artifact_paths_resolve_against_the_campaign_root(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    write(tmp_path / "out" / "a.txt", b"a")
    record = campaign.register_artifact("a", "out/a.txt", producer_attempt="a1", evidence_scope="SYNTHETIC")
    assert record.path == "out/a.txt"


def test_reregistering_identical_content_is_idempotent(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"payload")
    first = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC")
    before = campaign.registry_path.read_bytes()
    again = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC")
    assert again == first
    assert campaign.registry_path.read_bytes() == before
    explicit = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC", version=1)
    assert explicit == first


def test_changed_content_appends_a_new_version(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"one")
    first = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC")
    report.write_bytes(b"two")
    second = campaign.register_artifact("x", report, producer_attempt="a2", evidence_scope="SYNTHETIC")
    assert (first.version, second.version) == (1, 2)
    assert campaign.artifacts() == (first, second)
    assert campaign.artifact("x") == second
    assert campaign.artifact("x", 1) == first
    with pytest.raises(InvalidDataError):  # version 1's bytes are gone from disk
        campaign.verify_artifact("x", 1)


def test_same_version_with_a_different_hash_is_refused(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"one")
    first = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC", version=1)
    report.write_bytes(b"tampered")
    with pytest.raises(ImmutableRecordError) as info:
        campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC", version=1)
    assert info.value.code == "immutable_record"
    assert isinstance(info.value, InvalidDataError)
    assert campaign.artifacts() == (first,)


def test_versions_must_be_contiguous(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"one")
    with pytest.raises(InvalidDataError):
        campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC", version=2)
    assert campaign.artifacts() == ()


def test_identical_bytes_from_a_new_attempt_are_a_new_version(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"same")
    first = campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope="SYNTHETIC")
    second = campaign.register_artifact("x", report, producer_attempt="a2", evidence_scope="SYNTHETIC")
    assert second.version == 2 and second.sha256 == first.sha256
    assert second.producer_attempt == "a2"


def test_directory_hash_is_a_stable_manifest_of_its_files(tmp_path: Path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    write(one / "a.txt", b"alpha")
    write(one / "sub" / "b.bin", b"\x00\x01")
    write(two / "sub" / "b.bin", b"\x00\x01")  # created in another order
    write(two / "a.txt", b"alpha")
    write(two / "__pycache__" / "x.pyc", b"bytecode is not content")
    manifest = {
        "schema": "financeai.directory_manifest",
        "version": 1,
        "files": [["a.txt", 5, sha256_bytes(b"alpha")], ["sub/b.bin", 2, sha256_bytes(b"\x00\x01")]],
    }
    expected = sha256_json(manifest)
    assert content_sha256(one) == content_sha256(two) == expected
    write(two / "a.txt", b"alphA")
    assert content_sha256(two) != expected
    write(two / "a.txt", b"alpha")
    (two / "sub" / "b.bin").rename(two / "sub" / "c.bin")
    assert content_sha256(two) != expected
    campaign = make_campaign(tmp_path)
    record = campaign.register_artifact("bundle", one, producer_attempt="a1", evidence_scope="SYNTHETIC")
    assert (record.kind, record.sha256, record.size_bytes, record.path) == ("directory", expected, 7, "one")


def test_directory_artifacts_refuse_symlinks(tmp_path: Path) -> None:
    write(tmp_path / "d" / "a.txt", b"a")
    (tmp_path / "d" / "link").symlink_to(tmp_path / "d" / "a.txt")
    with pytest.raises(InvalidDataError):
        content_sha256(tmp_path / "d")


def test_artifacts_must_exist_inside_the_campaign_root(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path / "root")
    outside = write(tmp_path / "elsewhere.txt", b"x")
    with pytest.raises(InvalidConfigError):
        campaign.register_artifact("x", outside, producer_attempt="a1", evidence_scope="SYNTHETIC")
    with pytest.raises(InvalidDataError):
        campaign.register_artifact(
            "x", tmp_path / "root" / "missing.txt", producer_attempt="a1", evidence_scope="SYNTHETIC"
        )


def test_dependencies_are_pinned_to_registered_versions(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    protocol = write(tmp_path / "protocol.json", b"{}")
    results = write(tmp_path / "results.json", b"[]")
    base = campaign.register_artifact("protocol", protocol, producer_attempt="a1", evidence_scope=["NONE"])
    child = campaign.register_artifact(
        "results", results, producer_attempt="a2", evidence_scope=["DEVELOPMENT"], dependencies=["protocol"]
    )
    assert child.depends_on == (ArtifactRef("protocol", 1, base.sha256),)
    pinned = campaign.register_artifact(
        "other", results, producer_attempt="a2", evidence_scope="SYNTHETIC", dependencies=[base.ref, child]
    )
    assert pinned.depends_on == (base.ref, child.ref)
    refused: list[list[Any]] = [
        ["unknown"],
        [ArtifactRef("protocol", 1, "0" * 64)],
        [ArtifactRef("protocol", 2, base.sha256)],
        [{"artifact_id": "protocol", "version": True}],
        [{"version": 1}],
        [7],
    ]
    for bad in refused:
        with pytest.raises(InvalidDataError):
            campaign.register_artifact(
                "bad", results, producer_attempt="a2", evidence_scope="SYNTHETIC", dependencies=bad
            )


@pytest.mark.parametrize("scope", [[], ["NONE", "SYNTHETIC"], ["SYNTHETIC", "SYNTHETIC"], ["synthetic"], ["TEST"], [1]])
def test_evidence_scope_is_validated(tmp_path: Path, scope: Any) -> None:
    campaign = make_campaign(tmp_path)
    report = write(tmp_path / "r.bin", b"x")
    with pytest.raises(InvalidDataError):
        campaign.register_artifact("x", report, producer_attempt="a1", evidence_scope=scope)


def test_registry_detects_rewritten_or_torn_history(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    for name in ("x", "y"):
        campaign.register_artifact(
            name, write(tmp_path / f"{name}.bin", name.encode()), producer_attempt="a1", evidence_scope="SYNTHETIC"
        )
    original = campaign.registry_path.read_bytes()
    lines = original.split(b"\n")
    doc = json.loads(lines[0])
    doc["sha256"] = "0" * 64
    lines[0] = canonical_json(doc).encode()
    campaign.registry_path.write_bytes(b"\n".join(lines))
    with pytest.raises(InvalidDataError):  # the next record's hash chain no longer matches
        campaign.artifacts()
    campaign.registry_path.write_bytes(original.replace(b'","', b'", "', 1))
    with pytest.raises(InvalidDataError):  # still valid JSON, but no longer canonical
        campaign.artifacts()
    campaign.registry_path.write_bytes(original[:-1])
    with pytest.raises(InvalidDataError):  # torn final line
        campaign.artifacts()


@given(
    ops=st.lists(
        st.tuples(st.sampled_from(["a", "b", "c"]), st.binary(max_size=6), st.sampled_from(["p1", "p2"])),
        min_size=1,
        max_size=12,
    )
)
@settings(max_examples=40, deadline=None)
def test_registry_is_append_only_with_contiguous_versions(
    tmp_path_factory: pytest.TempPathFactory, ops: list[tuple[str, bytes, str]]
) -> None:
    root = tmp_path_factory.mktemp("registry")
    campaign = make_campaign(root)
    previous = b""
    returned = {}
    for artifact_id, content, attempt in ops:
        source = write(root / "data" / f"{artifact_id}.bin", content)
        record = campaign.register_artifact(artifact_id, source, producer_attempt=attempt, evidence_scope="SYNTHETIC")
        data = campaign.registry_path.read_bytes()
        assert data.startswith(previous)
        previous = data
        returned[(record.artifact_id, record.version)] = record
    records = campaign.artifacts()
    assert {(r.artifact_id, r.version): r for r in records} == returned
    for artifact_id in {r.artifact_id for r in records}:
        chain = [r for r in records if r.artifact_id == artifact_id]
        assert [r.version for r in chain] == list(range(1, len(chain) + 1))
        for older, newer in itertools.pairwise(chain):
            assert (older.sha256, older.producer_attempt) != (newer.sha256, newer.producer_attempt)


# ---------------------------------------------------------------------------
# Data-exposure ledger
# ---------------------------------------------------------------------------

EXPOSURE: dict[str, Any] = {
    "source": "recorder:venue_a+venue_b",
    "period": "2024-01-01T00:00Z/2023-01-31T00:00Z",
    "assigned_role": "development",
    "previously_exposed": True,
    "exposure_kind": "HISTORICAL_PRIOR_CAMPAIGN",
    "reason": "opened by earlier campaigns",
}


def test_exposure_rows_use_the_template_header_and_round_trip(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    record = campaign.record_exposure("2024-01-01", **EXPOSURE, manifest_hash="b" * 64)
    text = campaign.exposure_ledger_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "campaign_id,dataset_id,source,period,assigned_role,previously_exposed,exposure_kind,"
        "exposed_at,reason,protocol_id,manifest_hash"
    )
    row = next(csv.DictReader(io.StringIO(text)))
    assert row["previously_exposed"] == "true"
    assert row["protocol_id"] == ""
    assert row["exposed_at"] == "2026-10-21T12:00:01.000000+00:00"
    assert campaign.exposures() == (record,)


@pytest.mark.parametrize(
    "override",
    [
        {"assigned_role": "test"},
        {"exposure_kind": "PEEK"},
        {"previously_exposed": "yes"},
        {"exposed_at": datetime(2026, 1, 1)},
        {"exposed_at": "2026-01-01T00:00:00"},
        {"manifest_hash": "XYZ"},
        {"reason": "multi\nline"},
        {"reason": ""},
        {"source": " padded"},
        {"previously_exposed": False},  # a historical prior-campaign row documents earlier exposure
    ],
)
def test_exposure_fields_are_validated(tmp_path: Path, override: dict[str, Any]) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):
        campaign.record_exposure("2024-01-01", **(EXPOSURE | override))
    assert not campaign.exposure_ledger_path.exists()


def test_exposure_timestamps_are_normalised_to_utc(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    local = datetime.fromisoformat("2026-10-21T14:00:00+02:00")
    record = campaign.record_exposure("2024-01-01", **EXPOSURE, exposed_at=local)
    assert record.exposed_at == "2026-10-21T12:00:00.000000+00:00"


def test_exposure_history_can_never_be_reset(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    meta = EXPOSURE | {"previously_exposed": False, "exposure_kind": "METADATA_ONLY"}
    campaign.record_exposure("2023-05-01", **(meta | {"assigned_role": "reserved"}))
    # Metadata-only lookups are not exposure: the dataset may still become fresh test data.
    campaign.record_exposure("2023-05-01", **(meta | {"assigned_role": "fresh_test"}))
    campaign.record_exposure("2023-04-01", **(EXPOSURE | {"exposure_kind": "RAW_VALUES"}))
    with pytest.raises(InvalidDataError):  # opened data is never fresh again
        campaign.record_exposure("2023-04-01", **(meta | {"assigned_role": "fresh_test"}))
    with pytest.raises(InvalidDataError):  # and its exposure flag cannot be reset
        campaign.record_exposure(
            "2023-04-01", **(EXPOSURE | {"previously_exposed": False, "exposure_kind": "DESCRIPTIVE_SUMMARY"})
        )
    with pytest.raises(InvalidDataError):  # an opening is only ever a fresh-test opening
        campaign.record_exposure("2023-06-01", **(meta | {"exposure_kind": "FRESH_TEST_OPENING"}))
    with pytest.raises(InvalidDataError):  # a fresh role cannot carry prior exposure
        campaign.record_exposure(
            "2023-07-01", **(EXPOSURE | {"assigned_role": "fresh_test", "exposure_kind": "RAW_VALUES"})
        )
    assert len(campaign.exposures()) == 3


# ---------------------------------------------------------------------------
# Trial ledger
# ---------------------------------------------------------------------------

TRIAL: dict[str, Any] = {
    "protocol_id": "p-1",
    "trial_id": "t-0001",
    "stage_id": "06",
    "policy_config_id": "cfg-1",
    "account_id": "acct-1",
    "scenario": "base",
    "timing_mode": "HISTORICAL_REPLAY",
    "dataset_id": "synthetic-7",
    "seed": 12_345,
    "status": "COMPLETED",
    "validity": "VALID",
    "net_pnl": 0.0012345,
    "fees": 0.0001,
    "turnover_units": 2_000_000,
    "residual_inventory": {"venue_a/ALPHA-USD": 0, "venue_b/ALPHA-PERP": 0},
    "evidence_scope": "SYNTHETIC",
    "artifact_path": "runs/c-001/stages/06/a1/replay.json",
}


def test_trial_rows_use_the_template_header_and_round_trip(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    record = campaign.record_trial(**TRIAL)
    text = campaign.trial_ledger_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == (
        "campaign_id,protocol_id,trial_id,stage_id,fit_job_id,policy_config_id,account_id,"
        "scenario,timing_mode,dataset_id,seed,status,validity,net_pnl,fees,"
        "turnover_units,residual_inventory,evidence_scope,artifact_path,notes"
    )
    row = next(csv.DictReader(io.StringIO(text)))
    assert row["net_pnl"] == "0.0012345"
    assert row["fit_job_id"] == "" and row["notes"] == ""
    assert row["residual_inventory"] == '{"venue_a/ALPHA-USD":0,"venue_b/ALPHA-PERP":0}'
    assert campaign.trials() == (record,)


def test_invalid_accounts_have_an_empty_score_never_nan_or_zero(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    invalid = TRIAL | {
        "validity": "INVALID",
        "net_pnl": None,
        "notes": "not_flat",
        "residual_inventory": {"venue_a/ALPHA-USD": 1_000, "venue_b/ALPHA-PERP": 0},
    }
    campaign.record_trial(**invalid)
    rows = list(csv.DictReader(io.StringIO(campaign.trial_ledger_path.read_text(encoding="utf-8"))))
    assert rows[0]["net_pnl"] == ""
    assert all(v.lower() not in {"nan", "inf", "-inf", "infinity"} for v in rows[0].values())
    refused = [
        invalid | {"trial_id": "t-2", "net_pnl": 0.0},  # no ordinary profit for an invalid account
        TRIAL | {"trial_id": "t-3", "net_pnl": float("nan")},
        TRIAL | {"trial_id": "t-4", "net_pnl": float("inf")},
        TRIAL | {"trial_id": "t-5", "residual_inventory": {"venue_a/ALPHA-USD": 100}},  # not flat
        TRIAL | {"trial_id": "t-6", "status": "FAILED"},  # a failed run has no score
        TRIAL | {"trial_id": "t-7", "account_id": None, "policy_config_id": "cfg-1"},
        # VALID alone is no evidence of a flat account: a score needs the recorded terminal inventory
        TRIAL | {"trial_id": "t-8", "residual_inventory": None},
        TRIAL | {"trial_id": "t-9", "residual_inventory": {}},
    ]
    for fields in refused:
        with pytest.raises(InvalidDataError):
            campaign.record_trial(**fields)
    assert len(campaign.trials()) == 1
    # Without a score the terminal inventory may be unknown (an aborted replay).
    campaign.record_trial(
        **(
            TRIAL
            | {
                "trial_id": "t-10",
                "status": "ABORTED",
                "validity": "INCOMPLETE_OBSERVATION",
                "net_pnl": None,
                "residual_inventory": None,
            }
        )
    )


@pytest.mark.parametrize(
    "override",
    [
        {"status": "DONE"},
        {"validity": "OK"},
        {"timing_mode": "REALTIME"},
        {"evidence_scope": "PRODUCTION"},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
        {"turnover_units": -5},
        {"turnover_units": 1.5},
        {"fees": float("nan")},
        {"notes": ""},
        {"notes": "two\nlines"},
        {"trial_id": " t"},
        {"account_id": "acct-1", "policy_config_id": None},
        {"fit_job_id": None, "policy_config_id": None, "account_id": None, "net_pnl": None},
        {"residual_inventory": {"": 0}},
        {"residual_inventory": {"venue_a/ALPHA-USD": 0.5}},
        # recorded paths are relative to the campaign root: no user-specific absolute paths, no escapes
        {"artifact_path": "/Users/someone/private/replay.json"},
        {"artifact_path": "../other-campaign/replay.json"},
        {"artifact_path": "runs/c-001/../../replay.json"},
        # and canonical: no ".", no empty or "." component, no trailing slash
        {"artifact_path": "."},
        {"artifact_path": "runs/"},
        {"artifact_path": "./runs/x"},
        {"artifact_path": "a//b"},
        {"artifact_path": "runs/./x"},
    ],
)
def test_trial_fields_are_validated(tmp_path: Path, override: dict[str, Any]) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):
        campaign.record_trial(**(TRIAL | override))
    assert not campaign.trial_ledger_path.exists()


def test_trial_ids_are_unique_and_retries_are_new_rows(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    with pytest.raises(ImmutableRecordError):
        campaign.record_trial(**TRIAL)
    campaign.record_trial(**(TRIAL | {"trial_id": "t-0002", "notes": "RETRY_OF=t-0001"}))
    assert [t.trial_id for t in campaign.trials()] == ["t-0001", "t-0002"]


def test_retries_are_not_new_trials_in_the_budget_counts(tmp_path: Path) -> None:
    """Pack stage 04: an idempotent retry (``RETRY_OF=<trial_id>``) is a new ledger row but not a new trial, so
    it must not consume the account-replay budget; a citation of an unrecorded id is an ordinary new trial."""
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    campaign.record_trial(**(TRIAL | {"trial_id": "t-0002", "notes": "RETRY_OF=t-0001 after a worker restart"}))
    campaign.record_trial(**(TRIAL | {"trial_id": "t-0003", "notes": "RETRY_OF=t-9999"}))  # cites nothing recorded
    campaign.record_trial(
        **(
            TRIAL
            | {
                "trial_id": "t-0004",
                "notes": "RETRY_OF=t-0003",
                "status": "FAILED",
                "validity": "INVALID",
                "net_pnl": None,
            }
        )
    )
    counts = campaign.trial_counts()
    assert (counts.rows, counts.account_replays, counts.valid_accounts, counts.retries) == (4, 2, 2, 2)
    assert (counts.fit_jobs, counts.policy_configs) == (0, 1)


def test_a_valid_retry_of_a_failed_trial_counts_one_validly_replayed_account(tmp_path: Path) -> None:
    """Regression: a retry row was dropped from the counters altogether, so the motivating case (the original
    FAILED after a worker restart, the retry COMPLETED/VALID) reported account_replays=1 but valid_accounts=0.
    Retry chains are resolved to their root trial and count one object per chain."""
    campaign = make_campaign(tmp_path)
    failed = {"status": "FAILED", "validity": "INVALID", "net_pnl": None}
    campaign.record_trial(**(TRIAL | failed))
    campaign.record_trial(**(TRIAL | failed | {"trial_id": "t-0002", "notes": "RETRY_OF=t-0001 (restart)"}))
    counts = campaign.trial_counts()
    assert (counts.rows, counts.account_replays, counts.valid_accounts, counts.retries) == (2, 1, 0, 1)
    campaign.record_trial(**(TRIAL | {"trial_id": "t-0003", "notes": "RETRY_OF=t-0002"}))  # a retry of a retry
    counts = campaign.trial_counts()
    assert (counts.rows, counts.account_replays, counts.valid_accounts, counts.retries) == (3, 1, 1, 2)
    assert (counts.fit_jobs, counts.policy_configs) == (0, 1)
    # A later failed retry does not take the valid replay away: the account was validly replayed once.
    campaign.record_trial(**(TRIAL | failed | {"trial_id": "t-0004", "notes": "RETRY_OF=t-0001"}))
    counts = campaign.trial_counts()
    assert (counts.account_replays, counts.valid_accounts, counts.retries) == (1, 1, 3)


@pytest.mark.parametrize(
    "override",
    [
        {"account_id": "acct-other"},
        {"policy_config_id": "cfg-other"},
        {"stage_id": "07"},
        {"scenario": "stress"},
        {"dataset_id": "synthetic-8"},
        {"seed": 1},
        {"timing_mode": "FULL_PIPELINE"},
        {"fit_job_id": "fit-1"},
        {"evidence_scope": "NONE"},
    ],
)
def test_a_retry_citation_must_name_the_same_trial(tmp_path: Path, override: dict[str, Any]) -> None:
    """Regression: any row whose notes said ``RETRY_OF=<earlier id>`` left the budget, whatever its identity
    columns, so a free-text note could run a second account or configuration without consuming budget."""
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    with pytest.raises(InvalidDataError):
        campaign.record_trial(**(TRIAL | override | {"trial_id": "t-0002", "notes": "RETRY_OF=t-0001 (other)"}))
    assert [t.trial_id for t in campaign.trials()] == ["t-0001"]
    # The same mismatch inside one batch is refused as a whole.
    with pytest.raises(InvalidDataError):
        campaign.record_trials(
            [
                campaign.make_trial(**(TRIAL | {"trial_id": "t-0003"})),
                campaign.make_trial(**(TRIAL | override | {"trial_id": "t-0004", "notes": "RETRY_OF=t-0003"})),
            ]
        )
    assert [t.trial_id for t in campaign.trials()] == ["t-0001"]


def test_counts_treat_a_mislabelled_retry_in_an_older_ledger_as_a_new_trial(tmp_path: Path) -> None:
    """A ledger written before the identity check may carry a RETRY_OF citation of a different trial: the
    counters do not trust the note and charge the row to the budget as a new trial."""
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    other = campaign.make_trial(
        **(TRIAL | {"trial_id": "t-0002", "account_id": "acct-other", "notes": "RETRY_OF=t-0001"})
    )
    with campaign.trial_ledger_path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(other.to_row())
    counts = campaign.trial_counts()
    assert (counts.rows, counts.account_replays, counts.valid_accounts, counts.retries) == (2, 2, 2, 0)


def test_trials_on_real_data_require_a_recorded_exposure(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    dev = TRIAL | {"dataset_id": "2024-01-01", "evidence_scope": "DEVELOPMENT"}
    with pytest.raises(InvalidDataError):
        campaign.record_trial(**dev)
    with pytest.raises(InvalidDataError):  # development evidence must name its dataset
        campaign.record_trial(**(dev | {"dataset_id": None}))
    campaign.record_exposure("2024-01-01", **(EXPOSURE | {"exposure_kind": "RAW_VALUES"}))
    campaign.record_trial(**dev)
    fresh = TRIAL | {"trial_id": "t-9", "dataset_id": "2024-01-01", "evidence_scope": "FRESH_TEST"}
    with pytest.raises(InvalidDataError):  # no fresh-test opening of that dataset
        campaign.record_trial(**fresh)


def test_trial_counts_separate_fit_jobs_configurations_and_replays(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    fit = {
        "fit_job_id": "fit-1",
        "policy_config_id": None,
        "account_id": None,
        "net_pnl": None,
        "residual_inventory": None,
        "turnover_units": None,
        "fees": None,
    }
    campaign.record_trials([])  # an empty batch writes nothing
    assert not campaign.trial_ledger_path.exists()
    campaign.record_trial(**(TRIAL | fit | {"trial_id": "f1"}))
    campaign.record_trial(**(TRIAL | {"trial_id": "r1"}))
    campaign.record_trial(**(TRIAL | {"trial_id": "r2", "account_id": "acct-2"}))
    campaign.record_trial(
        **(
            TRIAL
            | {
                "trial_id": "r3",
                "policy_config_id": "cfg-2",
                "account_id": "acct-3",
                "status": "FAILED",
                "validity": "INVALID",
                "net_pnl": None,
            }
        )
    )
    counts = campaign.trial_counts()
    assert (counts.rows, counts.fit_jobs, counts.policy_configs, counts.account_replays, counts.valid_accounts) == (
        4,
        1,
        2,
        3,
        2,
    )


def test_trial_batches_are_written_atomically(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    first = campaign.make_trial(**(TRIAL | {"trial_id": "b1"}))
    second = campaign.make_trial(**(TRIAL | {"trial_id": "b2"}))
    duplicate = campaign.make_trial(**(TRIAL | {"trial_id": "b1"}))
    with pytest.raises(ImmutableRecordError):  # one bad row rejects the whole batch
        campaign.record_trials([first, second, duplicate])
    assert campaign.trials() == ()
    campaign.record_trials([first, second])
    assert campaign.trials() == (first, second)


TEXT = st.text(alphabet=st.characters(exclude_categories=("Cc", "Cs", "Zl", "Zp")), min_size=1, max_size=12).filter(
    lambda s: s == s.strip()
)


@st.composite
def trial_fields(draw: st.DrawFn) -> dict[str, Any]:
    account = draw(st.one_of(st.none(), TEXT))
    status = draw(st.sampled_from(["COMPLETED", "FAILED", "ABORTED", "CANCELLED", "INTERRUPTED"]))
    validity = draw(st.sampled_from(["VALID", "INVALID", "INCOMPLETE_OBSERVATION"]))
    scored = account is not None and status == "COMPLETED" and validity == "VALID"
    net = draw(st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))) if scored else None
    inventory = (
        st.dictionaries(TEXT, st.just(0), min_size=1, max_size=3)
        if net is not None  # a score: flat
        else st.one_of(st.none(), st.dictionaries(TEXT, st.integers(-(10**12), 10**12), max_size=3))
    )
    relative = TEXT.filter(
        lambda s: not s.startswith("/") and ".." not in s.split("/") and PurePosixPath(s).as_posix() == s and s != "."
    )
    return {
        "protocol_id": draw(TEXT),
        "trial_id": draw(TEXT),
        "stage_id": draw(TEXT),
        "fit_job_id": draw(st.one_of(st.none(), TEXT)),
        "policy_config_id": draw(TEXT),
        "account_id": account,
        "scenario": draw(st.one_of(st.none(), TEXT)),
        "timing_mode": draw(st.sampled_from([None, "HISTORICAL_REPLAY", "FULL_PIPELINE"])),
        "dataset_id": draw(st.one_of(st.none(), TEXT)),
        "seed": draw(st.one_of(st.none(), st.integers(0, 2**64 - 1))),
        "status": status,
        "validity": validity,
        "net_pnl": net,
        "fees": draw(st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False))),
        "turnover_units": draw(st.one_of(st.none(), st.integers(0, 10**15))),
        "residual_inventory": draw(inventory),
        "evidence_scope": draw(st.sampled_from(["SYNTHETIC", "NONE"])),
        "artifact_path": draw(st.one_of(st.none(), relative)),
        "notes": draw(st.one_of(st.none(), TEXT)),
    }


@given(rows=st.lists(trial_fields(), min_size=1, max_size=4, unique_by=lambda f: f["trial_id"]))
@settings(max_examples=50, deadline=None)
def test_trial_ledger_csv_round_trip(tmp_path_factory: pytest.TempPathFactory, rows: list[dict[str, Any]]) -> None:
    campaign = make_campaign(tmp_path_factory.mktemp("trials"))
    written = [campaign.record_trial(**fields) for fields in rows]
    assert campaign.trials() == tuple(written)
    with campaign.trial_ledger_path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.reader(handle))
    assert parsed[0] == list(TRIAL_LEDGER_HEADER)
    assert all(len(row) == len(TRIAL_LEDGER_HEADER) for row in parsed)
    assert [row[2] for row in parsed[1:]] == [fields["trial_id"] for fields in rows]


def _append_trials(root: str, worker: int, count: int) -> None:
    campaign = Campaign(Path(root), "c-001")
    for i in range(count):
        campaign.record_trial(**(TRIAL | {"trial_id": f"w{worker}-{i}"}))


def test_concurrent_writers_never_lose_rows(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_append_trials, args=(str(tmp_path), w, 8)) for w in range(4)]
    for process in workers:
        process.start()
    for process in workers:
        process.join(timeout=120)
        assert process.exitcode == 0
    ids = [t.trial_id for t in Campaign(tmp_path, "c-001").trials()]
    assert sorted(ids) == sorted(f"w{w}-{i}" for w in range(4) for i in range(8))


def test_ledgers_refuse_rows_of_another_campaign(tmp_path: Path) -> None:
    source = make_campaign(tmp_path, "c-001")
    source.record_trial(**TRIAL)
    source.record_exposure("2024-01-01", **EXPOSURE)
    target = make_campaign(tmp_path, "c-002")
    for path in (source.trial_ledger_path, source.exposure_ledger_path):
        write(target.directory / "registry" / path.name, path.read_bytes())
    with pytest.raises(InvalidDataError):
        target.trials()
    with pytest.raises(InvalidDataError):
        target.exposures()


def test_ledger_snapshots_detect_rewritten_history(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    snapshot = campaign.ledger_snapshot("trial-ledger")
    campaign.record_trial(**(TRIAL | {"trial_id": "t-0002"}))
    campaign.verify_append_only(snapshot)
    text = campaign.trial_ledger_path.read_text(encoding="utf-8").replace("t-0001", "t-000X")
    campaign.trial_ledger_path.write_text(text, encoding="utf-8")
    with pytest.raises(InvalidDataError):
        campaign.verify_append_only(snapshot)
    with pytest.raises(InvalidConfigError):
        campaign.ledger_snapshot("scratch")


def test_failed_writes_leave_no_partial_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    before = campaign.trial_ledger_path.read_bytes()

    def disk_full(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", disk_full)
    with pytest.raises(OSError):
        campaign.record_trial(**(TRIAL | {"trial_id": "t-0002"}))
    monkeypatch.undo()
    assert campaign.trial_ledger_path.read_bytes() == before
    assert not list(campaign.directory.rglob("*.tmp"))


# ---------------------------------------------------------------------------
# Stage records
# ---------------------------------------------------------------------------


def test_stage_record_has_every_template_field(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    written = campaign.write_stage_record(
        "03",
        "a1",
        execution_status="COMPLETE",
        engineering_status="PASS",
        evidence_scope=["SYNTHETIC", "DEVELOPMENT"],
        economic_outcome="NOT_EVALUATED",
        commands=[{"cmd": "pytest -q", "exit_code": 0, "passed": 53}],
        started_at=T0,
        finished_at=T0 + timedelta(minutes=5),
    )
    assert written.path == campaign.directory / "stages" / "03" / "a1" / "stage-record.json"
    assert written.sha256 == sha256_bytes(written.path.read_bytes())
    doc = loads_strict(written.path.read_text(encoding="utf-8"))
    assert tuple(sorted(doc)) == tuple(sorted(STAGE_RECORD_FIELDS))
    assert (doc["schema_version"], doc["campaign_id"], doc["stage_id"], doc["attempt_id"]) == (5, "c-001", "03", "a1")
    assert doc["evidence_scope"] == ["SYNTHETIC", "DEVELOPMENT"]
    assert doc["started_at"] == "2026-10-21T12:00:00.000000+00:00"
    assert doc["finished_at"] == "2026-10-21T12:05:00.000000+00:00"
    assert doc["delivery_profile"] == "research_run" and doc["failed_checks"] == []
    assert campaign.read_stage_record("03", "a1") == doc


@pytest.mark.parametrize(
    "fields",
    [
        {"execution_status": "DONE"},
        {"engineering_status": "PASSED"},
        {"evidence_scope": []},
        {"evidence_scope": ["NONE", "DEVELOPMENT"]},
        {"evidence_scope": "SYNTHETIC"},
        {"economic_outcome": "PROFITABLE"},
        {"execution_status": "COMPLETE", "engineering_status": "FAIL"},
        {"execution_status": "DEFERRED", "defer_or_block_reason": "needs a GPU"},
        {"execution_status": "BLOCKED"},
        {"economic_outcome": "TEST_PASSED", "evidence_scope": ["DEVELOPMENT"]},
        {"economic_outcome": "DEVELOPMENT_POSITIVE", "evidence_scope": ["SYNTHETIC"]},
        {"unknown_field": 1},
        {"schema_version": 4},
        {"campaign_id": "other"},
        {"started_at": "2026-10-21T12:00:00"},
        {"started_at": T0 + timedelta(hours=1), "finished_at": T0},
        {"commands": [float("nan")]},
        {"commands": "pytest"},
        {"delivery_profile": "demo"},
    ],
)
def test_stage_record_refuses_inconsistent_content(tmp_path: Path, fields: dict[str, Any]) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):
        campaign.write_stage_record("03", "a1", **fields)
    assert not campaign.stage_record_path("03", "a1").exists()


def test_deferred_and_blocked_stages_carry_their_reasons(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.write_stage_record(
        "08",
        "a1",
        execution_status="DEFERRED",
        engineering_status="FAIL",
        defer_or_block_reason="sequence model failed conformance",
        reentry_criterion="adapter passes the fill-response suite",
    )
    campaign.write_stage_record(
        "09",
        "a1",
        execution_status="BLOCKED",
        engineering_status="FAIL",
        defer_or_block_reason="ledger invariant violated",
    )


def test_stage_attempts_are_never_overwritten(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    first = campaign.write_stage_record("03", "a1", execution_status="RUNNING")
    before = first.path.read_bytes()
    with pytest.raises(ImmutableRecordError):
        campaign.write_stage_record("03", "a1", execution_status="COMPLETE", engineering_status="PASS")
    assert first.path.read_bytes() == before
    second = campaign.write_stage_record("03", "a2", execution_status="COMPLETE", engineering_status="PASS")
    assert second.path.parent.name == "a2"
    assert campaign.stage_records() == (first.path, second.path)


@pytest.mark.parametrize(
    ("stage", "attempt"), [("../x", "a1"), ("03", "a/b"), ("03", ""), (".hidden", "a1"), ("03", "a1\n")]
)
def test_stage_and_attempt_ids_must_be_safe_path_components(tmp_path: Path, stage: str, attempt: str) -> None:
    with pytest.raises(InvalidConfigError):
        make_campaign(tmp_path).write_stage_record(stage, attempt)


def test_stage_record_artifact_references_must_match_the_registry(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    spec = campaign.register_artifact(
        "execution_spec",
        write(tmp_path / "spec.json", b"{}"),
        producer_attempt="a0",
        producer_stage="03",
        evidence_scope=["SYNTHETIC"],
    )
    controller = campaign.register_artifact(
        "controller_spec",
        write(tmp_path / "ctl.json", b"[]"),
        producer_attempt="a1",
        producer_stage="06",
        evidence_scope=["SYNTHETIC"],
    )
    written = campaign.write_stage_record(
        "06", "a1", consumed_artifacts=["execution_spec"], produced_artifacts=[controller]
    )
    doc = loads_strict(written.path.read_text(encoding="utf-8"))
    assert doc["consumed_artifacts"] == [
        {"artifact_id": "execution_spec", "version": 1, "sha256": spec.sha256, "producer_attempt": "a0"}
    ]
    assert doc["produced_artifacts"] == [
        {"artifact_id": "controller_spec", "version": 1, "sha256": controller.sha256, "producer_attempt": "a1"}
    ]
    with pytest.raises(InvalidDataError):  # produced by a different attempt
        campaign.write_stage_record("06", "a2", produced_artifacts=[controller])
    with pytest.raises(InvalidDataError):
        campaign.write_stage_record("07", "b1", consumed_artifacts=[ArtifactRef("execution_spec", 1, "0" * 64)])
    with pytest.raises(InvalidDataError):
        campaign.write_stage_record("07", "b2", consumed_artifacts=["missing"])


# ---------------------------------------------------------------------------
# Freeze seal, opening and evaluation records
# ---------------------------------------------------------------------------


def seal_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {name: hashlib.sha256(name.encode()).hexdigest() for name in REQUIRED_SEAL_HASHES}
    # Sealed before the deterministic campaign clock (T0): openings and dispositions stamped by that clock follow it.
    payload.update(
        admission_decision="READY_TO_OPEN",
        sealed_at="2026-10-21T11:30:00+00:00",
        scenario_order=["base", "higher_fees", "longer_delay", "combined_stress"],
        seed_policy="seed_from(campaign_id, trial_id)",
    )
    payload.update(overrides)
    return payload


FRESH = OpenedDataset(
    dataset_id="2023-05-01",
    sha256="c" * 64,
    source="recorder:venue_a+venue_b",
    period="2023-05-01T00:00Z/2023-05-02T00:00Z",
)


def test_seal_is_canonical_json_whose_hash_is_the_file_hash(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())
    data = seal.path.read_bytes()
    doc = loads_strict(data)
    assert data == canonical_json(doc).encode("utf-8")
    assert seal.sha256 == sha256_bytes(data) == sha256_json(doc)
    assert (doc["schema_version"], doc["status"], doc["phase"], doc["campaign_id"]) == (
        5,
        "SEALED",
        "PRE_EVALUATION",
        "c-001",
    )
    assert doc["sealed_at"] == "2026-10-21T11:30:00.000000+00:00"
    assert tuple(sorted(doc)) == tuple(sorted(FREEZE_FIELDS))
    assert campaign.read_freeze().sha256 == seal.sha256
    assert campaign.read_freeze().document["admission_decision"] == "READY_TO_OPEN"


def test_seal_defaults_its_time_to_the_campaign_clock(tmp_path: Path) -> None:
    payload = seal_payload()
    del payload["sealed_at"]
    seal = make_campaign(tmp_path).seal_freeze(payload)
    assert seal.document["sealed_at"] == "2026-10-21T12:00:01.000000+00:00"


def test_a_campaign_has_exactly_one_seal(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())
    before = seal.path.read_bytes()
    with pytest.raises(ImmutableRecordError):
        campaign.seal_freeze(seal_payload(reason="second thoughts"))
    assert seal.path.read_bytes() == before


@pytest.mark.parametrize(
    "override",
    [
        {"code_hash": None},
        {"protocol_hash": "ABC" * 21 + "D"},
        {"initial_model_hash": "123"},
        {"admission_decision": "NOT_EVALUATED"},
        {"economic_decision": "TEST_PASSED"},
        {"opening_record": None},
        {"result_paths": []},
        {"decision": "promote"},
        {"status": "OPENED"},
        {"phase": "EVALUATION_DISPOSITION"},
        {"campaign_id": "other"},
        {"schema_version": 4},
        {"sealed_at": "2026-10-21T12:30:00"},
        {"surprise": 1},
        {"evaluation_mode": "ONLINE"},
        {"delivery_profile": "demo"},
        {"reason": float("nan")},
        {"scenario_order": ["base", "base"]},
        {"scenario_order": [""]},
        {"scenario_order": "base"},
    ],
)
def test_seal_refuses_incomplete_or_future_content(tmp_path: Path, override: dict[str, Any]) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):
        campaign.seal_freeze(seal_payload(**override))
    assert not campaign.freeze_path.exists()


def test_seal_closure_pins_registered_artifact_hashes(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    spec = campaign.register_artifact(
        "execution_spec", write(tmp_path / "spec.json", b"{}"), producer_attempt="a3", evidence_scope=["SYNTHETIC"]
    )
    for bad in (
        ["execution_spec"],
        [{"artifact_id": "execution_spec", "sha256": "0" * 64}],
        [{"artifact_id": "unregistered", "sha256": spec.sha256}],
    ):
        with pytest.raises(InvalidDataError):
            campaign.seal_freeze(seal_payload(selected_artifact_closure=bad))
    assert not campaign.freeze_path.exists()
    seal = campaign.seal_freeze(
        seal_payload(selected_artifact_closure=[spec, {"artifact_id": "execution_spec", "sha256": spec.sha256}])
    )
    assert seal.document["selected_artifact_closure"] == [
        {"artifact_id": "execution_spec", "version": 1, "sha256": spec.sha256},
        {"artifact_id": "execution_spec", "sha256": spec.sha256},
    ]


def test_hash_citations_pin_the_matching_version_after_newer_ones(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    spec = write(tmp_path / "spec.json", b"{}")
    v1 = campaign.register_artifact("execution_spec", spec, producer_attempt="a1", evidence_scope="SYNTHETIC")
    spec.write_bytes(b"[]")
    v2 = campaign.register_artifact("execution_spec", spec, producer_attempt="a2", evidence_scope="SYNTHETIC")
    pin: dict[str, Any] = {"artifact_id": "execution_spec", "sha256": v1.sha256}  # record shape
    child = campaign.register_artifact(
        "child",
        write(tmp_path / "child.bin", b"x"),
        producer_attempt="a3",
        evidence_scope="SYNTHETIC",
        dependencies=[pin],
    )
    assert child.depends_on == (v1.ref,)
    stage = campaign.write_stage_record("07", "b1", consumed_artifacts=[pin])
    assert loads_strict(stage.path.read_text(encoding="utf-8"))["consumed_artifacts"] == [
        v1.ref.to_json() | {"producer_attempt": "a1"}
    ]
    assert campaign.artifact("execution_spec") == v2  # an id alone still means the latest version
    with pytest.raises(InvalidDataError):  # a hash that no registered version carries
        campaign.write_stage_record("07", "b2", consumed_artifacts=[pin | {"sha256": "0" * 64}])
    seal = campaign.seal_freeze(seal_payload(selected_artifact_closure=[pin]))
    assert seal.document["selected_artifact_closure"] == [pin]
    campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    evaluation = campaign.append_evaluation(
        seal.sha256,
        status="COMPLETED",
        economic_decision="TEST_FAILED",
        reason="lost",
        opening_record_id="open-1",
        result_artifacts=[pin],
    )
    assert loads_strict(evaluation.path.read_text(encoding="utf-8"))["result_artifacts"] == [v1.ref.to_json()]
    spec.write_bytes(b"{}")  # v1's bytes registered again by a later attempt
    v3 = campaign.register_artifact("execution_spec", spec, producer_attempt="a4", evidence_scope="SYNTHETIC")
    later = campaign.register_artifact(
        "child",
        write(tmp_path / "child.bin", b"y"),
        producer_attempt="a5",
        evidence_scope="SYNTHETIC",
        dependencies=[pin],
    )
    assert v3.sha256 == v1.sha256 and later.depends_on == (v3.ref,)  # the latest such version


def test_opening_references_the_unchanged_seal_and_records_exposure(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())
    sealed = seal.path.read_bytes()
    opening = campaign.append_opening(
        seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="user approval recorded 2026-10-21"
    )
    doc = loads_strict(opening.path.read_text(encoding="utf-8"))
    assert doc["freeze_record_sha256"] == seal.sha256
    assert doc["opening_record_id"] == "open-1"
    assert doc["datasets"] == [
        {
            "dataset_id": "2023-05-01",
            "sha256": "c" * 64,
            "source": "recorder:venue_a+venue_b",
            "period": "2023-05-01T00:00Z/2023-05-02T00:00Z",
        }
    ]
    exposure = campaign.exposures()[-1]
    assert (
        exposure.dataset_id,
        exposure.assigned_role,
        exposure.exposure_kind,
        exposure.previously_exposed,
        exposure.manifest_hash,
    ) == ("2023-05-01", "fresh_test", "FRESH_TEST_OPENING", False, "c" * 64)
    assert seal.path.read_bytes() == sealed
    # Trials on the opened data are now admissible as fresh-test evidence.
    campaign.record_trial(**(TRIAL | {"dataset_id": "2023-05-01", "evidence_scope": "FRESH_TEST"}))


def test_only_one_opening_and_only_when_admitted(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):  # nothing sealed yet
        campaign.append_opening("0" * 64, opening_record_id="o", datasets=[FRESH], authorization="ok")
    seal = campaign.seal_freeze(seal_payload())
    with pytest.raises(InvalidDataError):  # wrong seal hash
        campaign.append_opening("0" * 64, opening_record_id="o", datasets=[FRESH], authorization="ok")
    campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    with pytest.raises(ImmutableRecordError):
        campaign.append_opening(
            seal.sha256,
            opening_record_id="open-2",
            datasets=[OpenedDataset("2023-06-01", "d" * 64, "recorder", "x")],
            authorization="ok",
        )
    closed = make_campaign(tmp_path, "c-002")
    no_test = closed.seal_freeze(seal_payload(admission_decision="NO_TEST_OPENING"))
    with pytest.raises(InvalidDataError):
        closed.append_opening(no_test.sha256, opening_record_id="o", datasets=[FRESH], authorization="ok")


def test_a_previously_exposed_dataset_cannot_be_opened_as_fresh(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.record_exposure("2023-04-01", **(EXPOSURE | {"exposure_kind": "RAW_VALUES"}))
    seal = campaign.seal_freeze(seal_payload())
    stale = OpenedDataset("2023-04-01", "e" * 64, "recorder", "2023-04-01")
    with pytest.raises(InvalidDataError):
        campaign.append_opening(seal.sha256, opening_record_id="o1", datasets=[stale], authorization="ok")
    assert len(campaign.exposures()) == 1
    assert campaign.openings() == ()


def test_no_test_disposition_never_fabricates_an_opening(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload(admission_decision="NO_TEST_OPENING"))
    written = campaign.append_evaluation(
        seal.sha256, status="NOT_EVALUATED", economic_decision="NO_CANDIDATE_PASSED", reason="pre-test gate failed"
    )
    doc = loads_strict(written.path.read_text(encoding="utf-8"))
    created = doc.pop("created_at")
    assert datetime.fromisoformat(created).utcoffset() == timedelta(0)
    assert doc == {
        "schema_version": 1,
        "phase": "EVALUATION_DISPOSITION",
        "status": "NOT_EVALUATED",
        "freeze_record_sha256": seal.sha256,
        "opening_record_id": None,
        "result_artifacts": [],
        "economic_decision": "NO_CANDIDATE_PASSED",
        "reason": "pre-test gate failed",
    }
    refused: list[dict[str, Any]] = [
        {"status": "NOT_EVALUATED", "economic_decision": "NOT_EVALUATED", "opening_record_id": "open-1"},
        {"status": "COMPLETED", "economic_decision": "TEST_PASSED"},
        {"status": "NOT_EVALUATED", "economic_decision": "TEST_PASSED"},
        {"status": "PASSED", "economic_decision": "NOT_EVALUATED"},
        {"status": "NOT_EVALUATED", "economic_decision": "NOT_EVALUATED", "reason": ""},
    ]
    for fields in refused:
        with pytest.raises(InvalidDataError):
            campaign.append_evaluation(seal.sha256, **({"reason": "r"} | fields))
    assert campaign.evaluations() == (written.path,)


def test_completed_evaluation_links_the_opening_and_registered_results(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())
    campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    result = campaign.register_artifact(
        "evaluation_report",
        write(tmp_path / "results" / "suite.json", b"{}"),
        producer_attempt="a14",
        producer_stage="14",
        evidence_scope=["FRESH_TEST"],
    )
    first = campaign.append_evaluation(
        seal.sha256,
        status="INTERRUPTED",
        economic_decision="INCONCLUSIVE",
        reason="power loss",
        opening_record_id="open-1",
        result_artifacts=[result],
    )
    second = campaign.append_evaluation(
        seal.sha256,
        status="COMPLETED",
        economic_decision="TEST_FAILED",
        reason="base case lost",
        opening_record_id="open-1",
        result_artifacts=[result.ref],
    )
    assert (first.path.name, second.path.name) == ("evaluation-0001.json", "evaluation-0002.json")
    doc = loads_strict(second.path.read_text(encoding="utf-8"))
    assert doc["result_artifacts"] == [{"artifact_id": "evaluation_report", "version": 1, "sha256": result.sha256}]
    refused: list[dict[str, Any]] = [
        {"opening_record_id": "open-2", "result_artifacts": [result]},
        {"opening_record_id": "open-1", "result_artifacts": []},
        {"opening_record_id": "open-1", "result_artifacts": [ArtifactRef("evaluation_report", 1, "0" * 64)]},
    ]
    for fields in refused:
        with pytest.raises(InvalidDataError):
            campaign.append_evaluation(
                seal.sha256, status="COMPLETED", economic_decision="TEST_FAILED", reason="r", **fields
            )
    with pytest.raises(InvalidDataError):  # an opened test cannot be filed as not evaluated
        campaign.append_evaluation(
            seal.sha256, status="NOT_EVALUATED", economic_decision="NOT_EVALUATED", reason="hide it"
        )
    assert campaign.evaluations() == (first.path, second.path)


OPENING_ROW: dict[str, Any] = EXPOSURE | {
    "assigned_role": "fresh_test",
    "previously_exposed": False,
    "exposure_kind": "FRESH_TEST_OPENING",
    "reason": "opened by hand",
}
FRESH_TRIAL: dict[str, Any] = TRIAL | {"trial_id": "run-1", "dataset_id": "2023-05-01", "evidence_scope": "FRESH_TEST"}


def open_test(campaign: Campaign) -> str:
    """Seal (READY_TO_OPEN) and record the declared opening of FRESH; returns the seal hash."""
    seal = campaign.seal_freeze(seal_payload())
    campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    return seal.sha256


def forge_opening_row(campaign: Campaign, dataset_id: str) -> None:
    """Append a FRESH_TEST_OPENING exposure row behind the API (a hand-edited ledger)."""
    row = ExposureRecord(
        campaign_id=campaign.campaign_id,
        dataset_id=dataset_id,
        source="recorder",
        period=dataset_id,
        assigned_role=DataRole.FRESH_TEST,
        previously_exposed=False,
        exposure_kind=ExposureKind.FRESH_TEST_OPENING,
        exposed_at=T0.isoformat(),
        reason="hand-edited",
    )
    path = campaign.exposure_ledger_path
    rows = [row.to_row()] if path.exists() else [list(EXPOSURE_LEDGER_HEADER), row.to_row()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def test_only_the_declared_opening_records_a_fresh_test_opening(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):  # no seal and no opening record: the gate is not skipped
        campaign.record_exposure("2023-05-01", **OPENING_ROW)
    assert not campaign.exposure_ledger_path.exists()
    open_test(campaign)
    with pytest.raises(InvalidDataError):  # nor can an undeclared second opening follow later
        campaign.record_exposure("2023-06-01", **OPENING_ROW)
    assert [(e.dataset_id, e.exposure_kind) for e in campaign.exposures()] == [("2023-05-01", "FRESH_TEST_OPENING")]


def test_fresh_test_trials_need_the_sealed_opening_of_their_dataset(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    with pytest.raises(InvalidDataError):  # nothing sealed or opened
        campaign.record_trial(**FRESH_TRIAL)
    seal = campaign.seal_freeze(seal_payload())
    with pytest.raises(InvalidDataError):  # sealed, but the test was never opened
        campaign.record_trial(**FRESH_TRIAL)
    campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    campaign.record_trial(**FRESH_TRIAL)
    forge_opening_row(campaign, "2023-06-01")
    with pytest.raises(InvalidDataError):  # a hand-written ledger row is not part of the opening
        campaign.record_trial(**(FRESH_TRIAL | {"trial_id": "run-2", "dataset_id": "2023-06-01"}))
    assert [t.trial_id for t in campaign.trials()] == ["run-1"]


@pytest.mark.parametrize(("admission", "cited"), [("NO_TEST_OPENING", None), ("READY_TO_OPEN", "0" * 64)])
def test_hand_written_openings_admit_no_fresh_test_trials(tmp_path: Path, admission: str, cited: str | None) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload(admission_decision=admission))
    forge_opening_row(campaign, "2023-05-01")
    write(
        campaign.directory / "freeze" / "openings" / "open-1.json",
        json.dumps(
            {
                "schema_version": 1,
                "campaign_id": "c-001",
                "opening_record_id": "open-1",
                "freeze_record_sha256": cited or seal.sha256,
                "datasets": [FRESH.to_json()],
                "opened_at": T0.isoformat(),
                "authorization": "hand-written",
            }
        ).encode(),
    )
    with pytest.raises(InvalidDataError):  # a no-test seal, or an opening citing another seal
        campaign.record_trial(**FRESH_TRIAL)
    assert campaign.trials() == ()


def test_an_opened_test_admits_no_tuning_only_labelled_diagnostics(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    open_test(campaign)
    on_test = TRIAL | {"dataset_id": "2023-05-01"}
    refused: list[dict[str, Any]] = [
        on_test | {"trial_id": "dev-1", "evidence_scope": "DEVELOPMENT"},  # tuning on the test
        on_test
        | {
            "trial_id": "dev-2",
            "evidence_scope": "DEVELOPMENT",
            "notes": "DIAGNOSTIC: still tuning",
        },  # a label does not admit development
        on_test | {"trial_id": "syn-1", "evidence_scope": "SYNTHETIC"},  # opened data is real
        on_test | {"trial_id": "rerun-1", "evidence_scope": "NONE"},  # an unlabelled rerun
        on_test
        | {
            "trial_id": "rerun-2",
            "evidence_scope": "FRESH_TEST",
            "notes": "DIAGNOSTIC: corrected fees",
        },  # a diagnostic is never test evidence
    ]
    for fields in refused:
        with pytest.raises(InvalidDataError):
            campaign.record_trial(**fields)
    diagnostic = campaign.record_trial(
        **(on_test | {"trial_id": "diag-1", "evidence_scope": "NONE", "notes": "DIAGNOSTIC: corrected fee units"})
    )
    assert campaign.trials() == (diagnostic,)


def test_completed_evaluations_are_final(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = open_test(campaign)
    first = campaign.register_artifact(
        "suite_v1", write(tmp_path / "r1.json", b"{}"), producer_attempt="a14", evidence_scope=["FRESH_TEST"]
    )
    retuned = campaign.register_artifact(
        "suite_v2", write(tmp_path / "r2.json", b"[1]"), producer_attempt="a15", evidence_scope=["FRESH_TEST"]
    )
    failed = campaign.append_evaluation(
        seal,
        status="COMPLETED",
        economic_decision="TEST_FAILED",
        reason="base case lost",
        opening_record_id="open-1",
        result_artifacts=[first],
    )
    for status, decision in (
        ("COMPLETED", "TEST_PASSED"),
        ("COMPLETED", "TEST_FAILED"),
        ("FAILED", "INCONCLUSIVE"),
        ("INTERRUPTED", "INCONCLUSIVE"),
    ):
        with pytest.raises(InvalidDataError):  # the suite is evaluated once: no re-scored rerun
            campaign.append_evaluation(
                seal,
                status=status,
                economic_decision=decision,
                reason="retuned",
                opening_record_id="open-1",
                result_artifacts=[retuned],
            )
    with pytest.raises(InvalidDataError):  # and no further test evidence after the disposition
        campaign.record_trial(**FRESH_TRIAL)
    # A correction may still be rerun, as a labelled diagnostic that claims no evidence.
    campaign.record_trial(
        **(
            FRESH_TRIAL
            | {"trial_id": "diag-1", "evidence_scope": "NONE", "notes": "DIAGNOSTIC: rerun with corrected fees"}
        )
    )
    assert campaign.evaluations() == (failed.path,)


def test_failed_or_interrupted_evaluations_may_be_continued(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = open_test(campaign)
    partial = campaign.register_artifact(
        "evaluation_report",
        write(tmp_path / "suite.json", b"{}"),
        producer_attempt="a14",
        evidence_scope=["FRESH_TEST"],
    )
    campaign.record_trial(**FRESH_TRIAL)
    campaign.append_evaluation(
        seal,
        status="FAILED",
        economic_decision="INCONCLUSIVE",
        reason="worker crashed",
        opening_record_id="open-1",
        result_artifacts=[partial],
    )
    campaign.record_trial(**(FRESH_TRIAL | {"trial_id": "run-2"}))  # the continuation
    campaign.append_evaluation(
        seal,
        status="INTERRUPTED",
        economic_decision="INCONCLUSIVE",
        reason="power loss",
        opening_record_id="open-1",
        result_artifacts=[partial],
    )
    done = campaign.append_evaluation(
        seal,
        status="COMPLETED",
        economic_decision="TEST_FAILED",
        reason="base case lost",
        opening_record_id="open-1",
        result_artifacts=[partial],
    )
    assert len(campaign.evaluations()) == 3 and campaign.evaluations()[-1] == done.path


def test_record_times_follow_the_seal_and_the_opening(tmp_path: Path) -> None:
    """The evidence must never show a test opened before the freeze that admits it, or a disposition
    created before the seal or the opening it cites (timestamps compared as UTC instants)."""
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())  # sealed_at 11:30:00 UTC
    with pytest.raises(InvalidDataError):
        campaign.append_opening(
            seal.sha256,
            opening_record_id="open-1",
            datasets=[FRESH],
            authorization="ok",
            opened_at="2026-10-21T11:29:59.999999+00:00",
        )
    assert campaign.openings() == () and campaign.exposures() == ()
    campaign.append_opening(
        seal.sha256,
        opening_record_id="open-1",
        datasets=[FRESH],
        authorization="ok",
        opened_at="2026-10-21T15:00:00+02:00",
    )  # 13:00 UTC
    result = campaign.register_artifact(
        "evaluation_report",
        write(tmp_path / "suite.json", b"{}"),
        producer_attempt="a14",
        evidence_scope=["FRESH_TEST"],
    )
    for stamp in ("2026-10-21T11:00:00+00:00", "2026-10-21T12:59:59+00:00"):  # before the seal; before the opening
        with pytest.raises(InvalidDataError):
            campaign.append_evaluation(
                seal.sha256,
                status="COMPLETED",
                economic_decision="TEST_FAILED",
                reason="base case lost",
                opening_record_id="open-1",
                result_artifacts=[result],
                created_at=stamp,
            )
    assert campaign.evaluations() == ()
    campaign.append_evaluation(
        seal.sha256,
        status="COMPLETED",
        economic_decision="TEST_FAILED",
        reason="base case lost",
        opening_record_id="open-1",
        result_artifacts=[result],
        created_at="2026-10-21T13:00:00+00:00",
    )  # at the opening instant: allowed
    closed = make_campaign(tmp_path, "c-002")
    no_test = closed.seal_freeze(seal_payload(admission_decision="NO_TEST_OPENING"))
    with pytest.raises(InvalidDataError):  # a no-test audit cannot predate the seal it disposes of
        closed.append_evaluation(
            no_test.sha256,
            status="NOT_EVALUATED",
            economic_decision="NO_CANDIDATE_PASSED",
            reason="gate failed",
            created_at="2026-10-21T11:00:00+00:00",
        )
    assert closed.evaluations() == ()
    future = make_campaign(tmp_path, "c-003")  # the campaign clock reads 12:00:01 at the first stamp
    with pytest.raises(InvalidDataError):  # a seal cannot claim a time later than the campaign clock
        future.seal_freeze(seal_payload(sealed_at="2026-10-21T12:00:30+00:00"))
    assert not future.freeze_path.exists()


def test_a_no_test_disposition_is_final(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())  # admitted, but the pre-test gate then failed
    disposition = campaign.append_evaluation(
        seal.sha256, status="NOT_EVALUATED", economic_decision="NO_CANDIDATE_PASSED", reason="pre-test gate failed"
    )
    with pytest.raises(InvalidDataError):  # a no-test audit is not followed by opening the test
        campaign.append_opening(seal.sha256, opening_record_id="open-1", datasets=[FRESH], authorization="ok")
    with pytest.raises(InvalidDataError):  # nor by a second disposition
        campaign.append_evaluation(
            seal.sha256, status="NOT_EVALUATED", economic_decision="NOT_EVALUATED", reason="again"
        )
    assert campaign.openings() == () and campaign.exposures() == ()
    assert campaign.evaluations() == (disposition.path,)


def test_a_tampered_seal_blocks_every_later_record(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    seal = campaign.seal_freeze(seal_payload())
    doc = loads_strict(seal.path.read_bytes())
    doc["reason"] = "edited after sealing"
    seal.path.write_bytes(canonical_json(doc).encode("utf-8"))
    with pytest.raises(InvalidDataError):
        campaign.append_evaluation(seal.sha256, status="NOT_EVALUATED", economic_decision="NOT_EVALUATED", reason="r")
    with pytest.raises(InvalidDataError):
        campaign.append_opening(seal.sha256, opening_record_id="o", datasets=[FRESH], authorization="ok")
    seal.path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    with pytest.raises(InvalidDataError):  # a re-serialised seal is no longer canonical
        campaign.read_freeze()


# ---------------------------------------------------------------------------
# Registry document
# ---------------------------------------------------------------------------


def test_registry_document_follows_the_record_schema(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    record = campaign.register_artifact(
        "baseline_results",
        write(tmp_path / "b.json", b"{}"),
        producer_attempt="a1",
        producer_stage="06",
        evidence_scope=["SYNTHETIC"],
    )
    campaign.write_stage_record("06", "a1", produced_artifacts=[record])
    written = campaign.write_registry_document(protected_paths=["data/raw"])
    doc = loads_strict(written.path.read_text(encoding="utf-8"))
    assert (doc["schema_version"], doc["record_format"], doc["campaign_id"]) == (4, "1", "c-001")
    assert doc["artifact_record_fields"] == list(ARTIFACT_RECORD_FIELDS)
    assert set(ARTIFACT_RECORD_FIELDS) <= set(doc["artifacts"][0])
    assert doc["artifacts"][0]["sha256"] == record.sha256
    assert doc["stage_records"] == ["runs/c-001/stages/06/a1/stage-record.json"]
    assert doc["trial_ledger"] == "runs/c-001/registry/trial-ledger.csv"
    assert doc["exposure_ledger"] == "runs/c-001/registry/data-exposure.csv"
    assert doc["protected_paths"] == ["data/raw"]
    assert doc["implementation_root"] == "."
    again = campaign.write_registry_document(protected_paths=["data/raw"])  # a derived view
    assert again.sha256 == written.sha256


def test_records_never_leave_temporary_files_behind(tmp_path: Path) -> None:
    campaign = make_campaign(tmp_path)
    campaign.record_trial(**TRIAL)
    campaign.record_exposure("2024-01-01", **EXPOSURE)
    campaign.write_stage_record("03", "a1")
    seal = campaign.seal_freeze(seal_payload(admission_decision="NO_TEST_OPENING"))
    campaign.append_evaluation(seal.sha256, status="NOT_EVALUATED", economic_decision="NOT_EVALUATED", reason="r")
    leftovers = [p for p in campaign.directory.rglob("*") if p.name.endswith(".tmp")]
    assert leftovers == []
    with pytest.raises(FinanceAIError):
        campaign.read_stage_record("03", "missing")
