"""HTTP conformance of the local job API: health, Host/token guards, strict JSON, validation,
idempotency, bounded admission, cancellation, ID isolation and the error envelope."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from financeai.domains.market.synthetic import SyntheticMarketSpec
from financeai.service.coordinator import Coordinator
from financeai.service.store import new_id

from .runners import blocking_runner, release_file, stubborn_runner
from .support import (
    AUTH,
    TOKEN,
    SettingsFactory,
    assert_error,
    coordinator_of,
    get_run,
    post_run,
    run_request,
    service,
    synthetic_job,
    wait_progress,
    wait_status,
)
from .test_jobs import write_caches

VENUE_B = "venue_b/ALPHA-PERP"
LEAD_GAP: dict[str, Any] = synthetic_job()["strategy"]["signal"]
QUOTE_CONFIRM: dict[str, Any] = {"kind": "quote_confirm", "venue": VENUE_B, "window_ms": 20, "ticks": 2}
OFI: dict[str, Any] = {"kind": "ofi", "venue": VENUE_B, "levels": 3, "window_ms": 20, "normalize": "none"}


def test_health_is_unauthenticated_minimal_and_reports_readiness(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, blocking_runner) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "readiness": "READY", "api_version": "1.0"}
        assert response.headers["x-request-id"] and response.headers["cache-control"] == "no-store"
        assert str(settings.state_root) not in response.text and TOKEN not in response.text
    with service(make_settings(max_workers=0)) as client:
        assert client.get("/healthz").json()["readiness"] == "NOT_READY"


def test_host_allowlist_rejects_rebinding_hosts(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        assert client.get("/healthz", headers={"Host": "localhost:8765"}).status_code == 200
        assert_error(client.get("/healthz", headers={"Host": "evil.example"}), 400, "invalid_host")
        assert_error(
            client.get("/v1/runs/" + new_id("run"), headers={**AUTH, "Host": "attacker.test"}), 400, "invalid_host"
        )


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong-token-" + "y" * 30},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": "Bearer"},
    ],
)
def test_every_v1_route_requires_the_bearer_token(make_settings: SettingsFactory, headers: dict[str, str]) -> None:
    with service(make_settings(), blocking_runner) as client:
        for method, path in (
            ("post", "/v1/runs"),
            ("get", f"/v1/runs/{new_id('run')}"),
            ("post", f"/v1/runs/{new_id('run')}/cancel"),
            ("get", f"/v1/artifacts/{new_id('art')}"),
            ("get", "/v1/unknown"),
        ):
            response = client.request(
                method.upper(), path, headers=headers, content=json.dumps(run_request(synthetic_job()))
            )
            body = assert_error(response, 401, "unauthorized")
            assert response.headers["www-authenticate"] == "Bearer"
            assert TOKEN not in json.dumps(body)


def test_unknown_routes_and_methods_use_the_envelope(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        assert_error(client.get("/nope"), 404, "not_found")
        assert_error(client.get("/v1/runs", headers=AUTH), 405, "method_not_allowed")
        assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        '{"schema_version": "1.0", "job": {"kind": "synthetic_replay", "seed": NaN}}',
        '{"schema_version": "1.0", "job": {"kind": "synthetic_replay", "seed": Infinity}}',
        '{"schema_version": "1.0", "job": {"kind": "synthetic_replay", "seed": -Infinity}}',
        '{"schema_version": "1.0", "schema_version": "1.0", "job": {}}',
        json.dumps(run_request(synthetic_job()))[:-1] + ', "job": {"kind": "strategy_replay"}}',
        '{"schema_version": "1.0", "job": {"kind": "synthetic_replay", "strategy": {"name": "a", "name": "b"}}}',
        '{"schema_version": "1.0", "job": ',
        # Decoder failures that are not JSONDecodeErrors used to escape as 500 internal_error:
        json.dumps(run_request(synthetic_job())).replace('"seed": 7', '"seed": ' + "1" * 5000),  # int-string limit
        b'{"schema_version": "1.0", "job": "\xff"}',  # invalid UTF-8
        "[" * 60_000 + "]" * 60_000,  # 120 KB of nesting: under the body limit, beyond the recursion limit
    ],
)
def test_duplicate_keys_nonfinite_numbers_and_malformed_json_are_rejected(
    make_settings: SettingsFactory, body: str | bytes
) -> None:
    with service(make_settings(), blocking_runner) as client:
        assert_error(post_run(client, body), 422, "invalid_config")
        assert coordinator_of(client).store.queued_runs(10) == []


def _job(**changes: Any) -> dict[str, Any]:
    return run_request(synthetic_job(**changes))


@pytest.mark.parametrize(
    "body",
    [
        {**_job(), "extra": True},
        run_request({**synthetic_job(), "path": "/etc/passwd"}),
        run_request({**synthetic_job(), "module": "os"}),
        _job(seed=1.0),
        _job(seed="7"),
        _job(seed=True),
        _job(seed=2**53),
        _job(start_ns=1_700_000_000_000_000_000),
        _job(start_ns="1.7e18"),
        _job(start_ns="01"),
        _job(start_ns="-1"),
        _job(start_ns="9223372036854775808"),
        _job(start_ns=str(2**62 + 1)),  # past the synthetic start cap: derived replay clocks could wrap int64
        _job(start_ns="9223372036793785807"),  # 2**63 - 1 - max_span_ns(10): was admitted and wrapped arrival clocks
        _job(rows_per_venue=10**6),
        _job(scenario="moon"),
        _job(venues=["binance/ASSETUSD"]),
        _job(venues=[]),
        {"schema_version": "2.0", "job": synthetic_job()},
        {"schema_version": "1.0", "campaign_id": "../etc", "job": synthetic_job()},
        {"schema_version": "1.0", "job": {"kind": "train_model"}},
        {
            "schema_version": "1.0",
            "job": {
                "kind": "strategy_replay",
                "strategy_artifact_id": "../../strategy.json",
                "cache_artifact_ids": [new_id("art")],
                "scenario": "base",
            },
        },
        {"schema_version": "1.0"},
        [],
    ],
)
def test_transport_contract_violations_are_422(make_settings: SettingsFactory, body: Any) -> None:
    with service(make_settings(), blocking_runner) as client:
        response = assert_error(post_run(client, body), 422, "invalid_request")
        assert response["error"]["details"]["errors"]


@pytest.mark.parametrize(
    ("strategy_change", "status", "code"),
    [
        ({"extra": 1}, 422, "invalid_config"),
        ({"session": {"warmup_s": -1, "entry_stop_s": 0, "flatten_s": 0, "exec_stop_s": 0}}, 422, "invalid_config"),
        ({"schema_version": "9.0"}, 409, "incompatible_schema"),
    ],
)
def test_strategy_semantics_use_the_shared_validator(
    make_settings: SettingsFactory, strategy_change: dict[str, Any], status: int, code: str
) -> None:
    strategy = {**synthetic_job()["strategy"], **strategy_change}
    with service(make_settings(), blocking_runner) as client:
        assert_error(post_run(client, _job(strategy=strategy)), status, code)
        assert_error(post_run(client, _job(venues=["venue_a/ALPHA-USD"])), 422, "invalid_config")


def _strategy(*, policy: dict[str, Any] | None = None, **replace: Any) -> dict[str, Any]:
    """The example strategy with ``policy`` entries merged in and other top-level sections replaced."""
    strategy: dict[str, Any] = synthetic_job()["strategy"]
    strategy["policy"].update(policy or {})
    strategy.update(replace)
    return strategy


@pytest.mark.parametrize(
    ("strategy_change", "field"),
    [
        ({"policy": {"direction": {"venue_a/ALPHA-USD": 0.9, VENUE_B: 0}}}, "policy.direction.venue_a/ALPHA-USD"),
        ({"policy": {"size_units": {"venue_a/ALPHA-USD": 1000.9, VENUE_B: 0}}}, "policy.size_units.venue_a/ALPHA-USD"),
        ({"policy": {"allow_short": 1}}, "policy.allow_short"),
        ({"policy": {"enter": "0.5"}}, "policy.enter"),
        ({"policy": {"enter": None}}, "policy.enter"),  # escaped as a TypeError (HTTP 500) before
        # optional policy keys (bool()/int() coercions in threshold_params)
        ({"policy": {"reverse": "false"}}, "policy.reverse"),
        ({"policy": {"refresh_hold": 1}}, "policy.refresh_hold"),
        ({"policy": {"max_clips": 2.5}}, "policy.max_clips"),
        ({"policy": {"size_units_cap": 2.0}}, "policy.size_units_cap"),
        ({"policy": {"clip_spacing_ms": 0.5}}, "policy.clip_spacing_ms"),
        ({"policy": {"depth_ticks": "3"}}, "policy.depth_ticks"),
        # episode-clock knobs: 202 before (2.5 ran as 2 ms, True as 1 ms, "5" as 5 ms; 10**15 FAILED in the worker)
        ({"policy": {"abs_cap_ms": 2.5}}, "policy.abs_cap_ms"),
        ({"policy": {"abs_cap_ms": True}}, "policy.abs_cap_ms"),
        ({"policy": {"abs_cap_ms": 10**15}}, "policy.abs_cap_ms"),
        ({"policy": {"early_check_ms": "5"}}, "policy.early_check_ms"),
        ({"policy": {"early_markout_bps": "1"}}, "policy.early_markout_bps"),
        # the exit recipe and nested signal objects
        ({"exit_signal": 5}, "exit_signal"),
        ({"exit_signal": {**LEAD_GAP, "window_ms": 20.0}}, "exit_signal.window_ms"),
        ({"signal": {**LEAD_GAP, "gate": {"venue": VENUE_B, "tau_ms": 1.5, "min_units": 0}}}, "signal.gate.tau_ms"),
        ({"signal": {**LEAD_GAP, "fresh": {"max_age_ms": "50"}}}, "signal.fresh.max_age_ms"),
        ({"signal": {"kind": "any", "components": [{"spec": LEAD_GAP, "scale": "1"}]}}, "signal.components.0.scale"),
        # unsuffixed integer recipe keys: both recipes were rejected as "must be a string" before
        ({"signal": {**QUOTE_CONFIRM, "ticks": 2.0}}, "signal.ticks"),
        ({"signal": {**OFI, "levels": "3"}}, "signal.levels"),
    ],
)
def test_lossy_or_mistyped_strategy_values_are_422(
    make_settings: SettingsFactory, strategy_change: dict[str, Any], field: str
) -> None:
    with service(make_settings(), blocking_runner) as client:
        body = assert_error(post_run(client, _job(strategy=_strategy(**strategy_change))), 422, "invalid_config")
        assert body["error"]["details"]["field"] == field
        assert coordinator_of(client).store.queued_runs(10) == []


def test_equivalent_numeric_spellings_replay_one_canonical_run(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        first = post_run(client, _job(strategy=_strategy(policy={"exit": -1000000000.0, "stop_bps": 0.0})))
        assert first.status_code == 202, first.text
        run_id = first.json()["run_id"]
        for spelling in ({"exit": -1000000000}, {"stop_bps": 0}, {"exit": -1000000000, "stop_bps": 0}):
            again = post_run(client, _job(strategy=_strategy(policy=spelling)))
            assert again.status_code == 202, again.text
            assert again.json()["run_id"] == run_id and again.json()["replayed"] is True
        recorded = get_run(client, run_id)["job"]["strategy"]
        assert recorded["policy"]["exit"] == -1000000000.0 and type(recorded["policy"]["exit"]) is float
        assert type(recorded["policy"]["stop_bps"]) is float and type(recorded["session"]["warmup_s"]) is float
        assert recorded["policy"]["direction"] == {"venue_a/ALPHA-USD": 1, "venue_b/ALPHA-PERP": 0}


def test_episode_clock_knobs_and_unsuffixed_recipe_counts_are_admitted_canonically(
    make_settings: SettingsFactory,
) -> None:
    knobs = {"abs_cap_ms": 30, "early_check_ms": 10, "early_markout_bps": -2}
    with service(make_settings(), blocking_runner) as client:
        first = post_run(client, _job(strategy=_strategy(policy=knobs)))
        assert first.status_code == 202, first.text
        again = post_run(client, _job(strategy=_strategy(policy={**knobs, "early_markout_bps": -2.0})))
        assert again.status_code == 202 and again.json()["run_id"] == first.json()["run_id"]  # one canonical form
        policy = get_run(client, first.json()["run_id"])["job"]["strategy"]["policy"]
        assert {k: policy[k] for k in knobs} == {**knobs, "early_markout_bps": -2.0}
        assert type(policy["abs_cap_ms"]) is int and type(policy["early_markout_bps"]) is float
        for key, signal in (("quote", QUOTE_CONFIRM), ("ofi", OFI)):
            accepted = post_run(client, _job(strategy=_strategy(signal=signal)), key=key)
            assert accepted.status_code == 202, accepted.text
            assert get_run(client, accepted.json()["run_id"])["job"]["strategy"]["signal"] == signal


def test_the_latest_synthetic_start_clock_is_admitted(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        accepted = post_run(client, _job(start_ns=str(2**62)))
        assert accepted.status_code == 202, accepted.text
        assert get_run(client, accepted.json()["run_id"])["job"]["start_ns"] == str(2**62)


@pytest.mark.parametrize(
    "headers", [{"Idempotency-Key": ""}, {"Idempotency-Key": "bad key with spaces"}, {"Idempotency-Key": "k" * 200}]
)
def test_idempotency_key_is_required_and_bounded(make_settings: SettingsFactory, headers: dict[str, str]) -> None:
    with service(make_settings(), blocking_runner) as client:
        no_key = client.post(
            "/v1/runs", content=json.dumps(_job()), headers={"Content-Type": "application/json", **AUTH}
        )
        assert_error(no_key, 422, "invalid_request")
        assert_error(post_run(client, _job(), headers=headers), 422, "invalid_request")


def test_oversized_bodies_are_rejected(make_settings: SettingsFactory) -> None:
    with service(make_settings(max_body_bytes=4096), blocking_runner) as client:
        big = _job(strategy={**synthetic_job()["strategy"], "name": "x" * 5000})
        assert_error(post_run(client, big), 413, "request_too_large")
        chunks = (json.dumps(big).encode()[i : i + 1000] for i in range(0, 6000, 1000))
        streamed = client.post(
            "/v1/runs", content=chunks, headers={"Content-Type": "application/json", "Idempotency-Key": "s", **AUTH}
        )
        assert_error(streamed, 413, "request_too_large")


def test_idempotent_replay_normalization_and_conflict(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        first = post_run(client, _job())
        assert first.status_code == 202, first.text
        accepted = first.json()
        assert accepted["status"] in {"QUEUED", "RUNNING"} and accepted["replayed"] is False
        assert accepted["status_url"] == f"/v1/runs/{accepted['run_id']}" == first.headers["location"]
        equivalent = run_request(
            synthetic_job(venues=["venue_b/ALPHA-PERP", "venue_a/ALPHA-USD"], rows_per_venue=300, start_ns="0")
        )
        again = post_run(client, json.dumps(equivalent, sort_keys=True))
        assert again.status_code == 202 and again.json()["run_id"] == accepted["run_id"]
        assert again.json()["replayed"] is True
        conflict = assert_error(post_run(client, _job(seed=8)), 409, "idempotency_conflict")
        assert conflict["error"]["details"]["run_id"] == accepted["run_id"]
        other_campaign = post_run(client, run_request(synthetic_job(seed=8), campaign="other"))
        assert other_campaign.status_code == 202 and other_campaign.json()["run_id"] != accepted["run_id"]
        new_key = post_run(client, _job(), key="key-2")
        assert new_key.status_code == 202 and new_key.json()["run_id"] != accepted["run_id"]


def test_concurrent_duplicate_posts_admit_one_run(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        with ThreadPoolExecutor(8) as pool:
            responses = list(pool.map(lambda _: post_run(client, _job(), key="same"), range(12)))
        assert {r.status_code for r in responses} == {202}
        assert len({r.json()["run_id"] for r in responses}) == 1
        assert sum(not r.json()["replayed"] for r in responses) == 1


def test_conflicting_concurrent_posts_admit_exactly_one_request(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        with ThreadPoolExecutor(8) as pool:
            responses = list(pool.map(lambda seed: post_run(client, _job(seed=seed), key="race"), range(1, 9)))
        assert sorted(r.status_code for r in responses) == [202] + [409] * 7
        winner = next(r for r in responses if r.status_code == 202).json()["run_id"]
        assert all(r.json()["error"]["details"]["run_id"] == winner for r in responses if r.status_code == 409)


def test_full_queue_is_429_and_replays_still_succeed(make_settings: SettingsFactory) -> None:
    with service(make_settings(queue_capacity=1), blocking_runner) as client:
        running = post_run(client, _job(seed=1), key="a").json()["run_id"]
        wait_progress(client, running, "blocked")
        queued = post_run(client, _job(seed=2), key="b")
        assert queued.status_code == 202 and queued.json()["status"] == "QUEUED"
        assert_error(post_run(client, _job(seed=3), key="c"), 429, "queue_full")
        assert post_run(client, _job(seed=2), key="b").json()["replayed"] is True


def test_cancel_before_start_is_final_and_idempotent(make_settings: SettingsFactory) -> None:
    with service(make_settings(), blocking_runner) as client:
        running = post_run(client, _job(seed=1), key="a").json()["run_id"]
        wait_progress(client, running, "blocked")
        queued = post_run(client, _job(seed=2), key="b").json()["run_id"]
        response = client.post(f"/v1/runs/{queued}/cancel", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "CANCELLED" and body["attempt_id"] is None and body["generation"] == 0
        assert body["economic_status"] == "NOT_EVALUATED" and body["artifact_ids"] == []
        again = client.post(f"/v1/runs/{queued}/cancel", headers=AUTH)
        assert again.status_code == 200 and again.json()["status"] == "CANCELLED"
        assert get_run(client, queued)["finished_at_ns"] is not None


def test_cooperative_cancel_during_work_quarantines_partial_output(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, blocking_runner) as client:
        run_id = post_run(client, _job()).json()["run_id"]
        running = wait_progress(client, run_id, "blocked")
        assert running["status"] == "RUNNING" and running["attempt_id"].startswith("att_")
        response = client.post(f"/v1/runs/{run_id}/cancel", headers=AUTH)
        assert response.status_code == 202 and response.json()["status"] == "CANCEL_REQUESTED"
        done = wait_status(client, run_id, {"CANCELLED"})
        assert done["failure"] is None and done["economic_status"] == "INCOMPLETE" and done["artifact_ids"] == []
        assert client.post(f"/v1/runs/{run_id}/cancel", headers=AUTH).status_code == 200
    quarantined = list((settings.quarantine_dir / run_id).glob("*-cancelled-*/partial.txt"))
    assert len(quarantined) == 1
    assert not (settings.artifacts_dir / "runs" / run_id).exists()
    assert not (settings.staging_dir / run_id).exists()


def test_non_cooperative_worker_is_terminated_after_the_grace_period(make_settings: SettingsFactory) -> None:
    with service(make_settings(cancel_grace_s=0.5), stubborn_runner) as client:
        run_id = post_run(client, _job()).json()["run_id"]
        wait_status(client, run_id, {"RUNNING"})
        assert client.post(f"/v1/runs/{run_id}/cancel", headers=AUTH).status_code == 202
        done = wait_status(client, run_id, {"CANCELLED"}, timeout_s=60)
        assert done["failure"]["code"] == "terminated" and done["artifact_ids"] == []


def test_unavailable_workers_are_503(make_settings: SettingsFactory) -> None:
    with service(make_settings(max_workers=0)) as client:
        assert_error(post_run(client, _job()), 503, "worker_unavailable")


@pytest.mark.parametrize(
    "bad_id",
    [
        "art_" + "0" * 31,
        "art_" + "A" * 32,
        "..",
        "%2e%2e",
        "art_..%2F..%2Fetc",
        "..%2F..%2Fetc%2Fpasswd",
        "C:%5Cwindows",
        "art_" + "0" * 32 + "%00",
        "%20",
    ],
)
def test_unknown_and_path_like_ids_are_404(make_settings: SettingsFactory, bad_id: str) -> None:
    with service(make_settings(), blocking_runner) as client:
        for path in (
            f"/v1/artifacts/{bad_id}",
            f"/v1/runs/{bad_id.replace('art_', 'run_')}",
            f"/v1/runs/{bad_id.replace('art_', 'run_')}/cancel",
        ):
            method = "POST" if path.endswith("/cancel") else "GET"
            response = client.request(method, path, headers=AUTH)
            assert_error(response, 404, "not_found")
            assert "passwd" not in response.text and "root:" not in response.text
        assert_error(client.get(f"/v1/artifacts/{new_id('art')}", headers=AUTH), 404, "not_found")
        assert_error(client.get(f"/v1/runs/{new_id('run')}", headers=AUTH), 404, "not_found")


def test_strategy_replay_resolves_only_registered_artifacts_and_guards_test_data(
    make_settings: SettingsFactory, tmp_path: Path
) -> None:
    caches = write_caches(SyntheticMarketSpec(seed=7), tmp_path / "caches")
    strategy_file = tmp_path / "strategy.json"
    strategy_file.write_text(json.dumps(synthetic_job()["strategy"]))
    with service(make_settings(), blocking_runner) as client:
        coordinator: Coordinator = coordinator_of(client)
        strategy = coordinator.register_strategy(strategy_file)
        dev = coordinator.register_tape_cache(caches[0], data_role="DEVELOPMENT")
        reserved = coordinator.register_tape_cache(caches[1], data_role="RESERVED_TEST")

        def job(strategy_id: str, cache_ids: list[str]) -> dict[str, Any]:
            return run_request(
                {
                    "kind": "strategy_replay",
                    "strategy_artifact_id": strategy_id,
                    "cache_artifact_ids": cache_ids,
                    "scenario": "base",
                }
            )

        forbidden = assert_error(
            post_run(client, job(strategy.artifact_id, [dev.artifact_id, reserved.artifact_id])),
            403,
            "forbidden_data_role",
        )
        assert forbidden["error"]["details"]["data_role"] == "RESERVED_TEST"
        assert_error(post_run(client, job(new_id("art"), [dev.artifact_id]), key="k2"), 404, "not_found")
        assert_error(post_run(client, job(dev.artifact_id, [dev.artifact_id]), key="k3"), 422, "invalid_request")
        assert_error(post_run(client, job(strategy.artifact_id, [dev.artifact_id]), key="k4"), 422, "invalid_request")
        assert coordinator.store.queued_runs(10) == [] and coordinator.active_runs == ()
        metadata = client.get(f"/v1/artifacts/{reserved.artifact_id}", headers=AUTH).json()
        assert metadata["data_role"] == "RESERVED_TEST" and metadata["kind"] == "tape_cache"
        assert "rel_path" not in metadata and str(tmp_path) not in json.dumps(metadata)


def test_unexpected_errors_are_500_without_internals(
    make_settings: SettingsFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(self: Coordinator, run_id: str) -> None:
        raise RuntimeError("secret detail /Users/someone/state")

    with service(make_settings(), blocking_runner, raise_server_exceptions=False) as client:
        monkeypatch.setattr(Coordinator, "status", explode)
        response = client.get(f"/v1/runs/{new_id('run')}", headers=AUTH)
        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "internal_error" and body["error"]["message"] == "internal error"
        assert body["error"]["details"]["correlation_id"] == body["error"]["request_id"]
        assert "secret" not in response.text and "Traceback" not in response.text and "/Users/" not in response.text


def test_released_blocking_job_runs_the_real_replay(make_settings: SettingsFactory) -> None:
    settings = make_settings()
    with service(settings, blocking_runner) as client:
        run_id = post_run(client, _job(start_ns="1700000000000000001")).json()["run_id"]
        wait_progress(client, run_id, "blocked")
        release_file(settings.state_root, run_id).touch()
        done = wait_status(client, run_id, {"SUCCEEDED", "FAILED"})
        assert done["status"] == "SUCCEEDED", done
        assert done["job"]["start_ns"] == "1700000000000000001"
        assert int(done["result"]["first_event_ns"]) > 2**53 and isinstance(done["result"]["first_event_ns"], str)
