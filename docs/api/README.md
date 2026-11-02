# FinanceAI local research job API (v1)

A **local, single-user research job API**: it admits bounded replay jobs, runs them in worker
processes and reports their durable state and committed artifacts. It is not a live exchange
API and not a per-event inference service. The contract is `openapi-v1.json` in this
directory, generated from the Pydantic transport models (`financeai.service.models`) and the
routes (`financeai.service.app`); `tests/service/test_contract.py` regenerates it and fails on
any unexplained diff.

Installation stays optional (`python -m pip install -e '.[api]'` from the repository root). `import financeai`, the
numerical domain, the CLI and the service internals (`jobs`, `store`, `coordinator`,
`worker`) never import FastAPI, Pydantic, Starlette, Uvicorn or httpx (tested in a subprocess).

## Run it

```bash
export FINANCEAI_API_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
python -m financeai.service --state-root ~/financeai-state      # serve is the default command
curl -s http://127.0.0.1:8765/healthz
```

* Binds loopback only (`--host` must be a loopback address), one Uvicorn worker, no CORS,
  no docs UI and no live OpenAPI route.
* Bearer token: `FINANCEAI_API_TOKEN`, else a private token file (`--token-file`,
  `FINANCEAI_API_TOKEN_FILE` or `STATE_ROOT/api-token`, created with mode 0600 when absent;
  group/world-readable files are refused). Tokens are compared in constant time.
* Host allowlist (DNS-rebinding protection): `127.0.0.1` and `localhost` by default
  (`--allowed-host` to change). Other Host headers get `400 invalid_host`.
* Bounds: request body (`--max-body-bytes`, 256 KiB), admission queue (`--queue-capacity`, 16
  QUEUED jobs), worker processes (`--max-workers`, 1; `0` disables admission), cancellation
  grace (`--cancel-grace-s`, 5 s).
* Other commands: `openapi [--out FILE]`, `register-strategy FILE --state-root DIR`,
  `register-cache DIR --state-root DIR --data-role {SYNTHETIC,DEVELOPMENT,RESERVED_TEST}`.

## Endpoints

| Method / path | Auth | Success | Purpose |
|---|---|---|---|
| `GET /healthz` | none | 200 | `{"status": "ok", "readiness": "READY"\|"NOT_READY", "api_version": "1.0"}`; no paths or config |
| `POST /v1/runs` | bearer | 202 | Submit a job; `Idempotency-Key` header required; returns run ID, status, status URL (`Location`) |
| `GET /v1/runs/{run_id}` | bearer | 200 | State, attempt, progress, failure, result, artifact IDs, evidence scope, economic status |
| `POST /v1/runs/{run_id}/cancel` | bearer | 202 / 200 | Idempotent cancel request: 202 while pending, 200 once terminal |
| `GET /v1/artifacts/{artifact_id}` | bearer | 200 | Metadata only: sha256, size, media/schema type, data role, provenance (no file serving, no paths) |

## Jobs (allowlist)

```json
{"schema_version": "1.0", "campaign_id": "demo",
 "job": {"kind": "synthetic_replay", "seed": 7, "venues": ["venue_a/ALPHA-USD", "venue_b/ALPHA-PERP"],
         "rows_per_venue": 300, "start_ns": "1700000000000000000", "scenario": "base",
         "strategy": {"schema": "financeai.strategy", "schema_version": "1.0", "...": "..."}}}
```

* `synthetic_replay`: a deterministic synthetic market (`financeai.domains.market.synthetic`)
  plus an inline strategy, checked strictly and canonicalized (see *Canonical strategies*)
  and then validated by `StrategyConfig.from_mapping`. `start_ns` is at most `2**62`
  (`4611686018427387904`, 2116-02-20 UTC). The worker writes the generated tapes as
  immutable caches and then runs exactly the `financeai run` path
  (`load_market` -> `run_strategy` -> `write_run_outputs`).
* `strategy_replay`: `strategy_artifact_id`, `cache_artifact_ids`, `scenario`. Only artifacts
  registered under `STATE_ROOT` are resolved (never a path); kinds, coverage and data roles are
  checked at admission and again at start, and digests are re-verified in the worker.
  `RESERVED_TEST` data is refused with `403 forbidden_data_role`: opening a reserved test needs
  a recorded freeze and authorization that an HTTP request cannot grant.
* Scenarios are the four example cases (`base`, `higher_fees`, `longer_delay`,
  `combined_stress`). No request may carry code, module paths, shell commands or file paths.

A successful run registers its immutable outputs as artifacts: `run_summary`, `fills_csv`,
`orders_csv`, `strategy_config`, `run_manifest` and (synthetic jobs) one `tape_cache` per venue.
These are valid inputs for a later `strategy_replay`.

## Wire rules and errors

* Unknown fields are rejected; request models are strict (no int/str/float coercion).
* Duplicate JSON keys, `NaN`/`Infinity` and malformed JSON are rejected before validation
  (`loads_strict` on the raw body) with `422 invalid_config`. "Malformed" covers every decoder
  failure, not only syntax errors: an integer literal beyond CPython's int-string limit, a finite
  spelling of an infinite number (`1e400`), invalid UTF-8 bytes and nesting deeper than the
  interpreter allows are `422`, never `500`.
* Clocks are UTC epoch nanoseconds as canonical decimal **strings** (`^(0|[1-9][0-9]{0,18})$`,
  int64-checked; no sign, exponent or leading zeros). Exact fees are decimal strings. `net_pnl`
  is a binary64 number (exact JSON round trip) or `null` for an invalid account.
* Derived clocks never wrap: the synthetic `start_ns` is capped at `2**62`, which leaves about
  146 years of int64 headroom above every generated clock, and every replay (both job kinds)
  checks that the later of its last market clock and session end, plus the replay horizon
  (order latency, the longer of acknowledgement latency and consumption TTL, cooldown),
  fits int64 below the no-event sentinel; a registered cache
  that fails this check ends `FAILED` with `failure.code = invalid_data` instead of producing
  wrapped order clocks.
* Errors: `{"error": {"code", "message", "request_id", "details"}}` with bounded, sanitised
  details (no paths, tokens or tracebacks); `X-Request-ID` echoes `request_id`.

| Status | Codes |
|---|---|
| 400 | `invalid_host` |
| 401 | `unauthorized` (with `WWW-Authenticate: Bearer`) |
| 403 | `forbidden_data_role` |
| 404 | `not_found` (unknown, malformed or path-like IDs, unknown routes) |
| 409 | `idempotency_conflict`, `state_conflict`, `incompatible_schema` |
| 413 | `request_too_large` |
| 422 | `invalid_request` (transport contract), `invalid_config` / `invalid_data` (shared validators) |
| 429 | `queue_full` |
| 503 | `worker_unavailable` |
| 500 | `internal_error` (logged with `correlation_id` = `request_id`) |

Failures after admission are reported in the run status (`failure.code`), never by changing
the POST response retroactively.

## Canonical strategies

`StrategyConfig` checks policy types and model documents for Python and CLI callers.
The service adds bounded, lossless canonicalization (`financeai.service.canonical`) — for inline strategies at
admission, for strategy files at registration and again when a `strategy_replay` loads them:

* Integers (policy `direction` in {-1, 0, 1}, `size_units`, `*_ms` — the episode-clock knobs
  `abs_cap_ms` and `early_check_ms` included — `*_ticks`, `size_units_cap`, `max_clips`; signal
  parameters with a unit suffix `_ms`/`_units`/`_ticks` and the unsuffixed counts `ticks`
  (`quote_confirm`) and `levels` (`ofi`); model `n_train`) must be JSON integers: `0.9`, `1000.0`
  and `true` are `422 invalid_config`. They are non-negative and at most `10**12` (as nanoseconds
  still inside int64), except `depth_ticks` (`-1` disables depth sizing), `max_hold_ms` (at most
  `2**62 // 10**6`, the engine's "never" as research candidates export it) and `size_units_cap` /
  `max_clips` (1 to 1000, so the engine's size products stay inside int64).
* Binary64 values (policy `enter`, `exit`, `stop_bps`, `early_markout_bps`; a composite component's
  `scale`; session seconds, at most `10**9`; model `alpha`, `coef`, `intercept`) accept `0` or `0.0`
  and are stored as floats (`-0.0` as `0.0`); integers that binary64 cannot represent exactly are
  rejected.
* `allow_short`, `refresh_hold` and `reverse` are JSON booleans; names, labels, features, venues
  and the other signal parameters (`kind`, `normalize`) are strings. Malformed shapes
  (`"policy": []`, `"enter": null`, `"exit_signal": 5`) are `422`, never `500`.
* The canonicaliser's policy key sets mirror `StrategyConfig`'s: an untyped key would be admitted,
  hashed and published as written but run through `int()`/`float()` (`abs_cap_ms: 2.5` once ran as
  2 ms), so `tests/service/test_canonical.py` fails until a new engine knob is typed here.
* Signal recipes nest as in `financeai.strategies.leadlag`: `gate` and `fresh` objects, composite
  `components` (each a `spec` recipe and a `scale`) and the optional `exit_signal` recipe follow
  the same rules, at most 8 objects deep.
* The error names the offending leaf in `details.field`, e.g. `policy.direction.venue_a/ALPHA-USD` or
  `signal.components.1.spec.window_ms`.

The canonical document is what is hashed for idempotency (`0` and `0.0` are one request),
returned as `job.strategy`, written to `run/strategy.json` and executed.

## Idempotency

`UNIQUE(campaign_id, idempotency_key)` plus the SHA-256 of the **normalized** command (defaults
filled in, venues and cache IDs sorted, canonical strategy document; no credentials or wall-clock
metadata). Same key + same normalized request returns the same run (`replayed: true`, even when
the queue is full); same key + different request is `409 idempotency_conflict`. Admission,
the queue-capacity check and the insert are one SQLite transaction, so concurrent duplicate
POSTs create exactly one run. Mappings are retained indefinitely.

## States, workers and recovery

States: `QUEUED`, `RUNNING`, `CANCEL_REQUESTED`, `SUCCEEDED`, `FAILED`, `CANCELLED`,
`INTERRUPTED`. Legal moves: QUEUED -> RUNNING | CANCELLED | FAILED; RUNNING ->
CANCEL_REQUESTED | SUCCEEDED | FAILED | INTERRUPTED; CANCEL_REQUESTED -> CANCELLED | SUCCEEDED |
FAILED | INTERRUPTED. Terminal states never change; every transition is appended to an audit log.

* One `spawn`-context process per attempt (never a global start method, never
  `BackgroundTasks`), at most `max_workers`, one active job per campaign, a private pipe and
  cancel event per job. Workers set `OMP/OPENBLAS/MKL/VECLIB/NUMEXPR/NUMBA` thread variables to 1
  before importing NumPy and become the leader of their own process group.
* Dispatch is fair across campaigns: candidates are the oldest QUEUED runs of campaigns that have
  no active job, selected in SQL (`campaign_id NOT IN (busy)`), so a campaign with a long backlog
  never hides another campaign's run behind a scan window while a worker slot is free.
* Cancellation: QUEUED jobs are cancelled at once. Running jobs stop cooperatively at safe
  checkpoints; after the grace period the whole process group (worker and descendants) gets
  SIGTERM, then SIGKILL, and the run ends `CANCELLED` (`failure.code = terminated`).
* Publication is owned by the coordinator only. Workers write into
  `STATE_ROOT/staging/<run>/<attempt>`; the coordinator checks the active attempt, the expected
  file set and every digest, writes `manifest.json`, fsyncs every staged file and directory (the
  database is `synchronous=FULL`; a durable `SUCCEEDED` must never reference output bytes the OS
  has not written), records a PENDING outbox operation, renames the attempt atomically into
  `artifacts/runs/<run>/<attempt>` and then commits artifacts + `SUCCEEDED` in one transaction. Commit/cancel order is total (one lock): a completion
  processed before cancellation wins (SUCCEEDED, even if a cancel request arrived); once the
  coordinator terminated a worker, late output is rejected. Rejected, partial or stale output
  is moved to `STATE_ROOT/quarantine` and never registered.
* Restart: PENDING publications are completed exactly once (if the renamed directory and its
  manifest verify) or aborted; RUNNING / CANCEL_REQUESTED runs become `INTERRUPTED` and are never
  resubmitted; QUEUED runs stay queued. A graceful shutdown marks unfinished runs
  `INTERRUPTED` (`service_shutdown`).
* One coordinator owns a `STATE_ROOT` (exclusive `flock`); a second service on the same root
  refuses to start instead of interrupting the owner's live runs. If the coordinator thread
  itself fails, admission reports `503` / `NOT_READY` until restart.
* A cancelled or interrupted replay is an operational abort, not a liquidation: its
  `economic_status` is `INCOMPLETE`. `economic_status` only reports account validity
  (`VALID_ACCOUNT` / `INVALID_ACCOUNT`); it is never a claim of economic success.

## STATE_ROOT

```
STATE_ROOT/service.sqlite3        runs, artifacts, outbox operations, transitions (WAL, synchronous=FULL)
STATE_ROOT/service.lock           exclusive flock of the owning coordinator (one service per root)
STATE_ROOT/artifacts/inputs/      registered strategies and tape caches (private read-only copies, verified)
STATE_ROOT/artifacts/runs/        committed run attempts with manifest.json
STATE_ROOT/staging/               attempt directories of running workers
STATE_ROOT/quarantine/            rejected, partial or stale output (never registered)
```

Registered inputs are copied, never hard-linked, and the copies are mode `0444`: the caller keeps
its own inode, so editing the source file later cannot change (or corrupt) the registered version
behind its recorded digest. Every use re-verifies that digest.

Use a local, unsynchronised, single-host directory (not iCloud/Dropbox/network storage).
Back up the database with SQLite's backup mechanism, not by copying an active file.

## Limitations

* No campaign registry, trial budget or exposure ledger: `campaign_id` scopes idempotency and
  the one-active-job rule only. Reserved test data is refused rather than gated.
* One process per attempt (about one second of interpreter/NumPy/Numba start-up per job);
  workers are trusted code, not a sandbox for hostile plugins.
* Signal parameter *types* are checked at admission, but recipe-specific keys and values (an
  unknown recipe, a missing `window_ms`) are validated when the worker builds the signal; such
  errors appear as `FAILED` with `failure.code = invalid_config`.
* Canonical normalization and service-specific size/depth bounds are enforced at the HTTP
  boundary. Python and CLI callers share strict strategy validation but do not acquire
  the HTTP service's request limits.
* Both numerical replay entry points check clock headroom (see *Wire rules*), including
  direct Python calls. A custom reference policy remains responsible for its own internal
  state; the shared checks cover engine-derived clocks and the built-in policy's cooldown.
* Determinism of synthetic markets holds per NumPy version (Generator streams are not frozen).
* Tested on macOS (APFS) with CPython 3.12; other platforms are unsupported until tested.
