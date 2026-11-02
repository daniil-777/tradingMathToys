"""``python -m financeai.service``: run the local job API or manage its state root.

Commands (``serve`` is the default)::

    python -m financeai.service [serve] --state-root DIR [--port 8765]
    python -m financeai.service openapi [--out docs/api/openapi-v1.json]
    python -m financeai.service register-strategy FILE --state-root DIR
    python -m financeai.service register-cache DIR --state-root DIR --data-role DEVELOPMENT

``serve`` binds loopback only (one Uvicorn worker) and uses ``FINANCEAI_API_TOKEN`` or a
private token file (created as ``STATE_ROOT/api-token`` with mode 0600 when absent).
Exit codes: 0 success, 2 typed FinanceAI error (one JSON line on stderr).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from financeai.core.errors import FinanceAIError, InvalidConfigError, OptionalDependencyError
from financeai.service.settings import (
    DEFAULT_ALLOWED_HOSTS,
    STATE_ROOT_ENV,
    TOKEN_ENV,
    TOKEN_FILE_NAME,
    ServiceSettings,
    is_loopback_host,
)
from financeai.service.worker import THREAD_ENV

_COMMANDS = ("serve", "openapi", "register-strategy", "register-cache")


def _state_root(args: argparse.Namespace) -> Path:
    root = args.state_root or os.environ.get(STATE_ROOT_ENV)
    if not root:
        raise InvalidConfigError(f"--state-root (or {STATE_ROOT_ENV}) is required")
    return Path(root).expanduser().resolve()


def _serve(args: argparse.Namespace) -> int:
    for key, value in THREAD_ENV.items():  # the control plane needs no native thread pools either
        os.environ.setdefault(key, value)
    if not is_loopback_host(args.host):
        raise InvalidConfigError(
            "only loopback binding is supported; an external service needs its own "
            "deployment and authentication design",
            context={"host": args.host},
        )
    settings = ServiceSettings.from_environment(
        _state_root(args),
        token_file=args.token_file,
        create_token=True,
        allowed_hosts=tuple(args.allowed_host or DEFAULT_ALLOWED_HOSTS),
        queue_capacity=args.queue_capacity,
        max_workers=args.max_workers,
        max_body_bytes=args.max_body_bytes,
        cancel_grace_s=args.cancel_grace_s,
    )
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - exampled only without the extra
        raise OptionalDependencyError("serving needs the 'api' extra: pip install 'financeai[api]'") from exc
    from financeai.service.app import create_app

    source = TOKEN_ENV if os.environ.get(TOKEN_ENV) else (args.token_file or str(settings.state_root / TOKEN_FILE_NAME))
    print(
        json.dumps({"listening": f"http://{args.host}:{args.port}", "token_source": source}),
        file=sys.stderr,
        flush=True,
    )
    uvicorn.run(
        create_app(settings),
        host=args.host,
        port=args.port,
        workers=1,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        log_level=args.log_level,
        access_log=False,
        timeout_graceful_shutdown=int(settings.cancel_grace_s) + 5,
    )
    return 0


def _openapi(args: argparse.Namespace) -> int:
    from financeai.service.app import render_openapi

    text = render_openapi()
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


def _register(args: argparse.Namespace) -> int:
    from financeai.service.artifacts import ArtifactStore, register_strategy, register_tape_cache
    from financeai.service.store import JobStore

    root = _state_root(args)
    store = JobStore(root / "service.sqlite3")
    try:
        artifacts = ArtifactStore(root / "artifacts")
        if args.command == "register-strategy":
            record = register_strategy(store, artifacts, args.source)
        else:
            record = register_tape_cache(store, artifacts, args.source, data_role=args.data_role)
    finally:
        store.close()
    print(
        json.dumps(
            {
                "artifact_id": record.artifact_id,
                "kind": record.kind,
                "sha256": record.sha256,
                "size_bytes": record.size_bytes,
                "data_role": record.data_role,
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m financeai.service", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="run the local API (loopback, one worker)")
    p.add_argument("--state-root", help=f"local unsynchronised work root (default ${STATE_ROOT_ENV})")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token-file", help=f"private token file (default STATE_ROOT/{TOKEN_FILE_NAME})")
    p.add_argument("--allowed-host", action="append", help="Host header allowlist entry (repeatable)")
    p.add_argument("--queue-capacity", type=int, default=16)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-body-bytes", type=int, default=256 * 1024)
    p.add_argument("--cancel-grace-s", type=float, default=5.0)
    p.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"])
    p.set_defaults(func=_serve)
    p = sub.add_parser("openapi", help="print or write the OpenAPI v1 document")
    p.add_argument("--out")
    p.set_defaults(func=_openapi)
    p = sub.add_parser("register-strategy", help="register a strategy config file as an immutable artifact")
    p.add_argument("source")
    p.add_argument("--state-root")
    p.set_defaults(func=_register)
    p = sub.add_parser("register-cache", help="verify and register a tape cache directory")
    p.add_argument("source")
    p.add_argument("--state-root")
    p.add_argument("--data-role", required=True, choices=["SYNTHETIC", "DEVELOPMENT", "RESERVED_TEST"])
    p.set_defaults(func=_register)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] not in (*_COMMANDS, "-h", "--help"):
        args_list = ["serve", *args_list]
    args = build_parser().parse_args(args_list)
    try:
        return int(args.func(args))
    except FinanceAIError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
