# Contributing

Use Python 3.12 and a virtual environment. Install `.[dev,fast,report,api]` for the complete
check suite, or `.[dev]` to exercise the NumPy-only core.

```bash
python -m pytest -q -m 'not slow and not data'
python -m ruff check src tests examples scripts
python -m mypy src
python scripts/check_docs.py
python -m build
```

Test changes at the behavioral boundary: independently calculated cash, reference-engine parity,
causal information access, corrupt inputs, and configuration round trips. Use synthetic fixtures
that run without credentials or external datasets. Optional dependencies may skip their own tests
in a core-only environment. Keep tests deterministic and record random seeds.

Declare units and clocks explicitly. Do not silently coerce fractional quantities, sum different
currencies, use future observations as a signal, or apply a specialized payoff formula to another
instrument convention. New adapters must preserve receipt/exchange timestamps and validate their
source schema.

The package uses a `src/` layout. Dependency-boundary tests keep the core independent of plots and
HTTP tooling. Import optional packages inside their owning capability rather than at package import.

Keep large data, caches, generated experiment outputs, credentials, and local environment files
outside version control. Update the relevant guide when changing a public API. Describe the behavior
and validation in commits and pull requests. Rendering instructions are in the
[animation guide](docs/trading_animation.md).
