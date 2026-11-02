# Normalized snapshot and trade ingestion

`read_tardis_snapshot_trades(book_path, trades_path, instrument)` converts normalized
`book_snapshot_25` and `trades` CSV files into immutable `MarketTapes`. Supply your own
`InstrumentSpec`; symbols, exchanges, currencies, tick sizes, and quantity steps are configurable.

The adapter accepts unquoted ASCII CSV with LF line endings, either plain or gzip-compressed.
It retains every snapshot and the first ten of the required 25 levels. Receipt and exchange clocks
remain separate. Integer microsecond timestamps become integer nanoseconds after range checks.

Prices and native quantities must lie exactly on the declared grids. Missing trailing levels remain
missing; missing interior levels, invalid ordering, unknown sides, wrong instrument identifiers,
overflow, decreasing receipt clocks, and malformed numeric fields are rejected. Scientific notation
and quoted/CRLF dialects are outside this adapter's normalized input contract.

A specialized integer decoder accelerates half-unit ticks with integer native quantities. Other
instruments use the exact-decimal decoder. Both produce the same tape contract and validation rules.

`buy` aggressors map to +1 and `sell` to −1. Normalized CSV cannot reconstruct disconnect markers
or a cross-file message ordering absent from the source. Crossed top quotes are retained and marked
unusable by the market-validity layer.

```bash
financeai ingest book.csv.gz --format tardis --trades trades.csv.gz \
  --instrument-spec instrument.json --out cache/my-instrument
```

For recorder text, use `read_recorder_file(path, instrument)` or omit `--format tardis`.
The exact-decimal recorder fallback streams input lines and supports fractional quantity grids.
