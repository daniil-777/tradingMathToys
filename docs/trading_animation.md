# From signal to trade

[Watch the movie](assets/trading/trading-methods.mp4) or
[open the GIF](assets/trading/trading-methods.gif). The MP4 provides pause and seek controls.
All examples use fictional positive-price linear contracts, one unit of exposure per contract,
100-unit positions, one-cent ticks, 1 bp taker fees, 5 ms order latency, and 5 ms acknowledgement latency.

The movie uses the library's actual replay engine. Its generator checks every order/fill field and
settlement against the independent reference implementation. The
[numerical trace](assets/trading/trace.json) contains inputs, policies, orders, fills, and asset hashes.

## Transcript

| Movie time | What happens | Why |
| --- | --- | --- |
| 0–5 s | Introduce flow, lead–lag, and execution | A signal is an input to a policy, not a fill |
| 5–8 s | Wait flat | No sufficiently strong trade-flow burst has arrived |
| 8–12 s | Buy to open a long | Decayed signed trade flow reaches the +1,500-unit threshold |
| 12–16 s | Hold the long | Falling below the entry threshold alone does not close this policy |
| 16–19 s | Sell to close, then sell to open a short | Opposite pressure triggers a reversal; the close must be acknowledged first |
| 19–22 s | Hold the short | Negative exposure benefits from a price decline |
| 22–25 s | Buy to close | The session's flattening rule returns the account to zero |
| 25–28 s | Compare two venues | Compare their midpoint returns over the same one-second window |
| 28–32 s | Buy the follower | The leader-minus-follower return gap exceeds 0.75 bp |
| 32–35 s | Hold | Only received observations are available; future prices remain hidden |
| 35–39 s | Sell to close | The follower catches up and the gap falls below the exit threshold |
| 39–48 s | An order misses | The ask changes at the exchange before the order arrives; the limit prevents a fill |
| 48–55 s | Reconcile costs | A positive price result becomes a negative result after two taker fees |

**Flow** adds buyer-initiated quantity, subtracts seller-initiated quantity, and exponentially decays
older prints with a one-second time constant. **Lead–lag** subtracts the follower's recent return
from the leader's, expressed in basis points; 1 bp is 0.01%.

The position plot shows **acknowledged** inventory. Fill markers show exchange-arrival prices.
These occur at different times. The missed-fill scene enlarges this distinction: the ask moves
at +3 ms, the order arrives at +5 ms, and its no-fill report reaches the controller at +10 ms.

The final trade buys 100 units at $100.01 and sells at $100.02. Its price result is $1.00;
fees are $2.0003, and the net result is −$1.0003. On-screen amounts are rounded to cents.
The simplified policies explain mechanisms; no ranking or expected return on real markets is implied.

## Reproduce and review

Install the plotting extra and FFmpeg. Numerical verification needs only the core dependencies:

```bash
python examples/trading_movie.py --verify-only
python examples/trading_movie.py --preview --out /tmp/financeai-movie-review
python examples/trading_movie.py --out docs/assets/trading
```

`--preview` creates nine checkpoint frames and a contact sheet for inspecting captions, buy/sell
annotations, and scales. The rendered version follows two visual review passes of the generalized
examples. Rendering streams frames to FFmpeg to keep memory bounded. It produces a GIF, MP4, poster,
and numerical trace; the source is in [the renderer](../examples/trading_movie.py) and
[the teaching markets](../examples/trading_movie_data.py).
