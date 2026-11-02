"""Render the README's trading explainer from verified synthetic replay ledgers.

    python examples/trading_movie.py --verify-only
    python examples/trading_movie.py --preview --out /tmp/movie-preview
    python examples/trading_movie.py --out docs/assets/trading

Rendering requires the report extra and ffmpeg on PATH. Neither is required by the
numerical verification. Movies are silent, captioned, and use no external market data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from trading_movie_data import NS, Demo, build_demos, evidence, position_at, signal_at

BG, PANEL, INK, MUTED = "#0b1220", "#111e31", "#f3f5f8", "#a9b9cb"
GRID, BUY, SELL, BLUE, GOLD = "#26364a", "#53dcb4", "#ff978a", "#80b7ff", "#f1cf83"


@dataclass(frozen=True)
class Beat:
    chapter: str
    seconds: float
    start: float
    end: float
    badge: str
    explanation: str


# Display time is intentionally different from market time. Pauses explain exact states;
# prices and signals are never interpolated between market observations.
BEATS = (
    Beat("intro", 5, 0, 0, "", ""),
    Beat("flow", 3, 0, 2.95, "WAIT", "No strong burst yet.\nStay flat."),
    Beat(
        "flow",
        4,
        3.010,
        3.010,
        "BUY TO OPEN",
        "Buying pressure crossed\n+1.5k units. Open a long\nposition: +100 units.",
    ),
    Beat(
        "flow",
        4,
        3.010,
        10.5,
        "HOLD LONG",
        "Older trades fade out.\nFalling below +1.5k units\ndoes not itself close\nthis policy's position.",
    ),
    Beat(
        "flow",
        3,
        11.010,
        11.010,
        "CLOSE, THEN REVERSE",
        "An opposite burst.\nSELL closes the long.\nOnce acknowledged,\nSELL opens a short.",
    ),
    Beat(
        "flow",
        3,
        11.020,
        17.95,
        "HOLD SHORT",
        "A short position is\nnegative exposure.\nIt benefits if the\nprice falls.",
    ),
    Beat("flow", 3, 18.010, 20, "BUY TO CLOSE", "The session ends.\nBUY covers the short.\nThe account is flat."),
    Beat(
        "lead_gap",
        3,
        0,
        1.49,
        "WATCH TWO VENUES",
        "Compare their recent\nprice changes over\nthe same 1-second\nwindow.",
    ),
    Beat(
        "lead_gap",
        4,
        1.510,
        1.510,
        "BUY THE FOLLOWER",
        "Venue B moved first.\nThe gap crossed\n+0.75 bp: BUY Venue A.\n1 bp = 0.01%.",
    ),
    Beat(
        "lead_gap",
        3,
        1.510,
        2.299,
        "HOLD LONG",
        "The signal uses only\nreceived prices.\nFuture observations\nremain hidden.",
    ),
    Beat(
        "lead_gap",
        4,
        2.310,
        2.39,
        "SELL TO CLOSE",
        "Venue A caught up.\nThe gap fell to zero,\ntriggering this\nexample's exit rule.",
    ),
    Beat("missed_fill", 7, 0, 10, "", ""),
    Beat("missed_fill", 2, 10, 10, "", ""),
    Beat("costs", 7, 0, 0, "", ""),
)
DURATION = sum(b.seconds for b in BEATS)


def scene_at(seconds: float) -> tuple[Beat, float, float]:
    offset = 0.0
    for beat in BEATS:
        if seconds < offset + beat.seconds:
            progress = max(0.0, (seconds - offset) / beat.seconds)
            return beat, beat.start + progress * (beat.end - beat.start), progress
        offset += beat.seconds
    return BEATS[-1], 0.0, 1.0


class Movie:
    def __init__(self, demos: dict[str, Demo], width: int = 1280):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        self.plt, self.demos = plt, demos
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "text.color": INK,
                "axes.labelcolor": MUTED,
                "xtick.color": MUTED,
                "ytick.color": MUTED,
                "axes.edgecolor": GRID,
                "font.size": 12,
                "path.simplify": False,
            }
        )
        self.fig = plt.figure(figsize=(12.8, 8), dpi=width / 12.8, facecolor=BG)

    def text(
        self, x: float, y: float, text: str, size: float = 14, color: str = INK, weight: str = "normal", **kwargs: Any
    ) -> Any:
        return self.fig.text(x, y, text, fontsize=size, color=color, weight=weight, va="top", **kwargs)

    def box(self, x: float, y: float, w: float, h: float, color: str = PANEL) -> None:
        from matplotlib.patches import FancyBboxPatch

        self.fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=self.fig.transFigure,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                facecolor=color,
                edgecolor=GRID,
                linewidth=0.7,
                zorder=0,
            )
        )

    def axis(self, rect: list[float]) -> Any:
        ax = self.fig.add_axes(rect, facecolor=PANEL)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, alpha=0.75, linewidth=0.7)
        ax.tick_params(labelsize=11.5, length=0, pad=7)
        ax.set_axisbelow(True)
        return ax

    def header(self, chapter: str, title: str, subtitle: str, seconds: float) -> None:
        self.text(0.047, 0.968, "FINANCEAI   /   METHODS IN MOTION", 10.5, MUTED, "bold")
        self.text(0.955, 0.968, "SYNTHETIC MARKET · REAL REPLAY ENGINE", 10, GOLD, ha="right")
        self.text(0.047, 0.911, title, 27, weight="bold")
        self.text(0.047, 0.853, subtitle, 13.5, MUTED)
        self.text(0.047, 0.052, chapter, 10.5, GOLD, "bold")
        self.text(0.955, 0.052, "Teaching policies · no historical performance claim", 10.5, MUTED, ha="right")
        from matplotlib.lines import Line2D

        self.fig.add_artist(
            Line2D([0.047, 0.955], [0.019, 0.019], transform=self.fig.transFigure, color=GRID, linewidth=2)
        )
        self.fig.add_artist(
            Line2D(
                [0.047, 0.047 + 0.908 * min(seconds / DURATION, 1)],
                [0.019, 0.019],
                transform=self.fig.transFigure,
                color=BLUE,
                linewidth=2,
            )
        )

    def intro(self, seconds: float) -> None:
        self.header(
            "THE IDEA",
            "When does a signal become a trade?",
            "Two causal signal families, followed all the way to orders, fills, and costs.",
            seconds,
        )
        cards = [
            (
                "01",
                "FLOW",
                "Follow trading pressure",
                BUY,
                "Buyer-initiated trades add.\nSeller-initiated trades subtract.\nOlder trades fade away.",
            ),
            (
                "02",
                "LEAD–LAG",
                "Watch who moves first",
                BLUE,
                "Compare two venues.\nLook for a delayed move.\nTrade the follower.",
            ),
            (
                "03",
                "EXECUTION",
                "Check what really fills",
                GOLD,
                "Orders take time to arrive.\nPrice limits still apply.\nFees change the result.",
            ),
        ]
        for i, (number, name, title, color, copy) in enumerate(cards):
            x = 0.059 + 0.305 * i
            self.box(x, 0.257, 0.279, 0.492)
            self.text(x + 0.018, 0.716, number, 32, color, "bold")
            self.text(x + 0.018, 0.609, name, 12, color, "bold")
            self.text(x + 0.018, 0.557, title, 15, weight="bold")
            self.text(x + 0.018, 0.471, copy, 14, MUTED, linespacing=1.9)
        self.text(0.5, 0.172, "Observe → decide → send → fill → acknowledge", 20, ha="center")
        self.text(
            0.5,
            0.115,
            "BUY can open a long or close a short. SELL can close a long or open a short.",
            12.5,
            MUTED,
            ha="center",
        )

    def replay(self, beat: Beat, t: float, seconds: float) -> None:
        from matplotlib.ticker import FuncFormatter

        demo = self.demos[beat.chapter]
        flow = beat.chapter == "flow"
        title = "Flow: follow buying and selling pressure" if flow else "Lead–lag: trade the slower response"
        subtitle = (
            "Signal = decayed aggressive buys − sells. Thresholds: ±1,500 units; decay time: 1 second."
            if flow
            else "Signal = Venue B's 1-second return − Venue A's 1-second return, in basis points."
        )
        self.header("01 / FLOW" if flow else "02 / LEAD–LAG", title, subtitle, seconds)
        now = round(t * NS)
        xmax = 20.0 if flow else 2.5
        self.box(0.063, 0.132, 0.622, 0.645)
        self.box(0.729, 0.132, 0.214, 0.645)
        price = self.axis([0.115, 0.548, 0.543, 0.182])
        sig = self.axis([0.115, 0.335, 0.543, 0.138])
        pos = self.axis([0.115, 0.171, 0.543, 0.089])
        self.text(0.085, 0.765, "RECEIVED MIDPOINT  /  USD", 10.5, MUTED, "bold")
        self.text(
            0.085, 0.508, "SIGNED FLOW  /  1,000 UNITS" if flow else "RETURN GAP  /  BASIS POINTS", 10.5, MUTED, "bold"
        )
        self.text(0.085, 0.296, "ACKNOWLEDGED POSITION  /  UNITS", 10.5, MUTED, "bold")
        for ax in (price, sig, pos):
            ax.set_xlim(0, xmax)
            ax.axvline(t, color=INK, alpha=0.45, linewidth=1, linestyle=(0, (2, 3)))
        for v, tape in enumerate(demo.market.data.venues):
            x = tape.book.recv_ns / NS
            y = (tape.book.bid_px[:, 0].astype(float) + tape.book.ask_px[:, 0]) * tape.instrument.tick_float / 2
            keep = tape.book.recv_ns <= now
            price.step(
                x[keep],
                y[keep],
                where="post",
                color=BUY if v == 0 else BLUE,
                linewidth=2.1,
                linestyle="-" if v == 0 else "--",
                label="Venue A" if v == 0 else "Venue B",
            )
            if np.any(keep):
                price.scatter([x[keep][-1]], [y[keep][-1]], s=18, color=BUY if v == 0 else BLUE, zorder=5)
        price.set_ylim((99.97, 100.1) if flow else (99.994, 100.032))
        price.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.2f}"))
        price.legend(loc="upper left", fontsize=10.5, frameon=False, labelcolor=INK, ncols=2)
        fills = demo.result.fills
        for i in np.flatnonzero(fills.arrival_ns <= now):
            side = int(fills.side[i])
            x, y = fills.arrival_ns[i] / NS, fills.px_ticks[i] * demo.market.data.venues[0].instrument.tick_float
            price.scatter(
                [x],
                [y],
                marker="^" if side > 0 else "v",
                s=88,
                facecolors=BUY if side > 0 else SELL,
                edgecolors=BG,
                linewidths=1,
                zorder=6,
            )
        # Label actions at real fill coordinates; the two reversal sells are almost
        # coincident at this scale, so one explicit label explains both orders.
        labels = (
            [
                (0, "BUY · open", (3.8, 99.986)),
                (2, "SELL · close → open", (12.0, 100.084)),
                (3, "BUY · close", (14.0, 99.977)),
            ]
            if flow
            else [(0, "BUY · open", (0.4, 100.013)), (1, "SELL · close", (1.75, 100.03))]
        )
        for i, label, location in labels:
            if fills.arrival_ns[i] <= now:
                color = BUY if fills.side[i] > 0 else SELL
                price.annotate(
                    label,
                    (fills.arrival_ns[i] / NS, fills.px_ticks[i] * demo.market.data.instruments[0].tick_float),
                    xytext=location,
                    fontsize=10.5,
                    color=color,
                    weight="bold",
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
                    bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.5},
                )
        # One sample per receipt cohort: this is the signal actually used at decisions.
        recv = demo.market.stream.recv_ns
        ends = np.r_[recv[:-1] != recv[1:], True]
        seen = ends & (recv <= now)
        x, y = recv[seen] / NS, demo.signal[seen] / (1000 if flow else 1)
        threshold = demo.params.enter / (1000 if flow else 1)
        sig.axhline(threshold, color=BUY, linestyle=(0, (4, 3)), linewidth=1)
        if flow:
            sig.axhline(-threshold, color=SELL, linestyle=(0, (4, 3)), linewidth=1)
        else:
            sig.axhline(demo.params.exit, color=GOLD, linestyle=(0, (4, 3)), linewidth=1)
        sig.axhline(0, color=MUTED, alpha=0.35, linewidth=0.7)
        sig.step(x, y, where="post", color=GOLD, linewidth=1.8)
        sig.fill_between(x, y, 0, step="post", color=GOLD, alpha=0.10)
        sig.set_ylim((-2.6, 2.4) if flow else (-0.3, 2.5))
        sig.set_yticks([-1.5, 0, 1.5] if flow else [0, 0.75, 2.0])
        sig.text(
            0.99,
            0.94,
            "BUY ≥ +1.5" if flow else "BUY ≥ +0.75",
            color=BUY,
            transform=sig.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.5},
        )
        sig.text(
            0.99,
            0.06,
            "SELL ≤ −1.5" if flow else "EXIT ≤ +0.05",
            color=SELL if flow else GOLD,
            transform=sig.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.5},
        )
        count = int(np.searchsorted(fills.ack_ns, now, side="right"))
        px = np.r_[0, fills.ack_ns[:count] / NS, t]
        positions = np.r_[0, np.cumsum(fills.side[:count].astype(np.int64) * fills.qty_units[:count])]
        py = np.r_[positions, positions[-1]]
        pos.step(px, py, where="post", color=BLUE, linewidth=2)
        pos.fill_between(px, py, 0, step="post", color=BLUE, alpha=0.17)
        pos.set_ylim(-140, 140)
        pos.set_yticks([-100, 0, 100], ["−100", "0", "+100"])
        pos.set_xlabel("Market time (seconds)", fontsize=11, labelpad=5)
        price.tick_params(labelbottom=False)
        sig.tick_params(labelbottom=False)

        p = position_at(demo, now)
        color = BUY if p > 0 else SELL if p < 0 else GOLD
        self.text(0.742, 0.753, beat.badge, 10.5, color, "bold")
        self.text(0.742, 0.701, "LONG" if p > 0 else "SHORT" if p < 0 else "FLAT", 25, color, "bold")
        self.text(0.742, 0.638, f"{p:+,} units" if p else "0 units", 13, MUTED)
        self.text(0.742, 0.566, "WHY THIS ACTION?", 10.5, MUTED, "bold")
        self.text(0.742, 0.521, beat.explanation, 13, linespacing=1.7)
        signal = signal_at(demo, now)
        self.text(0.742, 0.344, "CURRENT SIGNAL", 10.5, MUTED, "bold")
        flow_text = f"{signal:+.1f} units" if abs(signal) < 50 else f"{signal / 1000:+.2f}k units"
        self.text(0.742, 0.302, flow_text if flow else f"{signal:+.2f} bp", 22, GOLD, "bold")
        self.text(0.742, 0.233, f"t = {t:.3f} s", 13, INK)
        self.text(0.742, 0.193, "Paused to explain" if beat.start == beat.end else "Replay advancing", 10.5, MUTED)
        self.text(
            0.115,
            0.091,
            "▲ BUY fill   ▼ SELL fill     Fill prices shown; position changes at acknowledgement.",
            10.5,
            MUTED,
        )

    def missed_fill(self, milliseconds: float, seconds: float) -> None:
        self.header(
            "03 / EXECUTION",
            "A buy signal does not guarantee a fill",
            "Zoom into one order: 5 ms to the exchange, then 5 ms for its acknowledgement.",
            seconds,
        )
        demo = self.demos["missed_fill"]
        order = demo.result.orders
        tick = demo.market.data.venues[0].instrument.tick_float
        limit = float(order.limit_ticks[0] * tick)
        arrival_ask = float(demo.market.apx[0][order.book_row[0], 0] * tick)
        self.box(0.065, 0.365, 0.872, 0.388)
        ax = self.fig.add_axes([0.115, 0.41, 0.775, 0.22], facecolor=PANEL)
        ax.set_xlim(-0.6, 10.6)
        ax.set_ylim(-1, 1)
        ax.axis("off")
        ax.plot([0, 10], [0, 0], color=GRID, linewidth=5)
        ax.plot([0, milliseconds], [0, 0], color=BLUE, linewidth=5)
        for at, label, detail in (
            (0, "SEND BUY", "0 ms"),
            (3, "ASK MOVES", "+3 ms\nexchange only"),
            (5, "ORDER ARRIVES", "+5 ms"),
            (10, "REPORT RECEIVED", "+10 ms"),
        ):
            color = GOLD if at <= milliseconds else MUTED
            ax.scatter(
                [at],
                [0],
                s=120,
                facecolors=color if at <= milliseconds else PANEL,
                edgecolors=color,
                linewidth=2,
                zorder=4,
            )
            ax.text(at, 0.45, label, ha="center", color=color, fontsize=11, weight="bold")
            ax.text(at, -0.38, detail, ha="center", va="top", color=MUTED, fontsize=12)
        self.text(0.087, 0.722, f"TIME SINCE SUBMISSION   {milliseconds:04.1f} ms", 12, BLUE, "bold")
        self.box(0.065, 0.155, 0.405, 0.151)
        self.box(0.505, 0.155, 0.432, 0.151)
        self.text(0.084, 0.286, "BUY PRICE LIMIT · SET FROM RECEIVED BOOK", 10.5, MUTED, "bold")
        self.text(0.084, 0.240, f"${limit:,.2f}", 25, BUY, "bold")
        self.text(0.526, 0.286, "ASK AT EXCHANGE ARRIVAL", 10.5, MUTED, "bold")
        self.text(
            0.526,
            0.240,
            f"${arrival_ask:,.2f}" if milliseconds >= 5 else "Not yet reached",
            25,
            SELL if milliseconds >= 5 else MUTED,
            "bold",
        )
        if milliseconds >= 5:
            self.text(0.915, 0.225, "NO FILL", 14, SELL, "bold", ha="right")
        if milliseconds < 3:
            copy = "BUY is sent at the last received ask. The order is still traveling."
        elif milliseconds < 5:
            copy = "The controller sees the old quote. The exchange has already changed it."
        elif milliseconds < 10:
            copy = "The new ask exceeds the limit. The order expires unfilled at the exchange."
        else:
            copy = "The no-fill report arrives. The controller confirms that its position is still flat."
        self.text(0.5, 0.111, copy, 13, MUTED, ha="center")

    def costs(self, seconds: float) -> None:
        self.header(
            "04 / COSTS",
            "Right direction. Negative net result.",
            "The lead–lag example bought before the rise. Spread and taker fees still matter.",
            seconds,
        )
        result = self.demos["lead_gap"].result
        fee = sum(result.settlement.fees)
        net = float(result.settlement.net_pnl)
        values = [
            ("PRICE RESULT", "After crossing the spread", net + fee, BUY),
            ("TAKER FEES", "1 bp = 0.01% per fill", -fee, GOLD),
            ("NET RESULT", "Both fills; account ends flat", net, SELL),
        ]
        for i, (title, detail, value, color) in enumerate(values):
            x = 0.059 + i * 0.305
            self.box(x, 0.385, 0.279, 0.33)
            self.text(x + 0.018, 0.682, title, 12, MUTED, "bold")
            self.text(x + 0.018, 0.611, f"{value:+.2f}", 38, color, "bold")
            self.text(x + 0.018, 0.521, "USD", 14, color)
            self.text(x + 0.018, 0.442, detail, 11.5, MUTED)
        self.text(0.5, 0.305, "A signal is a reason to act—not a promise of profit.", 23, weight="bold", ha="center")
        self.text(
            0.5,
            0.232,
            "Check the signal, execution, costs, risk limits, and previously unseen data.",
            14,
            MUTED,
            ha="center",
        )
        self.text(
            0.5, 0.164, "Linear contracts · 100 units per position · Amounts rounded to cents.", 12.5, GOLD, ha="center"
        )

    def frame(self, seconds: float) -> np.ndarray:
        self.fig.clear()
        beat, t, _ = scene_at(seconds)
        if beat.chapter == "intro":
            self.intro(seconds)
        elif beat.chapter in ("flow", "lead_gap"):
            self.replay(beat, t, seconds)
        elif beat.chapter == "missed_fill":
            self.missed_fill(t, seconds)
        else:
            self.costs(seconds)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()

    def save_frame(self, seconds: float, path: Path) -> None:
        self.frame(seconds)
        self.fig.savefig(path, dpi=self.fig.dpi, facecolor=BG)


def render(demos: dict[str, Demo], out: Path, *, preview: bool, fps: int, width: int) -> None:
    from matplotlib import pyplot as plt

    if shutil.which("ffmpeg") is None:
        raise SystemExit("Rendering needs ffmpeg on PATH. Numerical checks work with --verify-only.")
    out.mkdir(parents=True, exist_ok=True)
    movie = Movie(demos, width)
    movie.save_frame(21.0, out / "poster.png")
    if preview:
        checkpoints = [1.0, 9.0, 14.5, 18.0, 21.0, 30.0, 37.0, 45.0, 51.0]
        for i, seconds in enumerate(checkpoints):
            movie.save_frame(seconds, out / f"review-{i + 1:02}.png")
        # Review typography and narrative transitions together, without shipping draft assets.
        sheet, axes = plt.subplots(3, 3, figsize=(19.2, 12), facecolor=BG)
        for i, ax in enumerate(axes.flat):
            ax.imshow(plt.imread(out / f"review-{i + 1:02}.png"))
            ax.set_axis_off()
        sheet.subplots_adjust(0, 0, 1, 1, wspace=0.015, hspace=0.025)
        sheet.savefig(out / "contact-sheet.png", dpi=100, facecolor=BG)
        plt.close(sheet)
        plt.close(movie.fig)
        return
    # Stream frames: memory stays bounded instead of retaining a full uncompressed movie.
    with tempfile.TemporaryDirectory(prefix="financeai-movie-") as temp:
        staging = Path(temp)
        height = round(width * 8 / 12.8)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(staging / "trading-methods.mp4"),
        ]
        with (staging / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                assert process.stdin is not None
                count = round(DURATION * fps)
                for i in range(count):
                    process.stdin.write(movie.frame(i / fps).tobytes())
                    if i % (fps * 5) == 0:
                        print(f"Rendered {i / fps:.0f}/{DURATION:.0f} seconds", flush=True)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError((staging / "ffmpeg.log").read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(staging / "trading-methods.mp4"),
                "-filter_complex",
                "fps=12,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop",
                "0",
                str(staging / "trading-methods.gif"),
            ],
            check=True,
        )
        for name in ("trading-methods.mp4", "trading-methods.gif"):
            shutil.copyfile(staging / name, out / name)
    manifest = evidence(demos)
    manifest["render"] = {
        "duration_seconds": DURATION,
        "width": width,
        "height": height,
        "mp4_fps": fps,
        "gif_fps": 12,
        "captions": "embedded",
        "audio": False,
    }
    manifest["files_sha256"] = {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in ("trading-methods.mp4", "trading-methods.gif", "poster.png")
    }
    manifest["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("trading_movie.py", "trading_movie_data.py")
    }
    (out / "trace.json").write_text(json.dumps(manifest, indent=2, default=str, allow_nan=False) + "\n")
    plt.close(movie.fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/trading"))
    parser.add_argument("--preview", action="store_true", help="render review frames without encoding movies")
    parser.add_argument("--verify-only", action="store_true", help="verify real replays; no plotting dependencies")
    parser.add_argument("--width", type=int, default=1280, choices=(640, 960, 1280, 1600))
    parser.add_argument("--fps", type=int, default=16, choices=(8, 12, 16, 24))
    args = parser.parse_args()
    demos = build_demos()
    print("Verified three synthetic replays against every reference order/fill field.", flush=True)
    if not args.verify_only:
        render(demos, args.out, preview=args.preview, fps=args.fps, width=args.width)
        print(f"Wrote animation assets to {args.out}", flush=True)


if __name__ == "__main__":
    main()
