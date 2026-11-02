"""Synthetic recorder-format fixtures (values are invented, not market data)."""

from __future__ import annotations

import pytest

# Two leading log lines, a warm-up snapshot with no levels, a normal snapshot, two
# trade lines (one with two prints), a snapshot whose declared depth exceeds the ten
# printed levels, and a final one-level snapshot.
SYNTHETIC_RECORDER_TEXT = (
    b"23-03-02 14:18:33.372532 [INFO simulation/any_file_event_provider.cpp:24] "
    b"Read events from binary data: /x/synthetic.bin\n"
    b"23-03-02 14:18:33.372640 [INFO base/logging.cpp:120] Starting thread for logging io "
    b"service id: 1\n"
    b"23-01-30 00:00:00.000100 type: OrderbookSnapshot symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:00.000100 server_time: 1675036800000000 orderbook:  "
    b"bid_size: 0 ask_size: 0\n"
    b"\n"
    b"23-01-30 00:00:00.001000 type: OrderbookSnapshot symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:00.001000 server_time: 1675036799998000 orderbook:  "
    b"bid_size: 2 ask_size: 2\n"
    b"Ask price: 20001.500000 volume: 300.000000\n"
    b"Ask price: 20000.500000 volume: 1000.000000\n"
    b"Bid price: 20000.000000 volume: 2000.000000\n"
    b"Bid price: 19999.000000 volume: 500.000000\n"
    b"\n"
    b"23-01-30 00:00:00.002500 type: Trades symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:00.002500 server_time: 1675036800001000 "
    b"trades: [ { Bid 20000.500000 100.000000 } ]\n"
    b"23-01-30 00:00:00.002500 type: Trades symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:00.002500 server_time: 1675036800001000 "
    b"trades: [ { Ask 20000.000000 200.000000 } { Ask 19999.000000 300.000000 } ]\n"
    b"23-01-30 00:00:00.003000 type: OrderbookSnapshot symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:00.003000 server_time: 1675036800002000 orderbook:  "
    b"bid_size: 12 ask_size: 1\n"
    b"Ask price: 20000.500000 volume: 900.000000\n"
    + b"".join(b"Bid price: %d.000000 volume: %d.000000\n" % (19999 - k, 100 * (k + 1)) for k in range(10))
    + b"\n"
    b"23-01-30 00:00:01.000000 type: OrderbookSnapshot symbol: none symbol_id: none "
    b"receive_time: 23-01-30 00:00:01.000000 server_time: 1675036800995000 orderbook:  "
    b"bid_size: 1 ask_size: 1\n"
    b"Ask price: 20003.000000 volume: 10.000000\n"
    b"Bid price: 20002.500000 volume: 20.000000\n"
    b"\n"
)


@pytest.fixture
def synthetic_recorder_text() -> bytes:
    return SYNTHETIC_RECORDER_TEXT
