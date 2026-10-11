"""Event movement traces (for the dashboard chart) and the startup self-check."""

import json
import socket

from monitor import selfcheck
from monitor.trace import EventTrace, trace_path


def test_trace_captures_before_and_after_the_event(tmp_path):
    tr = EventTrace(tmp_path, pre_s=5, post_s=3, threshold=0.04)
    for i in range(300):                        # 20 s at 15 fps
        t = i / 15
        if i == 150:                            # event detected at 10 s, movement began at 7 s
            tr.start(event_id=7, start=7.0, detected_at=10.0)
        if i == 225:                            # event ends at 15 s
            tr.end(15.0)
        tr.add(t, 0.1 if 7 <= t < 15 else 0.001, 0.0)
    data = json.loads(trace_path(tmp_path, 7).read_text())
    ts = [s[0] for s in data["samples"]]
    assert data["threshold"] == 0.04 and data["detected_at"] == 3.0 and data["end"] == 8.0
    assert -2.1 <= ts[0] <= -1.9                # 5 s of history before detection (10 s) = from 5 s = 2 s before movement start
    assert 10.9 <= ts[-1] <= 11.2               # until 3 s after the end (15 + 3 - 7)
    assert not tr.capturing


def test_trace_saved_on_pause_even_without_end(tmp_path):
    tr = EventTrace(tmp_path, pre_s=5, post_s=30, threshold=None)
    tr.add(0.0, None, 0.0)
    tr.start(3, 0.0, 1.0)
    tr.add(1.0, None, 2.5)
    tr.stop_now()
    data = json.loads(trace_path(tmp_path, 3).read_text())
    assert data["end"] is None and data["samples"][-1] == [1.0, None, 2.5]


def test_bed_area_check():
    assert selfcheck.check_bed_area(None, True).status == selfcheck.WARN
    assert selfcheck.check_bed_area([0.0, 0.0, 1.0, 1.0], True).status == selfcheck.WARN    # whole picture
    assert selfcheck.check_bed_area([0.25, 0.11, 0.69, 1.0], True).status == selfcheck.OK    # hospital-like box


def test_telegram_check_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert selfcheck.check_telegram(True).status == selfcheck.WARN
    assert selfcheck.check_telegram(False).status == selfcheck.WARN


def test_dashboard_port_in_use():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        c = selfcheck.check_dashboard({"enabled": True, "host": "127.0.0.1", "port": port})
        assert c.status == selfcheck.FAIL
    assert selfcheck.check_dashboard({"enabled": True, "host": "0.0.0.0", "port": port}).status == selfcheck.WARN


def test_disk_and_print(tmp_path, capsys):
    checks = [selfcheck.check_disk(tmp_path, min_gb=0), selfcheck.Check("X", selfcheck.FAIL, "broken")]
    assert selfcheck.print_checks(checks) == 1
    assert "1 failed" in capsys.readouterr().out
