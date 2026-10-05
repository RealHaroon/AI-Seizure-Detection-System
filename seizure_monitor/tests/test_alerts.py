"""Alert logic tests: fake clock, fake alarm, fake Telegram (no network, no sound)."""

from pathlib import Path

import pytest
import yaml

from monitor.alerts import AlertManager, format_duration

CFG = yaml.safe_load((Path(__file__).parent.parent / "config.yaml").read_text())["alerts"]


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeAlarm:
    def __init__(self):
        self.active = False
        self.starts = 0

    def start(self):
        self.active = True
        self.starts += 1

    def stop(self):
        self.active = False


class FakeTelegram:
    def __init__(self):
        self.on_ack = None
        self.alerts, self.texts, self.unbuttoned = [], [], []

    def send_alert(self, key, text, photo):
        self.alerts.append((key, text, photo))

    def send_text(self, text):
        self.texts.append(text)

    def remove_buttons(self, key):
        self.unbuttoned.append(key)

    def close(self):
        pass


@pytest.fixture
def env():
    clock, alarm, tg = FakeClock(), FakeAlarm(), FakeTelegram()
    mgr = AlertManager(CFG, "Bedroom camera", alarm, tg, clock=clock, run_timer=False)
    return mgr, clock, alarm, tg


def advance(mgr, clock, seconds):
    clock.t += seconds
    mgr.tick()


def test_alert_sent_with_alarm_and_button(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(start_wall=1_700_000_000, snapshot=b"jpeg")
    assert alarm.active
    assert len(tg.alerts) == 1
    key, text, photo = tg.alerts[0]
    assert "Bedroom camera" in text and photo == b"jpeg"
    assert mgr.status()["incident_active"] and not mgr.status()["call"]


def test_escalates_when_nobody_responds(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    advance(mgr, clock, CFG["escalate_after_s"] - 1)
    assert len(tg.alerts) == 1 and not mgr.status()["call"]
    advance(mgr, clock, 2)
    assert len(tg.alerts) == 2 and "URGENT" in tg.alerts[1][1]
    assert mgr.status()["call"]                      # Android app should call now
    advance(mgr, clock, 60)
    assert len(tg.alerts) == 2                       # escalates once, no repeated spam


def test_telegram_ack_stops_alarm_and_clears_call(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    advance(mgr, clock, CFG["escalate_after_s"] + 1)
    key = tg.alerts[0][0]
    clock.t += 9
    assert tg.on_ack(key, "Ayesha Khan") is True
    st = mgr.status()
    assert st["acknowledged_by"] == "Ayesha Khan" and not st["call"]
    assert not alarm.active
    assert tg.unbuttoned == [key]
    assert any("Ayesha Khan is with them" in t for t in tg.texts)


def test_ack_before_escalation_prevents_it(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    clock.t += 10
    tg.on_ack(tg.alerts[0][0], "Ali")
    advance(mgr, clock, 60)
    assert len(tg.alerts) == 1 and not mgr.status()["escalated"]


def test_double_press_and_old_buttons(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    key = tg.alerts[0][0]
    assert tg.on_ack(key, "Ali") and tg.on_ack(key, "Sara")   # second press accepted quietly
    assert mgr.status()["acknowledged_by"] == "Ali"
    assert sum("is with them" in t for t in tg.texts) == 1
    assert tg.on_ack("oldrun-7", "Stranger") is False          # button from an earlier alert


def test_local_stop_counts_as_acknowledgement(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    mgr.acknowledge_local()
    assert not alarm.active and mgr.status()["acknowledged_by"]
    advance(mgr, clock, 60)
    assert len(tg.alerts) == 1                                  # no escalation


def test_end_message_after_cooldown(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    tg.on_ack(tg.alerts[0][0], "Ali")
    mgr.on_event_end(1_700_000_065)
    advance(mgr, clock, CFG["cooldown_s"] - 1)
    assert not any("Event ended" in t for t in tg.texts)
    advance(mgr, clock, 2)
    assert any("Event ended, lasted 1 min 5 sec" in t for t in tg.texts)
    assert not mgr.status()["incident_active"]


def test_events_within_cooldown_are_one_incident(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    mgr.on_event_end(1_700_000_020)
    advance(mgr, clock, 10)
    mgr.on_event_start(1_700_000_030, None)     # rhythm returns within the cooldown
    mgr.on_event_end(1_700_000_050)
    advance(mgr, clock, CFG["cooldown_s"] + 1)
    assert len([a for a in tg.alerts if "Possible seizure detected" in a[1]]) == 1
    ended = [t for t in tg.texts if "Event ended" in t]
    assert len(ended) == 1 and "50 sec" in ended[0] and "2 episodes" in ended[0]


def test_new_incident_after_cooldown(env):
    mgr, clock, alarm, tg = env
    mgr.on_event_start(1_700_000_000, None)
    mgr.on_event_end(1_700_000_020)
    advance(mgr, clock, CFG["cooldown_s"] + 1)
    mgr.on_event_start(1_700_000_500, None)
    starts = [a for a in tg.alerts if "Possible seizure detected" in a[1]]
    assert len(starts) == 2 and starts[0][0] != starts[1][0]


def test_works_without_telegram():
    clock, alarm = FakeClock(), FakeAlarm()
    mgr = AlertManager(CFG, "cam", alarm, None, clock=clock, run_timer=False)
    mgr.on_event_start(1_700_000_000, None)
    advance(mgr, clock, CFG["escalate_after_s"] + 1)
    assert alarm.active and mgr.status()["call"]     # local alarm + call flag still work offline


def test_format_duration():
    assert format_duration(65) == "1 min 5 sec"
    assert format_duration(4.6) == "0 min 5 sec"
