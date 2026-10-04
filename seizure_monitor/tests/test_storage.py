import pytest

from monitor.storage import EventStore


def test_event_lifecycle(tmp_path):
    store = EventStore(tmp_path / "events.db")
    eid = store.start_event("2026-10-03T02:14:05", 0.6, "clips/event_1.mp4", "Bedroom camera")

    ev = store.get_event(eid)
    assert ev["end_time"] is None                    # recorded as soon as it starts
    assert ev["review_label"] == "unreviewed"

    store.end_event(eid, "2026-10-03T02:15:10", 65.0, 0.72)
    store.set_review(eid, "real_seizure", "Typical GTC, matches EEG")
    ev = store.get_event(eid)
    assert (ev["duration_s"], ev["confidence"]) == (65.0, 0.72)
    assert (ev["review_label"], ev["reviewer_note"]) == ("real_seizure", "Typical GTC, matches EEG")


def test_invalid_label_rejected(tmp_path):
    store = EventStore(tmp_path / "events.db")
    eid = store.start_event("2026-10-03T02:14:05", 0.6, None, "cam")
    with pytest.raises(ValueError):
        store.set_review(eid, "maybe")


def test_list_newest_first(tmp_path):
    store = EventStore(tmp_path / "events.db")
    store.start_event("2026-10-01T10:00:00", 0.5, None, "cam")
    store.start_event("2026-10-02T10:00:00", 0.5, None, "cam")
    assert [e["start_time"][:10] for e in store.list_events()] == ["2026-10-02", "2026-10-01"]
