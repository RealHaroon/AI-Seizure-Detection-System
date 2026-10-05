"""Alerts: local alarm, Telegram messages, acknowledgement and escalation.

Design rules:
  - The local alarm needs no internet, so it always works and starts first.
  - Network calls run in background threads with timeouts. A slow or dead
    connection can never block or crash the detection loop; failures are logged.
  - Events closer together than `cooldown_s` belong to one *incident*: the
    family gets one alert and one "ended" message, not a burst of messages.

Incident timeline:
  event starts -> alarm + Telegram alert with "I'm with them" button
  nobody responds within escalate_after_s -> urgent resend + `call` flag set
      (the dashboard exposes it at GET /api/status for an Android automation
      app to place a phone call)
  someone presses the button (or stops the alarm at the monitor) -> everyone
      is told who responded, alarm stops, `call` flag clears
  event ends and no new event for cooldown_s -> "Event ended, lasted X min Y sec"
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import requests

log = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m} min {s} sec"


# --------------------------------------------------------------- local alarm

class LocalAlarm:
    """Loud two-tone beeping through the laptop speakers until stopped."""

    def __init__(self, cfg: dict):
        self.enabled = cfg["enabled"]
        self.cfg = cfg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if not self.enabled or self.active:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="alarm")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            # Without this, SDL (pygame's audio library) swallows Ctrl+C / SIGTERM once the alarm has
            # played, and the monitor can no longer be stopped from the terminal.
            os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
            import pygame
            pygame.mixer.init()
            rate, _, channels = pygame.mixer.get_init()
            beeps = [self._tone(pygame, hz, rate, channels) for hz in self.cfg["tones_hz"]]
            log.warning("Local alarm ON (press 's' in the monitor window to stop)")
            for beep in itertools.cycle(beeps):
                if self._stop.is_set():
                    break
                beep.play()
                self._stop.wait(self.cfg["beep_s"] + self.cfg["gap_s"])
            pygame.mixer.stop()
            pygame.mixer.quit()
            log.info("Local alarm stopped")
        except Exception:
            log.exception("Local alarm failed (no audio device?)")

    def _tone(self, pygame, hz, rate, channels):
        t = np.arange(int(rate * self.cfg["beep_s"])) / rate
        wave = (np.sin(2 * np.pi * hz * t) * 32767 * self.cfg["volume"]).astype(np.int16)
        if channels == 2:
            wave = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(np.ascontiguousarray(wave))


# ------------------------------------------------------------------ telegram

class TelegramClient:
    """Minimal Telegram Bot API client.

    One thread sends messages from a queue (with retries); another long-polls
    getUpdates for button presses. Long polling means no public webhook/server
    is needed: the laptop asks Telegram "any button presses?" in a loop.
    """

    def __init__(self, token: str, chat_id: str, cfg: dict):
        self._base = f"https://api.telegram.org/bot{token}/"
        self.chat_id = str(chat_id)
        self.cfg = cfg
        self.on_ack = None                 # set by AlertManager: fn(incident_key, who) -> bool
        self._q: queue.Queue = queue.Queue()
        self._button_msgs: dict[str, list[int]] = {}   # incident key -> message ids that carry the button
        self._closing = threading.Event()
        self._threads = [threading.Thread(target=self._send_loop, daemon=True, name="tg-send"),
                         threading.Thread(target=self._poll_loop, daemon=True, name="tg-poll")]

    def start(self):
        for th in self._threads:
            th.start()

    def close(self, wait_s: float = 5.0):
        """Give queued messages a moment to go out, then stop."""
        self._q.put(None)
        self._threads[0].join(timeout=wait_s)
        self._closing.set()

    # Public API: only enqueues, never blocks the caller on the network.
    def send_alert(self, key: str, text: str, photo: bytes | None):
        self._q.put(("alert", key, text, photo))

    def send_text(self, text: str):
        self._q.put(("text", None, text, None))

    def remove_buttons(self, key: str):
        self._q.put(("unbutton", key, None, None))

    # Sender thread
    def _send_loop(self):
        while (job := self._q.get()) is not None:
            kind, key, text, photo = job
            for attempt in range(1, self.cfg["send_retries"] + 1):
                try:
                    self._do(kind, key, text, photo)
                    break
                except Exception as e:
                    log.error("Telegram %s failed (attempt %d/%d): %s", kind, attempt, self.cfg["send_retries"], e)
                    time.sleep(self.cfg["retry_backoff_s"] * attempt)

    def _do(self, kind, key, text, photo):
        if kind == "alert":
            markup = json.dumps({"inline_keyboard": [[{"text": "✅ I'm with them", "callback_data": f"ack:{key}"}]]})
            if photo:
                msg = self._call("sendPhoto", data={"chat_id": self.chat_id, "caption": text, "reply_markup": markup},
                                 files={"photo": ("snapshot.jpg", photo, "image/jpeg")})
            else:
                msg = self._call("sendMessage", data={"chat_id": self.chat_id, "text": text, "reply_markup": markup})
            self._button_msgs.setdefault(key, []).append(msg["message_id"])
        elif kind == "text":
            self._call("sendMessage", data={"chat_id": self.chat_id, "text": text})
        elif kind == "unbutton":
            for mid in self._button_msgs.pop(key, []):
                try:
                    self._call("editMessageReplyMarkup", data={"chat_id": self.chat_id, "message_id": mid,
                                                               "reply_markup": json.dumps({"inline_keyboard": []})})
                except Exception as e:
                    log.info("Could not remove button from message %s: %s", mid, e)

    def _call(self, method, data=None, files=None, timeout=None):
        r = requests.post(self._base + method, data=data, files=files,
                          timeout=timeout or self.cfg["request_timeout_s"])
        body = r.json()
        if not body.get("ok"):
            raise RuntimeError(f"{method}: {body.get('description', r.status_code)}")
        return body["result"]

    # Poller thread
    def _poll_loop(self):
        offset = None
        try:
            # Skip button presses left over from before this run (old alerts).
            old = self._call("getUpdates", data={"offset": -1, "timeout": 0})
            if old:
                offset = old[-1]["update_id"] + 1
        except Exception as e:
            log.error("Telegram getUpdates failed: %s", e)
        failing = False
        while not self._closing.is_set():
            try:
                updates = self._call("getUpdates",
                                     data={"offset": offset, "timeout": self.cfg["poll_timeout_s"],
                                           "allowed_updates": json.dumps(["callback_query"])},
                                     timeout=self.cfg["poll_timeout_s"] + self.cfg["request_timeout_s"])
                if failing:
                    log.info("Telegram polling recovered")
                    failing = False
                for u in updates:
                    offset = u["update_id"] + 1
                    if "callback_query" in u:
                        self._handle_callback(u["callback_query"])
            except Exception as e:
                # Typically no internet. Detection and the local alarm carry on; we keep retrying,
                # but log only the first failure so a night offline doesn't flood the log.
                if not failing:
                    log.error("Telegram polling failed (will keep retrying quietly): %s", e)
                    failing = True
                self._closing.wait(self.cfg["retry_backoff_s"] * 2)

    def _handle_callback(self, cq):
        chat = str(cq.get("message", {}).get("chat", {}).get("id"))
        data = cq.get("data", "")
        user = cq.get("from", {})
        who = " ".join(filter(None, [user.get("first_name"), user.get("last_name")])) or user.get("username", "someone")
        ok = False
        # Only accept presses from our own chat, so a stranger can't silence an alert.
        if chat == self.chat_id and data.startswith("ack:") and self.on_ack:
            ok = self.on_ack(data[4:], who)
        reply = "Thank you. Everyone has been told you're with them." if ok else "This alert is no longer active."
        try:
            self._call("answerCallbackQuery", data={"callback_query_id": cq["id"], "text": reply})
        except Exception as e:
            log.info("answerCallbackQuery failed: %s", e)


# ------------------------------------------------------------- alert manager

@dataclass
class Incident:
    key: str                      # unique id used in the Telegram button
    started_wall: float           # wall time the first event's rhythm began
    alerted_at: float             # monotonic time the first alert went out
    snapshot: bytes | None
    last_end_wall: float | None = None
    close_at: float | None = None # monotonic time to send "ended" if no new event arrives
    ended: bool = False
    escalated: bool = False
    acknowledged_by: str | None = None
    acknowledged_after_s: float | None = None
    events: int = 1


class AlertManager:
    def __init__(self, cfg: dict, source_name: str, alarm: LocalAlarm, telegram: TelegramClient | None,
                 clock=time.monotonic, run_timer: bool = True):
        self.cfg = cfg
        self.source = source_name
        self.alarm = alarm
        self.telegram = telegram
        self.clock = clock
        self._lock = threading.Lock()
        self._incident: Incident | None = None
        self._run_id = secrets.token_hex(3)       # makes button ids unique across restarts
        self._counter = itertools.count(1)
        if telegram:
            telegram.on_ack = self._telegram_ack
        # The timer runs escalation/close checks even if the video loop stalls (e.g. camera hiccup).
        self._stop = threading.Event()
        if run_timer:
            threading.Thread(target=self._timer_loop, daemon=True, name="alert-timer").start()

    # ---- called from the detection loop (must be fast, never raise) ----

    def on_event_start(self, start_wall: float, snapshot: bytes | None):
        with self._lock:
            inc = self._incident
            if inc and not inc.ended:
                # Rhythm came back within the cooldown: same incident, no new message (no spam).
                inc.events += 1
                inc.close_at = None
                log.info("Event %d joined the current incident (within cooldown)", inc.events)
                if inc.acknowledged_by is None:
                    self.alarm.start()
                return
            inc = Incident(key=f"{self._run_id}-{next(self._counter)}", started_wall=start_wall,
                           alerted_at=self.clock(), snapshot=snapshot)
            self._incident = inc
        self.alarm.start()
        self._send(inc.key, f"🚨 Possible seizure detected\n"
                            f"Camera: {self.source}\n"
                            f"Time: {_clock_str(start_wall)}\n"
                            f"Rhythmic convulsive movement detected. Please check on them now.\n"
                            f"(Automatic alert from the camera monitor.)", snapshot)

    def on_event_end(self, end_wall: float):
        with self._lock:
            if self._incident:
                self._incident.last_end_wall = end_wall
                self._incident.close_at = self.clock() + self.cfg["cooldown_s"]

    def acknowledge_local(self):
        """Alarm stopped with the keyboard at the monitor: someone is physically there."""
        self.alarm.stop()
        self._acknowledge(None, "Someone at the monitor")

    # ---- timers ----

    def tick(self):
        with self._lock:
            inc = self._incident
            if inc is None:
                return
            now = self.clock()
            escalate = (inc.acknowledged_by is None and not inc.escalated
                        and now - inc.alerted_at >= self.cfg["escalate_after_s"])
            if escalate:
                inc.escalated = True
            close = not inc.ended and inc.close_at is not None and now >= inc.close_at
            if close:
                inc.ended = True
        if escalate:
            log.warning("No response after %ss: escalating", self.cfg["escalate_after_s"])
            self._send(inc.key, f"‼️ URGENT: no one has responded for {self.cfg['escalate_after_s']} seconds.\n"
                                f"Possible seizure on {self.source} since {_clock_str(inc.started_wall)}.\n"
                                f"Please go to them or call immediately.", inc.snapshot)
        if close:
            msg = f"Event ended, lasted {format_duration(inc.last_end_wall - inc.started_wall)}."
            if inc.events > 1:
                msg += f" ({inc.events} episodes of movement close together.)"
            if inc.acknowledged_by is None:
                msg += "\nNo one has confirmed they are with them yet."
            self._text(msg)

    def _timer_loop(self):
        while not self._stop.wait(0.5):
            try:
                self.tick()
            except Exception:
                log.exception("Alert timer error")

    # ---- acknowledgement ----

    def _telegram_ack(self, key: str, who: str) -> bool:
        return self._acknowledge(key, who)

    def _acknowledge(self, key: str | None, who: str) -> bool:
        """key=None means a local acknowledgement (keyboard at the monitor)."""
        with self._lock:
            inc = self._incident
            if inc is None or (key is not None and key != inc.key):
                return False                      # button from an old alert
            if inc.acknowledged_by is not None:
                return True                       # already acknowledged: no duplicate message
            inc.acknowledged_by = who
            inc.acknowledged_after_s = self.clock() - inc.alerted_at
        if self.cfg["ack_stops_alarm"]:
            self.alarm.stop()
        log.warning("Acknowledged by %s after %.0f s", who, inc.acknowledged_after_s)
        if self.telegram:
            self.telegram.remove_buttons(inc.key)
        after = format_duration(inc.acknowledged_after_s)
        if key is None:
            self._text(f"✅ The alarm was stopped at the monitor after {after}: someone is there.")
        else:
            self._text(f"✅ {who} is with them (responded after {after}).")
        return True

    # ---- status for the dashboard / Android automation ----

    def status(self) -> dict:
        with self._lock:
            inc = self._incident
            return {
                "source": self.source,
                "alarm_active": self.alarm.active,
                "incident_active": bool(inc and not inc.ended),
                "incident_started": _iso(inc.started_wall) if inc else None,
                "escalated": bool(inc and inc.escalated),
                # The Android automation app should place a call when this is true.
                "call": bool(inc and inc.escalated and inc.acknowledged_by is None),
                "acknowledged_by": inc.acknowledged_by if inc else None,
            }

    def close(self):
        self._stop.set()
        self.alarm.stop()
        if self.telegram:
            self.telegram.close()

    # ---- helpers ----

    def _send(self, key, text, photo):
        if self.telegram:
            self.telegram.send_alert(key, text, photo if self.cfg["telegram"]["send_snapshot"] else None)

    def _text(self, text):
        if self.telegram:
            self.telegram.send_text(text)


def _clock_str(wall: float) -> str:
    return datetime.fromtimestamp(wall).strftime("%H:%M:%S, %d %b %Y")


def _iso(wall: float) -> str:
    return datetime.fromtimestamp(wall).isoformat(timespec="seconds")
