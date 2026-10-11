"""Arm / pause switch shared by the live window (key 'p') and the dashboard button.

A bed monitor should only watch while the person is in bed. Pausing it while
they are up and about (or while a caregiver is with them) removes the most
common source of false alarms: ordinary vigorous movement during the day.
"""

import threading


class ArmSwitch:
    def __init__(self, armed: bool = True):
        self._lock = threading.Lock()
        self._armed = armed

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def set(self, armed: bool):
        with self._lock:
            self._armed = bool(armed)

    def toggle(self) -> bool:
        with self._lock:
            self._armed = not self._armed
            return self._armed
