"""Startup self-check: find setup problems before a demo or a night of monitoring, not during it.

Each check returns OK, WARN (works, but something is degraded) or FAIL (a part
won't work). Printed as a checklist when main.py starts; `python main.py
--check` runs only the checks and exits.
"""

from __future__ import annotations

import os
import shutil
import socket
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_ICON = {OK: "✓", WARN: "!", FAIL: "✗"}


@dataclass
class Check:
    name: str
    status: str
    detail: str


def check_source(src) -> Check:
    import cv2
    w, h = int(src.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kind = "live camera/stream" if src.is_live else "video file"
    if w == 0 or h == 0:
        return Check("Video source", WARN, f"{kind} opened, but resolution unknown")
    return Check("Video source", OK, f"{kind}, {w}x{h}")


def check_model(path: Path) -> Check:
    if path.exists():
        return Check("Pose model", OK, f"{path.name} ({path.stat().st_size / 1e6:.0f} MB)")
    return Check("Pose model", FAIL, f"missing: {path}")


def check_ffmpeg() -> Check:
    if shutil.which("ffmpeg"):
        return Check("ffmpeg (browser-playable clips)", OK, "found")
    return Check("ffmpeg (browser-playable clips)", WARN, "not found: clips save, but browsers can't play them")


def check_bed_area(bed_area, motion_enabled: bool) -> Check:
    if not motion_enabled:
        return Check("Bed area", WARN, "movement detector disabled (motion.enabled: false)")
    if bed_area is None:
        return Check("Bed area", WARN, "not set: movement ANYWHERE in the picture counts. Run tools/set_bed_area.py")
    x0, y0, x1, y1 = bed_area
    share = (x1 - x0) * (y1 - y0)
    if share > 0.6:
        # A bed seen from a sensible distance fills ~30-45% of the picture (hospital videos: 33-44%).
        return Check("Bed area", WARN, f"covers {share:.0%} of the picture: draw a tight box around the bed only "
                                        "(tools/set_bed_area.py), or waving near the camera can trigger it")
    return Check("Bed area", OK, f"{share:.0%} of the picture")


def check_telegram(enabled: bool, timeout_s: float = 5.0) -> Check:
    if not enabled:
        return Check("Telegram alerts", WARN, "disabled in config: only the local alarm will sound")
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return Check("Telegram alerts", WARN, "no TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env: local alarm only")
    try:
        import requests
        base = f"https://api.telegram.org/bot{token}/"
        me = requests.get(base + "getMe", timeout=timeout_s).json()
        if not me.get("ok"):
            return Check("Telegram alerts", FAIL, f"bot token rejected: {me.get('description', 'unknown error')}")
        ch = requests.get(base + "getChat", params={"chat_id": chat}, timeout=timeout_s).json()
        if not ch.get("ok"):
            return Check("Telegram alerts", FAIL, f"bot can't reach chat {chat}: {ch.get('description', '')} "
                                                   "(is the bot added to the group?)")
        title = ch["result"].get("title") or ch["result"].get("first_name") or chat
        return Check("Telegram alerts", OK, f"bot @{me['result'].get('username')} -> '{title}'")
    except Exception as e:
        return Check("Telegram alerts", WARN, f"no connection to Telegram ({type(e).__name__}): alerts will retry, "
                                              "local alarm still works")


def check_alarm_audio(enabled: bool) -> Check:
    if not enabled:
        return Check("Alarm sound", WARN, "disabled in config (alerts.alarm.enabled: false)")
    try:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
        import pygame
        pygame.mixer.init()
        rate = pygame.mixer.get_init()
        pygame.mixer.quit()
        return Check("Alarm sound", OK, f"audio output ready ({rate[0]} Hz); check the volume is up")
    except Exception as e:
        return Check("Alarm sound", FAIL, f"no audio output: {e}")


def check_dashboard(dcfg: dict) -> Check:
    if not dcfg["enabled"]:
        return Check("Dashboard", WARN, "disabled in config")
    host, port = dcfg["host"], dcfg["port"]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host if host != "localhost" else "127.0.0.1", port))
    except OSError:
        return Check("Dashboard", FAIL, f"port {port} is already in use (another monitor running?)")
    if host not in ("127.0.0.1", "localhost"):
        return Check("Dashboard", WARN, f"http://{host}:{port} reachable from the network with NO login")
    return Check("Dashboard", OK, f"http://{host}:{port}")


def check_disk(data_dir: Path, min_gb: float = 2.0) -> Check:
    data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_dir).free / 1e9
    if free < min_gb:
        return Check("Disk space for clips", WARN, f"only {free:.1f} GB free")
    return Check("Disk space for clips", OK, f"{free:.0f} GB free")


def print_checks(checks: list[Check], extra: list[str] = ()):
    width = max(len(c.name) for c in checks)
    print("\n  Seizure Monitor: startup check")
    print("  " + "-" * 60)
    for c in checks:
        print(f"  {_ICON[c.status]} {c.status:<4}  {c.name:<{width}}  {c.detail}")
    for line in extra:
        print(f"    i   {line}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    print("  " + "-" * 60)
    print(f"  {'All checks passed.' if not (fails or warns) else f'{fails} failed, {warns} warning(s).'}\n",
          flush=True)
    return fails
