"""Send one fake incident through the real alert system (alarm + Telegram + escalation).

Lets you check your Telegram bot setup and the "I'm with them" button without
acting out a seizure in front of the camera. The messages say TEST clearly.

    python tools/test_alert.py              # with the local alarm
    python tools/test_alert.py --no-alarm   # Telegram only
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from main import build_alerts  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-alarm", action="store_true")
    ap.add_argument("--event-s", type=float, default=20, help="how long the fake event lasts")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv(BASE_DIR / ".env")
    cfg = yaml.safe_load((BASE_DIR / "config.yaml").read_text())
    if args.no_alarm:
        cfg["alerts"]["alarm"]["enabled"] = False
    alerts = build_alerts(cfg, f"TEST - {cfg['monitor']['source_name']}")

    img = np.full((240, 320, 3), 40, np.uint8)
    cv2.putText(img, "TEST ALERT", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    snapshot = cv2.imencode(".jpg", img)[1].tobytes()

    print(f"\nSending test alert. Press 'I'm with them' in Telegram to acknowledge, "
          f"or wait {cfg['alerts']['escalate_after_s']} s to see escalation. Ctrl+C to stop.\n")
    start = time.time()
    alerts.on_event_start(start, snapshot)
    ended = False
    try:
        while True:
            time.sleep(1)
            if not ended and time.time() - start >= args.event_s:
                alerts.on_event_end(time.time())
                ended = True
                print(f"Fake event ended; 'Event ended' message follows after cooldown "
                      f"({cfg['alerts']['cooldown_s']} s) if no new event.")
            st = alerts.status()
            print(f"  alarm={st['alarm_active']} escalated={st['escalated']} call={st['call']} "
                  f"acknowledged_by={st['acknowledged_by']}", end="\r")
            if ended and not st["incident_active"] and (st["acknowledged_by"] or st["escalated"]):
                time.sleep(2)   # let the last message go out
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        alerts.close()


if __name__ == "__main__":
    main()
