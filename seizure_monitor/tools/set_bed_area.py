"""Mark the bed area once per camera position.

    python tools/set_bed_area.py --source 0                  # webcam
    python tools/set_bed_area.py --source http://PHONE:8080/video
    python tools/set_bed_area.py --source video.mp4 --at 5   # frame 5 s into a video

A frame is shown: drag a box around the bed (include where arms and legs can
reach), press ENTER or SPACE to save, or C to cancel. The box is written to
motion.bed_area in config.yaml as fractions of the frame, so it stays correct
at any resolution. Re-run it whenever the camera is moved.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from monitor.capture import VideoSource  # noqa: E402


def grab_frame(source: str, width: int, at_s: float):
    src = VideoSource(source, width, 30)
    try:
        frame, start = None, None
        while (item := src.read()) is not None:
            frame, t = item
            start = t if start is None else start
            # Live cameras need a moment to settle exposure; files: skip to the requested time.
            if t - start >= (1.0 if src.is_live else at_s):
                break
        return frame
    finally:
        src.release()


def write_bed_area(config_path: Path, area: list):
    """Replace only the value on the bed_area line so all comments in config.yaml survive."""
    text = config_path.read_text()
    value = "[" + ", ".join(f"{v:.3f}" for v in area) + "]"
    new, n = re.subn(r"^(\s*bed_area:\s*)([^#\n]*?)(\s*(#.*)?)$", lambda m: m.group(1) + value + m.group(3),
                     text, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("could not find 'bed_area:' in config.yaml")
    config_path.write_text(new)
    assert yaml.safe_load(new)["motion"]["bed_area"] == [round(v, 3) for v in area]


def main():
    ap = argparse.ArgumentParser(description="Mark the bed area for the movement detector")
    ap.add_argument("--source", default="0", help="webcam index, video file, or RTSP/HTTP URL")
    ap.add_argument("--at", type=float, default=2.0, help="for video files: take the frame at this second")
    ap.add_argument("--config", type=Path, default=BASE_DIR / "config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    frame = grab_frame(args.source, cfg["capture"]["width"], args.at)
    if frame is None:
        sys.exit("Could not read a frame from the source.")

    h, w = frame.shape[:2]
    title = "Drag a box around the bed, then ENTER to save (C = cancel)"
    x, y, bw, bh = cv2.selectROI(title, frame, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    time.sleep(0.1)
    if bw < 10 or bh < 10:
        sys.exit("No box selected; config.yaml unchanged.")

    area = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    write_bed_area(args.config, area)
    print(f"Saved motion.bed_area = [{', '.join(f'{v:.3f}' for v in area)}] to {args.config}")


if __name__ == "__main__":
    main()
