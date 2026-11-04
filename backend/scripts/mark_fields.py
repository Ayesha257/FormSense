"""Draw each field's box on the blank template, one field at a time.

    python scripts/mark_fields.py                  # every field without a box
    python scripts/mark_fields.py --all            # redo every field
    python scripts/mark_fields.py --only cnic      # redo one field

In the window: drag a box around where the ANSWER is written (not the label),
then press ENTER or SPACE. Press C to skip a field. Progress is saved after
every field, so you can stop and continue later.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT = BACKEND / "data" / "templates" / "admission_form.json"
MAX_H = 900  # window height in pixels; lower it if the window is bigger than your screen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(DEFAULT))
    ap.add_argument("--all", action="store_true", help="redo every field")
    ap.add_argument("--only", help="redo one field by name")
    args = ap.parse_args()

    path = Path(args.template)
    data = json.loads(path.read_text(encoding="utf-8"))
    img_path = BACKEND / data["image_path"]
    img = cv2.imread(str(img_path))
    if img is None:
        sys.exit(f"Blank form image not found: {img_path}\n"
                 "Export the blank form as PNG (about 150-200 dpi) and save it there.")
    data["height"], data["width"] = img.shape[:2]

    scale = min(1.0, MAX_H / img.shape[0])
    shown = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def todo(f):
        if args.only:
            return f["name"] == args.only
        return args.all or f["bbox"][2] == 0 or f["bbox"][3] == 0

    fields = [f for f in data["fields"] if todo(f)]
    if not fields:
        print("Every field already has a box. Use --all or --only <name> to redo.")
        return

    win = "FormSense: mark fields"
    for i, f in enumerate(fields, 1):
        canvas = shown.copy()
        for other in data["fields"]:  # show boxes done so far
            x, y, w, h = (int(v * scale) for v in other["bbox"])
            if w and h:
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 170, 60), 1)
        msg = f"[{i}/{len(fields)}] {f['name']}  ({f['label']}, {f['type']})  ENTER=save  C=skip"
        cv2.putText(canvas, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(msg)
        x, y, w, h = cv2.selectROI(win, canvas, showCrosshair=False, fromCenter=False)
        if w == 0 or h == 0:
            print("  skipped")
            continue
        f["bbox"] = [round(x / scale), round(y / scale), round(w / scale), round(h / scale)]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  saved {f['bbox']}")

    cv2.destroyAllWindows()
    left = [f["name"] for f in data["fields"] if f["bbox"][2] == 0]
    print("Done." if not left else f"Still without a box: {', '.join(left)}")


if __name__ == "__main__":
    main()
