"""Start the FormSense web app:  python -m api  (from the backend folder)."""
import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description="FormSense web app (API + frontend)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.getenv("FORMSENSE_PORT", "8000")))
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser")
    ap.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    args = ap.parse_args()

    import uvicorn
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"\n  FormSense is starting on {url}\n  (the OCR models load in the background; the first check can take a minute)\n")
    if not args.no_browser:
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload, log_level="info")


if __name__ == "__main__":
    main()
