"""See which Groq models your API key can use, and check that the vision model really works.

    python scripts/groq_models.py            # list the models your key can use
    python scripts/groq_models.py --check    # send a tiny test image to the configured vision model

Groq retires models from time to time (Llama 4 Scout was shut down on 2026-07-17). When a
model_not_found error appears, run this, pick a current vision model, and set it in backend/.env:

    GROQ_VLM_MODEL=qwen/qwen3.8-27b
"""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from formsense import groq_util  # noqa: E402
from formsense.vlm.groq_vlm import encode_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="send a test image to the vision model")
    ap.add_argument("--model", help="vision model to check (default: GROQ_VLM_MODEL or the project default)")
    args = ap.parse_args()

    try:
        client = groq_util.make_client()
    except groq_util.GroqError as e:
        sys.exit(str(e))
    model = groq_util.vision_model(args.model)

    ids = groq_util.available_models(client)
    if not ids:
        print("Could not list models (network or key problem).")
    else:
        print(f"{len(ids)} models available to this key:\n")
        for i in ids:
            tags = []
            if i in groq_util.KNOWN_VISION_MODELS:
                tags.append("vision")
            if i in groq_util.RETIRED_MODELS:
                tags.append("RETIRED per Groq docs")
            if i == model:
                tags.append("<- configured vision model")
            print(f"  {i}" + (f"   [{', '.join(tags)}]" if tags else ""))
        if model not in ids:
            print(f"\nWARNING: the configured vision model '{model}' is not in this list.")
    print(f"\nConfigured vision model: {model}")
    print("Vision models per https://console.groq.com/docs/vision: " + ", ".join(groq_util.KNOWN_VISION_MODELS))

    if not args.check:
        print("\nRun with --check to test it with an image.")
        return
    img = np.full((140, 520, 3), 255, np.uint8)
    cv2.putText(img, "FormSense 2026", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    b64, _ = encode_image(img)
    print("\nSending a test image (expected text: FormSense 2026) ...")
    try:
        r = groq_util.chat(
            client, reasoning_effort="none", model=model, temperature=0.1, max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": 'Read the text in the image. Reply as JSON: {"text": "..."}'},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
    except groq_util.GroqError as e:
        sys.exit(f"FAILED: {e}")
    reply = r.choices[0].message.content or ""
    try:
        text = json.loads(reply).get("text", "")
    except ValueError:
        text = reply
    ok = "formsense" in str(text).lower()
    print(f"Model replied: {text!r}")
    print("OK: the vision model works." if ok else "The model answered but did not read the image correctly; try another model.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
