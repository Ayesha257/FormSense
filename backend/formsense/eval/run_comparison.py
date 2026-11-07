import cv2
from formsense.vision.template import load_template
from formsense.vision.pipeline import analyze_page
from formsense.vision.ocr import FieldOCR
import json, time
from pathlib import Path
from formsense.eval.harness import evaluate_dataset, full_pipeline_system, format_results
from formsense.eval.baseline_adapter import BaselineSystem
from formsense.schema import FormTemplate, FieldResult, FieldStatus

LABELS = "data/synthetic/test/labels"
IMAGES = Path("data/synthetic/test/images")
TPL = "data/templates/admission_form.json"
LIMIT = None
OUT = Path("data/eval_results"); OUT.mkdir(parents=True, exist_ok=True)


def find_image(form_id):
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES / f"{form_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {form_id}")


tpl = FormTemplate.model_validate_json(Path(TPL).read_text(encoding="utf-8"))
baseline = BaselineSystem(tpl)
rules = full_pipeline_system(TPL)
failed = []


def baseline_input(form):
    t = time.time()
    try:
        fields = baseline.extract(find_image(form.form_id))
    except Exception as e:  # ek kharab form poori raat ka run na roke
        failed.append((form.form_id, repr(e)))
        fields = [FieldResult(name=f.name, bbox=(0, 0, 0, 0), status=FieldStatus.MISSING) for f in form.fields]
    (OUT / f"baseline_{form.form_id}.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in fields], indent=2), encoding="utf-8")
    print(f"{form.form_id} done in {time.time() - t:.0f}s", flush=True)
    return fields

vtpl, vblank = load_template()
field_ocr = FieldOCR()


def vision_input(form):
    t = time.time()
    try:
        page = cv2.imread(str(find_image(form.form_id)))
        fields = analyze_page(page, vtpl, vblank, ocr=field_ocr).fields
    except Exception as e:
        failed.append((form.form_id, "vision: " + repr(e)))
        fields = [FieldResult(name=f.name, bbox=(0, 0, 0, 0), status=FieldStatus.MISSING) for f in form.fields]
    (OUT / f"full_{form.form_id}.json").write_text(
        json.dumps([f.model_dump(mode="json") for f in fields], indent=2), encoding="utf-8")
    print(f"{form.form_id} full done in {time.time() - t:.0f}s", flush=True)
    return fields

results = {
    "baseline": evaluate_dataset(LABELS, rules, limit=LIMIT, input_fn=baseline_input),
    "full (images)": evaluate_dataset(LABELS, rules, limit=LIMIT, input_fn=vision_input),
    "full (GT text)": evaluate_dataset(LABELS, rules, limit=LIMIT),
}
table = format_results(results)
print(table)
(OUT / "results_table.txt").write_text(table, encoding="utf-8")
(OUT / "results.json").write_text(json.dumps({k: v.as_dict() for k, v in results.items()}, indent=2), encoding="utf-8")
(OUT / "failed_forms.txt").write_text("\n".join(f"{a}: {b}" for a, b in failed), encoding="utf-8")