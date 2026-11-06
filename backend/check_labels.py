import json, glob
from formsense.schema import FieldResult, Issue

bad = 0
for p in glob.glob("data/synthetic/*/labels/*.json"):
    d = json.load(open(p, encoding="utf-8"))
    try:
        [FieldResult(**f) for f in d["fields"]]
        [Issue(**i) for i in d["issues"]]
    except Exception as e:
        bad += 1
        print(p, e)
print("failed:", bad)