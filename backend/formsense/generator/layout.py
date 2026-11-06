"""Form layout = single source of truth. Writes data/templates/admission_form.json + admission_form_blank.png (valid FormTemplate plus extra render keys).
Run:  python -m formsense.generator.layout      (bboxes are pixels on an A4 page at 150 dpi)"""
import json, pathlib
W, H = 1240, 1754
ROOT = pathlib.Path(__file__).resolve().parents[2]
LX, BX, RH = 80, 380, 56
fields, static = [], []


SEC = ["personal"]


def add(name, label, kind, x, y, w, h=RH, variants=(), required=True, group=None, hint=None, draw_label=True):
    """type is the schema FieldType; kind keeps the finer shape (number, multiline) used by renderer and validator"""
    ftype = {"multiline": "text", "number": "text"}.get(kind, kind)
    fields.append(dict(name=name, label=label, type=ftype, kind=kind, bbox=[x, y, w, h], required=required, group=group,
                       section=SEC[0], variants=list(variants), hint=hint, draw_label=draw_label))


def text(t, x, y, size=22, bold=False, color=40):
    static.append(dict(k="text", t=t, x=x, y=y, size=size, bold=bold, color=color))


# header
static.append(dict(k="border", box=[30, 30, 1180, 1694]))
static.append(dict(k="logo", c=[135, 130], r=58))
for cx, cy in [(60, 60), (1180, 60), (60, 1694), (1180, 1694)]:
    static.append(dict(k="mark", c=[cx, cy]))
static.append(dict(k="photo", box=[1000, 60, 160, 140]))
text("National Institute of Computing", 220, 62, 38, True)
text("Lahore Campus", 220, 112, 24)
text("APPLICATION FOR ADMISSION: BS PROGRAMS 2027", 220, 152, 28, True)
text("Form No. FS-ADM-2027-V1   Use blue or black ink, BLOCK or clear handwriting", 220, 192, 16, color=90)

# section A
static.append(dict(k="bar", t="A.  PERSONAL INFORMATION", y=232))
Y = 290
add("full_name", "Full Name", "text", BX, Y, 780, variants=["Name of Applicant", "Applicant's Name", "Student Name"])
add("father_name", "Father's Name", "text", BX, Y + 70, 780, variants=["Father Name", "Name of Father", "Guardian's Name"])
add("cnic", "CNIC No.", "text", BX, Y + 140, 380, variants=["CNIC Number", "National ID No.", "B-Form / CNIC"], hint="#####-#######-#")
add("date_of_birth", "Date of Birth", "date", BX, Y + 210, 300, variants=["D.O.B", "DOB", "Birth Date"], hint="DD/MM/YYYY")
text("Gender *", LX, Y + 280 + 14, 22)
add("gender_male", "Male", "checkbox", BX, Y + 280 + 6, 44, 44, group="gender", draw_label=True)
add("gender_female", "Female", "checkbox", BX + 220, Y + 280 + 6, 44, 44, group="gender")
add("email", "Email Address", "text", BX, Y + 350, 780, variants=["E-mail", "Email ID", "Email"])
add("phone", "Mobile No.", "text", BX, Y + 420, 380, variants=["Phone Number", "Cell No.", "Contact No."], hint="03XX-XXXXXXX")
add("address", "Postal Address", "multiline", BX, Y + 490, 780, 110, variants=["Home Address", "Address", "Mailing Address"])

# section B
SEC[0] = "program"
static.append(dict(k="bar", t="B.  PROGRAM APPLIED FOR  (tick ONE box only)", y=915))
add("program_bscs", "BS Computer Science", "checkbox", LX, 965, 44, 44, group="program")
add("program_bsse", "BS Software Engineering", "checkbox", 640, 965, 44, 44, group="program")
add("program_bsai", "BS Artificial Intelligence", "checkbox", LX, 1025, 44, 44, group="program")

# section C
SEC[0] = "academic"
static.append(dict(k="bar", t="C.  ACADEMIC RECORD", y=1105))
for t, x, w in [("Examination", LX, 280), ("Year", 380, 140), ("Total Marks", 540, 170), ("Obtained Marks", 730, 170), ("Percentage (%)", 920, 180)]:
    static.append(dict(k="cell", box=[x, 1155, w, 40], t=t))
for key, name, y in [("matric", "Matric / O Level", 1205), ("inter", "Intermediate / A Level", 1275)]:
    text(name, LX, y + 14, 20)
    add(f"{key}_year", f"{name.split(' /')[0]} Year", "number", 380, y, 140, draw_label=False)
    add(f"{key}_total", f"{name.split(' /')[0]} Total Marks", "number", 540, y, 170, draw_label=False)
    add(f"{key}_obtained", f"{name.split(' /')[0]} Obtained Marks", "number", 730, y, 170, draw_label=False)
    add(f"{key}_percentage", f"{name.split(' /')[0]} Percentage", "number", 920, y, 180, draw_label=False)

# section D
SEC[0] = "declaration"
static.append(dict(k="bar", t="D.  DECLARATION", y=1365))
text("I hereby declare that all information given above is true and complete to the best of my", LX, 1417, 20)
text("knowledge. I understand that false information can lead to cancellation of my admission.", LX, 1447, 20)
add("declaration_name", "Declarant Name", "text", BX, 1495, 780, variants=["Name of Declarant", "I, (full name)", "Applicant Name"])
add("declaration_date", "Date", "date", BX, 1575, 270, variants=["Declaration Date", "Dated", "Date Signed"], hint=None)
add("signature", "Signature of Applicant", "signature", 700, 1555, 460, 110, draw_label=False)
text("Signature of Applicant", 810, 1672, 18, color=70)
text("For office use only.   Application No: ____________   Received by: ____________", LX, 1698, 16, color=110)

tpl = dict(name="university_admission_form_v1", image_path="data/templates/admission_form_blank.png", width=W, height=H, dpi=150,
           groups={"gender": dict(label="Gender", required=True, exactly_one=True),
                   "program": dict(label="Program", required=True, exactly_one=True)},
           fields=fields, static=static)

if __name__ == "__main__":
    from formsense.generator.render import blank_form
    p = ROOT / "data" / "templates" / "admission_form.json"
    p.write_text(json.dumps(tpl, indent=1)); print("wrote", p, len(fields), "fields")
    blank_form(tpl)[0].save(ROOT / tpl["image_path"]); print("wrote", tpl["image_path"])
