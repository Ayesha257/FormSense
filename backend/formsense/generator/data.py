"""Fake but realistic applicant data (custom Pakistani name lists keep CNIC / phone / names plausible)."""
import datetime as dt, random

MALE = "Ali Ahmed Usman Hamza Bilal Hassan Hussain Talha Zain Saad Omar Faisal Imran Kamran Junaid Waqas Danish Shahzaib Arslan Fahad Haris Rehan Asad Taimoor Umair Zeeshan Adnan Nauman Salman".split()
FEMALE = "Ayesha Fatima Zainab Maryam Hira Sana Iqra Noor Areeba Laiba Mahnoor Amna Hafsa Khadija Rabia Sidra Anum Eman Mehwish Kinza Saba Nimra Sehrish Alina Tooba Rida Komal Bushra Hina Aiza".split()
SURNAMES = "Khan Malik Butt Sheikh Raza Chaudhry Qureshi Siddiqui Mirza Baig Awan Cheema Gill Hashmi Javed Mughal Niazi Rana Saleem Tariq Yousaf Zafar Abbasi Bhatti Dar Farooq Ghauri Iqbal Kayani Lodhi".split()
STREETS = ["Street 5", "Street 12", "Main Boulevard", "Canal Road", "Block C", "Lane 3", "Model Town Link Rd", "Garden Avenue"]
AREAS = ["Gulberg III", "Johar Town", "DHA Phase 5", "Model Town", "Wapda Town", "Iqbal Town", "Bahria Town", "Samanabad", "Township", "Faisal Town"]
CITIES = ["Lahore", "Lahore", "Lahore", "Islamabad", "Faisalabad", "Gujranwala", "Multan", "Sialkot", "Rawalpindi"]
DOMAINS = ["gmail.com", "gmail.com", "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "nic.edu.pk"]
DATE_STYLES = ["%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"]
PROGRAMS = ["program_bscs", "program_bsse", "program_bsai"]


def fmt_date(d, style):
    return d.strftime(style)


def add_years(d, y):
    try:
        return d.replace(year=d.year + y)
    except ValueError:  # 29 Feb
        return d.replace(year=d.year + y, day=28)


def pick_name(r, gender, surname=None):
    pool = MALE if gender == "male" else FEMALE
    given = r.choice(pool)
    if r.random() < 0.3:
        given += " " + r.choice(pool)
    return f"{given} {surname or r.choice(SURNAMES)}"


def make_person(r: random.Random, today: dt.date) -> dict:
    """Return a fully VALID form (every rule passes). Errors are injected afterwards."""
    gender = r.choice(["male", "female"])
    surname = r.choice(SURNAMES)
    name = pick_name(r, gender, surname)
    father = pick_name(r, "male", surname)
    while True:  # dates consistent with every cross-field rule
        age = r.randint(18, 22)
        dob = add_years(today, -age) - dt.timedelta(days=r.randint(0, 360))
        matric = dob.year + 16 + r.randint(0, 1)
        inter = matric + 2
        if inter <= today.year:
            break
    parity = r.choice([1, 3, 5, 7, 9]) if gender == "male" else r.choice([0, 2, 4, 6, 8])
    cnic = f"{r.choice('3333312')}{r.randint(0, 9999):04d}-{r.randint(0, 9999999):07d}-{parity}"
    phone = f"03{r.randint(0, 4)}{r.randint(0, 9)}-{r.randint(0, 9999999):07d}"
    local = f"{name.split()[0]}.{surname}".lower() if r.random() < .6 else f"{name.split()[0]}{surname}".lower()
    email = f"{local}{r.choice(['', str(r.randint(1, 99)), str(dob.year)])}@{r.choice(DOMAINS)}"
    address = f"House {r.randint(1, 450)}, {r.choice(STREETS)}, {r.choice(AREAS)}, {r.choice(CITIES)}"
    tot_m, tot_i = 1200, 1100
    ob_m, ob_i = int(tot_m * r.uniform(.6, .96)), int(tot_i * r.uniform(.6, .94))
    decl = today - dt.timedelta(days=r.randint(1, 30))
    return dict(gender=gender, dob=dob, decl=decl, text=dict(
        full_name=name, father_name=father, cnic=cnic, phone=phone, email=email, address=address,
        matric_year=str(matric), matric_total=str(tot_m), matric_obtained=str(ob_m), matric_percentage=f"{ob_m / tot_m * 100:.2f}",
        inter_year=str(inter), inter_total=str(tot_i), inter_obtained=str(ob_i), inter_percentage=f"{ob_i / tot_i * 100:.2f}",
        declaration_name=name))
