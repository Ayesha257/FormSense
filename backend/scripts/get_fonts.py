"""Download free handwriting + printed fonts (Google Fonts, OFL licence) into data/fonts/.
Run once:  python scripts/get_fonts.py   (fonts are git-ignored)"""
import pathlib, urllib.request
BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
FONTS = {
    "hand_caveat": "caveat/Caveat%5Bwght%5D.ttf",
    "hand_indie": "indieflower/IndieFlower-Regular.ttf",
    "hand_reenie": "reeniebeanie/ReenieBeanie.ttf",
    "hand_shadows": "shadowsintolight/ShadowsIntoLight.ttf",
    "hand_patrick": "patrickhand/PatrickHand-Regular.ttf",
    "hand_gochi": "gochihand/GochiHand-Regular.ttf",
    "print_main": "opensans/OpenSans%5Bwdth%2Cwght%5D.ttf",
}
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "fonts"
out.mkdir(parents=True, exist_ok=True)
for name, path in FONTS.items():
    try:
        urllib.request.urlretrieve(BASE + path, out / f"{name}.ttf"); print("ok  ", name)
    except Exception as e:
        print("skip", name, e)
