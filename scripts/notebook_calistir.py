"""NOTEBOOK'U BASTAN SONA CALISTIR -- ajanin iddiasini kendim dogrula.

nbconvert yok; kod hucrelerini sirayla TEK bir ad uzayinda exec ediyoruz.
Bu, "bastan sona calisiyor mu" sorusunun gercek testidir.

Her hucrenin suresi ve ciktisinin ilk satirlari raporlanir; hata alan
hucre numarasi ve tam izi basilir.
"""

import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

KOK = Path(r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX")
NB = KOK / "notebooks/TasnifX_final.ipynb"

with NB.open(encoding="utf-8") as fh:
    nb = json.load(fh)
kod = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
print(f"{len(kod)} kod hucresi\n")

os.chdir(KOK)
ns = {"__name__": "__main__", "__file__": str(NB)}
t0 = time.time()
hata = None
for sira, (i, src) in enumerate(kod, 1):
    tut = io.StringIO()
    t1 = time.time()
    try:
        with redirect_stdout(tut):
            exec(compile(src, f"<hucre {i}>", "exec"), ns)
    except Exception:
        hata = (sira, i, traceback.format_exc(), tut.getvalue())
        break
    sure = time.time() - t1
    cikti = tut.getvalue().strip().splitlines()
    ilk = cikti[0][:76] if cikti else ""
    print(f"  [{sira:2d}/{len(kod)}] hucre {i:2d}  {sure:6.2f}s  {len(cikti):3d} satir  {ilk}")

print(f"\ntoplam {time.time() - t0:.1f} saniye")
if hata:
    sira, i, iz, cikti = hata
    print(f"\nHATA -- kod hucresi {sira} (notebook hucre {i})")
    print(iz)
    if cikti.strip():
        print("hucrenin hatadan onceki ciktisi:")
        print(cikti[-2000:])
    sys.exit(1)
print("\nBASTAN SONA CALISTI, HATA YOK")
