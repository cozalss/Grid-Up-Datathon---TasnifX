"""y1_meta2: her blok icin trafo kume kodu + ay + hg + gun kaydet (X/meta ile AYNI SIRA)."""
import os, sys
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
import p27_ortak as PO

GEC = os.path.dirname(os.path.abspath(__file__))
AM = os.path.join(os.path.dirname(GEC), "am")

for bad in ("yaz25", "guz25", "kis26"):
    d = PO.blok(bad)
    m = np.load(os.path.join(AM, f"meta_{bad}.npz"))
    assert len(d) == len(m["r"]), (len(d), len(m["r"]))
    # sira dogrulamasi: r birebir esit olmali
    r_ref = m["r"]
    r_now = (d.y - d.p).to_numpy("float64")
    assert np.max(np.abs(r_ref - r_now)) < 1e-12, np.max(np.abs(r_ref - r_now))
    kod = pd.factorize(d.tanim.astype("string"))[0].astype("int32")
    tr = pd.to_datetime(d.tarih)
    np.savez(os.path.join(GEC, f"kume_{bad}.npz"),
             kume=kod, ay=tr.dt.month.to_numpy("int16"),
             hg=tr.dt.dayofweek.to_numpy("int16"),
             gun=(tr - tr.min()).dt.days.to_numpy("int32"))
    print(f"{bad}: n={len(d)} trafo={kod.max()+1} aylar={sorted(set(tr.dt.month))} "
          f"gun 0..{(tr-tr.min()).dt.days.max()}")
print("TAMAM")
