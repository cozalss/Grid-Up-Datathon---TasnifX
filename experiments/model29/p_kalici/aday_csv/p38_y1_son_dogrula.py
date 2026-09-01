"""y1_son_dogrula: uretilen yon dosyalarinin NIHAI denetimi."""
import json, os
import numpy as np
SCR = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
PK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX/experiments/model29/p_kalici"
taban = np.load(os.path.join(SCR, "taban_log.npy"))
V = np.load(os.path.join(PK, "p34_V30.npy")); BAZ = np.load(os.path.join(PK, "p34_dik_baz.npy"))
N = V.shape[0]; Gi = np.linalg.pinv((V.T @ V) / N, rcond=1e-6)
OUT = {}
for ad in ("ARTIK_RIDGE", "ARTIK_LGBM", "ARTIK_LGBM_MERK"):
    yol = os.path.join(SCR, f"yon_{ad}.npy")
    d = np.load(yol)
    assert d.shape == (714688,) and d.dtype == np.float64
    assert np.isfinite(d).all()
    dd = d - V @ (Gi @ ((V.T @ d) / N))
    for b in BAZ:
        dd = dd - float((dd * b).mean()) * b
    nn = float(np.sqrt((d * d).mean())); nd = float(np.sqrt((dd * dd).mean()))
    rec = dict(dosya=yol, n=714688, rms=nn, dik_norm=nd, dik_pay=nd / nn,
               span_ici_varyans=1 - (nd / nn) ** 2, NaN=0, inf=0)
    for etiket, vec, nrm in (("ham", d, nn), ("dik", dd, nd)):
        u = vec / nrm
        for kap in (0.15, -0.15):
            a = taban + kap * u
            v = np.expm1(a)
            rec[f"{etiket}_kappa{kap:+.2f}"] = dict(
                log_min=float(a.min()), expm1_min=float(v.min()),
                negatif_satir=int((v < 0).sum()),
                negatif_toplam_kwh=float(v[v < 0].sum()) if (v < 0).any() else 0.0,
                NaN=int(np.isnan(v).sum()))
    OUT[ad] = rec
    print(f"{ad}: rms={nn:.5f} dik_norm={nd:.5f} dik_pay={nd/nn:.4f} span_ici={1-(nd/nn)**2:.4f}")
    for k in rec:
        if k.startswith(("ham_kappa", "dik_kappa")):
            r = rec[k]
            print(f"   {k}: log_min={r['log_min']:+.4f} expm1_min={r['expm1_min']:+.5f} "
                  f"neg={r['negatif_satir']} negtoplam={r['negatif_toplam_kwh']:.2f} NaN={r['NaN']}")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "y1_dogrulama.json"), "w",
          encoding="utf-8") as fh:
    json.dump(OUT, fh, ensure_ascii=False, indent=1)
print("-> y1_dogrulama.json")
