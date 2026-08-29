"""g3b -- tau=2/3/4 adaylarinin agirliklari + aday CSV uretimi (GONDERILMEZ)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import g2_coz as C  # noqa: E402
from g3_sinav import l1_coz_genel  # noqa: E402

BURA = Path(__file__).resolve().parent
KOK = BURA.parents[1]
S = json.loads((BURA / "g3_sinav.json").read_text(encoding="utf-8"))
CEZA = {float(k.split("=")[1]): v for k, v in S["oniyimserlik"]["f=0.3"].items() for k in [k]}
CEZA = {
    float(k.split("=")[1]): v["oniyimserlik_ort"] for k, v in S["oniyimserlik"]["f=0.3"].items()
}
SD = {float(k.split("=")[1]): v["gercek_sd"] for k, v in S["oniyimserlik"]["f=0.3"].items()}

out = {}
for tau in (2.0, 3.0, 4.0, 5.0):
    w, _ = l1_coz_genel(C.A, C.aa, C.M, tau)
    d = C.rapor_w(w)
    ceza = CEZA.get(tau, np.interp(tau, sorted(CEZA), [CEZA[k] for k in sorted(CEZA)]))
    sd = SD.get(tau, np.interp(tau, sorted(SD), [SD[k] for k in sorted(SD)]))
    bek = d["mse"] + ceza
    d["duzeltilmis_mse"] = float(bek)
    d["duzeltilmis_rmsle"] = float(np.sqrt(max(bek, 0)))
    d["sd_mse"] = float(sd)
    d["ci95_rmsle"] = [float(np.sqrt(max(bek - 2 * sd, 0))), float(np.sqrt(max(bek + 2 * sd, 0)))]
    out["tau=%g" % tau] = d
    print("")
    print(
        "=== tau=%g  ongoru MSE=%.6f (RMSLE %.5f) | duzeltilmis MSE=%.6f (RMSLE %.5f) "
        "| %%95 CI RMSLE [%.5f, %.5f] | |w|_1=%.2f"
        % (
            tau,
            d["mse"],
            d["rmsle"],
            bek,
            d["duzeltilmis_rmsle"],
            d["ci95_rmsle"][0],
            d["ci95_rmsle"][1],
            d["l1"],
        )
    )
    for k_, v in d["w"].items():
        print("    %-22s %+9.5f" % (k_, v))
    if abs(tau - 3.0) < 1e-9:
        w3 = w.copy()

# ---- aday CSV (SADECE DISKE YAZILIR, KAGGLE'A GONDERILMEZ) ----
ids = pd.read_csv(KOK / "submissions" / C.DOSYALAR[0], usecols=["id"])["id"]
p = w3 @ C.X  # log1p uzayinda afin harman
tuketim = np.expm1(np.clip(p, 0.0, None))
csv = BURA / "g3_aday_tau3.csv"
pd.DataFrame({"id": ids, "tuketim": tuketim}).to_csv(csv, index=False)
kirp = int((p < 0).sum())
print("")
print("aday CSV yazildi: %s  (log1p<0 kirpilan satir: %d)" % (csv, kirp))
# kirpmanin etkisi
p2 = np.clip(p, 0.0, None)
mse_kirpsiz = C.F(w3)
fark = float(np.mean(p2**2 - p**2) - 2 * 0)  # sadece bilgi
print(
    "kirpma sonrasi ||dp||^2/N = %.3e (ihmal edilebilir mi kontrol)" % float(np.mean((p2 - p) ** 2))
)
out["kirpma"] = {"satir": kirp, "dnorm2": float(np.mean((p2 - p) ** 2))}

(BURA / "g3_aday.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("yazildi: %s" % (BURA / "g3_aday.json"))
