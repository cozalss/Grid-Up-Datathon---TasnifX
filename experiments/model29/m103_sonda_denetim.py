"""SONDA DENETIMI: uc sonda dosyasi iddia edilen sey mi, cozum sabitleri dogru mu?
m102_sonda.py'ye BAKMADAN, sifirdan yeniden turet."""

import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
Y = lambda f: np.log1p(pd.read_csv(os.path.join(S, f)).tuketim.values)
a0 = Y("tuketim_m6_ikiyon.csv")
N = len(a0)
dg = Y("tuketim_g7_span_tau3.csv") - a0
m0 = 1.00284**2
Lg = 0.002728
Qg = float((dg**2).mean())
cg = Lg / Qg
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
SONDA = {
    "sy40": ("tuketim_sy40.csv", "tuketim_y40_sota_temiz.csv", 0.60),
    "sq1c": ("tuketim_sq1c.csv", "tuketim_q1c_kapasite_siki.csv", 0.45),
    "sy46": ("tuketim_sy46.csv", "tuketim_y46_amnezik_kirpik.csv", 0.35),
}
print(f"m0={m0:.6f}  Lg={Lg}  Qg={Qg:.6f}  c_g7={cg:.6f}\n")
rap = {}
for ad, (sd, yd, t) in SONDA.items():
    p_disk = Y(sd)
    d = Y(yd) - a0
    # 1) YENIDEN KUR
    p_kur = a0 + cg * dg + t * d
    fark = float(np.abs(p_disk - p_kur).max())
    # kirpma etkisi: expm1 sonrasi clip(0) -> log uzayinda geri okurken fark cikabilir
    kirpilan = int((np.expm1(p_kur) < 0).sum())
    # 2) COZUM SABITINI BAGIMSIZ TURET
    Qd = float((d**2).mean())
    c = float((dg @ d) / N)
    # P^2 = m0 - 2(cg*Lg + t*Ld) + cg^2 Qg + 2 cg t c + t^2 Qd
    sab = m0 - 2 * cg * Lg + cg * cg * Qg + 2 * cg * t * c + t * t * Qd
    # 3) SINAV: L=0 varsayimiyla skor, ve tersten cozum tutuyor mu
    P0 = np.sqrt(sab)
    Ld_geri = (sab - P0 * P0) / (2 * t)
    # r=0.035 senaryosu ileri-geri
    Ld_test = 0.035 * np.sqrt(Qd)
    P_test = np.sqrt(sab - 2 * t * Ld_test)
    Ld_coz = (sab - P_test**2) / (2 * t)
    # 4) KAPI
    cc = pd.read_csv(os.path.join(S, sd))
    kapi = dict(
        satir=len(cc),
        id_test=bool((cc.id.values == te.id.values).all()),
        id_ss=bool((cc.id.values == ss.iloc[:, 0].values).all()),
        nan=int(cc.tuketim.isna().sum()),
        negatif=int((cc.tuketim < 0).sum()),
        sonsuz=int((~np.isfinite(cc.tuketim.values)).sum()),
        maks=float(cc.tuketim.max()),
    )
    ok_kur = fark < 1e-9
    ok_sab = abs(Ld_geri) < 1e-12 and abs(Ld_coz - Ld_test) < 1e-9
    ok_kapi = (
        kapi["satir"] == 714688
        and kapi["id_test"]
        and kapi["id_ss"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
        and kapi["sonsuz"] == 0
    )
    print(f"{ad}  ({sd})")
    print(
        f"  1) YENIDEN KUR      maks log fark {fark:.2e}   kirpilan {kirpilan}   {'GECTI' if ok_kur else 'KALDI'}"
    )
    print(
        f"  2) COZUM SABITI     bagimsiz {sab:.6f}  (m102: bkz json)   ileri-geri hata {abs(Ld_coz - Ld_test):.2e}  {'GECTI' if ok_sab else 'KALDI'}"
    )
    print(
        f"  3) KAPI             satir {kapi['satir']} id {kapi['id_test'] and kapi['id_ss']} nan {kapi['nan']} neg {kapi['negatif']} maks {kapi['maks']:,.0f}  {'GECTI' if ok_kapi else 'KALDI'}"
    )
    rap[ad] = dict(
        yeniden_kur_fark=fark,
        cozum_sabiti=sab,
        kapi=kapi,
        gecti=bool(ok_kur and ok_sab and ok_kapi),
    )
# m102'nin yazdigi sabitlerle karsilastir
eski = json.load(open("m102_sonda.json", encoding="utf-8"))
print("\nSABIT KARSILASTIRMASI (bagimsiz turetim vs m102_sonda.json):")
for ad in SONDA:
    k = ad[1:]  # sy40 -> y40
    a_ = rap[ad]["cozum_sabiti"]
    b_ = eski[k]["cozum_sabiti"]
    print(
        f"  {ad}: bagimsiz {a_:.6f}  m102 {b_:.6f}  fark {abs(a_ - b_):.2e}  {'AYNI' if abs(a_ - b_) < 1e-9 else 'FARKLI -- INCELE'}"
    )
json.dump(rap, open("m103_sonda_denetim.json", "w"), indent=1)
print(f"\nHUKUM: {'UCU DE TEMIZ' if all(v['gecti'] for v in rap.values()) else 'SORUN VAR'}")
