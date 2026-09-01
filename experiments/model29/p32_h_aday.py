"""p32-H: ADAY CSV URETIMI + DENETIM.

KATMANLAR
---------
K3e KUYRUK KAPAGI (dort kapiyi da gecen TEK katman):
    SICAK satirlarda log1p(tahmin) trafonun EGITIM GECMISI log1p ortalamasi
    mu_i +- k*sd_i bandina kirpilir. sd_i yoksa/0 ise dokunulmaz.
    Gecerli plato k in [8, 20]; k=8 agresif uc, k=12/20 muhafazakar uc.
K2 lgbm huber a=2.0 (varsa): DELTA yolu -- ayni p18 kosusundan uretilen
    taban ve huber lgbm test tahminlerinin farki, sicak harman payi
    1.0/6.4 ile olceklenip log uzayinda tabana bindirilir.
    (Birebir onbellek eslesmesi ARANMAZ: iki kol da AYNI kosudan geldigi
     icin delta ic tutarli; docs/80 §8'deki 0.325'lik onbellek sapmasi
     delta'da BIRINCI DERECEDEN iptal olur.)
K1 olu trafo: CURUDU (bkz. p32_a_olu / docs/52) -- URETILMEZ.
SOGUK winsorization: kis26'da G_3.0 disinda hepsi negatif, G_3.0 +0.001 --
    ihmal edilebilir, URETILMEZ.

DENETIM (her dosya): 714688 satir, id sirasi data/raw/test.csv ile birebir,
NaN/negatif/sonsuz yok, tabana gore degisen satir sayisi ve log kayma ozeti.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from p32_ortak import AC, KESIM, KOK, PK, _ham

TABAN = os.path.join(KOK, "submissions", "tuketim_YP_seviye.csv")
SP_L2 = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/p32/l2"
)
LGBM_PAY = 1.0 / 6.4  # uretim sicak harmani (3+1+1+1.4)


def capa_test(tanimlar):
    tr = _ham()
    g = tr[tr["tarih"] < pd.Timestamp(KESIM["TEST"])]
    lg = np.log1p(np.clip(g["tuketim"].to_numpy("float64"), 0, None))
    gb = pd.DataFrame({"tanim": g["tanim"].to_numpy(), "l": lg}).groupby("tanim")["l"]
    idx = pd.Index(tanimlar)
    return gb.mean().reindex(idx).to_numpy("float64"), gb.std().reindex(idx).to_numpy("float64")


def denetle(ad, x, taban, ids, IDS, soguk):
    L0, L1 = np.log1p(taban), np.log1p(x)
    f = L1 - L0
    d = np.abs(f) > 1e-12
    return {
        "n_satir": int(len(x)),
        "satir_dogru": bool(len(x) == 714688),
        "id_sirasi_birebir": bool(np.array_equal(ids, IDS)),
        "NaN": int(np.isnan(x).sum()),
        "negatif": int((x < 0).sum()),
        "sonsuz": int((~np.isfinite(x)).sum()),
        "degisen_satir": int(d.sum()),
        "degisen_pay": round(float(d.mean()), 5),
        "soguk_satirda_degisen": int((d & soguk).sum()),
        "log_kayma_ort": round(float(f[d].mean()), 5) if d.any() else 0.0,
        "log_kayma_min": round(float(f.min()), 4),
        "log_kayma_max": round(float(f.max()), 4),
        "toplam_tuketim_orani": round(float(x.sum() / taban.sum()), 6),
    }


def main() -> None:
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"),
                     usecols=["id", "tanim"], dtype={"id": str, "tanim": str})
    IDS = te["id"].to_numpy()
    tanim = te["tanim"].to_numpy()
    soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
    sub = pd.read_csv(TABAN, dtype={"id": str})
    assert np.array_equal(sub["id"].to_numpy(), IDS), "taban id sirasi"
    x0 = sub["tuketim"].to_numpy("float64")
    L0 = np.log1p(x0)
    mu, sd = capa_test(tanim)
    ok = (~soguk) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)

    R: dict = {
        "00_taban": {"dosya": "submissions/tuketim_YP_seviye.csv", "olculmus_LB": 1.00115},
        "01_capa_kapsami": {
            "sicak_satir": int((~soguk).sum()),
            "capasi_olan_sicak": int(ok.sum()),
            "capasiz_sicak": int(((~soguk) & ~ok).sum()),
        },
        "adaylar": {},
    }

    # --- L2 delta (varsa)
    d_l2 = None
    try:
        zt = np.load(os.path.join(SP_L2, "taban", "sicak_test.npz"))
        zh = np.load(os.path.join(SP_L2, "huber20", "sicak_test.npz"))
        ka = [k for k in zt.files if k.endswith("_lgbm")]
        kb = [k for k in zh.files if k.endswith("_lgbm")]
        if len(ka) == 3 and len(kb) == 3:
            pt = np.mean([zt[k].astype("float64") for k in ka], axis=0)
            ph = np.mean([zh[k].astype("float64") for k in kb], axis=0)
            d_sicak = LGBM_PAY * (ph - pt)
            d_l2 = np.zeros(len(x0))
            d_l2[~soguk] = d_sicak
            R["02_L2"] = {
                "durum": "HAZIR",
                "tohumlar": sorted(ka),
                "ham_lgbm_fark_maxabs": round(float(np.abs(ph - pt).max()), 5),
                "ham_lgbm_fark_rms": round(float(np.sqrt(((ph - pt) ** 2).mean())), 5),
                "harman_payi": LGBM_PAY,
                "bindirilecek_delta_rms": round(float(np.sqrt((d_sicak**2).mean())), 6),
                "CEKINCE": (
                    "p15 olcumu: harman kazanci +0.00111 dMSE -> tasima 0.5'te "
                    "LB +0.00056. KUCUK. Ayrica p18 sicak birebir dogrulamasi "
                    "TUTMUYOR (docs/80 §8); burada DELTA yolu kullanildi, iki kol "
                    "ayni kosudan geldigi icin karsilastirma ic tutarli."
                ),
            }
        else:
            R["02_L2"] = {"durum": f"EKSIK: taban {len(ka)}/3, huber {len(kb)}/3"}
    except FileNotFoundError as e:
        R["02_L2"] = {"durum": f"YOK ({e.filename})"}

    uret: dict[str, np.ndarray] = {}
    for k in (8.0, 12.0, 20.0):
        for yon in ("cift", "ust"):
            L = L0.copy()
            ust = mu[ok] + k * sd[ok]
            L[ok] = (np.clip(L0[ok], mu[ok] - k * sd[ok], ust) if yon == "cift"
                     else np.minimum(L0[ok], ust))
            uret[f"p32_kuyruk_{yon}_k{int(k)}"] = np.clip(np.expm1(np.maximum(L, 0.0)), 0, None)
    if d_l2 is not None:
        uret["p32_l2_lgbmhuber"] = np.clip(np.expm1(np.maximum(L0 + d_l2, 0.0)), 0, None)
        # SIRA: once K2 kaydirmasi, SONRA K3e kirpmasi. Tersi olsa kirpilan
        # satirlar K2 deltasiyla bandin DISINA tasardi (kirpma etkisiz kalirdi).
        for k in (8.0, 12.0):
            Lb = L0 + d_l2
            L = Lb.copy()
            L[ok] = np.clip(Lb[ok], mu[ok] - k * sd[ok], mu[ok] + k * sd[ok])
            uret[f"p32_kuyruk_cift_k{int(k)}_l2"] = np.clip(
                np.expm1(np.maximum(L, 0.0)), 0, None
            )

    # --- CAKISMA kontrolu
    m_k8 = np.abs(np.log1p(uret["p32_kuyruk_cift_k8"]) - L0) > 1e-12
    R["03_cakisma"] = {
        "kuyruk_k8_degisen": int(m_k8.sum()),
        "kuyruk_k8_soguk_kesisim": int((m_k8 & soguk).sum()),
        "L2_maskesi": "TUM sicak satirlar" if d_l2 is not None else "yok",
        "kesisim_kuyruk_L2": int(m_k8.sum()) if d_l2 is not None else 0,
        "not": (
            "K3e ve K2 KESISIYOR (ikisi de sicak). K3e bir KIRPMA (ust sinir), "
            "K2 kucuk bir kaydirma. Birlesik dosyada K2 K3e'den SONRA bindirildi; "
            "bu, kirpilan satirlarin bandin disina TASMASINA izin verir. "
            "Etkilenen satir sayisi kuyruk maskesi kadar ve delta rms'i kucuk."
        ),
    }
    os.makedirs(AC, exist_ok=True)
    for ad, x in uret.items():
        yol = os.path.join(AC, f"{ad}.csv")
        pd.DataFrame({"id": sub["id"], "tuketim": x}).to_csv(yol, index=False)
        R["adaylar"][ad] = {"yol": f"p_kalici/aday_csv/{ad}.csv",
                            **denetle(ad, x, x0, sub["id"].to_numpy(), IDS, soguk)}

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K9_adaylar"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
