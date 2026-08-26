"""H7 -- c* SAMPIYON TABANINA BAGLI MI? Sessiz hata avi.

Her aday taban icin gun ekseni sigma'sini (trafo etkisi cikarilmis, kural 6)
olcer; etiketsiz capayi (2025-04-01..07-31 GERCEK) yeniden hesaplar;
korelasyonu yeniden olcer; c*_formul ve LB'den cozulen c*'i karsilastirir.

Ciktilar reports/h7_cstar/ altina.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIK = KOK / "reports" / "h7_cstar"
CIK.mkdir(parents=True, exist_ok=True)

CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"

TABANLAR = [
    "tuketim_v50_nihai30.csv",
    "tuketim_v55_gunolcek.csv",
    "tuketim_v57_gunolcek175.csv",
    "tuketim_v66_c1335.csv",
    "tuketim_v67_c1335_olay.csv",
    "tuketim_v58_soguk_kalibre.csv",
]


def gun_etkisi(tanim: np.ndarray, gun: np.ndarray, r: np.ndarray) -> pd.Series:
    """Trafo etkisi cikarilmis gun ortalamasi (iki yonlu ayristirma)."""
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def main() -> int:
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")

    sicak_set = set(tr["tanim"])
    te = te.set_index("id").loc[ornek["id"]].reset_index()
    soguk = ~te["tanim"].isin(sicak_set).to_numpy()
    sicak = ~soguk
    log_guc = np.log1p(te["guc"].to_numpy(dtype="float64"))
    tanim_te = te["tanim"].to_numpy()
    tarih_te = te["tarih"].to_numpy()
    p_sicak = float(sicak.mean())
    p_soguk = float(soguk.mean())
    print(f"test {len(te):,} satir   p_sicak={p_sicak:.5f}  p_soguk={p_soguk:.5f}")

    # ---------- 2. ETIKETSIZ CAPA: 2025 Nis-Tem GERCEK gun ekseni ----------
    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    gun_g = g["tarih"].to_numpy()
    tan_g = g["tanim"].to_numpy()
    rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        g["guc"].to_numpy(dtype="float64")
    )
    x = pd.DataFrame({"t": tan_g, "g": gun_g})
    ngun = x["g"].nunique()
    tam_ser = x.groupby("t")["g"].nunique()
    tam = set(tam_ser[tam_ser >= 0.9 * ngun].index)
    sec = np.isin(tan_g, list(tam))
    b_gecen = gun_etkisi(tan_g[sec], gun_g[sec], rg[sec])
    sigma_gercek = float(b_gecen.std())
    # panel duyarliligi: esik %90 yerine %100 ve %50
    tam100 = set(tam_ser[tam_ser >= ngun].index)
    s100 = np.isin(tan_g, list(tam100))
    sg_100 = float(gun_etkisi(tan_g[s100], gun_g[s100], rg[s100]).std())
    tam50 = set(tam_ser[tam_ser >= 0.5 * ngun].index)
    s50 = np.isin(tan_g, list(tam50))
    sg_50 = float(gun_etkisi(tan_g[s50], gun_g[s50], rg[s50]).std())
    sg_hepsi = float(gun_etkisi(tan_g, gun_g, rg).std())
    print(
        f"CAPA {CAPA_BASI}..{CAPA_SONU}: {ngun} gun, {len(tam):,} tam-panel trafosu\n"
        f"  sigma_gercek (%90 esik) = {sigma_gercek:.4f}\n"
        f"  duyarlilik: %100 esik {sg_100:.4f} ({len(tam100):,} trafo) | "
        f"%50 esik {sg_50:.4f} ({len(tam50):,}) | esiksiz {sg_hepsi:.4f}"
    )

    # ---------- 1/3/4. Her taban ----------
    satirlar = []
    b_kayit: dict[str, pd.Series] = {}
    ort_r: dict[str, float] = {}
    tahmin_kayit: dict[str, np.ndarray] = {}
    for ad in TABANLAR:
        yol = KOK / "submissions" / ad
        if not yol.exists():
            print(f"  ATLANDI (yok): {ad}")
            continue
        sub = pd.read_csv(yol, encoding="utf-8")
        if not sub["id"].equals(ornek["id"]):
            raise RuntimeError(f"{ad}: id sirasi bozuk")
        tah = sub["tuketim"].to_numpy(dtype="float64")
        tahmin_kayit[ad] = tah
        r = np.log1p(tah) - log_guc

        b_s = gun_etkisi(tanim_te[sicak], tarih_te[sicak], r[sicak])
        b_c = gun_etkisi(tanim_te[soguk], tarih_te[soguk], r[soguk])
        b_kayit[ad] = b_s

        # SATIR AGIRLIKLI ikinci moment -- parabolun egriligi B icin dogru terim
        etki = pd.Series(tarih_te[sicak]).map(b_s).to_numpy(dtype="float64")
        etki = etki - etki.mean()
        rms_satir = float(np.sqrt((etki**2).mean()))
        B_sicak = p_sicak * rms_satir**2

        etki_c = pd.Series(tarih_te[soguk]).map(b_c).to_numpy(dtype="float64")
        etki_c = etki_c - etki_c.mean()
        rms_satir_c = float(np.sqrt((etki_c**2).mean()))

        # korelasyon: gun-of-year hizasi
        ia = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)
        ib = pd.Series(b_s.values, index=pd.to_datetime(b_s.index).dayofyear)
        ortak = ia.index.intersection(ib.index)
        kor_s = float(np.corrcoef(ia[ortak], ib[ortak])[0, 1])
        ic = pd.Series(b_c.values, index=pd.to_datetime(b_c.index).dayofyear)
        ortak_c = ia.index.intersection(ic.index)
        kor_c = float(np.corrcoef(ia[ortak_c], ic[ortak_c])[0, 1])

        sigma_s = float(b_s.std())
        sigma_c = float(b_c.std())
        c_formul_s = kor_s * sigma_gercek / sigma_s
        c_formul_c = kor_c * sigma_gercek / sigma_c
        ort_r[ad] = float(r.mean())

        satirlar.append(
            {
                "taban": ad,
                "sigma_sicak": sigma_s,
                "rms_satir_sicak": rms_satir,
                "B_sicak": B_sicak,
                "kor_sicak": kor_s,
                "c_formul_sicak": c_formul_s,
                "sigma_soguk": sigma_c,
                "rms_satir_soguk": rms_satir_c,
                "kor_soguk": kor_c,
                "c_formul_soguk": c_formul_c,
                "ort_r": ort_r[ad],
                "ort_log1p": float(np.log1p(tah).mean()),
                "ort_r_sicak": float(r[sicak].mean()),
                "ort_r_soguk": float(r[soguk].mean()),
                "ortak_gun": int(len(ortak)),
            }
        )
        print(
            f"\n{ad}\n"
            f"  SICAK sigma_gun {sigma_s:.4f}  satir-RMS {rms_satir:.4f}  B {B_sicak:.6f}\n"
            f"        kor {kor_s:+.4f} ({len(ortak)} ortak gun)  oran "
            f"{sigma_gercek / sigma_s:.3f}  ->  c_formul {c_formul_s:.3f}\n"
            f"  SOGUK sigma_gun {sigma_c:.4f}  satir-RMS {rms_satir_c:.4f}  "
            f"kor {kor_c:+.4f}  ->  c_formul {c_formul_c:.3f}\n"
            f"  ort r {ort_r[ad]:+.6f}  (sicak {r[sicak].mean():+.6f} / "
            f"soguk {r[soguk].mean():+.6f})"
        )

    df = pd.DataFrame(satirlar)
    df.to_csv(CIK / "tabanlar.csv", index=False, encoding="utf-8")

    # ---------- 5. c uygulamasi seviye sizdiriyor mu? ----------
    print("\n=== SEVIYE KAPISI: c uygulamasi ortalama log1p'yi bozdu mu? ===")
    kap = {}
    for a, b in [
        ("tuketim_v50_nihai30.csv", "tuketim_v55_gunolcek.csv"),
        ("tuketim_v50_nihai30.csv", "tuketim_v66_c1335.csv"),
        ("tuketim_v66_c1335.csv", "tuketim_v67_c1335_olay.csv"),
    ]:
        if a not in ort_r or b not in ort_r:
            continue
        ra = df.loc[df["taban"] == a].iloc[0]
        rb = df.loc[df["taban"] == b].iloc[0]
        d_all = rb["ort_r"] - ra["ort_r"]
        d_s = rb["ort_r_sicak"] - ra["ort_r_sicak"]
        d_c = rb["ort_r_soguk"] - ra["ort_r_soguk"]
        # soguk satirlar birebir ayni mi?
        ta, tb = tahmin_kayit[a], tahmin_kayit[b]
        sap_c = float(np.abs(tb[soguk] - ta[soguk]).max())
        degisen = int((np.abs(tb - ta) > 1e-9).sum())
        kap[f"{a}->{b}"] = {
            "d_ort_r": d_all,
            "d_ort_r_sicak": d_s,
            "d_ort_r_soguk": d_c,
            "soguk_maks_sapma": sap_c,
            "degisen_satir": degisen,
        }
        print(
            f"  {a} -> {b}\n"
            f"    d(ort r) hepsi {d_all:+.3e}  sicak {d_s:+.3e}  soguk {d_c:+.3e}\n"
            f"    soguk maks mutlak sapma {sap_c:.3e}   degisen satir {degisen:,}"
        )

    # ---------- 3. LB'den cozulen c* vs formul ----------
    print("\n=== LB PARABOLU: c* cozumu B'ye ne kadar duyarli? ===")
    v50 = df.loc[df["taban"] == "tuketim_v50_nihai30.csv"].iloc[0]
    v55 = df.loc[df["taban"] == "tuketim_v55_gunolcek.csv"].iloc[0]
    c_v55 = float(v55["sigma_sicak"] / v50["sigma_sicak"])
    mse50 = 1.01686**2
    mse55 = 1.01591**2
    print(f"  v55/v50 gun-sigma orani (uygulanan gercek c) = {c_v55:.4f}")
    print(f"  LB: v50 MSE {mse50:.6f}   v55 MSE {mse55:.6f}   fark {mse55 - mse50:+.6f}")
    coz = {}
    for etiket, B in [
        ("dokuman B=0,021839", 0.021839),
        ("olculen B (satir-RMS, v50)", float(v50["B_sicak"])),
        ("olculen B (std, v50)", p_sicak * float(v50["sigma_sicak"]) ** 2),
    ]:
        # mse55-mse50 = B[(c-c*)^2 - (1-c*)^2] = B(c-1)(c+1-2c*)
        d = mse55 - mse50
        cs = (c_v55 + 1.0 - d / (B * (c_v55 - 1.0))) / 2.0
        A = mse50 - B * (1.0 - cs) ** 2
        # v66/v67 = v50 uzerine 1,335 -> kalan olcek ihtiyaci
        kalan = cs / 1.335
        kazanc = B * ((1.335 - cs) ** 2 - (cs - cs) ** 2)
        coz[etiket] = {"B": B, "c_yildiz": cs, "A": A, "kalan_olcek": kalan, "v67_kaybi": kazanc}
        print(
            f"  {etiket:32s} B={B:.6f}  ->  c*={cs:.4f}   "
            f"v67 kalan olcek {kalan:.4f}   v67'nin optimumdan kaybi {kazanc:+.6f}"
        )

    # v67 kalan aciga gore duzeltici ek olcek dMSE tablosu
    print("\n=== DUZELTICI EK OLCEK: v67 uzerine k uygulanirsa dMSE ===")
    v67 = df.loc[df["taban"] == "tuketim_v67_c1335_olay.csv"]
    if len(v67):
        v67 = v67.iloc[0]
        B67 = float(v67["B_sicak"])
        print(f"  v67 satir-RMS {v67['rms_satir_sicak']:.4f}  B67 {B67:.6f}")
        print(f"  {'k':>6} {'(A) c*=1,335':>14} {'(B) c*=1,492':>14} {'formul c*':>14}")
        cf = float(v67["c_formul_sicak"])  # v67 tabanindan formulun istedigi ek olcek
        for k in [0.90, 0.95, 1.00, 1.05, 1.10, 1.1176, 1.20]:
            dA = B67 * ((k - 1.0) ** 2)
            dB = B67 * ((k - 1.492 / 1.335) ** 2 - (1.0 - 1.492 / 1.335) ** 2)
            dF = B67 * ((k - cf) ** 2 - (1.0 - cf) ** 2)
            print(f"  {k:6.4f} {dA:+14.6f} {dB:+14.6f} {dF:+14.6f}")
        print(f"  (formulun v67 tabanindan istedigi ek olcek c_formul = {cf:.4f})")

    with (CIK / "ozet.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sigma_gercek": sigma_gercek,
                "sigma_gercek_duyarlilik": {
                    "esik100": sg_100,
                    "esik90": sigma_gercek,
                    "esik50": sg_50,
                    "esiksiz": sg_hepsi,
                },
                "p_sicak": p_sicak,
                "p_soguk": p_soguk,
                "tabanlar": satirlar,
                "seviye_kapisi": kap,
                "lb_cozum": coz,
                "c_v55_uygulanan": c_v55,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nyazildi: {CIK / 'tabanlar.csv'} , {CIK / 'ozet.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
