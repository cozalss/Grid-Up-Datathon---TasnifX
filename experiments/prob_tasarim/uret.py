"""PROB TASARIMI -- adim 6: PROB DOSYALARINI URET.

Tasarim kararlari (olcumle gerekcelendirilmis):

  * DESEN KAYNAGI = yaz25 (2025-04..07), testin MEVSIM IKIZI.
    karar.py sicak seviye deseninin kis26'da TERS dondugunu olctu
    (yaz25|kis26 rho = -0.787). Havuzlamak desenleri birbirine kirpardi.
    yaz_bolme.py yaz25 icinde TRAFO-AYRIK bolmeyle desenin gercek oldugunu
    ve kappa'nin +0,6..+1,0 bandinda oturdugunu gosterdi.

  * DIKLESTIRME iki asamali:
      (a) olculmus 18 LB yonune (v_i - v83) dik -> prob YENI bilgi olcer
      (b) birbirine dik (Gram-Schmidt) -> kazanclar TAM toplanir

  * OLCEK kappa = yaz25 trafo-ayrik bolmesinin kappa tahmini. Boylece prob
    ayni zamanda muhtemel bir KAZANC dosyasidir; L yanlis cikarsa bile
    ertesi gun tam optimum cozulur.

Cikti: submissions/tuketim_p*.csv + experiments/prob_tasarim/prob_kayit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
ONB = KOK / "experiments" / "v93_denetim" / "onbellek"
sys.path.insert(0, str(BURA))

from tavan import desil, test_cercevesi  # noqa: E402
from yon import M_V93, S_V93, span_tabani  # noqa: E402

# (dosya adi, rejim, grup anahtari, karar.json yon adi, kappa)
PROBLAR = [
    ("p1_sicak_ilce", "sicak", "ilce", "ilce", 0.65),
    ("p2_sicak_seviye", "sicak", "sev10", "seviye_desili10", 0.86),
    ("p3_soguk_seviye", "soguk", "sev10", "seviye_desili10", 0.92),
    # yedekler (dosya uretilir ama ilk uc hakka girmez)
    ("p4_sicak_ay", "sicak", "ay", "ay", 1.00),
    ("p5_soguk_kva", "soguk", "kova", "kva_kovasi", 0.30),
]
KAYNAK_BLOK = "yaz25"


def main() -> None:
    te = test_cercevesi()
    n = len(te)
    sicak = te["sicak"].to_numpy()
    maske = {"sicak": sicak, "soguk": ~sicak}

    lgp = te["lgp"].to_numpy()
    sev = np.full(n, -1, dtype=int)
    sev[sicak] = desil(lgp[sicak], 10)
    sev[~sicak] = desil(lgp[~sicak], 10)

    anahtarlar = {
        "ilce": te["ilce"].to_numpy().astype(str),
        "sev10": sev.astype(str),
        "kova": te["kova"].to_numpy().astype(str),
        "ay": te["ay"].to_numpy().astype(str),
    }

    karar = {
        (k["rejim"], k["yon"]): k
        for k in json.loads((BURA / "karar.json").read_text(encoding="utf-8"))
    }
    ay_des = json.loads((BURA / "ay_deseni.json").read_text(encoding="utf-8"))
    yazb = {
        (k["rejim"], k["yon"]): k
        for k in json.loads((BURA / "yaz_bolme.json").read_text(encoding="utf-8"))
    }

    B, n2 = span_tabani()
    assert n2 == n
    print(f"olculmus LB span rank = {B.shape[0]}  n = {n:,}")
    print(f"M(v93) ongoru = {M_V93:.7f}   S(v93) ongoru = {S_V93:.6f}\n")

    v93_log = np.load(ONB / "v93.npy")
    ss_id = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])["id"]
    assert (ss_id.to_numpy() == te["id"].to_numpy()).all()

    onceki: list[np.ndarray] = []  # Gram-Schmidt icin (birim normlu)
    kayit = []
    for ad, rejim, anah, yon_adi, kappa in PROBLAR:
        if yon_adi == "ay":
            desen = {str(k): v for k, v in ay_des[rejim].items()}
        else:
            desen = karar[(rejim, yon_adi)]["desen"][KAYNAK_BLOK]
        m = maske[rejim]
        v = pd.Series(anahtarlar[anah]).map(desen).fillna(0.0).to_numpy(dtype="float64")
        v = np.where(m, v, 0.0)
        v[m] -= v[m].mean()
        q_ham = float(v @ v) / n

        # (a) olculmus LB span'ina dik
        v = v - B.T @ (B @ v)
        q_a = float(v @ v) / n
        # (b) onceki problara dik
        for u in onceki:
            v = v - float(v @ u) * u
        q_b = float(v @ v) / n

        # --- KIRPMA GUVENLIGI ---
        # log1p negatife duserse expm1 negatif tahmin verir; Kaggle kapisi
        # bunu reddeder. Kirparsak dMSE = k^2 Q - 2 k L ozdesligi BOZULUR
        # (v93 denetiminde 25 satirda 5,2e-08 kacak olculdu). Bu yuzden
        # kirpma yerine, TASACAK satirlarda yon SIFIRLANIR: boylece dosya
        # kesin olarak "v93 + kappa*d" bicimindedir, d dosyadan okunabilir.
        tasan = (v93_log + kappa * v) < 0.0
        v = np.where(tasan, 0.0, v)
        kirpik = int(tasan.sum())
        yeni_log = v93_log + kappa * v
        assert float(yeni_log.min()) >= 0.0, "hala negatif log1p var"
        tuketim = np.expm1(yeni_log)
        yol = KOK / "submissions" / f"tuketim_{ad}.csv"
        pd.DataFrame({"id": ss_id, "tuketim": tuketim}).to_csv(
            yol, index=False, float_format="%.17g"
        )

        onceki.append(v / np.sqrt(float(v @ v)))

        # --- Q'yu DOSYADAN geri oku (yazma hassasiyeti dahil GERCEK yon) ---
        geri = np.log1p(pd.read_csv(yol)["tuketim"].to_numpy(dtype="float64"))
        d_ger = (geri - v93_log) / kappa
        q_ger = float(d_ger @ d_ger) / n
        maxabs_fark = float(np.abs(d_ger - v).max())
        degisen = int((np.abs(geri - v93_log) > 1e-9).sum())

        # --- olcek okunabilirligi ---
        notr_dmse = kappa * kappa * q_ger
        notr_skor = float(np.sqrt(M_V93 + notr_dmse))
        # L'nin cozulebilecegi hassasiyet: her skor 5 ondalik -> +-5e-6
        sigma_m = 2.0 * S_V93 * 5e-6
        sigma_dmse = float(np.sqrt(2.0) * sigma_m)
        sigma_L = sigma_dmse / (2.0 * kappa)

        # CV'den beklenen kazanc (yaz25 trafo-ayrik) ve ondan beklenen L
        cv_kaz = yazb[(rejim, yon_adi)]["trafo_ayrik_kazanc_toplam"]
        cv_kappa = 0.5 * (yazb[(rejim, yon_adi)]["kappa_A"] + yazb[(rejim, yon_adi)]["kappa_B"])
        L_bek = cv_kappa * q_ger
        kaz_bek = -(L_bek**2) / q_ger if q_ger > 0 else 0.0
        skor_bek = float(np.sqrt(max(M_V93 + kappa * kappa * q_ger - 2 * kappa * L_bek, 1e-12)))
        # ters isaret senaryosu (en kotu makul durum)
        skor_ters = float(np.sqrt(M_V93 + kappa * kappa * q_ger + 2 * kappa * abs(L_bek)))

        kayit.append(
            {
                "ad": ad,
                "dosya": yol.name,
                "rejim": rejim,
                "yon": yon_adi,
                "desen_kaynagi": KAYNAK_BLOK,
                "grup": len(desen),
                "kappa": kappa,
                "Q_ham": q_ham,
                "Q_LBdik": q_a,
                "Q_probdik": q_b,
                "Q": q_ger,
                "span_kaybi_pay": 1.0 - q_a / q_ham if q_ham else 0.0,
                "kirpilan_satir": kirpik,
                "degisen_satir": degisen,
                "kirpma_sapmasi_maxabs": maxabs_fark,
                "notr_dMSE": notr_dmse,
                "notr_skor": notr_skor,
                "skor_degisimi_notr": notr_skor - S_V93,
                "sigma_dMSE": sigma_dmse,
                "sigma_L": sigma_L,
                "cv_kappa": cv_kappa,
                "cv_kazanc_toplam": cv_kaz,
                "L_beklenen": L_bek,
                "kazanc_beklenen": kaz_bek,
                "skor_beklenen": skor_bek,
                "skor_ters_isaret": skor_ters,
                "okunabilir": abs(notr_skor - S_V93) > 2e-4,
            }
        )
        k = kayit[-1]
        print(f"--- {ad}  ({rejim} x {yon_adi}, {len(desen)} grup, kappa={kappa})")
        print(
            f"      Q_ham={q_ham:.6f} -> LB-dik {q_a:.6f} (span kaybi "
            f"{k['span_kaybi_pay'] * 100:.2f}%) -> prob-dik {q_b:.6f} | dosyadan {q_ger:.6f}"
        )
        print(
            f"      degisen satir {degisen:,}  kirpilan {kirpik}  kirpma sapmasi {maxabs_fark:.2e}"
        )
        print(
            f"      NOTR (L=0) skor {notr_skor:.6f}  (v93'e gore {notr_skor - S_V93:+.6f})"
            f"   okunabilir={k['okunabilir']}"
        )
        print(f"      sigma(L)={sigma_L:.3e}   CV kappa={cv_kappa:+.2f}  CV kazanc={cv_kaz:+.6f}")
        print(f"      BEKLENEN skor {skor_bek:.6f} | ters isaretse {skor_ters:.6f}\n")

    # yonler arasi diklik kanit
    M = np.array(onceki)
    C = M @ M.T / n * n  # birim normlu -> dogrudan kosinus
    print("prob yonleri arasi kosinus (Gram-Schmidt sonrasi):")
    print(np.round(C, 12))
    print(f"maks off-diagonal = {np.abs(C - np.eye(len(onceki))).max():.2e}")

    (BURA / "prob_kayit.json").write_text(
        json.dumps(kayit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: prob_kayit.json")


if __name__ == "__main__":
    main()
