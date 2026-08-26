"""EKSEN 1 -- ASIL KARAR TESTI: erken TEMIZ kesmede uydurulan b_i kestiricisi
URETIM modelinin kis26 hatasini dusuruyor mu?

Neden bu test belirleyici:
  * kis26, uretim onbelleginin TEK temiz foldu ve GERCEK modeldir.
  * kesme 2025-07-31'in hedef penceresi (Agu-Kas) kis26'nin hedef penceresiyle
    (Ara-Mar) HIC ORTUSMEZ -> kestirici kis26'yi hic gormemis olur.
  * Bu tam olarak "kis26'da uydur, TEST'e uygula" adiminin adil provasidir.

Ayrica kesme 2025-03-31 (Nisan-Temmuz = TESTIN MEVSIMSEL IKIZI) ve 2025-05-31,
2025-09-30 kaynaklari da denenir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "scripts"))

from eksen1_coklu_kesme import (  # noqa: E402
    asof_oznitelik,
    b_modeli_egit,
    b_ozellik_matrisi,
    b_tablosu,
    kategoriler,
    kazanc,
    kirpma_tablosu,
    kumulatifler,
    panel_kur,
    veri_yukle,
)

CIKTI = KOK / "data" / "interim" / "eksen1_kesme"
ALFALAR = [0.0, 0.25, 0.5, 0.75, 1.0]
KAYNAKLAR = ["2025-03-31", "2025-05-31", "2025-07-31", "2025-09-30"]
ORTUSME = {
    "2025-03-31": "yok",
    "2025-05-31": "yok",
    "2025-07-31": "yok",
    "2025-09-30": "2 ay (Ara-Oca)",
}


def uretim_kis26(P):
    """Uretim modelinin kis26 satirlari: gercek, tahmin, trafo indeksi.

    DIKKAT: onbellek HAM train.csv SIRASINDA. veri_yukle() cerceveyi
    (tanim,tarih) siralar -- o cerceve burada KULLANILAMAZ."""
    d = pd.read_csv(KOK / "data" / "raw" / "train.csv", parse_dates=["tarih"])
    z = np.load(KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz")
    p = np.mean(
        [
            (3 * z[f"kis26_{s}_cat"] + z[f"kis26_{s}_xgb"] + z[f"kis26_{s}_lgbm"]) / 5
            for s in (1000, 1001, 1002)
        ],
        axis=0,
    )
    sicak = set(d.loc[(d.tarih >= "2025-01-01") & (d.tarih <= "2025-11-30"), "tanim"].unique())
    m = ((d.tarih >= "2025-12-01") & (d.tarih <= "2026-03-31") & d.tanim.isin(sicak)).to_numpy()
    idx = np.flatnonzero(m)
    assert len(idx) == len(p), f"satir uyusmuyor {len(idx)} vs {len(p)}"
    gercek = np.log1p(d["tuketim"].to_numpy()[idx])
    ti = P["t_ix"].reindex(d["tanim"].to_numpy()[idx]).to_numpy()
    return dict(gercek=gercek, tahmin=p, ti=ti)


def main():
    d = veri_yukle()
    P = panel_kur(d)
    kat, kat_ad = kategoriler(P)
    C = kumulatifler(P)
    ofs = d["ofs"].to_numpy(dtype=np.float64)
    n_t = P["n_t"]

    U = uretim_kis26(P)
    m0 = float(np.mean((U["gercek"] - U["tahmin"]) ** 2))
    print(f"[uretim kis26] {len(U['gercek']):,} satir  MSE={m0:.5f}")

    # kis26 kesmesinde (2025-11-30) as-of oznitelikler -- kestiriciyi UYGULAMAK icin
    k26 = int(P["g_ix"].loc[pd.Timestamp("2025-11-30")])
    Xt26, oz_ad, _ = asof_oznitelik(P, C, k26)
    R26 = dict(Xt=Xt26, oz_ad=oz_ad, ti=U["ti"], gercek=U["gercek"], tahmin=U["tahmin"])

    # uretim kis26'nin KENDI b_i'si -- tavan ve sabit referansi
    b26, say26 = b_tablosu(R26, n_t)
    w = say26 > 0
    delta26 = float(np.sum(say26[w] * b26[w]) / say26[w].sum())
    tavan = kazanc(U["gercek"], U["tahmin"], np.nan_to_num(b26)[U["ti"]], 1.0)
    print(f"  kendi b_i: agir.ort={delta26:+.4f}  TAVAN kazanc={tavan:.5f}")

    KAY = {}
    for ks in KAYNAKLAR:
        z = np.load(CIKTI / f"kesme_{ks}.npz", allow_pickle=True)
        KAY[ks] = dict(
            Xt=z["Xt"],
            oz_ad=[str(a) for a in z["oz_ad"]],
            ti=z["ti"],
            tahmin=z["tahmin"],
            gercek=z["gercek"],
        )
        print(f"[onbellek] {ks}: {len(z['gercek']):,} satir")

    print("\n=== ERKEN TEMIZ KESMEDE UYDUR -> URETIM kis26'YA UYGULA ===")
    for sade in (True, False):
        etiket = "SADE-9" if sade else "TAM"
        print(f"\n--- {etiket} ---")
        print(
            f"{'kaynak C1':<12}{'ortusme':<14}{'kor':>7}"
            + "".join(f"{'a=' + str(a):>9}" for a in ALFALAR)
            + f"{'SABIT@1':>10}"
        )
        for ks in KAYNAKLAR:
            r1 = KAY[ks]
            b1, say1 = b_tablosu(r1, n_t)
            X1, ad, kix = b_ozellik_matrisi(r1, kat, kat_ad, sade)
            mdl = b_modeli_egit(X1, ad, kix, b1, say1, say1 > 0)
            X26, _, _ = b_ozellik_matrisi(R26, kat, kat_ad, sade)
            bhat = mdl.predict(X26)
            kor = float(np.corrcoef(bhat[w], b26[w])[0, 1])
            kz = [kazanc(U["gercek"], U["tahmin"], bhat[U["ti"]], a) for a in ALFALAR]
            d1 = float(np.sum(say1[say1 > 0] * b1[say1 > 0]) / say1[say1 > 0].sum())
            sk = kazanc(U["gercek"], U["tahmin"], np.full(len(U["ti"]), d1), 1.0)
            print(
                f"{ks:<12}{ORTUSME[ks]:<14}{kor:>7.3f}"
                + "".join(f"{v:>9.5f}" for v in kz)
                + f"{sk:>10.5f}"
            )
            if sade:
                tab = kirpma_tablosu(R26, bhat[U["ti"]], 1.0, n_t)
                print(
                    "      kirpma K=0/1/5/10/25/50: "
                    + " ".join(f"{tab[K]:+.5f}" for K in (0, 1, 5, 10, 25, 50))
                    + f"   (C1 delta={d1:+.4f})"
                )


if __name__ == "__main__":
    main()
