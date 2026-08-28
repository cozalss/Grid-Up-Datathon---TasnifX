"""GOREV 1 -- temiz SOTA dosyasinin BAGIMSIZ dogrulamasi.

- grup B / grup A tanimini yeniden turet
- mevcut submissions/tuketim_v90_temiz_sota.csv beklenen icerikle birebir mi?
- grup A'da sota_v1'in sifirlamasinin etkisi
- mimari harmani (maske disi) kac satiri ne kadar degistiriyor
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import A_SINIRI, KOK, bagil_maske, hizala, test, train, yon_enerjisi

V83 = "tuketim_v83_sicak_optimum.csv"
V89 = "tuketim_v89_genis_taban.csv"
SOTA = "tuketim_sota_v1.csv"
V90 = "tuketim_v90_temiz_sota.csv"


def main() -> int:
    tr, te = train(), test()
    v83 = hizala(V83, te)
    v89 = hizala(V89, te)
    sota = hizala(SOTA, te)

    maske = bagil_maske(v89, v83)
    maske_tanim = sorted(te.loc[maske, "tanim"].unique())
    print(f"maske satir      = {int(maske.sum())}")
    print(f"maske trafo      = {len(maske_tanim)}")

    tr_son = tr.groupby("tanim")["tarih"].max()
    b_tanim = sorted(t for t in maske_tanim if tr_son.get(t, pd.NaT) < A_SINIRI)
    a_tanim = sorted(set(maske_tanim) - set(b_tanim))
    b_set, a_set = set(b_tanim), set(a_tanim)
    b_row = te["tanim"].isin(b_set).to_numpy()
    a_row = te["tanim"].isin(a_set).to_numpy()

    print(f"GRUP B  trafo={len(b_tanim)}  test satir={int(b_row.sum())}")
    print(f"GRUP A  trafo={len(a_tanim)}  test satir={int(a_row.sum())}")

    # ---- v90 = sota_v1 tabanli, grup B'nin TUM test satirlarinda v83
    beklenen = sota.copy()
    beklenen[b_row] = v83[b_row]

    sonuc: dict = {
        "grup_b_trafo": len(b_tanim),
        "grup_b_satir": int(b_row.sum()),
        "grup_a_trafo": len(a_tanim),
        "grup_a_satir": int(a_row.sum()),
        "maske_satir": int(maske.sum()),
        "maske_trafo": len(maske_tanim),
    }

    # mevcut v90 ile karsilastir
    try:
        mevcut = hizala(V90, te)
        fark = np.abs(mevcut - beklenen)
        bag = fark / np.maximum(np.abs(beklenen), 1e-9)
        sonuc["v90_mevcut_esles"] = {
            "maxabs": float(fark.max()),
            "maxbagil": float(bag.max()),
            "farkli_satir": int((bag >= 1e-9).sum()),
        }
        print(
            f"MEVCUT v90 vs beklenen: maxabs={fark.max():.3e} "
            f"farkli_satir={int((bag >= 1e-9).sum())}"
        )
    except FileNotFoundError:
        sonuc["v90_mevcut_esles"] = None
        print("MEVCUT v90 yok")

    # ---- yon enerjileri
    for ad, x in (("v90-v83", beklenen), ("sota_v1-v83", sota), ("v89-v83", v89)):
        sonuc.setdefault("olcumler", {})[ad] = {"ad": ad, **yon_enerjisi(x, v83)}
        o = sonuc["olcumler"][ad]
        print(f"{ad:14s} degisen={o['degisen_satir']:7d}  Q={o['Q']:.5f}  ort={o['ort_fark']:+.5f}")

    # ---- GRUP A: sota_v1'in sifirlamasi orada ne yapiyor
    lp = lambda v: np.log1p(np.clip(v, 0, None))  # noqa: E731
    for ad, sel in (("grupA", a_row), ("grupB", b_row)):
        d = lp(sota[sel]) - lp(v83[sel])
        print(
            f"[{ad}] n={sel.sum():6d} sota ort log1p={lp(sota[sel]).mean():.4f} "
            f"v83 ort log1p={lp(v83[sel]).mean():.4f} v89 ort={lp(v89[sel]).mean():.4f} "
            f"d_ort={d.mean():+.4f} ||d||^2={float(d @ d):.1f}"
        )
        sonuc.setdefault("grup_detay", {})[ad] = {
            "n": int(sel.sum()),
            "sota_ort_log1p": float(lp(sota[sel]).mean()),
            "v83_ort_log1p": float(lp(v83[sel]).mean()),
            "v89_ort_log1p": float(lp(v89[sel]).mean()),
            "sota_sifir_orani": float((sota[sel] <= 1e-9).mean()),
            "v83_sifir_orani": float((v83[sel] <= 1e-9).mean()),
            "d_ort": float(d.mean()),
            "sse_toplam_paydada": float(d @ d) / len(te),
        }

    # ---- grup A'da v83'e donmek: hipotetik dMSE sinirlari
    # gercek 0 varsayimi altinda MSE payi = mean(lp(x)^2 * 1{gercek=0})
    n = len(te)
    for etiket, sel in (("grupA", a_row), ("grupB", b_row)):
        pay_sota = float((lp(sota[sel]) ** 2).sum() / n)
        pay_v83 = float((lp(v83[sel]) ** 2).sum() / n)
        pay_v89 = float((lp(v89[sel]) ** 2).sum() / n)
        print(
            f"[{etiket}] TUMU-SIFIR varsayiminda MSE payi: "
            f"sota={pay_sota:.5f} v83={pay_v83:.5f} v89={pay_v89:.5f}"
        )
        sonuc["grup_detay"][etiket].update(
            {
                "pay_sifir_varsayimi_sota": pay_sota,
                "pay_sifir_varsayimi_v83": pay_v83,
                "pay_sifir_varsayimi_v89": pay_v89,
            }
        )

    # ---- mimari harmani: maske DISI satirlarda sota_v1 vs v83
    disi = ~(a_row | b_row)
    d = lp(sota[disi]) - lp(v83[disi])
    deg = bagil_maske(sota[disi], v83[disi])
    sonuc["mimari_harmani_maske_disi"] = {
        "n": int(disi.sum()),
        "degisen_satir": int(deg.sum()),
        "degisen_oran": float(deg.mean()),
        "ort_mutlak_fark_log1p": float(np.abs(d).mean()),
        "ort_fark_log1p": float(d.mean()),
        "Q_tam_paydada": float(d @ d / n),
        "p50_abs": float(np.percentile(np.abs(d), 50)),
        "p90_abs": float(np.percentile(np.abs(d), 90)),
        "p99_abs": float(np.percentile(np.abs(d), 99)),
        "maxabs": float(np.abs(d).max()),
    }
    h = sonuc["mimari_harmani_maske_disi"]
    print(
        f"[MIMARI] maske disi n={h['n']} degisen={h['degisen_satir']} "
        f"(%{100 * h['degisen_oran']:.2f}) |d|ort={h['ort_mutlak_fark_log1p']:.4f} "
        f"p50={h['p50_abs']:.4f} p90={h['p90_abs']:.4f} Q={h['Q_tam_paydada']:.5f}"
    )

    (KOK / "reports").mkdir(exist_ok=True)
    (KOK / "reports/g1_dogrulama.json").write_text(
        json.dumps(sonuc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # grup listelerini de yaz -- gorev 3 kullanacak
    (KOK / "experiments/rotus_envanteri/grup_b.txt").write_text(
        "\n".join(b_tanim), encoding="utf-8"
    )
    (KOK / "experiments/rotus_envanteri/grup_a.txt").write_text(
        "\n".join(a_tanim), encoding="utf-8"
    )
    print("yazildi: reports/g1_dogrulama.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
