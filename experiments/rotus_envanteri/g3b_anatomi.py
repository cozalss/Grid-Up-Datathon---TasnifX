"""GRUP B ANATOMISI -- brifingdeki "grup B DIRI" iddiasini sina.

Iddia: 251 maskeli trafonun 93'u (train son kaydi < 2026-03-27) aslinda DIRI,
v83'un onlara yazdigi ~402 kWh isabetli.

Bu betik o iddiayi train ETIKETLERIYLE dogrudan sinar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, hizala, test, train


def main() -> int:
    tr, te = train(), test()
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())
    A = set((KOK / "experiments/rotus_envanteri/grup_a.txt").read_text(encoding="utf-8").split())
    print(f"grup B {len(B)} trafo | grup A {len(A)} trafo")

    trB = tr[tr["tanim"].isin(B)]
    trA = tr[tr["tanim"].isin(A)]
    rap: dict = {}

    for ad, d, kume in (("B", trB, B), ("A", trA, A)):
        g = d.groupby("tanim")
        poz = g["tuketim"].apply(lambda s: int((s > 0).sum()))
        top = g["tuketim"].sum()
        rap[f"grup_{ad}"] = {
            "trafo": len(kume),
            "train_satir": int(len(d)),
            "train_sifir_orani": float((d["tuketim"] <= 0).mean()),
            "hic_pozitif_kaydi_olmayan_trafo": int((poz == 0).sum()),
            "pozitif_kayit_medyani": float(poz.median()),
            "pozitif_kayit_p90": float(poz.quantile(0.9)),
            "toplam_tuketimi_sifir_trafo": int((top <= 0).sum()),
            "ilk_kayit_min": str(d["tarih"].min().date()),
            "son_kayit_max": str(d["tarih"].max().date()),
        }
        r = rap[f"grup_{ad}"]
        print(
            f"[{ad}] train satir={r['train_satir']:6d} sifir%={100 * r['train_sifir_orani']:.2f} "
            f"HIC pozitifi olmayan trafo={r['hic_pozitif_kaydi_olmayan_trafo']}/{len(kume)} "
            f"poz kayit medyan={r['pozitif_kayit_medyani']:.0f}"
        )

    # --- B'nin train boyunca AY AY sifir orani
    trB2 = trB.copy()
    trB2["ay"] = trB2["tarih"].dt.to_period("M").astype(str)
    ay = trB2.groupby("ay").agg(
        satir=("tuketim", "size"),
        sifir=("tuketim", lambda s: (s <= 0).mean()),
        ort_lp=("lp", "mean"),
    )
    print("\n[B] ay bazinda train:")
    print(ay.to_string())
    rap["grup_B_aylik"] = {
        k: {
            "satir": int(v["satir"]),
            "sifir_orani": float(v["sifir"]),
            "ort_lp": float(v["ort_lp"]),
        }
        for k, v in ay.iterrows()
    }

    # --- 2026-05-11 olayi: B trafolari teste NE ZAMAN giriyor
    teB = te[te["tanim"].isin(B)]
    ilk_te = teB.groupby("tanim")["tarih"].min()
    rap["B_test_ilk_gun"] = {
        str(k.date()): int(v) for k, v in ilk_te.value_counts().head(8).items()
    }
    print("\n[B] test ilk gun dagilimi:", rap["B_test_ilk_gun"])

    # --- v83 / v89 ne yaziyor
    v83 = hizala("tuketim_v83_sicak_optimum.csv", te)
    v89 = hizala("tuketim_v89_genis_taban.csv", te)
    sel = te["tanim"].isin(B).to_numpy()
    selA = te["tanim"].isin(A).to_numpy()
    for ad, s in (("B", sel), ("A", selA)):
        rap[f"yazilan_{ad}"] = {
            "satir": int(s.sum()),
            "v83_ort_kwh": float(v83[s].mean()),
            "v83_medyan_kwh": float(np.median(v83[s])),
            "v83_ort_lp": float(np.log1p(v83[s]).mean()),
            "v89_ort_kwh": float(v89[s].mean()),
            "v89_ort_lp": float(np.log1p(v89[s]).mean()),
        }
        r = rap[f"yazilan_{ad}"]
        print(
            f"[{ad}] v83 ort={r['v83_ort_kwh']:.1f} kWh medyan={r['v83_medyan_kwh']:.1f} "
            f"lp={r['v83_ort_lp']:.3f} | v89 ort={r['v89_ort_kwh']:.2f} kWh lp={r['v89_ort_lp']:.3f}"
        )

    # --- docs/43'un GERCEK grup B'si (kuyrugu >=60 sifir + son kayit < 2026-03-27)
    d = tr.sort_values(["tanim", "tarih"])
    son_n = d.groupby("tanim").tail(60)
    olu = son_n.groupby("tanim")["tuketim"].apply(lambda s: bool((s <= 0).all()))
    yeterli = d.groupby("tanim").size() >= 60
    son_t = d.groupby("tanim")["tarih"].max()
    d43_B = set((olu & yeterli & (son_t < pd.Timestamp("2026-03-27"))).pipe(lambda s: s.index[s]))
    d43_B &= set(te["tanim"].unique())
    d43_A = set((olu & yeterli & (son_t >= pd.Timestamp("2026-03-27"))).pipe(lambda s: s.index[s]))
    d43_A &= set(te["tanim"].unique())
    rap["docs43_grupB"] = {
        "trafo": len(d43_B),
        "satir": int(te["tanim"].isin(d43_B).sum()),
        "v89_maskesiyle_kesisim": len(d43_B & B),
        "maske_disinda_kalan": len(d43_B - B - A),
    }
    print(
        f"\n[docs/43 grup B] {len(d43_B)} trafo / {int(te['tanim'].isin(d43_B).sum())} satir | "
        f"v89-maske B ile kesisim {len(d43_B & B)} | maskenin TAMAMEN disinda {len(d43_B - B - A)}"
    )
    # docs/43 B'nin ikiz penceresi
    ik = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & tr["tanim"].isin(d43_B)]
    print(
        f"[docs/43 grup B] 2025 Nis-Tem: {len(ik)} satir sifir%={100 * (ik['tuketim'] <= 0).mean():.1f} "
        f"ort_lp={ik['lp'].mean():.3f}"
    )
    rap["docs43_grupB_ikiz"] = {
        "satir": int(len(ik)),
        "sifir_orani": float((ik["tuketim"] <= 0).mean()),
        "ort_lp": float(ik["lp"].mean()),
    }
    # bu grubun tamami maske disinda mi -> v83 onlara ne yaziyor
    disB = d43_B - B - A
    if disB:
        s = te["tanim"].isin(disB).to_numpy()
        ikd = tr[
            (tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & tr["tanim"].isin(disB)
        ]
        print(
            f"[docs/43 B \\ v89maske] {len(disB)} trafo {int(s.sum())} satir  "
            f"v83 ort={v83[s].mean():.1f} kWh lp={np.log1p(v83[s]).mean():.3f} | "
            f"ikiz sifir%={100 * (ikd['tuketim'] <= 0).mean():.1f} ort_lp={ikd['lp'].mean():.3f}"
        )
        rap["docs43_B_maske_disi"] = {
            "trafo": len(disB),
            "test_satir": int(s.sum()),
            "v83_ort_lp": float(np.log1p(v83[s]).mean()),
            "ikiz_satir": int(len(ikd)),
            "ikiz_sifir_orani": float((ikd["tuketim"] <= 0).mean()),
            "ikiz_ort_lp": float(ikd["lp"].mean()),
        }

    (KOK / "reports/g3b_anatomi.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: reports/g3b_anatomi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
