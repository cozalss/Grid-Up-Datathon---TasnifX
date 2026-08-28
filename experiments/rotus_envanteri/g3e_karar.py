"""GRUP B KARARI -- toplu donus vs tekil donus, ve tohum gurultusu (rotus 6)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, hizala, test, train

TRAIN_SON = pd.Timestamp("2026-03-31")


def main() -> int:
    tr = train().sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    g = tr.groupby("tanim", observed=True)
    tr["bosluk"] = (tr["tarih"] - g["tarih"].shift(1)).dt.days
    tr["poz"] = (tr["tuketim"] > 0).astype(int)
    tr["kum_poz_once"] = g["poz"].cumsum() - tr["poz"]
    tr["kum_n_once"] = g.cumcount()
    rap: dict = {}

    # ---- TUM olu-donusleri tek tek listele (bosluk>=60g, oncesi TAMAMI sifir, >=60 kayit)
    don = tr[(tr["bosluk"] >= 60) & (tr["kum_poz_once"] == 0) & (tr["kum_n_once"] >= 60)]
    ilk = don.groupby("tanim").head(1)
    # o gun kac trafo donuyor (TUM trafolar uzerinden, parti buyuklugu)
    tum_don = tr[tr["bosluk"] >= 60].groupby("tanim").head(1)
    parti = tum_don.groupby("tarih").size()

    kayit = []
    for tanim, gun, n_once in zip(ilk["tanim"], ilk["tarih"], ilk["kum_n_once"], strict=True):
        w = tr[(tr["tanim"] == tanim) & (tr["tarih"] >= gun)]
        w = w[w["tarih"] < gun + pd.Timedelta(days=122)]
        kayit.append(
            {
                "tanim": tanim,
                "donus": str(gun.date()),
                "parti_n": int(parti.get(gun, 1)),
                "kalan_gun": int((TRAIN_SON - gun).days + 1),
                "onceki_kayit": int(n_once),
                "n": int(len(w)),
                "sifir_orani": float((w["tuketim"] <= 0).mean()),
                "ort_lp": float(w["lp"].mean()),
            }
        )
    k = pd.DataFrame(kayit).sort_values("donus")
    print("OLU DONUSLER (bosluk>=60g, oncesi TAMAMI sifir, >=60 kayit) -- HEPSI:")
    print(k.to_string(index=False))
    rap["olu_donusler"] = kayit

    for ad, sel in (("TOPLU parti>=20", k["parti_n"] >= 20), ("TEKIL parti<20", k["parti_n"] < 20)):
        d = k[sel]
        if d.empty:
            continue
        print(
            f"\n  {ad}: trafo={len(d)} satir={int(d['n'].sum())} "
            f"sifir%={100 * np.average(d['sifir_orani'], weights=d['n']):.1f} "
            f"lp={np.average(d['ort_lp'], weights=d['n']):.3f}"
        )
        rap[f"ozet_{ad}"] = {
            "trafo": int(len(d)),
            "satir": int(d["n"].sum()),
            "sifir_orani": float(np.average(d["sifir_orani"], weights=d["n"])),
            "ort_lp": float(np.average(d["ort_lp"], weights=d["n"])),
        }

    # ---- 2026-03-26 TOPLU olayinda donen TUM trafolar (olu/canli)
    olay = tr[(tr["bosluk"] >= 60) & (tr["tarih"] == pd.Timestamp("2026-03-26"))]
    for ad, sel in (("olu", olay["kum_poz_once"] == 0), ("canli", olay["kum_poz_once"] > 0)):
        d = olay[sel]
        if d.empty:
            continue
        w = tr[tr["tanim"].isin(set(d["tanim"])) & (tr["tarih"] >= pd.Timestamp("2026-03-26"))]
        print(
            f"[2026-03-26 TOPLU] {ad:6s} trafo={len(d):3d} sonraki {len(w):4d} satir "
            f"sifir%={100 * (w['tuketim'] <= 0).mean():5.1f} lp={w['lp'].mean():.3f}"
        )
        rap[f"olay_20260326_{ad}"] = {
            "trafo": int(len(d)),
            "satir": int(len(w)),
            "sifir_orani": float((w["tuketim"] <= 0).mean()),
            "ort_lp": float(w["lp"].mean()),
        }

    # ---- ROTUS 6: tohum gurultusu sigma ve 30->35 kazanci
    te = test()
    lp = lambda v: np.log1p(np.clip(v, 0, None))  # noqa: E731
    a30 = lp(hizala("tuketim_v50_ham30.csv", te))
    a5 = lp(hizala("tuketim_v51_ek1.csv", te))
    var_fark = float(np.var(a30 - a5, ddof=1))
    sigma2 = var_fark / (1 / 30 + 1 / 5)
    kaz_30_35 = sigma2 * (1 / 30 - 1 / 35)
    kaz_30_50 = sigma2 * (1 / 30 - 1 / 50)
    rap["rotus6_tohum"] = {
        "var(lp30-lp5)": var_fark,
        "sigma2_tohum": float(sigma2),
        "sigma": float(np.sqrt(sigma2)),
        "mevcut_tohum": 30,
        "kullanilabilir_ek_parti": 1,
        "dMSE_30_to_35": float(-kaz_30_35),
        "dMSE_30_to_50_HIPOTETIK": float(-kaz_30_50),
    }
    print(
        f"\n[ROTUS 6] tohum gurultusu sigma={np.sqrt(sigma2):.5f} "
        f"-> 30->35 dMSE={-kaz_30_35:+.6f}  (30->50 hipotetik {-kaz_30_50:+.6f})"
    )

    (KOK / "reports/g3e_karar.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: reports/g3e_karar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
