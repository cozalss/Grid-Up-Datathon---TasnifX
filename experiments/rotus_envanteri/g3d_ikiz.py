"""GRUP B'nin TAM IKIZI train icinde -- ve donus sonrasi gercek davranis.

GRUP B profili (olculdu):
  * train'de 9.713 satir, HEPSI sifir; 93/93 trafonun TEK pozitif kaydi yok
  * trafo basina ~104 kayit, hepsi 2025-01..2025-06 araliginda yogun
  * panelden dusus: 57'si 2025-06-17'de
  * teste donus: 2026-05-03 (33) / 2026-05-11 (37)  -> bosluk ~320 gun
  * donus TOPLU bir olay (ayni gun 30+ trafo)

Train ikizi icin ayni sartlari arayacagiz ve her sarti tek tek gevseterek
orneklem buyuklugu ile benzerlik arasindaki dengeyi ACIK gosterecegiz.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, train

TRAIN_SON = pd.Timestamp("2026-03-31")


def main() -> int:
    tr = train().sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    g = tr.groupby("tanim", observed=True)
    tr["bosluk"] = (tr["tarih"] - g["tarih"].shift(1)).dt.days
    tr["poz"] = (tr["tuketim"] > 0).astype(int)
    tr["kum_poz_once"] = g["poz"].cumsum() - tr["poz"]
    tr["kum_n_once"] = g.cumcount()

    # her trafonun ilk buyuk boslugu
    don = tr[tr["bosluk"].notna()].copy()
    rap: dict = {"senaryolar": {}}

    senaryolar = [
        ("S1 bosluk>=150g, oncesi TAMAMI sifir, >=60 kayit", 150, 60, True),
        ("S2 bosluk>=150g, oncesi TAMAMI sifir, >=20 kayit", 150, 20, True),
        ("S3 bosluk>=100g, oncesi TAMAMI sifir, >=60 kayit", 100, 60, True),
        ("S4 bosluk>= 60g, oncesi TAMAMI sifir, >=60 kayit", 60, 60, True),
        ("S5 bosluk>=150g, oncesi POZITIFLI, >=60 kayit (KIYAS)", 150, 60, False),
        ("S6 bosluk>= 60g, oncesi POZITIFLI, >=60 kayit (KIYAS)", 60, 60, False),
    ]
    for ad, gmin, nmin, olu in senaryolar:
        sel = (don["bosluk"] >= gmin) & (don["kum_n_once"] >= nmin)
        sel &= (don["kum_poz_once"] == 0) if olu else (don["kum_poz_once"] > 0)
        d = don[sel].groupby("tanim").head(1)
        if d.empty:
            rap["senaryolar"][ad] = {"trafo": 0}
            print(f"[{ad}] trafo=0")
            continue
        kayit = []
        for tanim, gun in zip(d["tanim"], d["tarih"], strict=True):
            w = tr[(tr["tanim"] == tanim) & (tr["tarih"] >= gun)]
            w122 = w[w["tarih"] < gun + pd.Timedelta(days=122)]
            kalan = (TRAIN_SON - gun).days + 1
            kayit.append(
                {
                    "tanim": tanim,
                    "donus": gun,
                    "kalan_gun": kalan,
                    "n": len(w122),
                    "sifir": float((w122["tuketim"] <= 0).mean()),
                    "lp": float(w122["lp"].mean()),
                }
            )
        k = pd.DataFrame(kayit)
        tam = k[k["kalan_gun"] >= 122]
        blok = {
            "trafo": int(len(k)),
            "satir": int(k["n"].sum()),
            "sifir_orani": float(np.average(k["sifir"], weights=k["n"])),
            "ort_lp": float(np.average(k["lp"], weights=k["n"])),
            "trafo_cogunlukla_sifir": float((k["sifir"] > 0.5).mean()),
            "TAM_PENCERE_trafo": int(len(tam)),
            "TAM_PENCERE_satir": int(tam["n"].sum()) if len(tam) else 0,
            "TAM_PENCERE_sifir_orani": float(np.average(tam["sifir"], weights=tam["n"]))
            if len(tam)
            else float("nan"),
            "TAM_PENCERE_ort_lp": float(np.average(tam["lp"], weights=tam["n"]))
            if len(tam)
            else float("nan"),
            "donus_gunleri": {
                str(x.date()): int(c) for x, c in k["donus"].value_counts().head(6).items()
            },
        }
        rap["senaryolar"][ad] = blok
        print(
            f"[{ad}]\n    trafo={blok['trafo']:4d} satir={blok['satir']:6d} "
            f"sifir%={100 * blok['sifir_orani']:5.1f} lp={blok['ort_lp']:6.3f} | "
            f"TAM PENCERE trafo={blok['TAM_PENCERE_trafo']:3d} satir={blok['TAM_PENCERE_satir']:5d} "
            f"sifir%={100 * blok['TAM_PENCERE_sifir_orani']:5.1f} lp={blok['TAM_PENCERE_ort_lp']:6.3f}"
        )
        print(f"    donus gunleri: {blok['donus_gunleri']}")

    # ---- BASA BAS analizi: grup B'de v83 (lp 5.479) vs v89 (lp 0.784)
    lp83, lp89 = 5.4787, 0.7836
    print("\n[BASA BAS] grup B, 7.149 satir. p = gercekten sifir olan satir orani.")
    tablo = {}
    for L in (5.0, 5.4787, 6.0, 6.5, 6.94):
        # p*(x^2) + (1-p)*(x-L)^2 esitligi
        a = lp83**2 - lp89**2
        b = (lp83 - L) ** 2 - (lp89 - L) ** 2
        p_bb = -b / (a - b) if (a - b) != 0 else float("nan")
        tablo[L] = float(p_bb)
        print(f"    canli seviye L={L:.3f} -> basa bas p = {p_bb:.3f}")
    rap["basa_bas_p"] = tablo

    # secili p'lerde dMSE (v83 - v89), tam paydada (714.688)
    n_b, N = 7149, 714688
    dm = {}
    for L in (5.4787, 6.48, 6.94):
        for p in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0):
            m83 = p * lp83**2 + (1 - p) * (lp83 - L) ** 2
            m89 = p * lp89**2 + (1 - p) * (lp89 - L) ** 2
            dm[f"L={L:.2f},p={p:.1f}"] = float(n_b * (m83 - m89) / N)
    rap["dMSE_v83_eksi_v89"] = dm
    print("\n[dMSE  v83 - v89]  (>0 ise v89 daha iyi), tam 714.688 paydada:")
    for L in (5.4787, 6.48, 6.94):
        satir = "  ".join(
            f"p={p:.1f}:{dm[f'L={L:.2f},p={p:.1f}']:+.4f}" for p in (0.0, 0.4, 0.6, 0.8, 1.0)
        )
        print(f"    L={L:.2f}   {satir}")

    (KOK / "reports/g3d_ikiz.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: reports/g3d_ikiz.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
