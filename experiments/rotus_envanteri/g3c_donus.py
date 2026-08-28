"""GRUP B'nin YONU -- panelden dusup GERI DONEN trafolar ne yapiyor?

Karar sorusu: 93 grup-B trafosu train'de 455 gun boyunca TEK kWh tuketmedi
(9.713 satir, %100 sifir). Testte 2026-05-03/05-11'de panele geri donuyorlar.
"Geri donus" bir DIRILME kaniti mi?

SIZINTISIZ SINAMA: train'in KENDI icindeki panel bosluk-donuslerini bul.
  * bosluk >= G gun
  * bosluktan ONCEKI tum kayitlar SIFIR      -> "olu donen"
  * bosluktan ONCEKI kayitlarda pozitif var  -> "canli donen" (kiyas)
Donus sonrasi ilk 122 gunde sifir orani ve ort log1p olculur.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, train

UFUK = 122


def main() -> int:
    tr = train().sort_values(["tanim", "tarih"], kind="mergesort").reset_index(drop=True)
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    g = tr.groupby("tanim", observed=True)
    tr["onceki"] = g["tarih"].shift(1)
    tr["bosluk"] = (tr["tarih"] - tr["onceki"]).dt.days
    # bosluktan onceki kumulatif pozitif sayisi (o satir HARIC)
    tr["poz"] = (tr["tuketim"] > 0).astype(int)
    tr["kum_poz_once"] = g["poz"].cumsum() - tr["poz"]

    rap: dict = {}
    for G in (30, 60, 120, 200):
        don = tr[tr["bosluk"] >= G]
        satirlar = []
        for etiket, sel in (
            ("olu_donen", don["kum_poz_once"] == 0),
            ("canli_donen", don["kum_poz_once"] > 0),
        ):
            d = don[sel]
            if d.empty:
                satirlar.append((etiket, 0, 0, float("nan"), float("nan"), float("nan")))
                continue
            # her trafonun ILK boyle donusu
            ilk = d.groupby("tanim").head(1)
            kayit = []
            for tanim, don_gun in zip(ilk["tanim"], ilk["tarih"], strict=True):
                w = tr[
                    (tr["tanim"] == tanim)
                    & (tr["tarih"] >= don_gun)
                    & (tr["tarih"] < don_gun + pd.Timedelta(days=UFUK))
                ]
                kayit.append(
                    (tanim, len(w), float((w["tuketim"] <= 0).mean()), float(w["lp"].mean()))
                )
            k = pd.DataFrame(kayit, columns=["tanim", "n", "sifir", "lp"])
            satirlar.append(
                (
                    etiket,
                    int(len(k)),
                    int(k["n"].sum()),
                    float(np.average(k["sifir"], weights=k["n"])),
                    float(np.average(k["lp"], weights=k["n"])),
                    float((k["sifir"] > 0.5).mean()),
                )
            )
        rap[f"bosluk>={G}g"] = {
            e: {"trafo": t, "satir": s, "sifir_orani": z, "ort_lp": l, "trafo_cogunlukla_sifir": m}
            for e, t, s, z, l, m in satirlar
        }
        for e, t, s, z, l, m in satirlar:
            print(
                f"[bosluk>={G:3d}g] {e:12s} trafo={t:4d} satir={s:6d} "
                f"donus sonrasi 122g: sifir%={100 * z:5.1f} ort_lp={l:6.3f} "
                f"trafo_cogunlukla_sifir={m:.2f}"
            )

    # ---- TOPLU PANEL GIRIS OLAYLARI: train'de kac trafo ayni gun donuyor
    ilk_don = tr[tr["bosluk"] >= 60].groupby("tanim").head(1)
    parti = ilk_don.groupby("tarih").size().sort_values(ascending=False).head(10)
    rap["toplu_donus_gunleri"] = {str(k.date()): int(v) for k, v in parti.items()}
    print("\n[toplu donus gunleri (bosluk>=60g)]", rap["toplu_donus_gunleri"])

    # ---- TOPLU gunlerde donen OLU trafolar ne yapiyor (parti >= 20)
    buyuk = set(parti[parti >= 20].index)
    sonuc = {}
    for etiket, sel in (
        ("olu_donen", ilk_don["kum_poz_once"] == 0),
        ("canli_donen", ilk_don["kum_poz_once"] > 0),
    ):
        for parti_ad, msk in (
            ("TOPLU (parti>=20)", ilk_don["tarih"].isin(buyuk)),
            ("TEKIL (parti<20)", ~ilk_don["tarih"].isin(buyuk)),
        ):
            d = ilk_don[sel & msk]
            if d.empty:
                continue
            top_n = top_z = top_l = 0.0
            for tanim, don_gun in zip(d["tanim"], d["tarih"], strict=True):
                w = tr[
                    (tr["tanim"] == tanim)
                    & (tr["tarih"] >= don_gun)
                    & (tr["tarih"] < don_gun + pd.Timedelta(days=UFUK))
                ]
                top_n += len(w)
                top_z += float((w["tuketim"] <= 0).sum())
                top_l += float(w["lp"].sum())
            sonuc[f"{etiket} | {parti_ad}"] = {
                "trafo": int(len(d)),
                "satir": int(top_n),
                "sifir_orani": top_z / top_n if top_n else float("nan"),
                "ort_lp": top_l / top_n if top_n else float("nan"),
            }
            s = sonuc[f"{etiket} | {parti_ad}"]
            print(
                f"[{etiket:12s} {parti_ad:18s}] trafo={s['trafo']:4d} satir={s['satir']:6d} "
                f"sifir%={100 * s['sifir_orani']:5.1f} ort_lp={s['ort_lp']:6.3f}"
            )
    rap["toplu_vs_tekil"] = sonuc

    (KOK / "reports/g3c_donus.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: reports/g3c_donus.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
