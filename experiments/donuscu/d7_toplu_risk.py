"""D7 -- TOPLU vs TEKIL DONUS, tam pencereli ve daha genis ornekle.

d6 toplu esigini >=20 aldi; o esikte tek olay (2026-03-26) kaldi ve train
bitisine 5 gun kala oldugu icin yalniz 170 satir olctu. Burada esik >=8 ve
YALNIZ 2026-01-01 ONCESI donusler alinir (>=90 gun tam pencere garantisi).

Ayrica: T1/T2 test nufusunun buyuk kismi 2026-05-03 ve 2026-05-11'de TOPLU
giriyor. Bu, olcumu gecersiz kilar mi -- net cevap icin.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, UFUK, train

TOPLU_ESIK = 8
SON_DONUS = pd.Timestamp("2026-01-01")  # tam pencere icin


def main() -> int:
    tr = train()
    d = tr.sort_values(["tanim", "tarih"]).copy()
    d["onceki"] = d.groupby("tanim")["tarih"].shift()
    d["bosluk"] = (d["tarih"] - d["onceki"]).dt.days

    rap: dict = {}
    for min_bosluk in (90, 180):
        donus = d[d["bosluk"] >= min_bosluk].copy()
        gun_say = donus["tarih"].value_counts()
        toplu_gun = set(gun_say.index[gun_say >= TOPLU_ESIK])
        dd = donus.drop_duplicates("tanim", keep="first")[["tanim", "tarih", "bosluk"]]
        dd = dd.rename(columns={"tarih": "donus_gunu"})
        dd["toplu"] = dd["donus_gunu"].isin(toplu_gun)
        dd = dd[dd["donus_gunu"] < SON_DONUS]

        # donusten onceki son 60 kaydin ofseti (olu mu?)
        onc = []
        for t, gd in dd.set_index("tanim")["donus_gunu"].items():
            g = tr[(tr["tanim"] == t) & (tr["tarih"] < gd)]
            onc.append(
                float(g.sort_values("tarih").tail(60)["ofs"].mean()) if len(g) >= 30 else np.nan
            )
        dd["son60_ofs"] = onc

        # donus sonrasi 122 gun
        idx = tr.set_index("tanim")
        parca = []
        for _, r in dd.iterrows():
            s = idx.loc[[r["tanim"]]]
            s = s[
                (s["tarih"] >= r["donus_gunu"])
                & (s["tarih"] < r["donus_gunu"] + pd.Timedelta(days=UFUK))
            ]
            if len(s):
                parca.append(
                    pd.DataFrame(
                        {
                            "tanim": r["tanim"],
                            "tarih": s["tarih"].to_numpy(),
                            "ofs": s["ofs"].to_numpy(),
                            "tuketim": s["tuketim"].to_numpy(),
                            "toplu": r["toplu"],
                            "son60_ofs": r["son60_ofs"],
                            "bosluk": r["bosluk"],
                        }
                    )
                )
        if not parca:
            continue
        P = pd.concat(parca, ignore_index=True)
        kontrol_gun = tr.groupby("tarih")["ofs"].mean()
        P["fazla"] = P["ofs"] - P["tarih"].map(kontrol_gun)

        blok = {}
        for olu_ad, msk in (
            ("hepsi", P["son60_ofs"].notna() | P["son60_ofs"].isna()),
            ("OLU son60_ofs<=-2", P["son60_ofs"] <= -2.0),
            ("CANLI son60_ofs>-2", P["son60_ofs"] > -2.0),
        ):
            for tp in (False, True):
                g = P[msk & (P["toplu"] == tp)]
                if len(g) < 20:
                    continue
                blok[f"{olu_ad} | {'TOPLU' if tp else 'TEKIL'}"] = {
                    "trafo": int(g["tanim"].nunique()),
                    "satir": int(len(g)),
                    "sifir_orani": round(float((g["tuketim"] <= 0).mean()), 4),
                    "ort_ofs": round(float(g["ofs"].mean()), 4),
                    "ort_FAZLA": round(float(g["fazla"].mean()), 4),
                    "trafo_bazli_FAZLA_medyan": round(
                        float(g.groupby("tanim")["fazla"].mean().median()), 4
                    ),
                }
        rap[f"bosluk>={min_bosluk}"] = {
            "toplu_gunler": {str(k.date()): int(v) for k, v in gun_say.items() if v >= TOPLU_ESIK},
            "bloklar": blok,
        }
        print(
            f"\n=== bosluk >= {min_bosluk} gun, donus < 2026-01-01, toplu esigi >= {TOPLU_ESIK} ==="
        )
        print("toplu gunler:", rap[f"bosluk>={min_bosluk}"]["toplu_gunler"])
        for k, v in blok.items():
            print(
                f"  {k:32s} trafo={v['trafo']:4d} satir={v['satir']:6d} "
                f"sifir%={100 * v['sifir_orani']:5.1f} ofs={v['ort_ofs']:+.4f} "
                f"FAZLA={v['ort_FAZLA']:+.4f} (trafo medyan {v['trafo_bazli_FAZLA_medyan']:+.4f})"
            )

    (CIK / "d7_toplu_risk.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d7_toplu_risk.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
