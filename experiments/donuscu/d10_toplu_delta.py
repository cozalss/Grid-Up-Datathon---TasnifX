"""D10 -- TOPLU donusun delta'si: organige gore FARK olcumu.

d5'in delta'si (1.0142-2.19) ORGANIK donuslerden olculdu. Test kohortunun
%85'i TOPLU giriyor. Ihtiyacimiz olan sayi:

    delta_toplu = delta_organik + (W_toplu - W_organik)

Ayni capaya (v102'nin kohort ofseti) gore olculdugu icin capa SADELESIR;
yalniz iki pencere ortalamasinin FARKI gerekir. Gun-kontrolu ile takvim
etkisi de temizlenir.

Kurallar:
  - yalniz TAM 122 gunluk ileri penceresi olan donus gunleri
  - toplu esigi >= 8 (d6/d7 ile ayni)
  - saglam istatistik: ortalama YANINDA medyan ve %10 budanmis ortalama
    (d7'de ortalama -0.466 iken trafo-medyani +0.186 idi -> aykiri baskin)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import CIK, UFUK, train

TOPLU_ESIK = 8
TRAIN_SON = pd.Timestamp("2026-03-31")


def budanmis(x: np.ndarray, p: float = 0.10) -> float:
    if len(x) == 0:
        return float("nan")
    lo, hi = np.quantile(x, [p, 1 - p])
    y = x[(x >= lo) & (x <= hi)]
    return float(y.mean()) if len(y) else float("nan")


def main() -> int:  # noqa: PLR0915
    tr = train()
    d = tr.sort_values(["tanim", "tarih"]).copy()
    d["onceki"] = d.groupby("tanim")["tarih"].shift()
    d["bosluk"] = (d["tarih"] - d["onceki"]).dt.days

    # gunluk kontrol: tum panelin o gunku ortalama ofseti
    gun_kontrol = tr.groupby("tarih")["ofs"].mean()

    rap: dict = {}
    for min_bosluk in (90, 180):
        donus = d[d["bosluk"] >= min_bosluk].copy()
        say = donus["tarih"].value_counts()
        toplu_gun = set(say.index[say >= TOPLU_ESIK])

        dd = donus.drop_duplicates("tanim", keep="first")[["tanim", "tarih", "bosluk"]]
        dd = dd.rename(columns={"tarih": "gun"})
        dd["toplu"] = dd["gun"].isin(toplu_gun)
        # TAM pencere sarti
        dd["tam"] = dd["gun"] + pd.Timedelta(days=UFUK) <= TRAIN_SON
        dd = dd[dd["tam"]]

        # donusten onceki son 60 kaydin ofseti -> katman tanimi icin
        idx = tr.sort_values(["tanim", "tarih"]).set_index("tanim")
        onc = {}
        nkay = {}
        for t, g in dd.set_index("tanim")["gun"].items():
            s = idx.loc[[t]]
            s = s[s["tarih"] < g]
            nkay[t] = len(s)
            onc[t] = float(s.tail(60)["ofs"].mean()) if len(s) >= 30 else np.nan
        dd["son60_ofs"] = dd["tanim"].map(onc)
        dd["n_kayit"] = dd["tanim"].map(nkay)

        # pencere ortalamasi, gun-kontrolu cikarilmis
        W = {}
        NS = {}
        for t, g in dd.set_index("tanim")["gun"].items():
            s = idx.loc[[t]]
            s = s[(s["tarih"] >= g) & (s["tarih"] < g + pd.Timedelta(days=UFUK))]
            if len(s) < 25:
                continue
            duz = s["ofs"].to_numpy() - gun_kontrol.reindex(s["tarih"]).to_numpy()
            W[t] = float(np.mean(duz))
            NS[t] = len(s)
        dd = dd[dd["tanim"].isin(W)]
        dd["W"] = dd["tanim"].map(W)
        dd["n_satir"] = dd["tanim"].map(NS)

        blok: dict = {}

        def olc(alt: pd.DataFrame, ad: str) -> None:
            if len(alt) == 0:
                blok[ad] = {"trafo": 0}
                return
            x = alt["W"].to_numpy()
            blok[ad] = {
                "trafo": int(len(alt)),
                "satir": int(alt["n_satir"].sum()),
                "W_ortalama": round(float(x.mean()), 4),
                "W_medyan": round(float(np.median(x)), 4),
                "W_budanmis10": round(budanmis(x), 4),
                "W_std": round(float(x.std(ddof=1)), 4) if len(x) > 1 else None,
                "gunler": sorted({str(g.date()) for g in alt["gun"]})[:6],
            }

        olc(dd[dd["toplu"]], "TOPLU")
        olc(dd[~dd["toplu"]], "ORGANIK")

        # katman esli (T1/T2 tanimlariyla)
        if min_bosluk == 90:
            kat = dd[(dd["son60_ofs"] <= -2.0) & (dd["n_kayit"] >= 30)]
            etiket = "T1-benzeri (son60_ofs<=-2 & n>=30)"
        else:
            kat = dd[dd["son60_ofs"] <= 0.0]
            etiket = "T2-benzeri (son60_ofs<=0)"
        olc(kat[kat["toplu"]], f"{etiket} | TOPLU")
        olc(kat[~kat["toplu"]], f"{etiket} | ORGANIK")

        # FARK: delta_toplu - delta_organik
        fark = {}
        for suffix in ("", f" | {etiket}"):
            a = blok.get("TOPLU" if suffix == "" else f"{etiket} | TOPLU", {})
            b = blok.get("ORGANIK" if suffix == "" else f"{etiket} | ORGANIK", {})
            if a.get("trafo") and b.get("trafo"):
                fark["hepsi" if suffix == "" else etiket] = {
                    "ortalama": round(a["W_ortalama"] - b["W_ortalama"], 4),
                    "medyan": round(a["W_medyan"] - b["W_medyan"], 4),
                    "budanmis10": round(a["W_budanmis10"] - b["W_budanmis10"], 4),
                }
        rap[f"bosluk>={min_bosluk}"] = {
            "toplu_gunler_tam_pencereli": sorted({str(g.date()) for g in dd[dd["toplu"]]["gun"]}),
            "bloklar": blok,
            "FARK_toplu_eksi_organik": fark,
        }

    print(json.dumps(rap, ensure_ascii=False, indent=1))
    (CIK / "d10_toplu_delta.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: experiments/donuscu/d10_toplu_delta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
