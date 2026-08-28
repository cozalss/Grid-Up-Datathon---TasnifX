"""GRUP B -- KAPASITE-NORMALIZE (ofset) uzayinda uc kesme + katsayi taramasi.

ofs = log1p(tuketim) - log1p(guc).  Analog trafolarin kVA'si grup B'ninkinden
farkli oldugu icin ham log1p karsilastirmasi elmayla armut. Ofset uzayinda
karsilastirma dogru olanidir.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, hizala, test, train

KESMELER = ("2025-06-30", "2025-08-31", "2025-11-30")
UFUK = 122


def main() -> int:
    tr = train()
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    tr["ofs"] = tr["lp"] - np.log1p(tr["guc"])
    te = test()
    v83 = hizala("tuketim_v83_sicak_optimum.csv", te)
    B = set((KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split())
    sel = te["tanim"].isin(B).to_numpy()
    ofs83 = np.log1p(np.clip(v83[sel], 0, None)) - np.log1p(te.loc[sel, "guc"].to_numpy(float))
    v83_ofs = float(ofs83.mean())
    lp83 = float(np.log1p(np.clip(v83[sel], 0, None)).mean())
    lg = np.log1p(te.loc[sel, "guc"].to_numpy(float))
    print(f"GERCEK grup B: {int(sel.sum())} satir  v83 ort ofs={v83_ofs:+.4f}  ort lp={lp83:.4f}")

    rap: dict = {"v83_grupb_ofs": v83_ofs, "v83_grupb_lp": lp83, "kesmeler": {}}
    deltalar = []
    for k in KESMELER:
        T = pd.Timestamp(k)
        yt = tr[tr["tarih"] < T]
        W = tr[(tr["tarih"] >= T) & (tr["tarih"] < T + pd.Timedelta(days=UFUK))]
        gy = yt.groupby("tanim")
        aday = set(
            (
                (gy["tuketim"].max() <= 0)
                & (gy.size() >= 60)
                & (gy["tarih"].max() < T - pd.Timedelta(days=5))
            ).pipe(lambda s: s.index[s])
        )
        w = W[W["tanim"].isin(aday & set(W["tanim"].unique()))]
        if w.empty:
            continue
        gercek_ofs = float(w["ofs"].mean())
        d = gercek_ofs - v83_ofs
        # trafo bazli
        per = w.groupby("tanim")["ofs"].mean()
        deltalar.append(d)
        rap["kesmeler"][k] = {
            "trafo": int(w["tanim"].nunique()),
            "satir": int(len(w)),
            "sifir_orani": float((w["tuketim"] <= 0).mean()),
            "gercek_ort_ofs": gercek_ofs,
            "v83_ort_ofs": v83_ofs,
            "delta_yildiz": d,
            "trafo_bazli_pozitif": int((per > v83_ofs).sum()),
            "trafo_sayisi": int(len(per)),
            "trafo_bazli_delta_ort": float((per - v83_ofs).mean()),
            "trafo_bazli_delta_medyan": float((per - v83_ofs).median()),
        }
        r = rap["kesmeler"][k]
        print(
            f"[{k}] trafo={r['trafo']} satir={r['satir']:4d} sifir%={100 * r['sifir_orani']:.1f} "
            f"gercek_ofs={gercek_ofs:+.4f} v83_ofs={v83_ofs:+.4f} "
            f"delta*={d:+.4f} | trafo bazli: {r['trafo_bazli_pozitif']}/{r['trafo_sayisi']} pozitif "
            f"(ort {r['trafo_bazli_delta_ort']:+.3f} medyan {r['trafo_bazli_delta_medyan']:+.3f})"
        )

    rap["isaret_uc_kesmede_ayni"] = bool(
        len(deltalar) == 3 and (np.sign(deltalar) == np.sign(deltalar[0])).all()
    )
    rap["delta_min"] = float(min(deltalar)) if deltalar else None
    rap["delta_ort"] = float(np.mean(deltalar)) if deltalar else None
    print(
        f"\nISARET UC KESMEDE AYNI MI: {rap['isaret_uc_kesmede_ayni']}  "
        f"delta min={rap['delta_min']:+.4f} ort={rap['delta_ort']:+.4f}"
    )

    # ---- KATSAYI TARAMASI: v83'e ek delta, gercek grup B'de beklenen dMSE
    # senaryo: grup B satirlarinin p'si gercekten SIFIR, (1-p)'si ofs = v83_ofs + delta_ort
    n_b, N = int(sel.sum()), len(te)
    tarama = {}
    print("\n[TARAMA] v83'e ek delta d; dMSE tam 714.688 paydada.")
    print("   p = grup B satirlarinin GERCEKTEN SIFIR olan orani")
    basliklar = [0.0, 0.1, 0.2, 0.3, 0.5]
    print("    d     " + "  ".join(f"p={p:.1f}" for p in basliklar))
    for dl in np.round(np.arange(0.0, 1.01, 0.1), 2):
        satir = []
        for p in basliklar:
            # canli seviye = v83 ofs + delta_ort (uc kesmenin min'i, muhafazakar)
            L = v83_ofs + rap["delta_min"]
            m_yeni = (
                p * (v83_ofs + dl + lg.mean() - lg.mean() + lp83 - v83_ofs + dl - dl) ** 2
            )  # placeholder
            # dogrudan lp uzayinda: tahmin lp = lp83 + dl
            x = lp83 + dl
            Ltrue = lp83 + rap["delta_min"]
            m_yeni = p * x**2 + (1 - p) * (x - Ltrue) ** 2
            x0 = lp83
            m_eski = p * x0**2 + (1 - p) * (x0 - Ltrue) ** 2
            satir.append(n_b * (m_yeni - m_eski) / N)
        tarama[float(dl)] = [float(v) for v in satir]
        print(f"  {dl:+.1f}   " + "  ".join(f"{v:+.4f}" for v in satir))
    rap["tarama_p_senaryolari"] = {"p_degerleri": basliklar, "tablo": tarama}

    (KOK / "reports/g3g_ofset.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: reports/g3g_ofset.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
