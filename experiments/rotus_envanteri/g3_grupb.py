"""GOREV 3 -- "grup B kaldirmasi" yonunu SIZINTISIZ ileri pencerede yeniden olc.

KURULUM
-------
Kesme T icin:
  * yalancı-train = train satirlari, tarih < T            (ETIKET GORULMEZ)
  * ileri pencere W = [T, T+122)                          (GERCEK)
  * yalancı grup B = >=60 kayit + son 60 kaydin TAMAMI sifir
                     + son kayit tarihi < T-5 gun (raporu KESILMIS)
                     + W'de satiri var

CAPA (anchor): olumden ONCEKI son 60 POZITIF kaydin ortalama ofseti
               (log1p(tuketim) - log1p(guc)) -- yalnizca yalancı-train'den.

OLCUM: W'deki gercek log1p ile capa temelli tahmin arasindaki fark.
       delta* = ort(lp_gercek - lp_capa)  -> optimum ek kaldirma.
       Katsayi taramasi: delta in [-1.0 .. +1.0].

Ayrica GERCEK grup B (93 trafo) icin ayni capa hesaplanip v83'un nerede
durdugu olculur; ve mevsimsel ikiz (2025 Nis-Tem) kontrolu yapilir.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, hizala, test, train

KUYRUK_ESIK = 60
UFUK = 122
KESMELER = ("2025-06-30", "2025-08-31", "2025-11-30")
TARAMA = np.round(np.arange(-1.0, 1.001, 0.1), 3)


def capa_seviyesi(d: pd.DataFrame, trafolar: set[str]) -> pd.Series:
    """Olumden onceki son 60 POZITIF kaydin ortalama ofseti."""
    poz = d[(d["tanim"].isin(trafolar)) & (d["tuketim"] > 0)].sort_values(["tanim", "tarih"])
    son = poz.groupby("tanim").tail(60)
    g = son.groupby("tanim")["ofs"]
    sev = g.mean()
    n = g.size()
    return sev.where(n >= 10)


def yalanci_grup_b(
    tr: pd.DataFrame, T: pd.Timestamp
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    yt = tr[tr["tarih"] < T]
    W = tr[(tr["tarih"] >= T) & (tr["tarih"] < T + pd.Timedelta(days=UFUK))]
    d = yt.sort_values(["tanim", "tarih"])
    son_n = d.groupby("tanim").tail(KUYRUK_ESIK)
    olu = son_n.groupby("tanim")["tuketim"].apply(lambda s: bool((s <= 0).all()))
    yeterli = d.groupby("tanim").size() >= KUYRUK_ESIK
    son_tarih = d.groupby("tanim")["tarih"].max()
    kesilmis = son_tarih < T - pd.Timedelta(days=5)
    aday = set((olu & yeterli & kesilmis).pipe(lambda s: s.index[s]))
    B = aday & set(W["tanim"].unique())
    return B, yt, W


def main() -> int:
    tr, te = train(), test()
    tr["ofs"] = np.log1p(tr["tuketim"].clip(lower=0)) - np.log1p(tr["guc"])
    tr["lp"] = np.log1p(tr["tuketim"].clip(lower=0))
    rap: dict = {"kesmeler": {}}

    # ---------- A) UC KESMEDE ILERI PENCERE ----------
    for k in KESMELER:
        T = pd.Timestamp(k)
        B, yt, W = yalanci_grup_b(tr, T)
        if not B:
            rap["kesmeler"][k] = {"trafo": 0, "not": "yalancı grup B bos"}
            print(f"[{k}] yalancı grup B BOS")
            continue
        sev = capa_seviyesi(yt, B)
        sev = sev.dropna()
        B2 = set(sev.index)
        w = W[W["tanim"].isin(B2)].copy()
        if w.empty:
            rap["kesmeler"][k] = {"trafo": 0, "not": "W bos"}
            continue
        w["capa_ofs"] = w["tanim"].map(sev)
        w["lp_capa"] = w["capa_ofs"] + np.log1p(w["guc"])
        d = w["lp"] - w["lp_capa"]
        delta_yildiz = float(d.mean())
        taban_mse = float((d**2).mean())
        tarama = {}
        for dl in TARAMA:
            tarama[float(dl)] = float(((d - dl) ** 2).mean()) - taban_mse
        rap["kesmeler"][k] = {
            "trafo": len(B2),
            "satir": int(len(w)),
            "sifir_orani": float((w["tuketim"] <= 0).mean()),
            "gercek_ort_lp": float(w["lp"].mean()),
            "capa_ort_lp": float(w["lp_capa"].mean()),
            "delta_yildiz": delta_yildiz,
            "delta_yildiz_SH": float(d.std(ddof=1) / np.sqrt(w["tanim"].nunique())),
            "trafo_bazli_delta_ort": float(d.groupby(w["tanim"]).mean().mean()),
            "trafo_bazli_pozitif_oran": float((d.groupby(w["tanim"]).mean() > 0).mean()),
            "kazanc_optimumda": float(-(delta_yildiz**2)),
            "tarama": tarama,
        }
        r = rap["kesmeler"][k]
        print(
            f"[{k}] trafo={r['trafo']:4d} satir={r['satir']:6d} sifir%={100 * r['sifir_orani']:.1f} "
            f"gercek_lp={r['gercek_ort_lp']:.3f} capa_lp={r['capa_ort_lp']:.3f} "
            f"delta*={delta_yildiz:+.4f} (SH {r['delta_yildiz_SH']:.4f}) "
            f"trafo+oran={r['trafo_bazli_pozitif_oran']:.2f}"
        )

    # ---------- B) GERCEK GRUP B: v83 capaya gore NEREDE ----------
    b_tanim = set(
        (KOK / "experiments/rotus_envanteri/grup_b.txt").read_text(encoding="utf-8").split()
    )
    v83 = hizala("tuketim_v83_sicak_optimum.csv", te)
    te2 = te.copy()
    te2["lp_v83"] = np.log1p(np.clip(v83, 0, None))
    te2["ofs_v83"] = te2["lp_v83"] - np.log1p(te2["guc"])
    sev_b = capa_seviyesi(tr, b_tanim).dropna()
    sel = te2["tanim"].isin(b_tanim)
    tb = te2[sel].copy()
    tb["capa_ofs"] = tb["tanim"].map(sev_b)
    kapsanan = tb["capa_ofs"].notna()
    fark = (tb.loc[kapsanan, "capa_ofs"] - tb.loc[kapsanan, "ofs_v83"]).to_numpy()
    rap["gercek_grup_b"] = {
        "trafo": len(b_tanim),
        "satir": int(sel.sum()),
        "capasi_olan_trafo": int(tb.loc[kapsanan, "tanim"].nunique()),
        "capasi_olan_satir": int(kapsanan.sum()),
        "v83_ort_ofs": float(tb["ofs_v83"].mean()),
        "capa_ort_ofs": float(tb.loc[kapsanan, "capa_ofs"].mean()),
        "capa_eksi_v83_ort": float(fark.mean()),
        "capa_eksi_v83_medyan": float(np.median(fark)),
        "pozitif_satir_orani": float((fark > 0).mean()),
    }
    g = rap["gercek_grup_b"]
    print(
        f"[GERCEK B] v83_ofs={g['v83_ort_ofs']:.4f} capa_ofs={g['capa_ort_ofs']:.4f} "
        f"capa-v83={g['capa_eksi_v83_ort']:+.4f} (poz oran {g['pozitif_satir_orani']:.2f})"
    )

    # ---------- C) MEVSIMSEL IKIZ: 2025 Nis-Tem gercegi ----------
    ikiz = tr[
        (tr["tarih"] >= pd.Timestamp("2025-04-01"))
        & (tr["tarih"] <= pd.Timestamp("2025-07-31"))
        & (tr["tanim"].isin(b_tanim))
    ]
    rap["mevsimsel_ikiz_2025"] = {
        "satir": int(len(ikiz)),
        "trafo": int(ikiz["tanim"].nunique()),
        "sifir_orani": float((ikiz["tuketim"] <= 0).mean()),
        "gercek_ort_lp": float(ikiz["lp"].mean()),
        "gercek_ort_ofs": float(ikiz["ofs"].mean()),
        "v83_ort_lp": float(tb["lp_v83"].mean()),
        "v83_ort_ofs": float(tb["ofs_v83"].mean()),
    }
    i = rap["mevsimsel_ikiz_2025"]
    print(
        f"[IKIZ 2025 Nis-Tem] satir={i['satir']} sifir%={100 * i['sifir_orani']:.1f} "
        f"gercek_lp={i['gercek_ort_lp']:.4f} vs v83_lp={i['v83_ort_lp']:.4f} "
        f"gercek_ofs={i['gercek_ort_ofs']:.4f} vs v83_ofs={i['v83_ort_ofs']:.4f}"
    )

    # ---------- D) GRUP B'nin son kayit tarihleri ----------
    son = tr[tr["tanim"].isin(b_tanim)].groupby("tanim")["tarih"].max()
    rap["grup_b_son_kayit_dagilimi"] = {
        str(k.date()): int(v) for k, v in son.value_counts().head(12).items()
    }
    print("[GRUP B son kayit tarihleri]", rap["grup_b_son_kayit_dagilimi"])

    (KOK / "reports/g3_grupb.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: reports/g3_grupb.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
