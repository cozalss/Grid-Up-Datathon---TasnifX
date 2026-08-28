"""v108 URETIMI -- onarilmis olcutle gecen tek yon: TAHMIN SEVIYESI OFSETI.

03/04'te kapiyi uc bolmede de gecen yon "seviye desili ofseti" idi. Orada
desil ``t_log_ort``den kuruluyordu; TEST'te ``t_log_ort`` 455 gunluk pencereden,
kis26'da 334 gunlukten hesaplaniyor -- dagilimlar birebir ayni degil. Bu yuzden
burada desil MODELIN KENDI TAHMIN SEVIYESINDEN (``log1p`` uzayinda) ve SIRA
(yuzdelik) tabanli kurulur; boylece kuresel seviye kaymasindan ve pencere
uzunlugundan bagimsiz olur.

Once ayni ILERI kapisiyla (uc bolme + trafo-kumeli bootstrap) yeniden sinanir,
sonra ``v102`` uzerine uygulanip dosya yazilir. Kaggle'a HICBIR SEY gonderilmez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
KOK = BURA.parents[1]
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import bootstrap, delta_coz, hazirla, zincir  # noqa: E402
from ortak import SICAK_PAY, bloklari_kur  # noqa: E402

KOVA = 10
N0 = 200.0
KAT = 0.5
BOLMELER = {"B1": "2026-02-01", "B2": "2026-01-01", "B3": "2026-03-01"}


def sira_kovasi(p: np.ndarray, m: np.ndarray, k: int = KOVA) -> np.ndarray:
    """``m`` icindeki satirlari kendi SIRA yuzdeligine gore k kovaya boler."""
    out = np.full(len(p), -1, dtype="int64")
    idx = np.flatnonzero(m)
    sira = np.argsort(np.argsort(p[idx], kind="stable"), kind="stable")
    out[idx] = np.minimum((sira * k) // len(idx), k - 1)
    return out


def ofset_ogren(e: np.ndarray, kov: np.ndarray, k: int = KOVA, n0: float = N0) -> np.ndarray:
    """Buzulmus, agirlikli-merkezlenmis kova ofseti."""
    o = np.zeros(k, dtype="float64")
    n = np.zeros(k, dtype="float64")
    em = e - e.mean()
    for j in range(k):
        s = kov == j
        n[j] = s.sum()
        if n[j]:
            o[j] = em[s].mean() * n[j] / (n[j] + n0)
    return o - float((o * n).sum() / n.sum())


def main() -> int:
    bl = bloklari_kur()
    b = bl["kis26"]
    hazirla(b)
    r = zincir(b)
    tar = pd.to_datetime(b.cerceve["tarih"]).to_numpy()

    print("=" * 104)
    print("A) ILERI KAPI -- TAHMIN SEVIYESI (sira desili) ofseti, uc bolme")
    print("=" * 104)
    print(
        f"{'bolme':8}{'kat':>6}{'dMSE_SINA':>12}{'GAalt':>10}{'GAust':>10}"
        f"{'kazanan':>9}{'testdMSE':>10}  karar"
    )
    print("-" * 104)
    kayit = []
    for etiket, kes_s in BOLMELER.items():
        kes = np.datetime64(kes_s)
        m_og, m_si = tar < kes, tar >= kes
        r0_og = r + delta_coz(b, r, m_og)
        p_og = r0_og + b.lgc
        kov_og = sira_kovasi(p_og, m_og)
        e_og = (b.lgy - np.maximum(p_og, 0.0))[m_og]
        o = ofset_ogren(e_og, kov_og[m_og])

        r0_si = r + delta_coz(b, r, m_si)
        p_si = r0_si + b.lgc
        kov_si = sira_kovasi(p_si, m_si)
        d = np.zeros(b.n)
        d[m_si] = o[kov_si[m_si]]
        r0 = r0_si
        for kat in (1.0, 0.75, 0.5):
            rr = r + kat * d
            r1 = rr + delta_coz(b, rr, m_si)
            dm, lo, hi, kaz, _ = bootstrap(b, r0, r1, m_si, B=1000)
            karar = "GECTI" if (dm < 0 and hi < 0 and kaz >= 0.60) else "red"
            print(
                f"{etiket:8}{kat:>6.2f}{dm:>+12.5f}{lo:>+10.5f}{hi:>+10.5f}"
                f"{100 * kaz:>8.1f}%{dm * SICAK_PAY:>+10.5f}  {karar}"
            )
            kayit.append(
                {
                    "bolme": etiket,
                    "kat": kat,
                    "dMSE": dm,
                    "GA_alt": lo,
                    "GA_ust": hi,
                    "kazanan": kaz,
                    "testdMSE": dm * SICAK_PAY,
                    "gecti": karar == "GECTI",
                }
            )

    # --- TUM kis26'dan ogrenilen nihai ofset
    m_all = np.ones(b.n, dtype=bool)
    r0 = r + delta_coz(b, r, m_all)
    p = r0 + b.lgc
    kov = sira_kovasi(p, m_all)
    o = ofset_ogren((b.lgy - np.maximum(p, 0.0)), kov)
    print()
    print("=" * 104)
    print("B) NIHAI OFSET (tum kis26 sicak satirlari, n0=200, sira desili)")
    print("=" * 104)
    print(f"{'desil':>7}{'n':>10}{'ort tahmin kWh':>18}{'ofset':>10}{'kat*ofset':>11}")
    for j in range(KOVA):
        s = kov == j
        print(
            f"{j:>7}{int(s.sum()):>10,}{np.expm1(p[s]).mean():>18,.1f}"
            f"{o[j]:>+10.4f}{KAT * o[j]:>+11.4f}"
        )

    # --- TEST'e uygula
    t = pd.read_parquet(
        KOK / "data/interim/deney/test.parquet", columns=["id", "tanim", "tarih", "soguk_mu"]
    )
    v102 = pd.read_csv(KOK / "submissions/tuketim_v102_kappa_optimum.csv")
    if not (t["id"].astype(str).to_numpy() == v102["id"].astype(str).to_numpy()).all():
        raise RuntimeError("id hizasi bozuk")
    lg = np.log1p(v102["tuketim"].to_numpy("float64"))
    sicak = t["soguk_mu"].to_numpy() == 0
    kov_t = sira_kovasi(lg, sicak)
    d_t = np.zeros(len(lg))
    d_t[sicak] = KAT * o[kov_t[sicak]]
    yeni = np.expm1(np.maximum(lg + d_t, 0.0))
    yeni = np.clip(yeni, 0.0, None)

    cik = KOK / "submissions" / "tuketim_v108_sicak_onarim.csv"
    pd.DataFrame({"id": v102["id"], "tuketim": yeni}).to_csv(cik, index=False)
    print(f"\nyazildi: {cik}")

    u = np.log1p(yeni) - lg
    print(f"  uygulanan satir: {int((d_t != 0).sum()):,} (sicak {int(sicak.sum()):,})")
    print(f"  Q(v108 - v102) = |u|^2/n = {float(u @ u) / len(u):.7f}")
    print(f"  kirpma nedeniyle kaybolan enerji: {float((u - d_t) @ (u - d_t)) / len(u):.3e}")

    json.dump(
        {
            "ofset": o.tolist(),
            "kat": KAT,
            "kova": KOVA,
            "n0": N0,
            "Q_v102": float(u @ u) / len(u),
            "kapi": kayit,
        },
        open(BURA / "06_uret.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
