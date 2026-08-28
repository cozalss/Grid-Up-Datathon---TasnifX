"""YON SEKILLENDIRME -- ayni adaydan hangi YON cikarilacagini secer.

NEDEN GEREKLI
-------------
Kazanc ``L^2/Q`` OLCEKTEN BAGIMSIZ: yonu 2 ile carpmak hicbir sey degistirmez.
Degistiren tek sey ACIDIR. Ham bir bagimsiz modelin ``aday - taban`` farki
buyuk bir ``Q`` tasir ama enerjisinin cogu MODELIN KENDI GURULTUSUDUR; o kisim
gercek artikla iliskisiz oldugu icin aciyi bozar.

Cozum: ayni adaydan gurultusu sonmus yonler cikarmak. Trafo bazli ortalama
almak (i) gunluk gurultuyu ``1/sqrt(gun)`` ile sondurur, (ii) modelin gercekten
farkli dedigi SEVIYE bilgisini birakir. Sekil yonu (trafo ortalamasi cikarilmis)
tam tersini yapar.

Her varyant hem CV'de (L, Q, kappa*, kazanc) hem test uzayinda (q_perp, q_yeni,
mevcut diklerle kosinus) olculur; secim CV kazancina gore yapilir.

Kaggle'a hicbir sey gondermez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import c_olc
import ortak

VARYANTLAR = (
    "ham",
    "sicak",
    "soguk",
    "trafo",
    "trafo_ay",
    "sekil",
    "hucre",
    "trafo_sicak",
    "trafo_soguk",
    "kirp010",
    "kirp020",
    "kirp030",
    "kirp050",
    "kirp030_trafo",
    "kirp030_sicak",
    "kirp030_soguk",
    "tanh030",
)


def _meta_cv() -> dict[str, pd.DataFrame]:
    e = ortak.egitim(["tanim", "tarih", "_blok", "soguk_mu", "ilce_key", "g_guc_kova"])
    cik = {}
    for b in ortak.BLOKLAR:
        d = e[e["_blok"].to_numpy() == b].reset_index(drop=True)
        cik[b] = pd.DataFrame(
            {
                "tanim": d["tanim"].astype(str).to_numpy(),
                "ay": d["tarih"].dt.month.to_numpy(),
                "soguk": d["soguk_mu"].to_numpy().astype(bool),
                "hucre": (
                    d["ilce_key"].fillna("NA").astype(str)
                    + "|"
                    + d["g_guc_kova"].fillna(-1).astype(int).astype(str)
                ).to_numpy(),
            }
        )
    return cik


def _meta_test() -> pd.DataFrame:
    t = ortak.test(["tanim", "tarih", "ilce_key", "g_guc_kova"])
    tr = pd.read_csv(ortak.KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
    gecmis = set(tr["tanim"].unique())
    return pd.DataFrame(
        {
            "tanim": t["tanim"].astype(str).to_numpy(),
            "ay": t["tarih"].dt.month.to_numpy(),
            "soguk": ~t["tanim"].astype(str).isin(gecmis).to_numpy(),
            "hucre": (
                t["ilce_key"].fillna("NA").astype(str)
                + "|"
                + t["g_guc_kova"].fillna(-1).astype(int).astype(str)
            ).to_numpy(),
        }
    )


def _grup_ort(d: np.ndarray, anahtar: np.ndarray) -> np.ndarray:
    s = pd.Series(d).groupby(anahtar).transform("mean")
    return s.to_numpy("float64")


def donustur(varyant: str, d: np.ndarray, m: pd.DataFrame) -> np.ndarray:
    tan = m["tanim"].to_numpy()
    sg = m["soguk"].to_numpy()
    if varyant == "ham":
        return d
    if varyant == "sicak":
        return np.where(sg, 0.0, d)
    if varyant == "soguk":
        return np.where(sg, d, 0.0)
    if varyant == "trafo":
        return _grup_ort(d, tan)
    if varyant == "trafo_ay":
        return _grup_ort(d, pd.Series(tan) + "|" + pd.Series(m["ay"].to_numpy()).astype(str))
    if varyant == "sekil":
        return d - _grup_ort(d, tan)
    if varyant == "hucre":
        return _grup_ort(d, m["hucre"].to_numpy())
    if varyant.startswith("kirp") and varyant[4:7].isdigit():
        c = int(varyant[4:7]) / 100.0
        k = np.clip(d, -c, c)
        if varyant.endswith("_trafo"):
            return _grup_ort(k, tan)
        if varyant.endswith("_sicak"):
            return np.where(sg, 0.0, k)
        if varyant.endswith("_soguk"):
            return np.where(sg, k, 0.0)
        return k
    if varyant == "tanh030":
        return 0.30 * np.tanh(d / 0.30)
    if varyant == "trafo_sicak":
        return np.where(sg, 0.0, _grup_ort(d, tan))
    if varyant == "trafo_soguk":
        return np.where(sg, _grup_ort(d, tan), 0.0)
    raise ValueError(varyant)


def main() -> None:
    adaylar = sorted(p.stem for p in ortak.ONB.glob("*.npz"))
    g = ortak.geo()
    tab = c_olc.uretim_tabani()
    mcv = _meta_cv()
    mte = _meta_test()
    v102 = g.v102

    sonuc = []
    for ad in adaylar:
        cv, tp = ortak.yukle_aday(ad)
        lgp = np.log1p(np.clip(tp, 0.0, None))
        d_test_ham = lgp - v102
        d_cv_ham = {b: np.log1p(np.clip(cv[b], 0.0, None)) - tab[b]["taban"] for b in cv}
        for v in VARYANTLAR:
            par = {}
            for b in ortak.BLOKLAR:
                dd = donustur(v, d_cv_ham[b], mcv[b])
                r = tab[b]["lgy"] - tab[b]["taban"]
                par[b] = (float(r @ dd), float(dd @ dd), len(dd))
            L = sum(p0 for p0, _, _ in par.values())
            Q = sum(p1 for _, p1, _ in par.values())
            n = sum(p2 for _, _, p2 in par.values())
            L /= n
            Q /= n
            if Q < 1e-9:
                continue
            kap = L / Q
            kaz = L * L / Q
            # DURUST TASIMA: kappa iki bloktan cozulur, ucuncude UYGULANIR.
            dis = []
            for disi in ortak.BLOKLAR:
                ic = [b for b in ortak.BLOKLAR if b != disi]
                Li = sum(par[b][0] for b in ic) / sum(par[b][2] for b in ic)
                Qi = sum(par[b][1] for b in ic) / sum(par[b][2] for b in ic)
                if Qi < 1e-12:
                    dis.append(0.0)
                    continue
                ki = Li / Qi
                Ld = par[disi][0] / par[disi][2]
                Qd = par[disi][1] / par[disi][2]
                dis.append(2.0 * ki * Ld - ki * ki * Qd)
            ut = donustur(v, d_test_ham, mte)
            geo_r = g.olc(f"{ad}|{v}", ut)
            sonuc.append(
                {
                    "ad": ad,
                    "varyant": v,
                    "cvL": L,
                    "cvQ": Q,
                    "kappa": kap,
                    "cv_kazanc": kaz,
                    "blok_disi_kazanc": float(np.mean(dis)),
                    "blok_disi_min": float(np.min(dis)),
                    "Q_test": geo_r["Q"],
                    "q_perp": geo_r["q_perp"],
                    "q_yeni": geo_r["q_yeni"],
                    "span_payi": geo_r["span_pay"],
                    "maks_kos_ad": geo_r["maks_kos_ad"],
                    "maks_kos": float(geo_r["maks_kos"]),
                    # LB'ye tasima: CV kappa ile test enerjisinde beklenen kazanc
                    "lb_kazanc_cvkappa": kap * kap * geo_r["q_perp"],
                    "lb_kazanc_f4115": 0.4115 * 0.4115 * geo_r["q_yeni"],
                }
            )
        print(f"{ad}: {len(VARYANTLAR)} varyant olculdu")

    sonuc.sort(key=lambda s: -s["blok_disi_kazanc"])
    _yaz(sonuc)
    json.dump(sonuc, open(ortak.CIK / "d_varyant.json", "w"), indent=2, default=float)


def _yaz(sonuc: list[dict]) -> None:
    print("\n" + "=" * 132)
    print("VARYANT TARAMASI -- BLOK-DISI tasima kazancina gore sirali")
    print("=" * 132)
    print(
        f"{'aday':22}{'varyant':13}{'cvL':>10}{'kappa*':>9}{'CVkaz':>9}{'blokdisi':>10}"
        f"{'bd_min':>9}{'q_perp':>9}{'q_yeni':>9}{'span%':>7}{'maks kos':>15}{'LB@f.41':>9}"
    )
    for s in sonuc:
        print(
            f"{s['ad'][:22]:22}{s['varyant']:13}{s['cvL']:>+10.5f}"
            f"{s['kappa']:>+9.3f}{s['cv_kazanc']:>9.5f}{s['blok_disi_kazanc']:>+10.5f}"
            f"{s['blok_disi_min']:>+9.5f}{s['q_perp']:>9.5f}"
            f"{s['q_yeni']:>9.5f}{100 * s['span_payi']:>7.1f}"
            f"{s['maks_kos_ad'] + ' ' + format(s['maks_kos'], '+.2f'):>15}"
            f"{s['lb_kazanc_f4115']:>9.5f}"
        )


if __name__ == "__main__":
    main()
