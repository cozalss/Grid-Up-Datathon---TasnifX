"""OFSET YONLERI -- yeni modellerin ACTIGI degiskenlere gore grup ofsetleri.

FIKIR
-----
``P1`` (ilce ofseti) ve ``P3`` (seviye desili) yonleri, CV artigini bir GRUP
degiskenine gore ortalayip test uzayina tasiyarak kuruldu. Prob kampanyasi bu
aileden; ama grup degiskenleri hep ESKI kanallardan geliyordu (ilce, kVA,
seviye, ay). Uretim modelinin gormedigi YENI bir degisken kullanilirsa, ayni
yontem SPAN DISI bir yon uretir.

Burada grup anahtari, yeni model siniflarinin uretimle ANLASMAZLIGI:

    d_C = log1p(aday C) - log1p(taban)

``d_C``nin desili "bu satirda klasik/farkli-hedefli model uretimden ne kadar ve
ne yonde ayriliyor" demektir. Uretimin hicbir ozelliginde bu yok.

Ham ``d_C`` yonunden farki: yonun DEGERI ``d_C`` degil, o kovadaki OLCULMUS CV
artik ortalamasidir. Modelin kendi gurultusu tamamen duser; geriye kova sayisi
kadar skaler kalir.

DURUSTLUK: kova kenarlari ve ofsetler YALNIZ CV'den (etiketli egitim verisi).
LB'den hicbir sey okunmuyor. Her yon icin serbest parametre = kova sayisi.
Ayrica her yon BLOK-DISI sinavdan gecer: ofsetler iki bloktan cozulur, kazanc
ucuncu blokta olculur. Tasimayan anahtar orada coker.

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
import d_varyant
import ortak

YON_DIZIN = ortak.CIK / "yon"
YON_DIZIN.mkdir(exist_ok=True)
KOVA = 10
BUZME = 200.0

#: (etiket, d donusumu, capraz eksen)
ANAHTARLAR = (
    ("ofs_d", "ham", None),
    ("ofs_dtrafo", "trafo", None),
    ("ofs_dsekil", "sekil", None),
    ("ofs_d_kova", "ham", "kova"),
    ("ofs_dtrafo_ay", "trafo", "ay"),
)


def _kenarlar(d: np.ndarray) -> np.ndarray:
    q = np.unique(np.quantile(d, np.linspace(0.0, 1.0, KOVA + 1)))
    q[0] = -np.inf
    return q[:-1]


def _kod(d: np.ndarray, m: pd.DataFrame, kenar: dict, capraz: str | None) -> np.ndarray:
    sg = m["soguk"].to_numpy()
    des = np.zeros(len(d), dtype="int64")
    des[~sg] = np.clip(np.searchsorted(kenar["sicak"], d[~sg], side="right") - 1, 0, KOVA - 1)
    des[sg] = np.clip(np.searchsorted(kenar["soguk"], d[sg], side="right") - 1, 0, KOVA - 1)
    kod = des + KOVA * sg.astype("int64")
    if capraz == "kova":
        kod = kod * 16 + m["kova"].to_numpy()
    elif capraz == "ay":
        kod = kod * 16 + m["ay"].to_numpy()
    return kod


def _ofsetler(kod: np.ndarray, r: np.ndarray) -> pd.Series:
    o = pd.Series(r).groupby(kod).mean()
    n = pd.Series(r).groupby(kod).size()
    return o * (n / (n + BUZME))


def _meta_ek(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["kova"] = [int(h.rsplit("|", 1)[1]) + 1 for h in m["hucre"]]
    return m


def main() -> None:
    adaylar = sorted(p.stem for p in ortak.ONB.glob("*.npz"))
    g = ortak.geo()
    tab = c_olc.uretim_tabani()
    mcv = {b: _meta_ek(v) for b, v in d_varyant._meta_cv().items()}
    mte = _meta_ek(d_varyant._meta_test())

    sonuc = []
    for ad in adaylar:
        cv, tp = ortak.yukle_aday(ad)
        d_te_ham = np.log1p(np.clip(tp, 0.0, None)) - g.v102
        d_cv_ham = {b: np.log1p(np.clip(cv[b], 0.0, None)) - tab[b]["taban"] for b in cv}
        for etiket, don, capraz in ANAHTARLAR:
            dcv = {b: d_varyant.donustur(don, d_cv_ham[b], mcv[b]) for b in ortak.BLOKLAR}
            dte = d_varyant.donustur(don, d_te_ham, mte)

            def kenar_kur(dd: dict, bloklar) -> dict:  # noqa: ANN001, ANN202
                h = np.concatenate([dd[b] for b in bloklar])
                s = np.concatenate([mcv[b]["soguk"].to_numpy() for b in bloklar])
                return {"sicak": _kenarlar(h[~s]), "soguk": _kenarlar(h[s])}

            kn = kenar_kur(dcv, ortak.BLOKLAR)
            kod = {b: _kod(dcv[b], mcv[b], kn, capraz) for b in ortak.BLOKLAR}
            rr = {b: tab[b]["lgy"] - tab[b]["taban"] for b in ortak.BLOKLAR}
            ofs = _ofsetler(
                np.concatenate([kod[b] for b in ortak.BLOKLAR]),
                np.concatenate([rr[b] for b in ortak.BLOKLAR]),
            )
            ycv = {
                b: pd.Series(kod[b]).map(ofs).fillna(0.0).to_numpy("float64") for b in ortak.BLOKLAR
            }
            Q = sum(float(ycv[b] @ ycv[b]) for b in ortak.BLOKLAR)
            L = sum(float(rr[b] @ ycv[b]) for b in ortak.BLOKLAR)
            n = sum(len(ycv[b]) for b in ortak.BLOKLAR)
            Q /= n
            L /= n
            if Q < 1e-12:
                continue

            # --- blok-disi sinav
            dis = []
            for disi in ortak.BLOKLAR:
                ic = [b for b in ortak.BLOKLAR if b != disi]
                kn2 = kenar_kur(dcv, ic)
                o2 = _ofsetler(
                    np.concatenate([_kod(dcv[b], mcv[b], kn2, capraz) for b in ic]),
                    np.concatenate([rr[b] for b in ic]),
                )
                yd = (
                    pd.Series(_kod(dcv[disi], mcv[disi], kn2, capraz))
                    .map(o2)
                    .fillna(0.0)
                    .to_numpy("float64")
                )
                Qd = float(yd @ yd) / len(yd)
                Ld = float(rr[disi] @ yd) / len(yd)
                # DURUST TASIMA OLCUTU: ic-katmandan gelen kappa (=1, ofsetler
                # zaten CV-optimum olcekte) disaridaki blokta UYGULANIR.
                # L^2/Q kullanmak yanlisti -- o hep >= 0 cikar, tasimayan yonu
                # de "kazanc" gosterir.
                dis.append(2.0 * Ld - Qd)

            yte = pd.Series(_kod(dte, mte, kn, capraz)).map(ofs).fillna(0.0).to_numpy("float64")
            r = g.olc(f"{ad}|{etiket}", yte)
            ismi = f"{ad}__{etiket}"
            np.savez_compressed(
                YON_DIZIN / f"{ismi}.npz",
                test=yte.astype("float32"),
                **{f"cv_{b}": ycv[b].astype("float32") for b in ortak.BLOKLAR},
            )
            sonuc.append(
                {
                    "ad": ismi,
                    "kova": int(len(ofs)),
                    "cvL": L,
                    "cvQ": Q,
                    "kappa": L / Q,
                    "cv_kazanc": L * L / Q,
                    "blok_disi_kazanc": float(np.mean(dis)),
                    "blok_disi": [float(x) for x in dis],
                    "blok_disi_min": float(np.min(dis)),
                    "q_perp": r["q_perp"],
                    "q_yeni": r["q_yeni"],
                    "span_payi": r["span_pay"],
                    "maks_kos_ad": r["maks_kos_ad"],
                    "maks_kos": float(r["maks_kos"]),
                }
            )
        print(f"{ad}: {len(ANAHTARLAR)} ofset yonu")

    sonuc.sort(key=lambda s: -s["blok_disi_kazanc"])
    print("\n" + "=" * 126)
    print("OFSET YONLERI -- grup anahtari YENI MODELIN URETIMLE ANLASMAZLIGI")
    print("=" * 126)
    print(
        f"{'yon':34}{'kova':>6}{'kappa':>8}{'CVkaz':>9}{'blokdisi':>10}{'bd_min':>9}"
        f"{'q_perp':>9}{'q_yeni':>9}{'span%':>7}  maks kos"
    )
    for s in sonuc:
        print(
            f"{s['ad'][:34]:34}{s['kova']:>6}{s['kappa']:>+8.3f}{s['cv_kazanc']:>9.5f}"
            f"{s['blok_disi_kazanc']:>+10.5f}{s['blok_disi_min']:>+9.5f}"
            f"{s['q_perp']:>9.5f}{s['q_yeni']:>9.5f}{100 * s['span_payi']:>7.1f}"
            f"  {s['maks_kos_ad']} {s['maks_kos']:+.2f}"
        )
    json.dump(sonuc, open(ortak.CIK / "f_ofset.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
