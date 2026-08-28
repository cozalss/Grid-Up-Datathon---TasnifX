"""SOGUK KOHORT -- IKINCI TUR ADAYLAR.

C10  genel 1-B kalibrasyon  g(r0) -> E[r_ger | r0], blok-disi (izotonik/kova)
C11  varlik deseni (blok icinde gozlenen gun sayisi) bazli seviye
C12  OLU (tamamen sifir) RISK SKORU: gozlenebilir ozniteliklerden blok-disi
     ogrenilen P(trafo tamamen sifir) ve ona gore asagi kaydirma
C13  sifir kutlesine karsi HEDGE: r' = log( (1-p) * expm1(r) ) benzeri
     kutle duzeltmesi (p sabit taranir)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaylar import kaba_kova, yazdir  # noqa: E402
from ortak import BLOKLAR, Blok, mse, taban_r, tum_bloklar  # noqa: E402

CIKTI = Path(__file__).resolve().parent / "adaylar2.jsonl"


def artik(b: Blok, r: np.ndarray) -> np.ndarray:
    return b.lgy - (r + b.lgc)


def gun_sayisi(b: Blok) -> np.ndarray:
    s = pd.Series(b.tanim)
    return s.map(s.value_counts()).to_numpy()


def c10_kalibrasyon(bloklar: dict[str, Blok]) -> list[dict]:
    """r0 uzerinde blok-disi, monoton olmayan kova kalibrasyonu."""
    out = []
    for nk in (5, 10, 20):
        satir = {"aday": f"C10 1-B kalibrasyon g(r0), {nk} kova (blok-disi)"}
        for ad in BLOKLAR:
            # blok-disi: r0'i kendi blogunun icinde yuzdeliklere cevir,
            # her yuzdelik icin E[r_ger] ogren
            hedef = np.zeros(nk)
            say = np.zeros(nk)
            for o in BLOKLAR:
                if o == ad:
                    continue
                bo = bloklar[o]
                r0o = taban_r(bo)
                q = np.clip((pd.Series(r0o).rank(pct=True) * nk).astype(int), 0, nk - 1).to_numpy()
                rg = bo.lgy - bo.lgc
                for i in range(nk):
                    m = q == i
                    if m.any():
                        hedef[i] += rg[m].mean()
                        say[i] += 1
            hedef = np.where(say > 0, hedef / np.maximum(say, 1), 0.0)
            b = bloklar[ad]
            r0 = taban_r(b)
            q = np.clip((pd.Series(r0).rank(pct=True) * nk).astype(int), 0, nk - 1).to_numpy()
            r1 = hedef[q]
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
        # yumusatilmis hali: taban ile kalibrasyonun harmani
        for w in (0.25, 0.50):
            satir = {"aday": f"C10 kalibrasyon {nk} kova, harman w={w:.2f}"}
            for ad in BLOKLAR:
                hedef = np.zeros(nk)
                say = np.zeros(nk)
                for o in BLOKLAR:
                    if o == ad:
                        continue
                    bo = bloklar[o]
                    r0o = taban_r(bo)
                    q = np.clip(
                        (pd.Series(r0o).rank(pct=True) * nk).astype(int), 0, nk - 1
                    ).to_numpy()
                    rg = bo.lgy - bo.lgc
                    for i in range(nk):
                        m = q == i
                        if m.any():
                            hedef[i] += rg[m].mean()
                            say[i] += 1
                hedef = np.where(say > 0, hedef / np.maximum(say, 1), 0.0)
                b = bloklar[ad]
                r0 = taban_r(b)
                q = np.clip((pd.Series(r0).rank(pct=True) * nk).astype(int), 0, nk - 1).to_numpy()
                r1 = (1 - w) * r0 + w * hedef[q]
                satir[ad] = mse(b, r0) - mse(b, r1)
            out.append(satir)
    return out


def c11_varlik(bloklar: dict[str, Blok]) -> list[dict]:
    """Gun sayisi kovasi bazli merkezli seviye (blok-disi)."""
    et = lambda g: np.asarray(["1-7", "8-21", "22-45", "46-75", "76+"])[  # noqa: E731
        np.digitize(g, [0, 8, 22, 46, 76]) - 1
    ]
    out = []
    for k in (50.0, 300.0):
        satir = {"aday": f"C11 gun sayisi kovasi (buzme k={k:.0f})"}
        for ad in BLOKLAR:
            birik: dict[str, list[float]] = {}
            for o in BLOKLAR:
                if o == ad:
                    continue
                bo = bloklar[o]
                e = artik(bo, taban_r(bo))
                d = pd.DataFrame({"g": et(gun_sayisi(bo)), "e": e})
                ort = d["e"].mean()
                t = d.groupby("g")["e"].agg(["mean", "size"])
                for gk, v in ((t["mean"] - ort) * (t["size"] / (t["size"] + k))).items():
                    birik.setdefault(str(gk), []).append(float(v))
            etki = {gk: float(np.mean(v)) for gk, v in birik.items()}
            b = bloklar[ad]
            r0 = taban_r(b)
            duz = np.array([etki.get(x, 0.0) for x in et(gun_sayisi(b))])
            satir[ad] = mse(b, r0) - mse(b, r0 + duz)
        out.append(satir)
    return out


def _olu_skor_ogren(bo: Blok) -> dict[str, float]:
    """Gozlenebilir hucrelerde P(trafo TAMAMEN SIFIR)."""
    d = pd.DataFrame(
        {
            "t": bo.tanim,
            "y": bo.y,
            "kova": kaba_kova(bo.guc),
            "ilce": bo.ilce,
        }
    )
    tr = d.groupby("t").agg(
        olu=("y", lambda s: float((s == 0).all())), kova=("kova", "first"), ilce=("ilce", "first")
    )
    tr["h"] = tr["kova"] + "|" + tr["ilce"]
    g = tr.groupby("h")["olu"].agg(["mean", "size"])
    taban = float(tr["olu"].mean())
    # ampirik-Bayes: k=20 trafo
    p = (g["mean"] * g["size"] + taban * 20) / (g["size"] + 20)
    return {str(i): float(v) for i, v in p.items()}, taban


def c12_olu_risk(bloklar: dict[str, Blok]) -> list[dict]:
    """Olu-risk skoruna gore asagi kaydirma; skor blok-disi ogrenilir."""
    out = []
    for kuvvet in (0.5, 1.0, 2.0):
        satir = {"aday": f"C12 olu-risk kaydirma kuvvet={kuvvet:.1f}"}
        for ad in BLOKLAR:
            birik: dict[str, list[float]] = {}
            tabanlar = []
            for o in BLOKLAR:
                if o == ad:
                    continue
                p, tb = _olu_skor_ogren(bloklar[o])
                tabanlar.append(tb)
                for hk, v in p.items():
                    birik.setdefault(hk, []).append(v)
            tb = float(np.mean(tabanlar))
            skor = {hk: float(np.mean(v)) for hk, v in birik.items()}
            b = bloklar[ad]
            h = np.char.add(np.char.add(kaba_kova(b.guc), "|"), b.ilce.astype(str))
            p = np.array([skor.get(x, tb) for x in h])
            r0 = taban_r(b)
            # merkezli: ortalama riskte kaydirma yok
            r1 = r0 + kuvvet * np.log1p(-np.clip(p, 0, 0.9)) - kuvvet * np.log1p(-tb)
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
    return out


def c13_kutle(bloklar: dict[str, Blok]) -> list[dict]:
    """Sabit sifir-kutlesi hedge'i: tahmin (1-p) ile carpilir (seviye uzayinda)."""
    out = []
    for p in (0.02, 0.05, 0.10, 0.20):
        satir = {"aday": f"C13 sifir kutlesi hedge p={p:.2f}"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            tah = np.expm1(r0 + b.lgc)
            r1 = np.log1p(np.clip((1 - p) * tah, 0, None)) - b.lgc
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
    return out


def main() -> int:
    bloklar = tum_bloklar()
    hepsi: list[dict] = []
    hepsi += yazdir("C10 -- GENEL 1-B KALIBRASYON g(r0)  (blok-disi)", c10_kalibrasyon(bloklar))
    hepsi += yazdir("C11 -- VARLIK DESENI (gun sayisi)", c11_varlik(bloklar))
    hepsi += yazdir("C12 -- OLU (tamamen sifir) RISK SKORU", c12_olu_risk(bloklar))
    hepsi += yazdir("C13 -- SIFIR KUTLESI HEDGE", c13_kutle(bloklar))
    with CIKTI.open("w", encoding="utf-8") as f:
        for s in hepsi:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nyazildi: {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
