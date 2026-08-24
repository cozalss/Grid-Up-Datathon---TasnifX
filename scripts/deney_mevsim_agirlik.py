"""SICAK UZMAN: egitim satirlari HEDEF MEVSIME gore agirliklandirilmali mi?

NEDEN
-----
Test tamamen Nisan-Temmuz. Ek kokenler egitim etiketlerinin ay dagilimini
bozuyor: Nisan-Temmuz payi ana bloklarda ~%33 iken ek kokenlerle ~%21'e
duşuyor (olculdu, bagimsiz denetim). Bu kimsenin verdigi bir karar degil,
tasarimin yan etkisi. Ve trafoya ozgu yaz/kis orani p5 0,62 - p95 4,92
(8 kat) -- mevsim karisimi zararsiz bir gurultu degil.

Model takvim kolonlarina sahip, yani ay etkisini ogrenebilir. Ama L2 kaybi
altinda hangi aylarin hatasini kucultmeye calistigi ORNEK DAGILIMINA bagli.
Ortak degisken kaymasi (covariate shift) duzeltmesinin klasik kurgusu.

NASIL SINANIR -- genellenebilir bicimde
Hedef blogun aylari ile egitim satirinin ayi arasindaki DAIRESEL uzaklik
hesaplanir ve agirlik exp(-uzaklik/tau) verilir. tau kucukse yalnizca
mevsimsel olarak yakin aylar sayilir.

    tau = sonsuz   duz agirlik (bugunku uretim)
    tau = 4, 2, 1  giderek daha keskin mevsim odagi

Soru "Nisan-Temmuz'u yukselt" degil, "HEDEFE mevsimsel yakin aylari
yukselt" -- boylece uc blokta da sinanabilir ve hukum teste tasinir.

NOT: bu tezgahta ek kokenler YOK (deney onbellegi yalnizca uc ana blok).
Yani olculen sey mekanizmanin KENDISI; ek kokenlerin getirdigi ek carpiklik
uretimde daha buyuk olabilir. Mekanizma burada calismiyorsa uretimde de
calismaz.

    python scripts/deney_mevsim_agirlik.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SICAK_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
SICAK_MASKE = 0.15
TAULAR = (None, 4.0, 2.0, 1.0)
KAYIT = KOK / "experiments" / "mevsim_agirlik.jsonl"


def dairesel_uzaklik(ay: np.ndarray, hedef_aylar: np.ndarray) -> np.ndarray:
    """Bir ayin hedef ay kumesine en kucuk dairesel (12'lik) uzakligi."""
    d = np.abs(ay[:, None] - hedef_aylar[None, :])
    return np.minimum(d, 12 - d).min(axis=1).astype("float64")


def egit_tahmin_agirlikli(egitim, hedef, kolonlar, tohum, agirlik):  # noqa: ANN001, ANN201
    """``di.egit_tahmin``in kopyasi -- tek fark: keyfi ornek agirligi."""
    y = np.log1p(egitim[tm.HEDEF].clip(lower=0.0)) - np.log1p(egitim["guc"])
    model = di.aile_modeli("cat", tohum, **SICAK_USTYAZIM)
    x_e, x_h = egitim[kolonlar].copy(), hedef[kolonlar].copy()
    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
    for k in kat:
        x_e[k] = x_e[k].astype(str)
        x_h[k] = x_h[k].astype(str)
    model.fit(x_e, y, sample_weight=agirlik, cat_features=kat)
    return model.predict(x_h) + np.log1p(hedef["guc"]).to_numpy()


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SICAK UZMAN: hedef mevsime gore ornek agirligi")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    for b in tm.BLOKLAR:
        parca, dogrulama, _, _ = parcalar[b.ad]
        ha = np.unique(pd.to_datetime(dogrulama["tarih"]).dt.month.to_numpy())
        ea = pd.to_datetime(parca["tarih"]).dt.month.to_numpy()
        u = dairesel_uzaklik(ea, ha)
        print(
            f"  {b.ad}: hedef aylar {list(ha)} | egitim uzakligi "
            f"ort {u.mean():.2f} maks {u.max():.0f}"
        )

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for tau in TAULAR:
        ad = "DUZ (uretim)" if tau is None else f"tau={tau:.0f}"
        t_bas = time.time()
        tekil[ad] = {}
        blok = {}
        for b in tm.BLOKLAR:
            parca, dogrulama, gercek, soguk = parcalar[b.ad]
            hedef_aylar = np.unique(pd.to_datetime(dogrulama["tarih"]).dt.month.to_numpy())
            sicak = ~soguk
            loglar = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(parca, kol, SICAK_MASKE, tohum)
                if tau is None:
                    w = None
                else:
                    ay = pd.to_datetime(maskeli["tarih"]).dt.month.to_numpy()
                    w = np.exp(-dairesel_uzaklik(ay, hedef_aylar) / tau)
                    w = w / w.mean()  # olcegi sabit tut
                log_t = egit_tahmin_agirlikli(maskeli, dogrulama, kol, tohum, w)
                loglar.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sicak], tek[sicak])
            harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
            blok[b.ad] = tm.rmsle(gercek[sicak], harman[sicak])
        ort = float(np.mean(list(blok.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok.items())
        print(f"  {ad:14} {ort:.5f}   {detay}  ({time.time() - t_bas:.0f} sn)")

    kayitlar = []
    taban_ad = "DUZ (uretim)"
    for ad in tekil:
        if ad == taban_ad:
            continue
        f = np.array([tekil[taban_ad][k] - tekil[ad][k] for k in tekil[taban_ad]])
        o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = o / sh if sh > 0 else 0.0
        hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")
        print(f"\n  {ad}: ESLENIK FARK {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
        for b in tm.BLOKLAR:
            bb = np.array([tekil[taban_ad][(b.ad, t)] - tekil[ad][(b.ad, t)] for t in di.TOHUMLAR])
            print(f"     {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum kazanc)")
        kayitlar.append({"kol": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
