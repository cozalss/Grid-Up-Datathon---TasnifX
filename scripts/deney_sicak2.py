"""SICAK TARAMA 2 -- EKSIK UYDURMA hipotezi, ESLENIK t testiyle.

BIRINCI TURUN SOYLEDIGI
-----------------------
Yedi aday tarandi (``deney_sicak.py``, 2026-08-22). Uc bagimsiz
degisiklik AYNI yone itti::

    l2_leaf_reg=1  (daha AZ duzenlileştirme)   +0,00153
    depth=6        (daha COK kapasite)          +0,00130
    bootstrap Bernoulli 0,8                     +0,00147

buna karsilik kapasiteyi KISAN iki aday net kaybetti::

    rsm=0,55       (kolon ornekleme daralt)    -0,00241
    langevin       (gradyana gurultu ekle)     -0,00228

Tek tek hepsi tohum gurultusunun (0,0105) altinda, yani biri bile kendi
basina delil degil. Ama DESEN delil: "kapasiteyi ac" diyen uc degisiklik
arti, "kapasiteyi kis" diyen iki degisiklik eksi. Sicak model eksik
uyduruyor olabilir.

Bu tur onu dogrudan sinar: en dogrudan kapasite kolu ``iterations``
(250'de duruyor, ``learning_rate`` 0,05).

OLCUM GUCU -- ESLENIK FARK
--------------------------
Birinci tur adaylarin MUTLAK skorlarinin yayilimini raporluyordu. Ama
butun adaylar ayni maskelenmis cerceveyi ve ayni ``random_seed``i
goruyor, yani aralarindaki fark ESLENIK. Eslenik farkin standart hatasi,
mutlak skorlarin yayilimindan cok daha kucuk -- ayni veriyle ayni deneyden
kat kat fazla ayirt etme gucu cikiyor.

Bu betik her adayi 9 hucrede (3 blok x 3 tohum) TABAN ile eslestirip
farkin ortalamasini, standart hatasini ve t degerini raporlar. Karar
kuralı ``|t| >= 2`` -- 8 serbestlik derecesinde kabaca %95.

Ayrica torbalanmis skor da veriliyor: uretim 3 tohumu log uzayinda
ortaliyor, yani gercekte kullanilan tahminci o.

Fit: 6 aday x 3 blok x 3 tohum = 54 CatBoost ~ 31 dakika (iki aday cift
iterasyonlu, onlar daha uzun).

    python scripts/deney_sicak2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SICAK_MASKE = 0.15
TABAN_USTYAZIM: dict[str, object] = {"random_strength": 4.0}

#: d(genel)/d(sicak) -- 0,01 sicak kazanci genel skora ne kadar geciyor.
SICAK_KALDIRAC = 0.590

#: Harman seyreltmesi: sicak tarafta cat tek 0,8128, harman 3/1/1 0,7979.
#: CatBoost'ta bulunan kazanc harmana kabaca bu oranda geciyor.
HARMAN_PAYI = 3.0 / 5.0

#: |t| >= 2 (8 serbestlik derecesi, ~%95) -- bunun altinda "olculemedi".
T_ESIGI = 2.0

ADAYLAR: tuple[tuple[str, dict[str, object]], ...] = (
    ("TABAN (uretim)", {}),
    ("l2=1", {"l2_leaf_reg": 1.0}),
    ("l2=1 + d6", {"l2_leaf_reg": 1.0, "depth": 6}),
    (
        "l2=1 + d6 + Bernoulli",
        {"l2_leaf_reg": 1.0, "depth": 6, "bootstrap_type": "Bernoulli", "subsample": 0.8},
    ),
    ("iterations=500", {"iterations": 500}),
    (
        "it=500 lr=0,035 l2=1 d6",
        {"iterations": 500, "learning_rate": 0.035, "l2_leaf_reg": 1.0, "depth": 6},
    ),
)

KAYIT = KOK / "experiments" / "sicak_tarama.jsonl"


def _hucre_skorlari(
    parcalar: dict, maskeli: dict, kolonlar: list[str], ustyazim: dict[str, object]
) -> tuple[dict[tuple[str, int], float], dict[str, float]]:
    """(blok, tohum) basina tekil skor ve blok basina TORBALANMIS skor."""
    tekil: dict[tuple[str, int], float] = {}
    torbali: dict[str, float] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = parcalar[b.ad]
        log_tahminler = []
        for tohum in di.TOHUMLAR:
            log_t = di.egit_tahmin(
                "cat", maskeli[(b.ad, tohum)], dogrulama, kolonlar, tohum, **ustyazim
            )
            log_tahminler.append(log_t)
            tek = np.clip(np.expm1(log_t), 0.0, None)
            tekil[(b.ad, tohum)] = tm.rmsle(gercek[~soguk], tek[~soguk])
        harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
        torbali[b.ad] = tm.rmsle(gercek[~soguk], harman[~soguk])
    return tekil, torbali


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 104)
    print("SICAK TARAMA 2 -- eksik uydurma hipotezi, ESLENIK fark testi")
    print("=" * 104)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    print(f"  {len(kolonlar)} kolon | taban {TABAN_USTYAZIM} | maske {SICAK_MASKE}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], kolonlar, SICAK_MASKE, tohum
            )

    baslik = f"{'aday':26} {'torbali':>8} {'eslenik fark':>13} {'SH':>8} {'t':>6} {'etki':>10}"
    print("\n" + baslik)
    print("-" * 104)

    taban_tekil: dict[tuple[str, int], float] = {}
    kayitlar = []
    for ad, ek in ADAYLAR:
        t0 = time.time()
        ustyazim = {**TABAN_USTYAZIM, **ek}
        tekil, torbali = _hucre_skorlari(parcalar, maskeli, kolonlar, ustyazim)
        torba_ort = float(np.mean(list(torbali.values())))

        if not taban_tekil:
            taban_tekil = tekil
            satir = f"{ad:26} {torba_ort:8.5f} {'TABAN':>13}"
            kayit_ek: dict[str, object] = {}
        else:
            farklar = np.array([taban_tekil[k] - tekil[k] for k in taban_tekil])
            ort = float(farklar.mean())
            sh = float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
            t_deger = ort / sh if sh > 0 else 0.0
            etki = ort * SICAK_KALDIRAC * HARMAN_PAYI
            hukum = "OLCULDU" if abs(t_deger) >= T_ESIGI else ""
            satir = (
                f"{ad:26} {torba_ort:8.5f} {ort:+13.5f} {sh:8.5f} "
                f"{t_deger:+6.2f} {etki:+11.5f}  {hukum}"
            )
            kayit_ek = {"eslenik_fark": ort, "sh": sh, "t": t_deger, "genel_etki": etki}

        detay = " ".join(f"{k} {v:.4f}" for k, v in torbali.items())
        print(f"{satir}\n{'':26} {detay}  ({time.time() - t0:.0f} sn)")
        kayitlar.append(
            {
                "tur": 2,
                "aday": ad,
                "ustyazim": {k: str(v) for k, v in ustyazim.items()},
                "torbali": torba_ort,
                "bloklar": torbali,
                **kayit_ek,
            }
        )

    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print("\n  'genel etki' = eslenik fark x 0,590 (kaldirac) x 0,60 (harman seyreltmesi)")
    print(f"  hukum icin |t| >= {T_ESIGI:.0f} gerekiyor -- bunun altindaki her sey olculememistir")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
