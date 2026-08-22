"""TAKVIM GERI GELSIN MI -- uretim modeli bugunun pazar oldugunu BILMIYOR.

BULGU
-----
Uretimdeki 105 kolonda hafta gunu ve ay YOK::

    'hafta' iceren kolon: []
    'tk_'   iceren kolon: []

Yalin set (2026-08-21 gecesi) 39 kolonu BIR DEMET halinde atti ve o demetin
icinde 25 takvim kolonu vardi. Demetin toplam etkisi olculdu (-0,0095, esik
alti) ama HAFTA GUNUNUN TEK BASINA degeri hic olculmedi.

Gunluk elektrik tuketiminde hafta gunu birinci dereceden bir surucudur:
ticari ve sanayi yuku pazar gunleri duser. Model bunu bilmiyor.

Bagimsiz destek: dis veri denetimi (2026-08-22) pozitif cikan dort dis
veri ailesinin DORDUNUN de aslinda takvim olctugunu buldu --
``gunes_deklinasyon`` (saf yilin gunu), ``turizm_il_doluluk`` (il x gun
R^2 = 1,0000), ``yaprak_mevsimi`` (R^2 = 1,0000). Yani model takvim
eksenini arka kapidan ariyor.

NEDEN IKI REJIM BIRDEN
Soguk uzmani maske 1,00'da calisiyor ve elinde trafoyu ayirt eden HICBIR
sey yok; hafta gunu onun icin nadir bulunan gercek bir sinyal olabilir.
Sicak uzmani icin ise gecmis ozetleri zaten haftalik deseni tasiyor
(``t_hg_genligi``, ``t_hg_sapma``) -- ama o desenin BUGUNE uygulanmasi icin
bugunun hangi gun oldugunu bilmek gerekir.

Adaylar::

    TABAN            uretim 105 kolon
    +HAFTA           tk_haftanin_gunu, tk_hafta_sonu
    +HAFTA+TAKVIM    ustune tk_ay, tk_yilin_gunu, tatil_mi

Fit: 3 aday x 2 rejim x 3 blok x 3 tohum = 54 CatBoost ~ 45 dakika.

    python scripts/deney_takvim.py
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

HAFTA = ("tk_haftanin_gunu", "tk_hafta_sonu")
TAKVIM = ("tk_ay", "tk_yilin_gunu", "tatil_mi")

ADAYLAR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TABAN (105)", ()),
    ("+HAFTA (2)", HAFTA),
    ("+HAFTA+TAKVIM (5)", HAFTA + TAKVIM),
)

#: (ad, maske, CatBoost ustyazimi, soguk satirlarda mi skorlanacak)
REJIMLER: tuple[tuple[str, float, dict[str, object], bool], ...] = (
    ("SICAK", 0.15, {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}, False),
    ("SOGUK", 1.00, {"depth": 7}, True),
)

KAYIT = KOK / "experiments" / "takvim.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("TAKVIM -- hafta gunu ve ay geri gelsin mi")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    eksik = [k for k in HAFTA + TAKVIM if k not in tum]
    if eksik:
        raise RuntimeError(f"onbellekte olmayan takvim kolonu: {eksik}")
    tm.kategorik_kodla(egitim, test)
    print(f"  taban {len(taban)} kolon | eklenecekler mevcut")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    kayitlar = []
    for rejim, maske, ustyazim, soguk_tarafi in REJIMLER:
        print(f"\n  ===== {rejim} (maske {maske}) =====")
        maskeli = {}
        for b in tm.BLOKLAR:
            for tohum in di.TOHUMLAR:
                maskeli[(b.ad, tohum)] = d.soguk_maskele(
                    parcalar[b.ad][0], taban + list(HAFTA + TAKVIM), maske, tohum
                )
        tekil: dict[str, dict[tuple[str, int], float]] = {}
        for ad, ek in ADAYLAR:
            t0 = time.time()
            kol = taban + list(ek)
            tekil[ad] = {}
            blok_skor = {}
            for b in tm.BLOKLAR:
                _, dogrulama, gercek, soguk = parcalar[b.ad]
                secim = soguk if soguk_tarafi else ~soguk
                log_tahminler = []
                for tohum in di.TOHUMLAR:
                    log_t = di.egit_tahmin(
                        "cat", maskeli[(b.ad, tohum)], dogrulama, kol, tohum, **ustyazim
                    )
                    log_tahminler.append(log_t)
                    tek = np.clip(np.expm1(log_t), 0.0, None)
                    tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[secim], tek[secim])
                harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
                blok_skor[b.ad] = tm.rmsle(gercek[secim], harman[secim])
            ort = float(np.mean(list(blok_skor.values())))
            detay = "  ".join(f"{k} {v:.5f}" for k, v in blok_skor.items())
            print(f"    {ad:20} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

        taban_ad = ADAYLAR[0][0]
        for ad, _ in ADAYLAR[1:]:
            f = np.array([tekil[taban_ad][k] - tekil[ad][k] for k in tekil[taban_ad]])
            o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
            t_d = o / sh if sh > 0 else 0.0
            hukum = ("KAZANDIRIYOR" if o > 0 else "ZARAR VERIYOR") if abs(t_d) >= 2 else "esik alti"
            print(f"    {ad} vs TABAN: {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
            for b in tm.BLOKLAR:
                bb = np.array(
                    [tekil[taban_ad][(b.ad, t)] - tekil[ad][(b.ad, t)] for t in di.TOHUMLAR]
                )
                print(f"        {b.ad:6} {bb.mean():+.5f}")
            kayitlar.append(
                {"rejim": rejim, "aday": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum}
            )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
