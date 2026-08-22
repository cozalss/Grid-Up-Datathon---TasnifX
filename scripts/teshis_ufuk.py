"""UFUK YANLILIGI UC BLOKTA -- tasinir mi, tasinmaz mi.

BULGU (teshis_sicak.py, yaz25)
------------------------------
Sicak hata ufukla tekduze yogunlasiyor ve yanlilik ISARET DEGISTIRIYOR::

    ufuk      satir%   HATA%   yogunluk   yanlilik
    1-30       25,0     11,7     0,47      +0,126
    31-60      24,9     17,4     0,70      +0,134
    61-90      24,5     27,9     1,14      -0,150
    90+        25,5     43,0     1,69      -0,374

Uzun ufukta log uzayinda 0,374 EKSIK tahmin -- yani 1,45 kat dusuk.

SORU
----
Bu bir UFUK etkisi mi (model uzaga bakarken ozet seviyesine geri cekiliyor)
yoksa MEVSIM etkisi mi (yaz25'te ufuk 90+ = Temmuz, ve model yazi eksik
tahmin ediyor)?

Ayrimin karari belirleyici::

    guz25   ozet 31 Tem'de biter -> ufuk 90+ = Ekim/Kasim  (sonbahar)
    kis26   ozet 30 Kas'da biter -> ufuk 90+ = Subat/Mart  (kis sonu)
    yaz25   ozet 31 Mar'ta biter -> ufuk 90+ = Temmuz      (yaz)
    TEST    ozet 31 Mar'ta biter -> ufuk 90+ = Temmuz      (yaz)

Uc blokta da ayni yonde ve benzer buyuklukteyse UFUK etkisidir ve
duzeltilebilir. Yalnizca yaz25'te varsa MEVSIM etkisidir -- o zaman da
test'e tasinir (yaz25 test'in mevsimsel ikizi) ama delil tek bloga
dayanir, yani cok daha riskli.

DUZELTMENIN DEGERI -- ONCEDEN HESAPLANIR
----------------------------------------
Kareli hatada, bir kovadaki ortalama yanliligi b ise, o kovayi b kadar
kaydirmak MSE'yi tam olarak b^2 azaltir. Betik her blok icin duzeltmenin
getirisini bu ozdeslikle onceden hesaplar; ayrica CAPRAZ dogrulama yapar:
BIR blokta olculen kaymalar DIGER bloklara uygulaninca ne oluyor. Kendi
blogunda kazandirip digerlerinde kaybettiren bir duzeltme, uydurmadir.

    python scripts/teshis_ufuk.py
"""

from __future__ import annotations

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

TOHUM = 1000
SICAK_MASKE = 0.15
USTYAZIM: dict[str, object] = {"random_strength": 4.0}

KAYNAK_BASLIK = "kaynak -> hedef"

KENAR = [0, 15, 30, 45, 60, 75, 90, 105, 200]
ETIKET = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90", "91-105", "106+"]


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("UFUK YANLILIGI -- uc blokta, sicak satirlarda")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    # blok -> (ufuk kovasi, log artik) ; artik = ln_gercek - ln_tahmin
    veri: dict[str, pd.DataFrame] = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, TOHUM)
        log_t = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, TOHUM, **USTYAZIM)
        sic = ~soguk
        ln_t = np.log1p(np.clip(np.expm1(log_t), 0.0, None))[sic]
        ln_y = np.log1p(gercek[sic])
        veri[b.ad] = pd.DataFrame(
            {
                "kova": pd.cut(
                    dogrulama.loc[sic, "ufuk_gun"].to_numpy(),
                    bins=KENAR,
                    labels=ETIKET,
                    include_lowest=True,
                ),
                "artik": ln_y - ln_t,  # + = model EKSIK tahmin etmis
            }
        )
        tarih = dogrulama.loc[sic, "tarih"]
        aralik = f"{tarih.min().date()}..{tarih.max().date()}"
        print(f"  {b.ad:6} sicak {sic.sum():>7,} satir | etiket {aralik}")

    print("\n  ORTALAMA ARTIK (+ = model EKSIK tahmin ediyor, duzeltme icin EKLENECEK kayma)")
    print(f"  {'ufuk':>8} " + " ".join(f"{b.ad:>10}" for b in tm.BLOKLAR) + f" {'YON':>6}")
    kaymalar: dict[str, dict[str, float]] = {b.ad: {} for b in tm.BLOKLAR}
    for et in ETIKET:
        satir = []
        for b in tm.BLOKLAR:
            alt = veri[b.ad][veri[b.ad]["kova"] == et]["artik"]
            v = float(alt.mean()) if len(alt) else float("nan")
            kaymalar[b.ad][et] = v
            satir.append(v)
        gecerli = [s for s in satir if not np.isnan(s)]
        yon = "AYNI" if len({np.sign(s) for s in gecerli}) == 1 else "farkli"
        print(f"  {et:>8} " + " ".join(f"{s:>+10.4f}" for s in satir) + f" {yon:>6}")

    print("\n  GENEL ORTALAMA ARTIK (kovasiz, tek sabit kayma)")
    for b in tm.BLOKLAR:
        print(f"    {b.ad:6} {veri[b.ad]['artik'].mean():+.4f}")

    print("\n  CAPRAZ DOGRULAMA -- 'A blogunda olculen kaymayi B'ye uygula'")
    print("  Deger: RMSLE degisimi (negatif = IYILESTI). Kosegen kendi blogu (uydurma).")
    print(f"  {KAYNAK_BASLIK:>14} " + " ".join(f"{b.ad:>10}" for b in tm.BLOKLAR))
    for kaynak in tm.BLOKLAR:
        satir = []
        for hedef in tm.BLOKLAR:
            df = veri[hedef.ad]
            duzeltme = df["kova"].map(kaymalar[kaynak.ad]).astype("float64").fillna(0.0)
            once = float(np.sqrt((df["artik"] ** 2).mean()))
            sonra = float(np.sqrt(((df["artik"] - duzeltme) ** 2).mean()))
            satir.append(sonra - once)
        print(f"  {kaynak.ad:>14} " + " ".join(f"{s:>+10.5f}" for s in satir))

    print("\n  SABIT KAYMA ile CAPRAZ DOGRULAMA (tek parametre, cok daha guvenli)")
    print(f"  {KAYNAK_BASLIK:>14} " + " ".join(f"{b.ad:>10}" for b in tm.BLOKLAR))
    for kaynak in tm.BLOKLAR:
        c = float(veri[kaynak.ad]["artik"].mean())
        satir = []
        for hedef in tm.BLOKLAR:
            a = veri[hedef.ad]["artik"]
            satir.append(float(np.sqrt(((a - c) ** 2).mean()) - np.sqrt((a**2).mean())))
        print(f"  {kaynak.ad:>14} " + " ".join(f"{s:>+10.5f}" for s in satir))

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
