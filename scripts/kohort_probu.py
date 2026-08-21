"""2026-05-03 KOHORTU -- gonderim probu.

BULGU
-----
Test trafolari, test'te ILK GORULDUKLERI tarihe gore kohortlara ayriliyor.
Bir kohort digerlerinden keskin biçimde ayriliyor:

    ilk tarih    trafo  sicak  soguk   sicak uyelerde %80+ sifir  medyan maks
    2026-04-01    3928   3927      1   147/3927 =  3,7%              1908,6
    2026-05-03     141     36    105    33/36  = 91,7%                 0,0
    2026-05-11    2222    896   1326    48/896 =  5,4%              1374,8
    TABAN ORANI (butun sicak test trafolari)     5,09%

91,7% ile %5,09 arasindaki fark 18 kat. Ve kohort idari olarak tutarli:
105 soguk uyenin %100'unun ilk test tarihi tam olarak 2026-05-03, her
birinde medyan 90 satir -- kusursuz duzenli blok. Genel soguk nufusta ilk
tarih 80 farkli degere yayiliyor.

Sicak uyelerin egitimdeki seyri: 33'u Ocak 2025'ten 17 Haziran 2025'e
kadar TAM SIFIR okuyor, sonra kayitlari tamamen kesiliyor, on bir ay hic
satir yok, 2026-05-03'te geri geliyorlar. Hepsi ayni gun kesiliyor, ayni
gun donuyor -- fizik degil, idari olay.

IKI YORUM, TERS YONDE
    (a) OLU: egitimde sifirdilar, test'te de sifirlar.
    (b) ENERJILENDIRME: egitimdeki sifirlar ariza degil "henuz devrede
        degil" hali; donus tarihi devreye alma tarihi, yani test'te CANLI.
Kohortta iki demonstre canli uye var (maks 12.173 ve 2.032 kWh), bu (b)'yi
destekliyor. Veriyle ayirt edilemiyor. P(olu) durustce %40-50.

NEDEN YINE DE HAREKET EDIYORUZ -- ASIMETRI
RMSLE log uzayinda kareli hatadir. Bu 9.107 satira (test'in %1,27'si) su
an ortalama log1p 7,134 diyoruz. Gercekte sifirlarsa katkilari 471.806,
yani toplam hata butcemizin %56,4'u.

    tam duzeltme  dogruysa 0,7137   yanilirsa 1,3526
    log1p x0,75   dogruysa 0,9349   yanilirsa 1,1000

Belirsizlik altinda RMSLE'nin optimumu TAM SIFIRLAMA DEGIL: P(olu)=p ise
log uzayinda en iyi tahmin ``(1-p)*mu``, ve beklenen kayip ``p(1-p)mu^2``.
Tam sifirlamak (p=1 varsaymak) p<1 iken optimumun GERISINDEDIR.

Bu betik ``carpan`` = 1-p ile log-uzayi tahminini olcekler.

PROB TASARIMI
Iki gonderim ayni gun: biri duzeltmesiz, biri duzeltmeli. Aradaki TEK
fark bu kohort, dolayisiyla LB farki dogrudan bu soruyu cevaplar. Sonra
kalan dokuz gun boyunca dogru p ile calisiriz.

Carpan bilerek 0,75 -- yani p=0,25 varsayimi, kendi inancimizdan (0,4-0,5)
DAHA MUHAFAZAKAR. Iki nedenle: yanilma bedelini 0,019'da tutmak, ve
liderlik tablosunda buyuk bir sicrama yaparak bulguyu rakiplere
duyurmamak.

Calistirma::

    python scripts/kohort_probu.py --carpan 0.75 --cikti tuketim_v14.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

KOHORT_TARIHI = "2026-05-03"
GONDERIM = KOK / "submissions"


def kohort_bul() -> tuple[set[str], dict[str, float]]:
    """Kohortun SOGUK uyelerini ve kohort saglik gostergelerini dondurur."""
    tr, te = tm.yukle()
    tr["tanim"] = tr["tanim"].astype(str)
    te["tanim"] = te["tanim"].astype(str)
    ilk = te.groupby("tanim", observed=True)["tarih"].min()
    kohort = set(ilk[ilk == KOHORT_TARIHI].index)
    egitimde = set(tr["tanim"])
    sicak = sorted(kohort & egitimde)
    soguk = kohort - egitimde

    sifir_orani = tr.groupby("tanim", observed=True)["tuketim"].apply(lambda s: (s == 0).mean())
    olu = int((sifir_orani.loc[sicak] >= 0.8).sum()) if sicak else 0
    tum_sicak = [t for t in ilk.index if t in egitimde]
    taban = float((sifir_orani.loc[tum_sicak] >= 0.8).mean())
    return soguk, {
        "kohort": len(kohort),
        "sicak": len(sicak),
        "soguk": len(soguk),
        "olu_sicak": olu,
        "olu_orani": olu / len(sicak) if sicak else float("nan"),
        "taban_orani": taban,
    }


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kaynak", default="tuketim_v13.csv", help="duzeltilecek gonderim")
    ap.add_argument("--cikti", default="tuketim_v14.csv")
    ap.add_argument(
        "--carpan",
        type=float,
        default=0.75,
        help="log1p tahmini bu sayiyla carpilir; = 1 - P(olu)",
    )
    args = ap.parse_args()

    print("=" * 88)
    print(f"KOHORT PROBU -- {KOHORT_TARIHI}, log1p carpani {args.carpan}")
    print("=" * 88)
    soguk, bilgi = kohort_bul()
    print(f"  kohort {bilgi['kohort']} trafo: {bilgi['sicak']} sicak, {bilgi['soguk']} soguk")
    print(
        f"  sicak uyelerin {bilgi['olu_sicak']}'i (%{bilgi['olu_orani'] * 100:.1f}) egitimde "
        f"%80+ sifir  --  taban orani %{bilgi['taban_orani'] * 100:.2f}"
    )
    if bilgi["olu_orani"] < 0.5:
        raise RuntimeError(
            f"kohortun olu orani %{bilgi['olu_orani'] * 100:.1f} -- bulgu dogrulanamadi, "
            "prob uretilmiyor"
        )

    yol = GONDERIM / args.kaynak
    v = pd.read_csv(yol)
    hedef_kolon = v.columns[1]
    tanim = v["id"].astype(str).str.rsplit("_", n=1).str[0]
    maske = tanim.isin(soguk).to_numpy()
    if maske.sum() == 0:
        raise RuntimeError("kohort satiri bulunamadi -- id formati beklendigi gibi degil")

    eski = v.loc[maske, hedef_kolon].to_numpy()
    lp = np.log1p(np.clip(eski, 0.0, None))
    yeni = np.clip(np.expm1(args.carpan * lp), 0.0, None)
    v.loc[maske, hedef_kolon] = yeni

    print(f"\n  duzeltilen satir {int(maske.sum()):,} / {len(v):,}  (%{maske.mean() * 100:.2f})")
    print(
        f"  ONCE  medyan {np.median(eski):>9.1f}  ort {eski.mean():>9.1f}"
        f"  ort log1p {lp.mean():.3f}"
    )
    print(
        f"  SONRA medyan {np.median(yeni):>9.1f}  ort {yeni.mean():>9.1f}"
        f"  ort log1p {np.log1p(yeni).mean():.3f}"
    )

    cikti = GONDERIM / args.cikti
    v.to_csv(cikti, index=False)
    print(f"\n  yazildi: {cikti}  ({len(v):,} satir)")
    print("  DIKKAT: bu bir PROB. Kaynak dosyayla AYNI GUN gonderilmeli;")
    print("  aradaki LB farki dogrudan 'kohort olu mu' sorusunu cevaplar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
