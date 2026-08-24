"""SON ISLEM (gun korumali) -- !!! LB'DE CURUTULDU, URETIMDE KULLANILMIYOR !!!

=====================================================================
HUKUM: BU DOSYA KULLANILMAZ. Uretim ``son_islem.py`` (beta=0,60) kullanir.
=====================================================================

Bu betik 23-24 Agustos gecesi kis26 uzerinde titizlikle olculerek yazildi ve
o blokta gercekten daha iyiydi: soguk RMSLE 1,83979 -> 1,82133. LB ise tam
tersini soyledi.

IZOLE OLCUM (24 Agustos, ayni model, tek fark son islem):

    v30 = v27 modeli (3 tohum) + son_islem.py      -> LB 1,02639
    v44 = v27 modeli (3 tohum) + son_islem_gun.py  -> LB 1,03053   +0,00414

Ve dogrulayan ucuncu nokta:

    v46 = 15 tohum + son_islem_gun.py  -> 1,02448
    v47 = 15 tohum + son_islem.py      -> 1,01750   BIRINCILIK

Yani buradaki her "iyilestirme" -- gun ekseni korumasi, ilce x kova hucre
etkisi, hucre/model agirlik ayrimi, seyrek gun kapisi, tablo penceresi --
kis26'da kazandirip TESTTE kaybettirdi. Toplam maliyet ~0,004.

NEDEN OLCUM YANILTTI
--------------------
kis26 soguk dilimi test soguk dilimine benzemiyor:
  kVA medyani     kis26 400   TEST 630
  kova 12 payi    kis26 %4,4  TEST %16,6
Ustelik ayni hucrede soguk-sicak ofset farkinin ISARETI mevsime gore
degisiyor (yaz25 -0,169 / guz25 +0,332 / kis26 +0,184) ve test yaz
tarafinda. kis26'da olculen her soguk yapisal duzeltme bu iki kayma
yuzunden test'e tasinmiyor.

Buna karsilik ayni gece tohum ortalamasi TAHMINCININ KENDI VARYANSINDAN
etiketsiz olculdu ve ongordugu kadar geldi (bkz. tohum_gurultusu.py).
Ders: model-disi bir nicelikten turetilen kestirim tasindi, tek bir
dogrulama blogundan turetilen kestirim tasinmadi.

DOSYA NEDEN SILINMEDI
---------------------
Curutulmus ama BELGELENMIS bir denemedir; icindeki olcum altyapisi
(butunluk_son_islem.py ile birlikte) ve cebir baskalarinin ayni yola
girmesini onlemek icin duruyor. Yeniden denenecekse once yukaridaki iki
kaymanin (kVA karisimi, mevsimsel isaret) nasil kapatildigi gosterilmeli.

--- ASAGISI ORIJINAL BELGE, TARIHSEL KAYIT ICIN DURUYOR ---

SON ISLEM (gun korumali): soguk tahminlere ZAMAN EKSENI BOZULMADAN buzme.

``son_islem.py``nin iki kusurunu birden kapatir. Yeniden egitim YOK.

KUSUR 1 -- ZAMAN EKSENI EZILIYORDU
----------------------------------
Eski kurgu ofsetin TAMAMINI tek bir sabite (tahminin kendi genel ortalamasi)
dogru cekiyordu. Ofsetin iki ekseni var ve buzme ikisini ayirmiyordu:

    trafo ekseni   -> asiri yayilmis, BUZULMELI
    zaman ekseni   -> gercek mevsim rampasi, KORUNMALI

Beta ``kis26`` (Ara-Mar) uzerinde ayarlandi ve orada ay ekseni varyansi
0,00113 -- kis ofseti duz oldugu icin zaman eksenini ezmek BEDAVA gorundu.
Test penceresinin (Nis-Tem) mevsimsel ikizinde ay ekseni varyansi 0,15298:

    **test penceresinde zaman ekseni 136 KAT guclu.**

Olculdu (gonderim dosyalari uzerinde, 158.369 soguk satir):

    ay   v27 ofset   v30 ofset   2025 ayni ay gercek
    04     +0,1219     +0,2611          +0,0408
    05     +0,1169     +0,2581          +0,0056
    06     +0,4138     +0,4363          +0,4706
    07     +0,7642     +0,6465          +0,9517

    gun-ortalamalarinin std'si  0,3003 -> 0,1802   (tam olarak x0,60 = beta)

v27'nin May->Tem rampasi zaten gercegin altindaydi (0,647 vs 0,946); buzme
onu 0,389'a indirip gercekten UZAKLASTIRDI.

Bu betik gun ortalamasini AYNEN korur: buzme yalnizca gun ICINDEKI trafolar
arasi yayilmaya uygulanir.

KUSUR 2 -- HEDEF COK KABA
-------------------------
Genel ortalama, bir tahminin cekilebilecegi en kaba hedef; trafonun ilcesini
ve kVA kademesini yok sayar. Efron-Morris (1975) James-Stein'i tam bu yonde
genellestirir: hedef ne kadar bilgiliyse buzmenin yanlilik maliyeti o kadar
kucuktur. Hedefe egitimden turetilen bir HUCRE ETKISI eklenir.

Hucre yapisi ``deney_taban_ince.py`` ile UC BLOKTA tarandi. Kazanan: ilce x
kova, EBEVEYN ``ilce`` (``kova`` degil -- ilce tek basina kovadan iyi, seyrek
hucre oraya dusmeli), 24 kova, M ~ 2000.

KURGU
-----
    r        = log1p(tahmin) - log1p(guc)
    gun_ort  = o GUNUN soguk satirlarindaki r ortalamasi   (modelden: model
               tarihi ve trendi biliyor)
    hucre    = ampirik-Bayes ilce x kova ofset ortalamasi  (egitimden)
    etki     = hucre - o gunun soguk satirlarindaki hucre ortalamasi
    taban    = gun_ort + etki
    r'       = taban + beta * (r - taban)

Gun ortalamasi TAM olarak korunur (etki gun icinde merkezlendigi icin).

OLCULDU (2026-08-23, ``deney_soguk_taban.py``, kis26 soguk, 3 tohum):

    URETIM (v30: kendi ortalamasi, beta=0,60)        1,83979
    GUN korumali, hucresiz,        beta=0,30         1,83438
    GUN + ilce x kova M=2000,      beta=0,25         1,83114   <- SECILEN
    GUN + ilce x kova M=5000,      beta=0,25         1,83079
    (kis26 ust siniri: OLS ile uydurulmus afin kalibrasyon 1,80850)

kis26'da gun korumasi neredeyse BEDAVA (kis ofseti duz) -- kazanci testte
ortaya cikar ve orada olculemez. Dogru yapisal duzeltmenin imzasi budur.

SOGUK TANIMI: test trafosunun ``tanim`` kodu ``train.csv``de HIC gecmiyor.

    python scripts/son_islem_gun.py --giris submissions/tuketim_v27_v18hedge.csv \
        --cikis submissions/tuketim_v31.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]

#: HUCRE tablosunun agirligi. Cebir: gun ortalamasi korundugu icin
#:     r' = gun + a*etki + b*(r - gun)
#: yani a EGITIMDEN gelen tabloya, b MODELIN kendi gun-ici sinyaline gider.
#: Eskiden ikisi tek bir beta ile bagliydi (a = 1-beta) ve bunun turetilmis
#: hicbir dayanagi yoktu. Iki bagimsiz olcum ayni yeri gosterdi:
#:   trafo-bazli holdout (2025-04..07 mevsimsel ikizi)   a* = 0,545
#:   kis26 izgarasi (deney_ikili_agirlik.py)             a* = 0,55
#: kis26'da a=0,75 -> 1,82250, a=0,55 -> 1,82131  (kazanc 0,00120).
#:
#: SONRA DAHA DA DUSURULDU (deney_soguk_seviye.py). kis26'nin HAM optimumu
#: 0,47 ama kis26 soguk kVA medyani 400, TESTinki 630 -- karisim testinkine
#: agirliklandirilinca optimum 0,40'a kayiyor:
#:                     a=0,30    0,40      0,47      0,55
#:     kis26 HAM      1,82198  1,82133   1,82117   1,82127
#:     TEST kVA kar.  2,01055  2,01044   2,01063   2,01111
#: Bagimsiz bir mevsim analizi de Nisan-Temmuz icin a* = 0,39-0,41 olcmus
#: (a* mevsime gore oynuyor: Nis-Tem 0,412, Ara-Mar 0,829). Test Nis-Tem.
#: Iki test-ilgili sinyal ayni yerde; kis26 HAM maliyeti yalnizca 0,00006.
A_HUCRE = 0.40

#: MODELIN gun-ici sinyalinin agirligi. kis26'da 0,20-0,30 arasi duz.
B_MODEL = 0.25

#: Gun ortalamasi AYLIK ortalamaya dogru ampirik-Bayes ile buzulur:
#: n soguk satirlik bir gun n/(n+M_GUN) agirlik alir. Neden gerekli --
#: testte 2026-04-01'de yalnizca 1, 2026-04-08'de 10 soguk satir var ve
#: gun ortalamasi o satirlarin KENDISINDEN hesaplaniyordu: islem seyrek
#: gunlerde tersine donuyor (n=2-5 bandinda yayilma 3,16 KAT artiyor),
#: n=1'de tam no-op oluyordu. Aylik seviye korundugu icin mevsim rampasi
#: bozulmaz; yalnizca ay ICINDEKI gun gurultusu yumusar.
#: kis26'da maliyeti 0,00007 (orada en seyrek gun 14 satir).
M_GUN = 50.0

#: HUCRE TABLOSU KAYNAGININ BASLANGICI. Tablo train.csv'nin TAMAMINDAN
#: kurulmamali: en eski aylar farkli bir rejimden geliyor. Trafo nufusu
#: donem boyunca neredeyse ikiye katlaniyor (gunluk 2.065 -> 3.896 satir),
#: yani ilk aylar hem seyrek hem baska bir populasyon.
#:
#: Modelden BAGIMSIZ olculdu (deney_tablo_pencere2.py) -- tablo etkisinin
#: gun ici gercek sapmayi acikladigi R^2 %:
#:
#:     pencere            kis26    guz25
#:     tumu               1,043    0,457    <- eski uretim, IKISINDE DE EN KOTU
#:     en eski 2 ay atik  1,167    0,508    <- ikisinde de daha iyi
#:     yalniz son 3 ay    0,823    0,578
#:
#: Iki blok pencerenin NE KADAR kisalacaginda ayrisiyor (kis26 uzun, guz25
#: kisa istiyor) ama "tamamini kullanma" hukmunde BIRLESIYOR. Ortak kural
#: secildi: veri setinin ilk 2 ayini at. kis26 RMSLE'sinde 1,82233 ->
#: 1,82127 (kazanc 0,00106); guz25'te de R^2 yukseliyor.
TABLO_BASLANGIC = "2025-03-01"

#: kVA kademesi sayisi. Kenarlar YALNIZ egitimden turetilir ve teste AYNI
#: kenarlarla uygulanir -- iki tarafta ayri hesaplamak kovalari kaydirir.
KOVA_SAYISI = 24

#: Ampirik-Bayes onsel agirligi: n satirlik hucre n/(n+M) agirlik alir.
#: Ana etkiler (ilce, kova) icin hafif, hucre icin guclu duzlestirme.
M_ANA = 200.0
M_HUCRE = 2000.0


def _kova(guc: np.ndarray, kenar: np.ndarray) -> np.ndarray:
    lg = np.log1p(np.clip(guc, 0.0, None))
    return np.clip(np.searchsorted(kenar, lg, side="right") - 1, 0, KOVA_SAYISI - 1)


def _eb(anahtar_e: np.ndarray, ofs_e: np.ndarray, anahtar_h: np.ndarray,
        ebeveyn: np.ndarray, m_once: float) -> np.ndarray:
    """Ampirik-Bayes hucre ortalamasi: (hucre_toplami + M*ebeveyn) / (n + M)."""
    s = pd.Series(ofs_e).groupby(anahtar_e).agg(["sum", "count"])
    top = np.nan_to_num(pd.Series(s["sum"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    n = np.nan_to_num(pd.Series(s["count"]).reindex(anahtar_h).to_numpy(dtype="float64"), nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def hucre_etkisi(tr: pd.DataFrame, hedef: pd.DataFrame) -> np.ndarray:
    """Egitimden ilce x kova ofset ortalamasi (ham, merkezlenmemis).

    ``tr`` cagiran tarafta ZATEN ``TABLO_BASLANGIC``a gore kirpilmis gelmeli.
    """
    ofs = (np.log1p(tr["tuketim"].clip(lower=0.0).to_numpy(dtype="float64"))
           - np.log1p(tr["guc"].to_numpy(dtype="float64")))
    lg_e = np.log1p(tr["guc"].to_numpy(dtype="float64"))
    kenar = np.linspace(float(lg_e.min()), float(lg_e.max()) + 1e-9, KOVA_SAYISI + 1)
    kv_e = _kova(tr["guc"].to_numpy(dtype="float64"), kenar)
    kv_h = _kova(hedef["guc"].to_numpy(dtype="float64"), kenar)
    il_e = tr["lokasyon"].astype(str).to_numpy()
    il_h = hedef["lokasyon"].astype(str).to_numpy()

    genel = np.full(len(hedef), float(ofs.mean()))
    ilce = _eb(il_e, ofs, il_h, genel, M_ANA)
    anahtar_e = pd.Series(il_e).to_numpy() + "|" + pd.Series(kv_e).astype(str).to_numpy()
    anahtar_h = pd.Series(il_h).to_numpy() + "|" + pd.Series(kv_h).astype(str).to_numpy()
    return _eb(anahtar_e, ofs, anahtar_h, ilce, M_HUCRE)


def main() -> int:
    a = argparse.ArgumentParser(description="soguk buzme -- gun ekseni korumali")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--a-hucre", type=float, default=A_HUCRE)
    a.add_argument("--b-model", type=float, default=B_MODEL)
    a.add_argument("--m-gun", type=float, default=M_GUN)
    a.add_argument("--tablo-baslangic", default=TABLO_BASLANGIC)
    a.add_argument("--hucresiz", action="store_true",
                   help="hucre etkisini kapat (yalniz gun korumasi)")
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["id", "tanim", "guc", "tarih", "lokasyon"],
                     encoding="utf-8", dtype={"tanim": str})
    tr = pd.read_csv(KOK / "data/raw/train.csv",
                     usecols=["tanim", "guc", "tuketim", "lokasyon", "tarih"],
                     encoding="utf-8", dtype={"tanim": str})
    giris = pd.read_csv(KOK / ar.giris, encoding="utf-8")

    m = ornek[["id"]].merge(giris, on="id", how="left").merge(te, on="id", how="left")
    if m["tuketim"].isna().any() or m["guc"].isna().any():
        raise RuntimeError("giris dosyasi ile ornek gonderim id kumesi ortusmuyor")

    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    ham = m["tuketim"].to_numpy(dtype="float64")
    r = np.log1p(ham) - log_guc
    gun = m["tarih"].to_numpy()

    # --- referans seviye: gun ortalamasi, aya dogru ampirik-Bayes buzulmus ---
    gun_a = gun[soguk]
    ay_a = pd.to_datetime(m.loc[soguk, "tarih"]).dt.to_period("M").astype(str).to_numpy()
    r_s = r[soguk]

    def gruplu(v, anahtar):  # noqa: ANN001, ANN202
        return pd.Series(v).groupby(anahtar).transform("mean").to_numpy()

    n_gun = pd.Series(gun_a).groupby(gun_a).transform("size").to_numpy().astype("float64")
    w = n_gun / (n_gun + ar.m_gun) if ar.m_gun > 0 else np.ones_like(n_gun)
    # Gun agirliklari ay icinde ESIT DEGIL (test'te 1 satirlik gun de var,
    # 1962 satirlik gun de). Duz bir harman aylik seviyeyi kaydirirdi --
    # olculdu: en buyuk sapma 0,0184. Ay icinde yeniden merkezleyerek
    # garantiyi TAM yapiyoruz: aylik seviye = mevsim rampasi, dokunulmaz.
    seviye = w * gruplu(r_s, gun_a) + (1.0 - w) * gruplu(r_s, ay_a)
    seviye = seviye - gruplu(seviye, ay_a) + gruplu(r_s, ay_a)

    if ar.hucresiz:
        etki = np.zeros(int(soguk.sum()))
    else:
        tablo_kaynak = tr[pd.to_datetime(tr["tarih"]) >= ar.tablo_baslangic]
        if len(tablo_kaynak) < 100_000:
            raise RuntimeError(f"tablo kaynagi cok seyrek: {len(tablo_kaynak):,} satir")
        hucre = hucre_etkisi(tablo_kaynak, m)[soguk]
        # Etkiyi AYNI referansa gore merkezle, sonra ay icinde sifirla.
        h_ref = w * gruplu(hucre, gun_a) + (1.0 - w) * gruplu(hucre, ay_a)
        etki = hucre - h_ref
        etki = etki - gruplu(etki, ay_a)

    r_yeni = r.copy()
    r_yeni[soguk] = seviye + ar.a_hucre * etki + ar.b_model * (r_s - seviye)
    yeni = ham.copy()  # sicak satirlar gidis-donusumsuz, birebir kopya
    yeni[soguk] = np.clip(np.expm1(r_yeni[soguk] + log_guc[soguk]), 0.0, None)

    # --- guvenlik kapilari ---
    if not np.array_equal(yeni[~soguk], ham[~soguk]):
        raise RuntimeError("SICAK satirlar degismis olmamaliydi")
    ay_once = pd.Series(r_s).groupby(ay_a).mean()
    ay_sonra = pd.Series(r_yeni[soguk]).groupby(ay_a).mean()
    ay_sapma = float(np.abs(ay_once - ay_sonra).max())
    if ay_sapma > 5e-3:
        raise RuntimeError(f"AYLIK seviye korunmadi: en buyuk sapma {ay_sapma:.3e}")
    # Kapi 2: islemin adi BUZME -- gun ici yayilma her ayda DUSMELI.
    ici_once = pd.Series(r_s - gruplu(r_s, gun_a)).groupby(ay_a).std()
    ici_sonra = pd.Series(r_yeni[soguk] - gruplu(r_yeni[soguk], gun_a)).groupby(ay_a).std()
    if (ici_sonra > ici_once + 1e-9).any():
        raise RuntimeError(f"gun ici yayilma ARTMIS: {dict(ici_once)} -> {dict(ici_sonra)}")
    if not np.isfinite(yeni).all() or (yeni < 0).any():
        raise RuntimeError("cikti NaN/sonsuz/negatif iceriyor")

    n_s = int(soguk.sum())
    hucre_ad = "KAPALI" if ar.hucresiz else f"ilce x kova (M={M_HUCRE:.0f})"
    trafo_sayisi = int(m.loc[soguk, "tanim"].nunique())
    print(f"  a(hucre) {ar.a_hucre:.2f}  b(model) {ar.b_model:.2f}  "
          f"M_gun {ar.m_gun:.0f}  hucre {hucre_ad}")
    if not ar.hucresiz:
        print(f"  tablo kaynagi {ar.tablo_baslangic}'ten itibaren "
              f"{len(tablo_kaynak):,} satir ({len(tr):,} icinden)")
    print(f"  soguk {n_s:,} satir (%{100 * n_s / len(m):.2f}), {trafo_sayisi:,} trafo")
    print(f"  en seyrek gun {n_gun.min():.0f} satir, medyan {np.median(n_gun):.0f}")
    print(f"  AYLIK seviye korundu (en buyuk sapma {ay_sapma:.2e})")
    print(f"  gun ici yayilma  {float(ici_once.mean()):.5f} -> {float(ici_sonra.mean()):.5f}")
    print(f"  toplam soguk ofset std   {r_s.std():.5f} -> {r_yeni[soguk].std():.5f}")
    ay_no = pd.to_datetime(m["tarih"]).dt.month.to_numpy()
    print("   ay   once    sonra")
    for k in np.unique(ay_no):
        d = soguk & (ay_no == k)
        print(f"   {k:2d}  {r[d].mean():+.4f}  {r_yeni[d].mean():+.4f}")

    cikti = pd.DataFrame({"id": m["id"], "tuketim": yeni})
    yol = KOK / ar.cikis
    yol.parent.mkdir(parents=True, exist_ok=True)
    cikti.to_csv(yol, index=False, encoding="utf-8")
    print(f"  yazildi: {ar.cikis}  ({len(cikti):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
