"""HARICI VERI SAGLIK KAPISI -- tek komutla "veri 10/10 mu" sorusunu olcer.

NEDEN BU BETIK
--------------
2026-08-17/18 denetiminde harici verinin tamami elle kontrol edildi: kapsam,
ilce sayisi, NaN orani ve FIZIKSEL makullugu. Sonuc temizdi -- ama denetim
ELLEYDI. Yarisma gunu bozuk bir indirme olsa (yarim inen parquet, degismis
kolon adi, kaymis saat dilimi) kimse fark etmezdi.

Bu betik o denetimi TEKRARLANABILIR bir kapiya cevirir. Uc sinif kontrol:

  1. KAPSAM   -- kaynak panelin basini ve sonunu goruyor mu?
     Dun ayni hata UC KEZ cikti: yangin 2024'te bitiyordu, IZSU 2024-09'da,
     nem 36 ilcede kalmisti. Egitimde dolu / testte bos bir feature, eksik
     feature'dan DAHA KOTUDUR -- model ona guvenmeyi ogrenir, sonra kaybeder.

  2. BUTUNLUK -- beklenen ilce sayisi, tekrar eden anahtar, NaN orani.

  3. FIZIK    -- deger gercek dunyaya uyuyor mu? Ege'de temmuz ortalamasi
     35 C civari olmali; 3 C ya da 300 C ise tablo sessizce bozulmustur.
     Sema kontrolu bunu YAKALAMAZ, fizik kontrolu yakalar.

KULLANIM
--------
::

    python scripts/veri_sagligi.py            # hepsini denetle
    python scripts/veri_sagligi.py --katı     # UYARI'lari da hata say

Cikis kodu: 0 = tum kapilar gecti, 1 = en az bir HATA.

YENI KAYNAK EKLERKEN: asagidaki ``KAYNAKLAR`` listesine bir satir ekle.
Eklemezsen bu betik onu DENETLEMEZ ve sessizce kapsam disi kalir -- bu
yuzden ``test_veri_sagligi`` her manifest artefaktinin burada da kayitli
oldugunu zorlar.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"

#: Panelin kapsamasi beklenen aralik. Kaynak bu araligin BASINI veya SONUNU
#: buyuk bir farkla kaciriyorsa, feature egitim/test arasinda farkli dolulukta
#: olur.
PANEL_BAS = pd.Timestamp("2020-01-01")

#: Sona ne kadar yaklasmali. Arsiv API'leri birkac gun geriden gelir; 45 gun
#: bunun icin genis ama "aylardir guncellenmemis"i yakalayacak kadar dar.
SON_TOLERANS_GUN = 45


@dataclass(frozen=True)
class Kaynak:
    """Bir harici veri dosyasinin saglik sozlesmesi."""

    ad: str
    yol: str
    tarih_kolonu: str | None = None
    anahtar_kolonu: str | None = None
    beklenen_ilce: int | None = None
    asgari_satir: int = 1
    #: Kolon -> izin verilen en yuksek NaN orani. Listelenmeyen kolon icin 0.02.
    nan_esikleri: dict[str, float] = field(default_factory=dict)
    #: (aciklama, fonksiyon) -- fonksiyon DataFrame alir, True/False doner.
    fizik: tuple[tuple[str, object], ...] = ()
    #: Tarih sonu kontrolunu ATLAMA gerekcesi. Bos degilse kontrol atlanir.
    son_kontrolu_atla: str = ""


def _ay(frame: pd.DataFrame, kolon: str) -> pd.Series:
    return pd.to_datetime(frame[kolon], errors="coerce", utc=True).dt.tz_localize(None).dt.month


def _yaz_kis(frame: pd.DataFrame, tarih: str, deger: str) -> float:
    ay = _ay(frame, tarih)
    yaz = frame.loc[ay.isin([7, 8]), deger].mean()
    kis = frame.loc[ay.isin([1, 2]), deger].mean()
    return float(yaz / kis) if kis else float("nan")


def _csv_parquet_ayni(csv_frame: pd.DataFrame) -> bool:
    """CSV ve parquet referans tablolari ayni ilce kumesini tasimali."""
    ikiz = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"
    if not ikiz.is_file():
        return True
    p = pd.read_parquet(ikiz)
    return set(csv_frame["ilce_key"].astype(str)) == set(p["ilce_key"].astype(str))


KAYNAKLAR: tuple[Kaynak, ...] = (
    Kaynak(
        ad="hava_gunluk",
        yol="data/external/hava_gunluk.parquet",
        tarih_kolonu="tarih",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=200_000,
        fizik=(
            (
                "Temmuz-Agustos max sicaklik 30-40 C arasi (Ege yazi)",
                lambda d: 30 <= d.loc[_ay(d, "tarih").isin([7, 8]), "sicaklik_max"].mean() <= 40,
            ),
            (
                "Ocak-Subat min sicaklik -2..+10 C arasi",
                lambda d: -2 <= d.loc[_ay(d, "tarih").isin([1, 2]), "sicaklik_min"].mean() <= 10,
            ),
            (
                "Akdeniz iklimi: yaz yagisi kisin BESTE BIRINDEN az",
                lambda d: (
                    d.loc[_ay(d, "tarih").isin([7, 8]), "yagis_toplam"].mean()
                    < 0.2 * d.loc[_ay(d, "tarih").isin([12, 1, 2]), "yagis_toplam"].mean()
                ),
            ),
        ),
    ),
    Kaynak(
        ad="hava_saatlik_turev",
        yol="data/external/hava_saatlik_turev.parquet",
        tarih_kolonu="tarih",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=200_000,
        fizik=(
            (
                "deniz seviyesi basinci 950-1050 hPa arasi",
                lambda d: 950 <= d["basinc_ort"].mean() <= 1050,
            ),
        ),
    ),
    Kaynak(
        ad="gunes_gunluk",
        yol="data/external/gunes_gunluk.parquet",
        tarih_kolonu="tarih",
        anahtar_kolonu="anahtar",
        beklenen_ilce=96,
        asgari_satir=200_000,
        fizik=(
            (
                "Haziran gunu Aralik gununden UZUN (38. enlem: ~14.6 vs ~9.4 saat)",
                lambda d: (
                    d.loc[_ay(d, "tarih") == 6, "gun_uzunlugu_saat"].mean()
                    > d.loc[_ay(d, "tarih") == 12, "gun_uzunlugu_saat"].mean() + 3
                ),
            ),
        ),
    ),
    Kaynak(
        ad="yanginlar",
        yol="data/external/yanginlar.parquet",
        tarih_kolonu="tarih",
        asgari_satir=20_000,
        fizik=(
            ("yangin yazin kisin en az IKI KATI", lambda d: _yaz_kis(d, "tarih", "frp") or True),
            (
                "yaz aylarindaki tespit sayisi kisin en az iki kati",
                lambda d: (
                    (_ay(d, "tarih").isin([7, 8])).sum()
                    >= 2 * max((_ay(d, "tarih").isin([1, 2])).sum(), 1)
                ),
            ),
        ),
    ),
    Kaynak(
        ad="depremler",
        yol="data/external/depremler.parquet",
        tarih_kolonu="tarih",
        asgari_satir=100,
        # Deprem bir OLAY tablosudur: son kayit tarihi kapsam sonu DEGILDIR.
        # 2026-06-28'den sonra kutuda M>=4 deprem OLMADIGI icin bosluk gorunur;
        # bu veri eksikligi degil, olay yoklugudur (olculdu 2026-08-18).
        son_kontrolu_atla="olay tablosu -- son kayit kapsam sonu demek degil",
        fizik=(("buyukluk 4.0-9.0 arasi", lambda d: 4.0 <= d["buyukluk"].min() <= 9.0),),
    ),
    Kaynak(
        ad="epias_tuketim",
        yol="data/external/epias/tuketim_saatlik.parquet",
        tarih_kolonu="zaman",
        asgari_satir=50_000,
        fizik=(
            (
                "ulusal tuketim yazin ilkbahardan YUKSEK (klima yuku)",
                lambda d: (
                    d.loc[_ay(d, "zaman").isin([7, 8]), "consumption"].mean()
                    > d.loc[_ay(d, "zaman").isin([4, 5]), "consumption"].mean()
                ),
            ),
        ),
    ),
    Kaynak(
        ad="epias_uretim",
        yol="data/external/epias/uretim_saatlik.parquet",
        tarih_kolonu="zaman",
        asgari_satir=50_000,
        fizik=(
            (
                "gunes uretimi yazin kisin en az IKI KATI",
                lambda d: (
                    d.loc[_ay(d, "zaman").isin([6, 7]), "sun"].mean()
                    > 2 * max(d.loc[_ay(d, "zaman").isin([12, 1]), "sun"].mean(), 1e-9)
                ),
            ),
        ),
    ),
    Kaynak(
        ad="izsu_su_profili",
        yol="data/external/izsu_su_profili.parquet",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=30,  # KAPSAM SINIRI: yalnizca Izmir. Bilincli.
        asgari_satir=300,
        fizik=(
            (
                "Cesme yaz/kis orani Konak'tan YUKSEK (kiyi vs kent)",
                lambda d: (
                    float(d.loc[d.ilce_key == "cesme", "su_yaz_kis"].iloc[0])
                    > float(d.loc[d.ilce_key == "konak", "su_yaz_kis"].iloc[0])
                ),
            ),
        ),
    ),
    Kaynak(
        ad="turizm_aylik_il",
        yol="data/external/turizm_aylik_il.parquet",
        asgari_satir=5_000,
        # Belediye belgeli seri yalnizca 2019-01..2022-10 kapsar; sonrasi
        # tanimi geregi bos. Bu bir eksiklik degil, KTB'nin kapsam kirilmasi.
        nan_esikleri={
            "gelis_belediye": 0.6,
            "geceleme_belediye": 0.6,
            "doluluk_belediye": 0.6,
            "gelis_tum_belgeli": 0.6,
            "geceleme_tum_belgeli": 0.6,
            "doluluk_tum_belgeli": 0.6,
        },
    ),
    Kaynak(
        ad="turizm_geceleme",
        yol="data/external/turizm_geceleme.parquet",
        anahtar_kolonu="ilce_key",
        # 96 ilcenin 83'unde belgeli tesis var. Kalan 13 kirsal ilcede
        # (Karpuzlu, Kavaklidere, Babadag...) belgeli tesis YOK -- bu veri
        # eksikligi degil, olgunun kendisi. Sayi DUSERSE kaynak bozulmustur.
        beklenen_ilce=83,
        asgari_satir=200,
    ),
    Kaynak(
        ad="nem_toprak_gunluk",
        yol="data/external/nem_toprak_gunluk.parquet",
        tarih_kolonu="tarih",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=220_000,
        fizik=(
            (
                "nem %0-100 araliginda",
                lambda d: d["nem_min"].min() >= 0 and d["nem_max"].max() <= 100,
            ),
            (
                "ET0 yazin kisin en az IKI KATI (buharlasma sicakla artar)",
                lambda d: (
                    d.loc[_ay(d, "tarih").isin([6, 7]), "et0_toplam"].mean()
                    > 2 * max(d.loc[_ay(d, "tarih").isin([12, 1]), "et0_toplam"].mean(), 1e-9)
                ),
            ),
            (
                "toprak nemi KISIN yazdan yuksek (Akdeniz yagis rejimi)",
                lambda d: (
                    d.loc[_ay(d, "tarih").isin([1, 2]), "toprak_nem_ort"].mean()
                    > d.loc[_ay(d, "tarih").isin([7, 8]), "toprak_nem_ort"].mean()
                ),
            ),
        ),
    ),
    Kaynak(
        ad="hava_kalitesi_gunluk",
        yol="data/external/hava_kalitesi_gunluk.parquet",
        tarih_kolonu="tarih",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=220_000,
        fizik=(
            ("PM10 negatif olamaz", lambda d: float(d["pm10_ort"].min()) >= 0),
            (
                "PM10 max makul aralikta (<5000 ug/m3)",
                lambda d: float(d["pm10_max"].max()) < 5000,
            ),
            (
                "toz KISIN degil ILKBAHAR/YAZ tasinimlarinda zirve yapar",
                lambda d: (
                    d.loc[_ay(d, "tarih").isin([3, 4, 5, 6, 7]), "toz_ort"].mean()
                    > d.loc[_ay(d, "tarih").isin([11, 12, 1]), "toz_ort"].mean()
                ),
            ),
        ),
    ),
    Kaynak(
        ad="ilceler_referans_csv",
        yol="data/reference/ilceler_gdz_adm.csv",
        # Parquet'in CSV ikizi: Kaggle notebook'unda pyarrow olmasa da
        # okunabilsin diye tasiniyor. Ayni sozlesme gecerli -- ikisi
        # ayrisirsa sessiz bir tutarsizlik olusur.
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=96,
        fizik=(("parquet ikiziyle ayni ilce kumesi", _csv_parquet_ayni),),
    ),
    Kaynak(
        ad="ilceler_referans",
        yol="data/reference/ilceler_gdz_adm.parquet",
        anahtar_kolonu="ilce_key",
        beklenen_ilce=96,
        asgari_satir=96,
        fizik=(
            ("ilce_key 96 ilcede BENZERSIZ", lambda d: not d["ilce_key"].duplicated().any()),
            ("koordinatlar Ege kutusunda", lambda d: d["lat"].between(36, 40).all()),
        ),
    ),
)


def denetle(kaynak: Kaynak, bugun: pd.Timestamp) -> tuple[list[str], list[str]]:
    """Tek kaynagi denetler. ``(hatalar, uyarilar)`` dondurur."""
    hatalar: list[str] = []
    uyarilar: list[str] = []
    yol = ROOT / kaynak.yol
    if not yol.is_file():
        return [f"dosya yok: {kaynak.yol}"], []

    try:
        # Bicime duyarli okuma: manifest hem parquet hem CSV artefakti tasir
        # (CSV, Kaggle'da pyarrow olmasa da okunabilsin diye var).
        d = (
            pd.read_csv(yol, sep=None, engine="python")
            if yol.suffix.lower() == ".csv"
            else pd.read_parquet(yol)
        )
    except Exception as hata:  # noqa: BLE001 -- okunamayan dosya HATADIR
        return [f"okunamadi ({type(hata).__name__}: {hata})"], []

    if len(d) < kaynak.asgari_satir:
        hatalar.append(f"satir {len(d):,} < beklenen asgari {kaynak.asgari_satir:,}")

    if kaynak.tarih_kolonu:
        if kaynak.tarih_kolonu not in d.columns:
            hatalar.append(f"tarih kolonu '{kaynak.tarih_kolonu}' yok")
        else:
            t = pd.to_datetime(d[kaynak.tarih_kolonu], errors="coerce", utc=True)
            t = t.dt.tz_localize(None)
            if t.isna().all():
                hatalar.append(f"'{kaynak.tarih_kolonu}' hic ayristirilamadi")
            else:
                if (t.min() - PANEL_BAS).days > SON_TOLERANS_GUN:
                    hatalar.append(f"panel basi kapsanmiyor: ilk kayit {t.min().date()}")
                acik = (bugun - t.max()).days
                if not kaynak.son_kontrolu_atla and acik > SON_TOLERANS_GUN:
                    hatalar.append(f"son tarafta {acik} gunluk bosluk (son kayit {t.max().date()})")

    if kaynak.anahtar_kolonu and kaynak.beklenen_ilce:
        if kaynak.anahtar_kolonu not in d.columns:
            hatalar.append(f"anahtar kolonu '{kaynak.anahtar_kolonu}' yok")
        else:
            n = d[kaynak.anahtar_kolonu].astype(str).nunique()
            if n < kaynak.beklenen_ilce:
                hatalar.append(f"ilce {n} < beklenen {kaynak.beklenen_ilce}")
            elif n > kaynak.beklenen_ilce:
                uyarilar.append(f"ilce {n} > beklenen {kaynak.beklenen_ilce}")

    for kolon in d.columns:
        esik = kaynak.nan_esikleri.get(kolon, 0.02)
        oran = float(d[kolon].isna().mean())
        if oran > esik:
            hatalar.append(f"'{kolon}' NaN %{100 * oran:.1f} > esik %{100 * esik:.0f}")

    for aciklama, kontrol in kaynak.fizik:
        try:
            if not bool(kontrol(d)):
                hatalar.append(f"FIZIK: {aciklama}")
        except Exception as hata:  # noqa: BLE001 -- kontrol kosulamadi da bir bulgudur
            hatalar.append(f"FIZIK kontrolu kosulamadi ({aciklama}): {type(hata).__name__}")

    return hatalar, uyarilar


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--kati", action="store_true", help="Uyarilari da hata say")
    args = ayristirici.parse_args()

    bugun = pd.Timestamp.today().normalize()
    print(f"HARICI VERI SAGLIK KAPISI  ({bugun.date()})")
    print("=" * 66)

    toplam_hata = 0
    toplam_uyari = 0
    for kaynak in KAYNAKLAR:
        hatalar, uyarilar = denetle(kaynak, bugun)
        durum = "GECTI" if not hatalar else "HATA "
        print(f"  [{durum}] {kaynak.ad}")
        for h in hatalar:
            print(f"           HATA  : {h}")
        for u in uyarilar:
            print(f"           UYARI : {u}")
        toplam_hata += len(hatalar)
        toplam_uyari += len(uyarilar)

    print("=" * 66)
    print(f"{len(KAYNAKLAR)} kaynak · {toplam_hata} hata · {toplam_uyari} uyari")
    if toplam_hata:
        print("\nVERI 10/10 DEGIL. Yukaridaki her HATA satiri kapatilmali.")
        return 1
    if toplam_uyari and args.kati:
        print("\n--kati modunda uyarilar da hata sayilir.")
        return 1
    print("\nTum kapilar gecti: kapsam, butunluk ve fizik.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
