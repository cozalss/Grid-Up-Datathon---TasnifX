"""DUSMANCA GUN-1 PROVASI: gercek kesinti verisinden HASIM bir yarisma dosyasi uretir.

NEDEN BU BETIK VAR
------------------
``real_data_rehearsal.py`` bizim URETTIGIMIZ parquet'leri okur -- yani hattin
yalnizca *temiz* girdiyle calistigini kanitlar. Yarisma dosyasi temiz gelmez.
Gecmis Turk veri setlerinde olculen tuzaklar:

    * cp1254 kodlama (Excel'den ihrac; UTF-8 sanip okumak mojibake uretir)
    * ``;`` ayirici + ondalik VIRGUL (Turkce Excel varsayilani)
    * Turkce basliklar: ``İl``, ``İlçe``, ``Tarih``, ``Kesinti Adedi``
    * Ilce adlari GORUNTU bicimiyle: ``ÇİĞLİ`` / ``Çiğli`` / ``CIGLI``
      -- bunlar bizim dis tablolarimizin anahtari ``cigli`` ile ancak
      ``join_key`` dogru calisirsa bulusur. Bulusmazsa 219 dis kolon
      SESSIZCE NaN olur ve model "bu ilcede orman yok" diye ogrenir.
    * hedef kolonun test dosyasinda OLMAMASI
    * ID'nin bilesik olmasi (``ilce_tarih``)
    * ise yaramaz kolonlar (aciklama metni, sabit kolon)

Bu betik bunlarin HEPSINI ayni anda uygular ve ``day_one.py``'i uzerinde
kosar. Amac skor degil: veri gunu ilk saatinde ne KIRILIR sorusuna bugun
cevap vermek.

Kaynak: ``data/external/epias/panel_ilce_gun_tam.parquet`` -- 96 ilce, gercek
GDZ+ADM plansiz kesinti sayilari. Yani sahte olan yalnizca BICIM; sayilar
gercek.

Calistirma::

    python scripts/dusmanca_prova.py              # dosyalari uret + day_one kos
    python scripts/dusmanca_prova.py --sadece-uret
    python scripts/dusmanca_prova.py --zorluk hafif   # sadece Turkce basliklar
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # kardes betikler

from epias_panel import MERKEZ_KURTARMA  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402
from gridup.turkish import (  # noqa: E402
    hizala_ilce_anahtarlari,
    join_key,
    strip_qualifier,
)

KOK = Path(__file__).resolve().parents[1]
KAYNAK_PANEL = KOK / "data" / "external" / "epias" / "panel_ilce_gun_tam.parquet"
KAYNAK_HAM = KOK / "data" / "external" / "epias" / "kesinti_plansiz.parquet"
CIKTI = KOK / "data" / "raw" / "_dusmanca"

#: Test dosyasinin basladigi gun. Egitim bunun ONCESI, test bunun sonrasi.
#: Gercek bir ileri-yonlu bolme -- rastgele degil.
BOLME_GUNU = "2026-07-01"

#: Turkce basliklar. ``İ`` (U+0130) ve ``ç`` bilerek var: ``normalize_columns``
#: bunlari ``il``/``ilce``/``kesinti_adedi``'ne cevirebiliyor mu, olculuyor.
BASLIKLAR = {
    "il_display": "İl",
    "ilce_display": "İlçe",
    "gun": "Tarih",
    "kesinti_adet": "Kesinti Adedi",
    "osm_iletim_hat_km": "Hat Uzunluğu (km)",
    "osm_kule": "Direk Sayısı",
}

#: Ondalik VIRGULLE yazilacak kolon. Bir tanesi yeter: ``sniff_dialect_shared``
#: ondalik=',' der ve NOKTA ondalikli hedef kolonun sessizce ``str`` kalmasi
#: tuzagi tetiklenir -- provanin yakalamak istedigi sey tam olarak budur.
ONDALIKLI_KOLON = BASLIKLAR["osm_iletim_hat_km"]

#: Statik ilce ozniteligi kaynagi. Kesinti SONUCUNDAN turetilmis hicbir kolon
#: kullanilmaz: ilk taslakta "Toplam Süre (dk)" ve "Etkilenen Abone" vardi ve
#: sizinti kapisi koşuyu HAKLI OLARAK durdurdu (Spearman 0,9935 / 0,9775 --
#: ikisi de hedefin ayni gunku monoton donusumu). Gercek bir test kumesinde
#: bu kolonlar bulunmaz. Yerine sebeke altyapisi konuldu: zamandan bagimsiz,
#: hedeften turetilmemis, ve gercek yarisma dosyalarinda bulunmasi makul.
OSM_TABLOSU = KOK / "data" / "external" / "osm_altyapi_ilce.parquet"

#: Join denetiminde kullanilan takma adlar -- panelin kendi haritasi.
_TAKMA_ADLAR = {ham: yeni for (_, ham), yeni in MERKEZ_KURTARMA.items()}

ZORLUKLAR = ("hafif", "orta", "tam")

#: SAATLIK senaryonun baslik ustyazimlari. Acilis sunumundan (2026-08-21)
#: birebir: veri "Profil Tarihi (tarih ve saat)" tasiyor, hedef ise "Aktif
#: Tuketim Gunluk". Ikisinin ayni cozunurlukte oldugu HICBIR YERDE
#: soylenmedi -- prova tam olarak o ihtimali sinar.
SAATLIK_BASLIKLARI = {
    "gun": "Profil Tarihi",
    "kesinti_adet": "Aktif Tüketim",
}

#: Gercekci gun-ici yuk profili (24 saat). Mutlak sekil onemli degil --
#: onemli olan bir (varlik, gun) ciftinin egitim dosyasinda 24 KEZ
#: gorunmesi. Kod icinde 1'e normalize edilir, yani saatlik degerlerin
#: toplami gunluk degere ESIT kalir; boylece "hat dogru topladi mi"
#: sorusu tahmin degil OLCUM olur.
SAATLIK_PROFIL = (
    0.030,
    0.028,
    0.027,
    0.026,
    0.026,
    0.028,
    0.033,
    0.039,
    0.044,
    0.047,
    0.048,
    0.048,
    0.047,
    0.046,
    0.046,
    0.047,
    0.050,
    0.055,
    0.058,
    0.057,
    0.053,
    0.047,
    0.040,
    0.034,
)


def _saatlige_ac(gunluk: pd.DataFrame, hedef: str) -> pd.DataFrame:
    """Her gunluk satiri 24 saatlik satira acar; hedefi profile boler.

    Gunluk toplam KORUNUR (profil 1'e normalize edilir). Dolayisiyla dogru
    davranis -- gune indirip toplamak -- gercek hedefi birebir geri verir.
    Yanlis davranis (24 kat sismis cerceve uzerinde lag/rolling, ya da
    tekrarlanan ID'yi sessizce ilk satira dusurmek) olculebilir bicimde
    farkli sonuc uretir.
    """
    pay = pd.Series(SAATLIK_PROFIL, dtype="float64")
    pay = (pay / pay.sum()).to_numpy()
    acik = gunluk.loc[gunluk.index.repeat(24)].reset_index(drop=True)
    saat = pd.Series(list(range(24)) * len(gunluk))
    acik["gun"] = acik["gun"] + pd.to_timedelta(saat, unit="h")
    acik[hedef] = acik[hedef].to_numpy(dtype="float64") * pay[saat.to_numpy()]
    return acik


def _ilce_goruntu_adlari() -> pd.DataFrame:
    """``ilce_key -> gercek EPIAS goruntu adi`` eslemesi.

    Uydurma degil: ham kesinti dosyasindaki ``province``/``district``
    kolonlarinin ta kendisi. Yarisma dosyasi da ayni kaynaktan gelecekse
    ayni yazimi kullanacaktir.
    """
    ham = pd.read_parquet(KAYNAK_HAM, columns=["province", "district", "il_key", "ilce_key"])
    # Panelin anahtar URETIMINI birebir tekrarla; kopyalama YOK.
    #
    # Iki ayri duzeltme gerekiyor ve ikisi de gercek veriden olculdu (2026-08-21):
    #   1) NITELIKLI ADLAR -- ham dosyada 'bozkurt / denizli', 'kale / denizli',
    #      'koprubasi / manisa', 'yenipazar / aydin' var. Turkiye'de ayni ilce
    #      adi birden fazla ilde bulundugu icin EPIAS ayirt edici ek koyuyor.
    #      Duzeltilmezse 5 ilce (8.450 satir) eslesmiyor.
    #   2) 2012 MERKEZ KURTARMASI -- 'aydin merkez' bugunku adiyla 'efeler'.
    #      Duzeltilmezse 1 ilce (1.690 satir) eslesmiyor.
    #
    # ``MERKEZ_KURTARMA`` panelin kendi sabiti; buradan ithal ediliyor ki
    # panel yarin degisirse prova SESSIZCE eskimesin.
    ham["ilce_key"] = ham["ilce_key"].map(lambda k: join_key(strip_qualifier(str(k))))
    ham["ilce_key"] = [
        MERKEZ_KURTARMA.get((il, ilce), ilce)
        for il, ilce in zip(ham["il_key"], ham["ilce_key"], strict=True)
    ]
    ham = ham.drop_duplicates(subset=["il_key", "ilce_key"])
    return ham.rename(columns={"province": "il_display", "district": "ilce_display"})


def _hasim_csv_yaz(
    frame: pd.DataFrame, path: Path, *, zorluk: str, ondalik_kolonlar: tuple[str, ...] = ()
) -> None:
    """Turkce Excel bicimiyle yazar: cp1254 + ``;`` + ondalik virgul."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metin = frame.copy()
    if zorluk == "tam":
        for kolon in ondalik_kolonlar:
            if kolon in metin.columns:
                metin[kolon] = metin[kolon].map(
                    lambda x: "" if pd.isna(x) else f"{float(x):.2f}".replace(".", ",")
                )
        ayirici, kodlama = ";", "cp1254"
    elif zorluk == "orta":
        ayirici, kodlama = ";", "utf-8"
    else:
        ayirici, kodlama = ",", "utf-8"
    metin.to_csv(path, sep=ayirici, index=False, encoding=kodlama, lineterminator="\n")


def uret(zorluk: str, senaryo: str = "sayim") -> dict[str, Path]:
    """Hasim train/test/sample uclusunu yazar ve yollarini dondurur."""
    panel = pd.read_parquet(KAYNAK_PANEL)
    panel["gun"] = pd.to_datetime(panel["gun"])
    goruntu = _ilce_goruntu_adlari()
    panel = panel.merge(goruntu, on=["il_key", "ilce_key"], how="left", validate="many_to_one")
    eksik = int(panel["ilce_display"].isna().sum())
    if eksik:
        raise RuntimeError(f"{eksik} satirda goruntu adi bulunamadi -- kaynaklar uyusmuyor")

    if senaryo == "ikili":
        # GDZ'22 Case-1 -- AYNI PROBLEM -- metrik olarak F1 kullandi, yani
        # hedef "kac kesinti" degil "kesinti oldu mu"ydu. 2026'da da boyle
        # olabilir ve o durumda butun MAE/sayim yolu devre disi kalir.
        panel["kesinti_adet"] = (panel["kesinti_adet"] > 0).astype(int)

    if zorluk == "tam":
        # Buyuk harf: 'Çiğli' -> 'ÇIĞLI' (Python varsayilan upper, I noktasiz).
        # join_key'in i-tuzagina dayanikli olup olmadigini SINAR.
        panel["ilce_display"] = panel["ilce_display"].str.upper()
        panel["il_display"] = panel["il_display"].str.upper()

    panel = panel.sort_values(["ilce_key", "gun"]).reset_index(drop=True)
    if senaryo == "gercek_geometri":
        # Yarismanin ILAN EDILEN takvimi. Hedef degerleri bu senaryoda
        # onemli degil -- sinanan sey GEOMETRI: 15 ay egitim, 4 ay test,
        # 122 gunluk ufuk.
        egitim = panel[panel["gun"].between("2025-01-01", "2026-03-31")].copy()
        test = panel[panel["gun"].between("2026-04-01", "2026-07-31")].copy()
        print(f"  GERCEK GEOMETRI: egitim {len(egitim):,} satir, test {len(test):,} satir")
    else:
        bolme = pd.Timestamp(BOLME_GUNU)
        egitim = panel[panel["gun"] < bolme].copy()
        test = panel[panel["gun"] >= bolme].copy()
    if egitim.empty or test.empty:
        raise RuntimeError(f"bolme gunu {BOLME_GUNU} paneli ikiye ayirmadi")

    if senaryo == "soguk_ilce":
        # SOGUK BASLANGIC: test'te olup train'de HIC OLMAYAN ilceler.
        # Yarisma ilceye gore bolerse (ya da yeni bir ilce eklenirse) bu olur.
        # Kategorik kodlama, hedef kodlama ve komsu feature'lari bu durumda
        # gorulmemis bir seviyeyle karsilasir; sessizce NaN ya da 0 uretmek
        # yerine ne yaptigini SOYLEMELI.
        soguk = sorted(panel["ilce_key"].unique())[:10]
        egitim = egitim[~egitim["ilce_key"].isin(soguk)].copy()
        print(f"  SOGUK ILCE: {len(soguk)} ilce train'den cikarildi -> {soguk[:4]} ...")

    if senaryo == "ic_ice":
        # IC ICE BOLME: test gunleri train gunlerinin ARASINA serpistirilmis.
        # forecast_geometry bunu 'interleaved' olarak isaretlemeli ve
        # day_one purged_time_series yerine KFold'a dusmeli.
        tum = panel.sort_values("gun")["gun"].unique()
        test_gunleri = set(tum[::7])  # her 7 gunde bir gun test'e
        egitim = panel[~panel["gun"].isin(test_gunleri)].copy()
        test = panel[panel["gun"].isin(test_gunleri)].copy()
        print(f"  IC ICE: {len(test_gunleri)} gun test'e serpistirildi")

    # Bilesik ID -- Kaggle'da cok yaygin ve day_one'in ``synthesize_id_column``
    # yolunu tetikler.
    for parca in (egitim, test):
        parca["ID"] = parca["ilce_key"] + "_" + parca["gun"].dt.strftime("%Y-%m-%d")

    if senaryo == "saatlik":
        # SAATLIK EGITIM / GUNLUK GONDERIM.
        #
        # Acilis sunumu "Profil Tarihi (tarih ve saat)" diyor ama hedefi
        # "Aktif Tuketim GUNLUK" diye adlandiriyor. Egitim dosyasi saatlik
        # gelirse bir (trafo, gun) cifti 24 KEZ tekrarlanir; gonderim ise
        # gunluktur. Bu sessiz bicimde olumcul:
        #   * ID 24 kez tekrar eder -- birebir gonderim anahtari degil artik
        #   * lag/rolling 24 kat sismis bir cerceve uzerinde hesaplanir,
        #     yani "1 gun onceki" aslinda "1 SAAT onceki" olur
        #   * CV grup kolonu ve ufuk hesabi gun degil saat sayar
        # Test ve sample GUNLUK birakiliyor -- gercek gonderim gunluk.
        egitim = _saatlige_ac(egitim, "kesinti_adet")
        print(f"  SAATLIK: egitim {len(egitim):,} satira acildi (gunluk x24)")

    disari = ["ID", "il_display", "ilce_display", "gun", "kesinti_adet"]
    if zorluk == "tam":
        # Statik altyapi ozniteligi: hedeften TURETILMEMIS, zamandan bagimsiz.
        # Tek gorevi ondalik-virgullu bir kolon tasimak; boylece ondalik
        # tespiti ',' der ve nokta ondalikli HEDEF kolonunun sessizce metin
        # kalmasi tuzagi tetiklenir.
        osm = pd.read_parquet(OSM_TABLOSU, columns=["ilce_key", "osm_iletim_hat_km", "osm_kule"])
        egitim = egitim.merge(osm, on="ilce_key", how="left", validate="many_to_one")
        test = test.merge(osm, on="ilce_key", how="left", validate="many_to_one")
        disari += ["osm_iletim_hat_km", "osm_kule"]

    basliklar = dict(BASLIKLAR)
    if senaryo == "saatlik":
        basliklar.update(SAATLIK_BASLIKLARI)
    gun_bicimi = "%Y-%m-%d %H:%M:%S" if senaryo == "saatlik" else "%Y-%m-%d"

    egitim_c = egitim[disari].rename(columns=basliklar)
    egitim_c[basliklar["gun"]] = egitim["gun"].dt.strftime(gun_bicimi)
    if zorluk == "tam":
        # Sabit kolon + bos kolon: gercek veri setleri bunlarla dolu ve
        # profil ikisini de "bilgi tasimaz" diye isaretlemeli.
        egitim_c["Dağıtım Bölgesi"] = "EGE"
        egitim_c["Açıklama"] = ""

    # Test GUNLUK kalir: saatlik senaryoda train ile test'in cozunurlugu
    # BILEREK ayrisiyor -- yakalanmak istenen arıza tam olarak bu.
    test_c = test[[c for c in disari if c != "kesinti_adet"]].rename(columns=basliklar)
    test_c[basliklar["gun"]] = test["gun"].dt.strftime("%Y-%m-%d")
    if zorluk == "tam":
        test_c["Dağıtım Bölgesi"] = "EGE"
        test_c["Açıklama"] = ""

    ornek = pd.DataFrame({"ID": test["ID"].to_numpy(), basliklar["kesinti_adet"]: 0})

    CIKTI.mkdir(parents=True, exist_ok=True)
    yollar = {
        "train": CIKTI / "train.csv",
        "test": CIKTI / "test.csv",
        "sample": CIKTI / "sample_submission.csv",
    }
    ondalikli = (ONDALIKLI_KOLON,)
    _hasim_csv_yaz(egitim_c, yollar["train"], zorluk=zorluk, ondalik_kolonlar=ondalikli)
    _hasim_csv_yaz(test_c, yollar["test"], zorluk=zorluk, ondalik_kolonlar=ondalikli)
    _hasim_csv_yaz(ornek, yollar["sample"], zorluk=zorluk)

    print(f"  zorluk={zorluk}  senaryo={senaryo}")
    print(f"  train : {len(egitim_c):>7,} satir  {yollar['train'].name}")
    print(f"  test  : {len(test_c):>7,} satir  {yollar['test'].name}")
    print(f"  sample: {len(ornek):>7,} satir  {yollar['sample'].name}")
    print(f"  hedef : {basliklar['kesinti_adet']!r}  (test dosyasinda YOK)")
    print(f"  ilce  : {egitim_c[basliklar['ilce_display']].iloc[0]!r} ornek")
    return yollar


def join_denetimi(zorluk: str) -> bool:
    """Uretilen dosyadaki ilce adlari dis tablolarin anahtarina donuyor mu?

    Bu, gun-1'in EN SESSIZ arizasidir: eslesme tutmazsa hata cikmaz, yalnizca
    219 dis kolon NaN olur.
    """
    print("\n[JOIN DENETIMI] ilce adi -> ilce_key")
    # DOSYANIN TAMAMI taranir, bas kismi degil.
    #
    # Onceki surum ``nrows=200_000`` ile okuyordu. Gunluk dosyada (162.240
    # satir) bu tesadufen dosyanin tamamiydi. SAATLIK senaryoda dosya
    # 3.783.168 satir ve ilce_key'e gore SIRALI -- olculdu (2026-08-21):
    #
    #     tam tarama : 3.783.168 satir, 96 benzersiz ilce, 1,5 sn
    #     nrows=200k :   200.000 satir,  6 benzersiz ilce, 0,2 sn
    #
    # Yani kapi 96 ilcenin 6'sina bakip 'TAMAM: hepsi eslesti' yaziyordu.
    # Bir denetim kapisinin verebilecegi en kotu cikti budur: YANLIS YESIL.
    # Yalnizca gereken kolon okunarak tam tarama 1,5 sn'ye iniyor, yani
    # kisaltmanin bir maliyet gerekcesi de yok.
    train = pd.read_csv(
        CIKTI / "train.csv",
        sep=";" if zorluk != "hafif" else ",",
        encoding="cp1254" if zorluk == "tam" else "utf-8",
        usecols=[BASLIKLAR["ilce_display"]],
    )
    adlar = train[BASLIKLAR["ilce_display"]].dropna().unique()
    print(f"  taranan satir          : {len(train):,}")
    arazi = pd.read_parquet(KOK / "data" / "external" / "arazi_ortusu_ilce.parquet")
    hedef = set(arazi["ilce_key"])

    # IKI KATMAN AYRI AYRI OLCULUR ve fark ONEMLIDIR.
    #
    # Onceki surum yalnizca ``join_key``e bakiyordu ve nitelikli adlar
    # yuzunden HER KOSUDA kirmizi yaziyordu -- oysa day_one o adlari
    # ``hizala_ilce_anahtarlari`` ile zaten kurtariyor. Hep kirmizi yanan bir
    # uyari, bir sure sonra "zaten hep kirmizi" diye gormezden gelinir; yani
    # yanlis alarm, alarmin KENDISINI ise yaramaz hale getirir.
    #
    # Simdi ham katman BILGI olarak, hizalanmis katman KARAR olarak raporlanir.
    ham_tutmayan = {ad: join_key(str(ad)) for ad in adlar if join_key(str(ad)) not in hedef}
    kurtarmalar = hizala_ilce_anahtarlari(adlar, referans=hedef, takma_adlar=_TAKMA_ADLAR)
    bulunamayan = {a: k for a, k in kurtarmalar.items() if k.yontem == "BULUNAMADI"}

    print(f"  benzersiz ilce adi     : {len(adlar)}")
    print(f"  ham join_key ile tutan : {len(adlar) - len(ham_tutmayan)}")
    print(f"  hizalama SONRASI tutan : {len(adlar) - len(bulunamayan)}")
    if ham_tutmayan and not bulunamayan:
        ornek = list(ham_tutmayan)[:3]
        print(f"  NOT: {len(ham_tutmayan)} ad ham anahtarla tutmuyordu, hizalama kurtardi.")
        print(f"       ornek: {ornek}")
    if bulunamayan:
        print(f"  BULUNAMAYAN ({len(bulunamayan)}) -- hizalama da kurtaramadi:")
        for ad, kayit in list(bulunamayan.items())[:15]:
            print(f"    {ad!r} -> {kayit.anahtar!r}")
        return False
    print("  TAMAM: hepsi eslesti.")
    return True


#: Senaryo -> day_one'a verilecek resmi metrik.
#:
#: 'ikili' icin F1: GDZ'22 Case-1 -- bizimkiyle AYNI problem -- tam olarak
#: bunu kullandi. Sayim yolunun butunu (Tweedie/Poisson objective, sifir
#: tabani, tamsayiya yuvarlama) o senaryoda devre disi kalir.
SENARYO_METRIGI = {
    "sayim": "MAE",
    "ikili": "f1",
    "soguk_ilce": "MAE",
    "ic_ice": "MAE",
    # GDZ 2023'un metrigi MAPE'ydi. MAPE sifir hedefte MATEMATIKSEL OLARAK
    # TANIMSIZDIR (y=0'a bolme) ve bu panelde gunlerin %65'i sifir. Yani
    # metrik MAPE cikarsa hedefin ucte ikisi metrigin disinda kalir --
    # hattin bunu SESSIZCE yapmamasi, yuksek sesle soylemesi gerekir.
    "mape": "mape",
    # 2023 ayrica log-olcekli hatalari da gundeme getirir; RMSLE negatif
    # tahminde tanimsizdir ve log1p donusumu ZORUNLU kilar.
    "rmsle": "rmsle",
    # GERCEK YARISMA GEOMETRISI (2026-08-21 acilis sunumu):
    #   egitim  Ocak 2025 - Mart 2026   (15 ay = 455 gun)
    #   test    Nisan 2026 - Temmuz 2026 (4 ay = 122 gun)
    # Simdiye kadarki provalar 48 gunluk ufukla kostu. 122 gunluk ufuk
    # lag/rolling kaydirmasini uce katlar ve 455 gunluk egitim setinde bu
    # ciddi bir pay -- ufuk kadar kaydirilan bir lag, egitimin ilk 122
    # gununu bos birakir. Kirilip kirilmadigi TAHMIN EDILMEZ, olculur.
    "gercek_geometri": "mape",
    # SAATLIK EGITIM / GUNLUK GONDERIM. Acilis sunumu "Profil Tarihi
    # (tarih ve saat)" diyor; hedef ise "Aktif Tuketim GUNLUK". Egitim
    # dosyasi saatlik gelirse bir (trafo, gun) cifti 24 kez tekrarlanir.
    # Olculdu (2026-08-21): day_one.py ve validation.py icinde HICBIR
    # cozunurluk indirgeme adimi yok -- ne resample, ne groupby-sum. Yani
    # bu sekil karsisinda ne yapacagi BILINMIYOR; prova onu olcer.
    "saatlik": "MAE",
}

SENARYOLAR = tuple(SENARYO_METRIGI)


def day_one_kos(yollar: dict[str, Path], ek: list[str], senaryo: str = "sayim") -> int:
    komut = [
        sys.executable,
        str(KOK / "scripts" / "day_one.py"),
        "--data",
        str(CIKTI),
        "--metric",
        SENARYO_METRIGI[senaryo],
        "--yes",
        *ek,
    ]
    print("\n[DAY_ONE] " + " ".join(komut[1:]))
    print("=" * 70)
    return subprocess.run(komut, cwd=KOK, check=False).returncode


def main() -> int:
    satir_tamponlu_cikti()
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--zorluk", choices=ZORLUKLAR, default="tam")
    ayristirici.add_argument(
        "--senaryo",
        choices=SENARYOLAR,
        default="sayim",
        help=(
            "Yarisma dosyasinin BICIMI degil SEKLI. 'sayim': gunluk kesinti "
            "adedi + MAE. 'ikili': kesinti oldu mu + F1 (GDZ'22 Case-1 boyleydi). "
            "'soguk_ilce': test'te train'de olmayan ilceler. 'ic_ice': test "
            "gunleri train gunlerinin arasina serpistirilmis. 'saatlik': "
            "egitim SAATLIK (gunluk x24), gonderim gunluk."
        ),
    )
    ayristirici.add_argument("--sadece-uret", action="store_true")
    ayristirici.add_argument("--temizle", action="store_true", help="Cikti dizinini sil ve cik")
    ayristirici.add_argument("day_one_ek", nargs="*", help="day_one.py'a aktarilacak ek bayraklar")
    args = ayristirici.parse_args()

    if args.temizle:
        if CIKTI.exists():
            shutil.rmtree(CIKTI)
            print(f"silindi: {CIKTI}")
        return 0

    if not KAYNAK_PANEL.exists():
        print(f"HATA: {KAYNAK_PANEL} yok. Once EPIAS panelini uret.", file=sys.stderr)
        return 2

    print("=" * 70)
    print("DUSMANCA GUN-1 PROVASI")
    print("=" * 70)
    yollar = uret(args.zorluk, args.senaryo)
    saglam = join_denetimi(args.zorluk)
    if not saglam:
        print("\nUYARI: join denetimi kirmizi -- day_one yine de kosuluyor,")
        print("       ama dis aileler buyuk ihtimalle NaN gelecek.")
    if args.sadece_uret:
        return 0
    return day_one_kos(yollar, args.day_one_ek, args.senaryo)


if __name__ == "__main__":
    raise SystemExit(main())
