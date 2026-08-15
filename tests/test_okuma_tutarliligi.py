"""Okuma katmaninin TUTARLILIK sozlesmesi.

Veri gununun ilk 10 dakikasi ``read_any`` uzerinden geciyor. Bu dosya iki
sozlesmeyi koruyor:

1. AYNI SEKILDE YAZILMIS iki dosya (train/test) AYNI ayristirma kurallariyla
   okunur -- ayni kolon iki dosyada farkli dtype'a dusemez.
2. AYNI VERI hangi formatta okunursa okunsun AYNI kolon adlarini verir --
   CSV, parquet ve JSON ayrisamaz.

Her testin docstring'i, duzeltmeden ONCE ve SONRA olculen somut sayiyi tasir.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.io_utils import (
    _binlik_kaniti,
    _kesin_en_ondalik,
    _looks_like_en_number,
    read_any,
    read_table,
    sniff_dialect,
    sniff_dialect_shared,
)

# Nokta-grupli trafo kodu: Turkce dosyada bu bir KOD'dur, ondalik sayi degil.
_TRAFO_KODLARI = [f"212.{index:03d}" for index in range(60)]


def _turkce_csv_yaz(yol, frame: pd.DataFrame) -> None:
    """Frame'i GDZ/EPDK bicimiyle yazar: ayirici ';', ondalik ',', kodlama cp1254."""
    metin = frame.to_csv(sep=";", index=False, decimal=",", lineterminator="\n")
    yol.write_bytes(metin.encode("cp1254"))


def _train_test_cifti(tmp_path) -> tuple:
    """Ayni kolonlari paylasan, YALNIZCA hedef kolonu farkli bir cift uretir."""
    ortak = {
        "TRAFO KODU": _TRAFO_KODLARI,
        "İLÇE": ["Ödemiş" if index % 2 else "Buca" for index in range(60)],
        "ABONE": list(range(100, 160)),
    }
    # Hedef FLOAT olmali: ``to_csv(decimal=",")`` yalnizca sayilari cevirir,
    # hazir string'e dokunmaz -- string verilirse dosya Turkce degil Ingilizce
    # ondalikli cikar ve senaryo test etmek istedigimiz sey olmaz.
    train = pd.DataFrame(
        {**ortak, "DAĞITILAN ENERJİ (MWh)": [round(1 + index / 10, 2) for index in range(60)]}
    )
    test = pd.DataFrame(ortak)

    train_yol, test_yol = tmp_path / "train.csv", tmp_path / "test.csv"
    _turkce_csv_yaz(train_yol, train)
    _turkce_csv_yaz(test_yol, test)
    return train_yol, test_yol


# --------------------------------------------------------------------------
# 1) train/test ayristirma tutarliligi
# --------------------------------------------------------------------------


def test_ortak_bicim_karariyla_train_ve_test_ayni_dtype_veriyor(tmp_path):
    """SOZLESME: hedef kolonunun yoklugu test'in dtype'larini degistiremez.

    OLCULDU (60 satir, cp1254, ';', ',' -- yalnizca train'de ondalikli hedef):
      BAGIMSIZ tespit : train binlik='.'  test binlik='yok'
                        trafo_kodu train=int64 test=str  -> UYUSMAYAN: 1
      ORTAK tespit    : iki dosya da AYNI bicim
                        trafo_kodu train=int64 test=int64 -> UYUSMAYAN: 0

    Sessiz varyanti oldurucudur: sayisal kolon test tarafinda kategoriye duser,
    CV temiz gorunur, model son adimda 'pandas dtypes must be int, float or
    bool' ile patlar veya daha kotusu tahmini bozar.

    DIKKAT -- ONCEKI SURUMDEKI YANLIS SOZLESME
    ------------------------------------------
    Bu test once ``sniff_dialect(train) == sniff_dialect(test)`` olmasini
    bekliyordu. O beklenti YALNIZCA bu fixture icin dogruydu: kanit dagilimi
    farkli bir cift (or. ortak kolon 4 haneli kesirli 'enlem') ayni bagimsiz
    tespitle yine ayrisiyor, sadece YONU degisiyordu -- olculdu, uyusmayan
    kolon sayisi ONCE 1, SONRA 1. Gercek cozum dosyalari BIRLIKTE koklamaktir.
    """
    train_yol, test_yol = _train_test_cifti(tmp_path)

    ortak = sniff_dialect_shared([train_yol, test_yol])
    train_frame = read_table(train_yol, verbose=False, dialect=ortak)
    test_frame = read_table(test_yol, verbose=False, dialect=ortak)

    ortak_kolonlar = [kolon for kolon in test_frame.columns if kolon in train_frame.columns]
    uyusmayan = [
        kolon
        for kolon in ortak_kolonlar
        if train_frame[kolon].dtype != test_frame[kolon].dtype
    ]
    assert uyusmayan == []
    assert train_frame["trafo_kodu"].dtype == test_frame["trafo_kodu"].dtype


def test_ortak_bicim_kanit_dagilimi_farkli_ciftte_de_tutuyor(tmp_path):
    """Denetimin gosterdigi karsi ornek: ortak kolon 4 haneli kesirli.

    Bagimsiz tespitte train ondalik=',' test ondalik='.' cikiyor ve 'enlem'
    train'de str, test'te float64 oluyordu (olculdu). Ortak karar bunu kapatir.
    """
    train_yol = tmp_path / "train.csv"
    test_yol = tmp_path / "test.csv"
    train_yol.write_text(
        "enlem;hedef\n" + "".join(f"38.4{i:03d};1,5\n" for i in range(40)),
        encoding="cp1254",
    )
    test_yol.write_text(
        "enlem\n" + "".join(f"38.4{i:03d}\n" for i in range(40)), encoding="cp1254"
    )

    bagimsiz_train = read_table(train_yol, verbose=False)
    bagimsiz_test = read_table(test_yol, verbose=False)
    assert bagimsiz_train["enlem"].dtype != bagimsiz_test["enlem"].dtype

    ortak = sniff_dialect_shared([train_yol, test_yol])
    birlikte_train = read_table(train_yol, verbose=False, dialect=ortak)
    birlikte_test = read_table(test_yol, verbose=False, dialect=ortak)
    assert birlikte_train["enlem"].dtype == birlikte_test["enlem"].dtype


def test_uc_ondalikli_mwh_binlik_sanilmiyor(tmp_path):
    """DENETIMIN BULDUGU KRITIK GERILEME.

    Ara surumde ';' ayiricili dosyada TAM 3 ondalikli Ingilizce sayilar binlik
    gruplama sanilip 1000 KAT buyuk TAMSAYIYA cevriliyordu -- sessizce:

        girdi  9.801   -> okunan np.int64(9801)   dtype=int64
        toplam 143.055 -> okunan 143055

    Hedef MWh oldugunda uc ondalik en olagan haldir; yani gercek yarisma
    verisinde neredeyse kesin tetiklenecekti. Nokta ancak POZITIF kanit varsa
    (virgul-ondalik ya da cok gruplu sayi) binlik sayilir.
    """
    yol = tmp_path / "mwh.csv"
    yol.write_text(
        "tarih;tuketim_mwh;trafo\n"
        + "".join(f"2024-01-{i:02d};9.80{i};TR000{i}\n" for i in range(1, 11)),
        encoding="cp1254",
    )
    frame = read_table(yol, verbose=False)
    assert frame["tuketim_mwh"].iloc[0] == pytest.approx(9.801)
    assert frame["tuketim_mwh"].dtype.kind == "f"


def test_gercek_turkce_bicim_hala_dogru_okunuyor(tmp_path):
    """Yanlis pozitif korumasi: '1.234,56' GERCEKTEN binlik + virgul ondalik."""
    yol = tmp_path / "tr.csv"
    yol.write_text(
        "tarih;tuketim;ad\n"
        + "".join(f"2024-01-{i:02d};1.234,{i:02d};ilce{i}\n" for i in range(1, 11)),
        encoding="cp1254",
    )
    frame = read_table(yol, verbose=False)
    assert frame["tuketim"].iloc[0] == pytest.approx(1234.01)


def test_cp1254_ve_noktali_virgul_tespiti_train_test_icin_ayni(tmp_path):
    """Kodlama ve ayirici karari da dosyadan dosyaya degismemeli.

    OLCULDU: her iki dosya da kodlama=cp1254/iso-8859-9, ayirici=';'.
    Turkce kolon adlari ('İLÇE', 'DAĞITILAN ENERJİ (MWh)') cp1254 baytlari
    urettigi icin utf-8 dali GURULTULU basarisiz olur -- istenen budur.
    """
    train_yol, test_yol = _train_test_cifti(tmp_path)

    train_bicim, test_bicim = sniff_dialect(train_yol), sniff_dialect(test_yol)

    assert train_bicim.encoding == test_bicim.encoding
    assert train_bicim.encoding in {"cp1254", "iso-8859-9"}
    assert train_bicim.delimiter == test_bicim.delimiter == ";"


def test_uc_haneli_kesir_ingilizce_ondalik_kaniti_sayilmaz():
    """'212.345' binlik gruplamadan AYIRT EDILEMEZ, ondalik kaniti olamaz.

    OLCULDU (60 adet '212.xxx' kodu, satir-satir tokenizasyon ile):
      eski kural _looks_like_en_number -> 60 kanit  (ondalik ','den '.'ya kayardi)
      yeni kural _kesin_en_ondalik     ->  0 kanit
      binlik kaniti                    -> 60
    """
    eski_kanit = sum(1 for kod in _TRAFO_KODLARI if _looks_like_en_number(kod))
    yeni_kanit = sum(1 for kod in _TRAFO_KODLARI if _kesin_en_ondalik(kod))
    binlik_kanit = sum(1 for kod in _TRAFO_KODLARI if _binlik_kaniti(kod))

    assert eski_kanit == 60
    assert yeni_kanit == 0
    assert binlik_kanit == 60


def test_belirsiz_nokta_kullanimi_sessizce_secilmiyor(tmp_path):
    """YANLIS-POZITIF DEGIL, GERCEK BELIRSIZLIK: uyari uretilmeli.

    Dosyada hem '1.234,56' (binlik) hem '12.5' (ondalik) varsa hangi kolonun
    hangi kurala tabi oldugunu dosyaya bakarak bilemeyiz. Sessiz secim yerine
    uyari veriyoruz -- OLCULDU: 2 belirsiz-olmayan ondalik token, 1 uyari.
    """
    yol = tmp_path / "belirsiz.csv"
    yol.write_bytes(
        "ad;tutar;oran\nA;1.234,56;12.5\nB;2.000,50;7.25\nC;3.000,75;9,5\n".encode("cp1254")
    )

    bicim = sniff_dialect(yol)

    assert len(bicim.warnings) == 1
    assert "binlik gruplama" in bicim.warnings[0]


# --------------------------------------------------------------------------
# 1b) YANLIS-POZITIF KORUMASI: masum dosyalar bozulmadi
# --------------------------------------------------------------------------


def test_gercek_binlik_ayirici_hala_dogru_sayiya_cevriliyor(tmp_path):
    """MASUM DURUM: '1.234.567,89' TEK sayidir ve oyle okunmaya devam etmeli.

    Binlik kaniti kurali genisletildi (artik ondaliksiz '212.345' de kanit);
    bu testin isi, genisletmenin ESKI dogru davranisi bozmadigini gostermek.
    OLCULDU: 1.234.567,89 -> 1234567.89 ve 12,5 -> 12.5 (ONCE de SONRA da).
    """
    yol = tmp_path / "tuik.csv"
    yol.write_bytes(
        "İL;TÜKETİM;ORAN\nİzmir;1.234.567,89;12,5\nMuğla;987.654,32;8,3\n".encode("cp1254")
    )

    frame = read_table(yol, verbose=False)

    assert frame["tuketim"].iloc[0] == pytest.approx(1_234_567.89)
    assert frame["tuketim"].iloc[1] == pytest.approx(987_654.32)
    assert frame["oran"].iloc[0] == pytest.approx(12.5)


def test_standart_virgul_csv_ne_degisti_ne_uyari_uretti(tmp_path):
    """MASUM DURUM: sirali utf-8 virgul CSV'sine hic dokunulmamali.

    OLCULDU: ayirici=',' ondalik='.' binlik=yok, uyari sayisi 0 (ONCE de SONRA da).
    """
    yol = tmp_path / "standart.csv"
    yol.write_text("a,b,c\n1,2.5,x\n3,4.5,y\n", encoding="utf-8")

    bicim = sniff_dialect(yol)

    assert (bicim.delimiter, bicim.decimal, bicim.thousands) == (",", ".", None)
    assert bicim.warnings == ()


def test_noktali_virgul_ama_gercek_ingilizce_ondalik_hala_taninir(tmp_path):
    """MASUM DURUM: ';' ayirici tek basina ondalik ','yi ZORUNLU kilmaz.

    Belirsiz-olmayan ondalik kanit baskinsa (OLCULDU: 5 kanit, 0 virgul kaniti)
    ondalik '.' secilir ve 1.5 gercekten 1.5 kalir -- 15 olmaz.
    """
    yol = tmp_path / "ingilizce.csv"
    yol.write_text("a;b\n1.5;2.25\n3.75;4.5\n5.125;6.5\n", encoding="utf-8")

    bicim = sniff_dialect(yol)
    frame = read_table(yol, verbose=False)

    assert bicim.decimal == "."
    assert frame["a"].tolist() == pytest.approx([1.5, 3.75, 5.125])


# --------------------------------------------------------------------------
# 2) format-bagimsiz kolon adlari
# --------------------------------------------------------------------------


def _ornek_frame() -> pd.DataFrame:
    return pd.DataFrame({"ID": ["R0", "R1"], "Dağıtılan Enerji (MWh)": [1.5, 2.5]})


def test_ayni_veri_csv_parquet_json_ayni_kolon_adlarini_veriyor(tmp_path):
    """SOZLESME: kolon adi normalizasyonu FORMATTAN bagimsizdir.

    OLCULDU (ayni iki kolon, uc formata yazilip read_any ile okundu):
      ONCE  csv     ['id', 'dagitilan_enerji_mwh']
            parquet ['ID', 'Dağıtılan Enerji (MWh)']
            json    ['ID', 'Dağıtılan Enerji (MWh)']
            -> CSV'den FARKLI kolon adi veren format sayisi: 2
      SONRA ucu de ['id', 'dagitilan_enerji_mwh'] -> FARKLI format sayisi: 0

    Zarar somuttu: parquet onbellekli train + CSV ornek submission karisiminda
    hedef kolon adi eslesmiyor ve pipeline 'hedef kolon belirlenemedi' ile
    duruyordu.
    """
    ham = _ornek_frame()
    csv_yol, parquet_yol, json_yol = (
        tmp_path / "veri.csv",
        tmp_path / "veri.parquet",
        tmp_path / "veri.json",
    )
    csv_yol.write_bytes(ham.to_csv(sep=";", index=False, decimal=",").encode("cp1254"))
    ham.to_parquet(parquet_yol, index=False)
    ham.to_json(json_yol, orient="records")

    kolonlar = {
        etiket: list(read_any(yol, verbose=False).columns)
        for etiket, yol in (
            ("csv", csv_yol),
            ("parquet", parquet_yol),
            ("json", json_yol),
        )
    }

    beklenen = ["id", "dagitilan_enerji_mwh"]
    farkli = [etiket for etiket, adlar in kolonlar.items() if adlar != beklenen]
    assert farkli == []


def test_parquet_ham_kolon_adlarini_attrs_icinde_sakliyor(tmp_path):
    """Submission basligini orijinale cevirebilmek icin esleme SART.

    OLCULDU: ONCE parquet'te attrs['original_columns'] YOK; SONRA VAR ve
    {'ID': 'id', 'Dağıtılan Enerji (MWh)': 'dagitilan_enerji_mwh'} esitligini
    tasiyor. submission.py bu sozlukten ters cevirip Kaggle basligini kuruyor.
    """
    parquet_yol = tmp_path / "veri.parquet"
    _ornek_frame().to_parquet(parquet_yol, index=False)

    frame = read_any(parquet_yol, verbose=False)

    esleme = frame.attrs["original_columns"]
    assert esleme["Dağıtılan Enerji (MWh)"] == "dagitilan_enerji_mwh"
    assert frame.attrs["source_path"] == str(parquet_yol)


def test_parquet_onbellegi_ikinci_okumada_adlari_tekrar_degistirmiyor(tmp_path):
    """YANLIS-POZITIF KORUMASI: normalizasyon idempotent olmali.

    Kalip 'ham CSV -> parquet onbellek -> parquet oku' oldugu icin normalize
    edilmis adlar ikinci kez normalize edilir. OLCULDU: ikinci okumada kolon
    adlari degismiyor ve esleme kimlik eslemesi ('id' -> 'id').
    """
    kaynak_yol, onbellek_yol = tmp_path / "kaynak.parquet", tmp_path / "onbellek.parquet"
    _ornek_frame().to_parquet(kaynak_yol, index=False)

    ilk = read_any(kaynak_yol, verbose=False)
    ilk.to_parquet(onbellek_yol, index=False)
    ikinci = read_any(onbellek_yol, verbose=False)

    assert list(ikinci.columns) == list(ilk.columns)
    assert ikinci.attrs["original_columns"] == {
        "id": "id",
        "dagitilan_enerji_mwh": "dagitilan_enerji_mwh",
    }


def test_normalize_kapatilinca_ham_adlar_her_formatta_korunuyor(tmp_path):
    """YANLIS-POZITIF KORUMASI: ``normalize_column_names=False`` hala kacis yolu.

    Yeni normalizasyon parquet/json dalina da girdigi icin, kapatma bayraginin
    o dallarda da calistigini gosteriyoruz. OLCULDU: iki formatta da ham adlar
    ['ID', 'Dağıtılan Enerji (MWh)'] ve attrs'te 'original_columns' YOK.
    """
    parquet_yol, csv_yol = tmp_path / "veri.parquet", tmp_path / "veri.csv"
    ham = _ornek_frame()
    ham.to_parquet(parquet_yol, index=False)
    csv_yol.write_bytes(ham.to_csv(sep=";", index=False, decimal=",").encode("cp1254"))

    for yol in (parquet_yol, csv_yol):
        frame = read_any(yol, normalize_column_names=False, verbose=False)
        assert list(frame.columns) == ["ID", "Dağıtılan Enerji (MWh)"]
        assert "original_columns" not in frame.attrs
