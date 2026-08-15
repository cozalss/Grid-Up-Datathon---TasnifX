"""KALAN BULGULAR -- submission, profiling, temporal, full_pipeline.

Bu dosya cekismeli dogrulama turunun dort ayri dosyada birakugi bulgulari
kapatir. Her testin docstring'inde OLCULEN ONCE/SONRA sayilari yazilidir;
sayilar ``scratchpad/olc_once.py`` ile duzeltmeden once ve sonra ayni betikle
alindi.

Dort bulgu:
  1. ``submission.write_submission`` 3+ kolonlu ornege karsi SESSIZCE 2 kolon
     yaziyordu (``validate=False`` yolunda).
  2. ``profiling`` ID-BENZERI isareti surekli float kolonlarda %100 yanlis
     pozitif uretiyor, TEKRARLI monoton sayaci ise hic yakalamiyordu.
  3. ``features.temporal`` AYNI zaman kolonunu iki farkli kuralla
     ayristiriyordu (``format="mixed"`` vs bicimsiz).
  4. ``scripts/full_pipeline.py`` hava merge'inde ``validate=`` -- ONCEKI
     turlarda ZATEN KAPATILMIS; burada yalnizca regresyon nobeti var.

Her testin yaninda bir YANLIS-POZITIF korumasi vardir: duzeltmenin masum
durumu bozmadigini gosteren test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.features.temporal import add_calendar_features, shared_origin
from gridup.profiling import profile
from gridup.submission import write_submission

DEPO = Path(__file__).resolve().parents[1]


def _uc_kolonlu_ornek() -> pd.DataFrame:
    """Bilesik anahtarli (tarih + ilce) ornek submission."""
    return pd.DataFrame(
        {
            "Tarih": ["2025-10-01", "2025-10-02", "2025-10-03"],
            "Ilce": ["konak", "bornova", "buca"],
            "Dagitilan Enerji (MWh)": [0.0, 0.0, 0.0],
        }
    )


# --------------------------------------------------------------------------- 1
class TestSubmissionKolonSayisi:
    """write_submission yapisal olarak 2 kolon uretir -- bunu SOYLEMELI."""

    def test_uc_kolonlu_ornek_dogrulama_kapaliyken_bile_hata_veriyor(self, tmp_path):
        """OLCULDU: validate=False ile diske yazilan kolon sayisi 2 -> istisna.

        ONCE : uc kolonlu ornege karsi "Yazildi: ... (3 satir)" basildi ve
               dosyanin basligi ['Tarih', 'Dagitilan Enerji (MWh)'] oldu --
               yani 3 kolonluk bir yarismaya 2 kolonluk dosya gonderilecekti,
               hicbir uyari cikmadan.
        SONRA: ValueError, ve dosya HIC olusmuyor.
        """
        hedef = tmp_path / "gonderim.csv"

        with pytest.raises(ValueError, match="3 kolonlu"):
            write_submission(
                np.array(["2025-10-01", "2025-10-02", "2025-10-03"]),
                np.array([1.0, 2.0, 3.0]),
                hedef,
                sample=_uc_kolonlu_ornek(),
                validate=False,
            )

        assert not hedef.exists(), "Hata firlatilmasina ragmen dosya yazilmis."

    def test_hata_mesaji_cikis_yolunu_gosteriyor(self, tmp_path):
        """Mesaj "eksik kolon" degil, "bu fonksiyon uretemez" demeli.

        ONCE : "Eksik kolon(lar): ['Ilce']" -- kullaniciyi KENDI verisinde
               olmayan bir hatayi aramaya yolluyordu.
        SONRA: mesaj write_submission'in sinirini ve elle kurulacak alternatifi
               (validate_submission) iceriyor.
        """
        with pytest.raises(ValueError) as bilgi:
            write_submission(
                np.array(["a", "b", "c"]),
                np.array([1.0, 2.0, 3.0]),
                tmp_path / "x.csv",
                sample=_uc_kolonlu_ornek(),
            )

        mesaj = str(bilgi.value)
        assert "write_submission yalnizca (id, hedef)" in mesaj
        assert "validate_submission" in mesaj

    def test_iki_kolonlu_ornek_normal_yazilmaya_devam_ediyor(self, tmp_path):
        """YANLIS-POZITIF KORUMASI: yeni kapi masum yolu kapatmamali.

        OLCULDU: 2 kolonlu ornekle yazilan dosya 3 satir, kolonlar
        ['ID', 'Dagitilan Enerji (MWh)'] -- duzeltmeden once de sonra da ayni.
        """
        ornek = pd.DataFrame(
            {"ID": ["a", "b", "c"], "Dagitilan Enerji (MWh)": [0.0, 0.0, 0.0]}
        )
        hedef = tmp_path / "gonderim.csv"

        write_submission(
            np.array(["a", "b", "c"]), np.array([1.0, 2.0, 3.0]), hedef, sample=ornek
        )

        yazilan = pd.read_csv(hedef)
        assert list(yazilan.columns) == ["ID", "Dagitilan Enerji (MWh)"]
        assert len(yazilan) == 3


# --------------------------------------------------------------------------- 2
def _panel_sayaclarla(satir_karistir: bool = False) -> pd.DataFrame:
    """96 ilce x 240 gun paneli; haftalik yukleme partisi sayaci ile."""
    rng = np.random.default_rng(7)
    ilceler = [f"ILCE_{i:02d}" for i in range(96)]
    gunler = pd.date_range("2024-01-01", periods=240, freq="D")
    frame = pd.DataFrame(
        [(ilce, gun) for gun in gunler for ilce in ilceler], columns=["ilce", "tarih"]
    )
    gun_no = (frame["tarih"] - gunler[0]).dt.days
    frame["gun_no"] = gun_no.astype("int32")
    frame["parti_no"] = (gun_no // 7).astype("int32")
    frame["kayit_no"] = np.arange(len(frame), dtype="int64")
    frame["sicaklik"] = rng.normal(18, 6, len(frame))
    frame["nem"] = rng.normal(60, 12, len(frame))
    frame["kesinti"] = rng.gamma(2.0, 30.0, len(frame))
    if satir_karistir:
        frame = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    return frame


def _isaretli(frame: pd.DataFrame, onek: str) -> list[str]:
    """Verilen isareti alan kolon adlari."""
    return [
        kolon.name
        for kolon in profile(frame).columns
        if any(isaret.startswith(onek) for isaret in kolon.flags)
    ]


class TestProfilingIdVeMonotonIsareti:
    """ID-BENZERI anlamli olmali, TEKRARLI monoton sayac kacmamali."""

    def test_surekli_float_kolonlar_artik_id_benzeri_isareti_almiyor(self):
        """OLCULDU: ID-BENZERI alan kolon sayisi 4 -> 1.

        ONCE : ['kayit_no', 'sicaklik', 'nem', 'kesinti'] -- ucu yanlis
               pozitif, ve 'kesinti' HEDEFIN KENDISIYDI. Her float kolon
               tanim geregi benzersizdir; uyari boylece anlamini yitiriyordu.
        SONRA: ['kayit_no'] -- yalnizca tamsayi kimlik kolonu.
        """
        isaretliler = _isaretli(_panel_sayaclarla(), "ID-BENZERI")

        assert isaretliler == ["kayit_no"]
        assert "sicaklik" not in isaretliler
        assert "kesinti" not in isaretliler

    def test_tekrarli_monoton_sayac_artik_yakalaniyor(self):
        """OLCULDU: 'parti_no' icin uretilen isaret sayisi 0 -> 1.

        parti_no = gun_no // 7 (haftalik yukleme partisi): 23.040 satirda
        yalnizca 35 farkli deger alir, yani benzersizlik orani 0.00152 --
        ID kapisina (0.98) hicbir zaman takilmaz. Ama satir sirasiyla monoton
        artar ve train/holdout araliklari neredeyse hic ortusmez
        (train [0, 28] / holdout [28, 34]): gercek bir zaman sizintisi.

        ONCE : TEKRARLI MONOTON isareti alan kolonlar = []
        SONRA: ['gun_no', 'parti_no']
        """
        isaretliler = _isaretli(_panel_sayaclarla(), "TEKRARLI MONOTON")

        assert "parti_no" in isaretliler
        assert "gun_no" in isaretliler

    def test_gercek_id_kolonu_hala_isaretleniyor(self):
        """YANLIS-POZITIF KORUMASI: dogru pozitif kaybolmamali.

        OLCULDU: 'kayit_no' (benzersiz int64) ONCE de SONRA da ID-BENZERI
        isareti aliyor; tamsayi kapisi metin ID'leri de gecirmeli.
        """
        frame = _panel_sayaclarla()
        frame["musteri_kodu"] = [f"MST{i:06d}" for i in range(len(frame))]

        isaretliler = _isaretli(frame, "ID-BENZERI")

        assert "kayit_no" in isaretliler
        assert "musteri_kodu" in isaretliler

    def test_karisik_sirali_sayac_monoton_isareti_almiyor(self):
        """YANLIS-POZITIF KORUMASI: isaret SIRAYA bagli, degere degil.

        Ayni parti_no kolonu satirlari karistirilmis bir frame'de artik
        monoton degildir -- yani satir sirasi uzerinden zaman sizdirmaz.
        OLCULDU: karistirilmis panelde TEKRARLI MONOTON isareti alan kolon
        sayisi = 0.
        """
        isaretliler = _isaretli(_panel_sayaclarla(satir_karistir=True), "TEKRARLI MONOTON")

        assert isaretliler == []


# --------------------------------------------------------------------------- 3
TR_TARIHLER = ["01.10.2025", "02.10.2025", "03.10.2025", "15.10.2025", "20.10.2025"]


class TestTemporalTekAyristirmaKurali:
    """shared_origin ve add_calendar_features AYNI takvimi uretmeli."""

    def test_tr_gg_aa_yyyy_kolonunda_iki_yol_ayni_takvimi_uretiyor(self):
        """OLCULDU: shared_origin 2025-01-10 -> 2025-10-01.

        ONCE (ayni kolon, iki kural):
          shared_origin (format="mixed") -> 2025-01-10  (ay-once okundu)
          add_calendar_features (bicimsiz) -> gun>12 olan 2 satir NaT,
            "%40.00 gecersiz tarih" uyarisi, ay kolonu [1, 2, 3, nan, nan]
        SONRA (ikisi de validation.parse_time_series):
          shared_origin -> 2025-10-01
          ay kolonu     -> [10, 10, 10, 10, 10]
          gun kolonu    -> [1, 2, 3, 15, 20]
        """
        frame = pd.DataFrame({"tarih": TR_TARIHLER})

        origin = shared_origin(frame, time_column="tarih")
        cikti = add_calendar_features(frame, "tarih", origin=origin)

        assert origin == pd.Timestamp("2025-10-01")
        assert list(cikti["tarih_ay"]) == [10, 10, 10, 10, 10]
        assert list(cikti["tarih_gun"]) == [1, 2, 3, 15, 20]
        assert list(cikti["tarih_gun_sayaci"]) == [0, 1, 2, 14, 19]

    def test_karisik_iso_ve_tr_kolonu_sessizce_cozulmuyor(self):
        """Ayni kolonda ISO ve TR satirlari birlikteyse DURULUR.

        Ara surum bu kolonu sessizce coziyordu ve olculen sonuc soydu:

            ham          : ['2024-03-01', '15.03.2024', '05.03.2024']
            dogru okuma  : ['2024-03-01', '2024-03-15', '2024-03-05']
            ara surum    : ay=[3, 3, 5]  gun=[1, 15, 3]
                           -> '15.03.2024' GUN-ONCE, '05.03.2024' AY-ONCE
                           -> UYARI YOK, ISTISNA YOK

        Yani AYNI kolona iki farkli kural uygulaniyordu -- tam da bulgunun
        kapatmayi vaat ettigi kusur. Ustelik daha eski surum ayni girdide
        gurultulu bir uyari veriyordu: sessiz yanlis deger, gorunur yanlis
        degerden KOTUDUR.

        Bir ISO satiri gorulunce toplanan gun-once kaniti artik cope
        atilmiyor; iki bicim birlikteyse "belirsiz" denir ve hata firlatilir.
        """
        frame = pd.DataFrame(
            {"tarih": ["2024-03-01", "2024-03-02", "15.03.2024", "2024-03-04"]}
        )

        with pytest.raises(ValueError, match="BELIRSIZ"):
            add_calendar_features(frame, "tarih")

    def test_saf_iso_ve_saf_tr_kolonlari_sorunsuz_cozuluyor(self):
        """Yanlis pozitif korumasi: TEK bicimli kolonlar calismaya devam eder."""
        iso = add_calendar_features(
            pd.DataFrame({"tarih": ["2024-03-01", "2024-03-02", "2024-03-15"]}), "tarih"
        )
        assert list(iso["tarih_gun"]) == [1, 2, 15]

        tr = add_calendar_features(
            pd.DataFrame({"tarih": ["01.03.2024", "02.03.2024", "15.03.2024"]}), "tarih"
        )
        assert list(tr["tarih_gun"]) == [1, 2, 15]

    def test_bicimi_kanitlanamayan_kolon_sessizce_tahmin_edilmiyor(self):
        """Butun gunleri <= 12 olan bir kolonda iki okuma da hatasiz calisir.

        '01.10.2025' hem 1 Ekim hem 10 Ocak olabilir ve pandas ikisini de
        sorunsuz uretir -- yanlis secim SESSIZ kalir. Repo kurali: belirsizse
        hata firlat.

        OLCULDU: ONCE add_calendar_features ay kolonu [1, 2] uretiyordu
        (kanitsiz ay-once secimi); SONRA ValueError.
        """
        frame = pd.DataFrame({"tarih": ["01.10.2025", "02.10.2025"]})

        with pytest.raises(ValueError, match="BELIRSIZ"):
            add_calendar_features(frame, "tarih")

    def test_karisik_saat_dilimi_hangi_frame_oldugunu_soyluyor(self):
        """OLCULDU: TypeError -> ValueError (frame kimligi mesajda).

        ONCE : TypeError "Cannot compare tz-naive and tz-aware timestamps" --
               hangi frame'in veya kolonun suclu oldugunu SOYLEMIYORDU.
        SONRA: ValueError "... frame#0=tz-aware, frame#1=tz-naive ..."
        """
        train = pd.DataFrame({"tarih": pd.to_datetime(["2024-03-01T00:00:00+03:00"])})
        test = pd.DataFrame({"tarih": pd.to_datetime(["2024-04-01"])})

        with pytest.raises(ValueError, match="TUTARSIZ saat dilimi"):
            shared_origin(train, test, time_column="tarih")

    def test_iso_kolonda_davranis_degismedi(self):
        """YANLIS-POZITIF KORUMASI: ISO tarihler dun ne veriyorsa bugun de o.

        OLCULDU: train 2024-01-01..2025-09-30, test 2025-10-01..2025-12-31 icin
        test gun sayacinin en kucugu (639) train'in en buyugunden (638) buyuk --
        duzeltmeden once de sonra da.
        """
        train = pd.DataFrame({"tarih": pd.date_range("2024-01-01", "2025-09-30", freq="D")})
        test = pd.DataFrame({"tarih": pd.date_range("2025-10-01", "2025-12-31", freq="D")})

        origin = shared_origin(train, test, time_column="tarih")
        train_cikti = add_calendar_features(train, "tarih", origin=origin)
        test_cikti = add_calendar_features(test, "tarih", origin=origin)

        assert origin == pd.Timestamp("2024-01-01")
        assert int(train_cikti["tarih_gun_sayaci"].max()) == 638
        assert int(test_cikti["tarih_gun_sayaci"].min()) == 639

    def test_bozuk_tarihler_hala_tolere_ediliyor(self):
        """YANLIS-POZITIF KORUMASI: bicim kapisi, bozuk hucre kapisi DEGIL.

        Gercek veride bos hucre ve '00.00.0000' kacinilmazdir; pipeline tek bir
        kotu hucre yuzunden durmamali. OLCULDU: 3 satirin 1'i bozukken cikti
        yine 3 satir ve tam olarak 1 NaN tasiyor -- duzeltmeden once de sonra da.
        """
        frame = pd.DataFrame({"tarih": ["2024-01-01", "bozuk-tarih", "2024-01-03"]})

        cikti = add_calendar_features(frame, "tarih")

        assert len(cikti) == 3
        assert int(cikti["tarih_ay"].isna().sum()) == 1


# --------------------------------------------------------------------------- 4
class TestHavaMergeCogalmaNobeti:
    """ZATEN KAPALI bulgu -- yalnizca geri donmesin diye nobetci."""

    def test_full_pipeline_hava_merge_validate_korumasini_tasiyor(self):
        """OLCULDU: kaynakta validate="many_to_one" gecen sayi 1 (ONCE de 1).

        Bulgu onceki turda kapatilmis. Koruma kaldirilirsa ne olacagi bu
        makinede olculdu: tekrarli (konum, tarih) tasiyan bir hava tablosuyla
        merge girdi 3 satiri 4 satira cikariyor ve hedef kutlesi 6.0 -> 7.0
        oluyor; validate="many_to_one" ile ayni cagri MergeError firlatiyor.
        """
        kaynak = (DEPO / "scripts" / "full_pipeline.py").read_text(encoding="utf-8")

        assert 'validate="many_to_one"' in kaynak

    def test_tekrarli_hava_satiri_gercekten_cogaltiyor(self):
        """Korumanin NEDEN gerektiginin olcumu -- pandas davranisi kanit.

        OLCULDU: validate yokken 3 satir -> 4 satir, hedef kutlesi 6.0 -> 7.0.
        """
        panel = pd.DataFrame(
            {
                "_il_key": ["izmir", "izmir", "manisa"],
                "tarih": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01"]),
                "hedef": [1.0, 2.0, 3.0],
            }
        )
        hava = pd.DataFrame(
            {
                "il_key": ["izmir", "izmir", "manisa"],
                "tarih": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-01"]),
                "isitma_derece_gun": [5.0, 6.0, 7.0],
            }
        )
        ortak = {
            "left_on": ["_il_key", "tarih"],
            "right_on": ["il_key", "tarih"],
            "how": "left",
        }

        korumasiz = panel.merge(hava, **ortak)

        assert len(korumasiz) == 4
        assert float(korumasiz["hedef"].sum()) == 7.0
        with pytest.raises(pd.errors.MergeError):
            panel.merge(hava, validate="many_to_one", **ortak)
