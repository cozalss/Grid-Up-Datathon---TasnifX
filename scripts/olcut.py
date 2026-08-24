"""TEST KARISIMINA AGIRLIKLANDIRILMIS OLCUT -- onem agirliklandirma.

NEDEN BU MODUL VAR
------------------
Gecen gece LB, ``kis26`` uzerinde iyilesen bir revizyonun butununu +0,00414
ile curuttu (docs/39 §3). Teshis: hata MODELDE degil OLCUTTE idi. ``kis26``in
kVA karisimi ve mevsimsel isareti testten farkli, dolayisiyla o blok uzerinde
olculen "iyilesme" testte gecerli degildi.

Ayni hastalik SICAK tarafta da duruyor ve HIC tedavi edilmedi (docs/39 §8):

    t_son_kayit_yasi >= 1     TEST sicak tarafin %15,5'i
                              CV bloklarinin %1,7'si
    kis26 sicak skoru         0,77882  ->  0,87811   bayatliga agirliklandirilinca

Yani sicak uzman icin yerel olcut testten 0,10 RMSLE iyimser. Rekor defterinde
sicak skorun v33'ten v46'ya kadar 0,74263'te CAKILI kalmasinin bir sebebi de bu:
yanlis karisimi olcen bir sayi, dogru degisiklige tepki vermez.

NE YAPIYOR
----------
Dogrulama satirlarini testin ORTAK dagilimina yeniden agirliklandirir:

    w_i = p_test(tabaka_i) / p_dogrulama(tabaka_i)

Kovaryat kayma altinda standart onem agirliklandirma kestiricisi. ``p(y|x)``
bloklar arasi ayni kaldigi surece YANSIZ; kayma yalnizca ``p(x)``de ise dogru
sayiyi verir.

Bedeli VARYANS. Bir tabaka dogrulamada seyrekse agirligi patlar ve kestirim
birkac satirin esiri olur. Bu yuzden modul her cagride ETKIN ORNEK BUYUKLUGU
bildirir ve agirliklari kirpar. ESS dusukse sayiya guvenilmez -- fonksiyon
bunu susarak gecmez, ``tani`` sozlugunde acikca doner.

KRITIK KURAL: KOVA KENARLARI TEK YERDEN
---------------------------------------
Kenarlar YALNIZCA test cercevesinden turetilir ve iki tarafa da ayni dizi
verilir. Kenarlari iki tarafta ayri hesaplamak hucreleri kaydirir ve sessizce
sahte sonuc uretir -- ``deney_soguk_taban.py``de tam olarak bu olmustu
(``ilce_kova`` uydurma bir 1,97083 skorlamisti).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Bayatlik kova kenarlari (gun). ``t_son_kayit_yasi`` = son kayittan ozet
#: penceresi sonuna gecen gun. Sabit kenar kullaniliyor cunku bu eksende
#: ilgilendigimiz esikler dogal: "bugun kayit var" (0), "bir hafta icinde",
#: "bir ay icinde", "bir ceyrek", "daha eski".
BAYATLIK_KENARLARI = (0.0, 1.0, 7.0, 30.0, 90.0, np.inf)

#: Ufuk kova kenarlari (gun). Test 1..122; aylik dilimler.
UFUK_KENARLARI = (0.0, 31.0, 61.0, 91.0, np.inf)

#: kVA kova sayisi. Kenarlar testin ``log1p(guc)`` KUANTILLERINDEN gelir,
#: yani her kova testte esit paya sahip olur -- seyrek tabaka uretmez.
GUC_KOVA_SAYISI = 8

#: Agirlik kirpma tavani (ortalamaya gore kat). Kirpma yanliligi biraz geri
#: getirir ama tek bir seyrek tabakanin skoru ele gecirmesini engeller.
KIRPMA = 20.0

#: ESS bu esigin altina duserse kestirim GURULTULU sayilir ve rapor uyarir.
ESS_ESIGI = 0.20


def guc_kenarlari(test: pd.DataFrame, kova: int = GUC_KOVA_SAYISI) -> np.ndarray:
    """Testin ``log1p(guc)`` kuantillerinden kova kenarlari.

    TEK kaynak testtir. Donen dizi hem test hem dogrulama icin kullanilir.
    """
    lg = np.log1p(test["guc"].to_numpy(dtype="float64"))
    q = np.linspace(0.0, 1.0, kova + 1)
    kenar = np.unique(np.quantile(lg, q))
    kenar[0] = -np.inf
    kenar[-1] = np.inf
    return kenar


def _kova(deger: np.ndarray, kenar) -> np.ndarray:  # noqa: ANN001
    """Yarim acik ``[alt, ust)`` kovalama. NaN -> -1 (kendi tabakasi)."""
    d = np.asarray(deger, dtype="float64")
    k = np.searchsorted(np.asarray(kenar, dtype="float64"), d, side="right") - 1
    k = np.clip(k, 0, len(kenar) - 2)
    return np.where(np.isnan(d), -1, k).astype("int64")


def tabaka_kodu(
    cerceve: pd.DataFrame,
    guc_kenar: np.ndarray,
    eksenler: tuple[str, ...] = ("bayatlik", "guc", "ufuk"),
) -> np.ndarray:
    """Satir basina tam sayi tabaka kodu.

    ``eksenler`` ile hangi eksenlerin carpima girecegi secilir. Tek eksen
    (marjinal) dusuk varyansli ama kismi; uc eksen (ortak) tam ama seyrek.
    Ikisini de olcup karsilastirmak dogru kullanimdir.
    """
    parcalar = []
    if "bayatlik" in eksenler:
        yas = (
            cerceve["t_son_kayit_yasi"].to_numpy(dtype="float64")
            if "t_son_kayit_yasi" in cerceve.columns
            else np.full(len(cerceve), np.nan)
        )
        parcalar.append(_kova(yas, BAYATLIK_KENARLARI) + 1)  # -1 -> 0
    if "guc" in eksenler:
        parcalar.append(_kova(np.log1p(cerceve["guc"].to_numpy(dtype="float64")), guc_kenar) + 1)
    if "ufuk" in eksenler:
        parcalar.append(_kova(cerceve["ufuk_gun"].to_numpy(dtype="float64"), UFUK_KENARLARI) + 1)
    if not parcalar:
        raise ValueError("en az bir eksen gerekli")
    kod = np.zeros(len(cerceve), dtype="int64")
    for p in parcalar:
        kod = kod * 64 + p
    return kod


def test_agirliklari(
    dogrulama: pd.DataFrame,
    test: pd.DataFrame,
    guc_kenar: np.ndarray,
    eksenler: tuple[str, ...] = ("bayatlik", "guc", "ufuk"),
    kirpma: float = KIRPMA,
) -> tuple[np.ndarray, dict]:
    """Dogrulama satirlarini testin dagilimina tasiyan agirliklar.

    Doner: ``(w, tani)``. ``w`` ortalamasi 1,0 olacak sekilde olceklenir --
    boylece agirlikli RMSLE, agirliksiz olanla ayni buyukluk mertebesinde
    okunur.

    ``tani`` icinde:
        ess_orani     etkin ornek buyuklugu / satir sayisi  (1,0 = kayma yok)
        kirpilan      tavana carpan satir orani
        kapsanmayan   testte var, dogrulamada HIC olmayan tabakalarin test payi
    """
    kod_d = tabaka_kodu(dogrulama, guc_kenar, eksenler)
    kod_t = tabaka_kodu(test, guc_kenar, eksenler)

    pay_t = pd.Series(kod_t).value_counts(normalize=True)
    pay_d = pd.Series(kod_d).value_counts(normalize=True)

    # Testte var ama dogrulamada YOK olan tabakalar: onem agirliklandirma bu
    # bolgeyi hicbir agirlikla temsil edemez. Susmak yerine olcup bildiriyoruz.
    eksik = pay_t.index.difference(pay_d.index)
    kapsanmayan = float(pay_t.reindex(eksik).sum()) if len(eksik) else 0.0

    w = pay_t.reindex(kod_d).to_numpy(dtype="float64") / pay_d.reindex(kod_d).to_numpy(
        dtype="float64"
    )
    w = np.nan_to_num(w, nan=0.0, posinf=0.0)
    if w.sum() <= 0:
        raise RuntimeError("butun agirliklar sifir -- tabaka tanimi uyusmuyor")
    w = w / w.mean()
    kirpilan = float((w > kirpma).mean())
    w = np.minimum(w, kirpma)
    w = w / w.mean()

    ess = float(w.sum() ** 2 / np.dot(w, w))
    tani = {
        "n": int(len(w)),
        "tabaka": int(len(pay_d)),
        "ess_orani": ess / len(w),
        "kirpilan": kirpilan,
        "kapsanmayan": kapsanmayan,
        "guvenilir": bool(ess / len(w) >= ESS_ESIGI and kapsanmayan < 0.02),
    }
    return w, tani


def agirlikli_rmsle(gercek: np.ndarray, tahmin: np.ndarray, w: np.ndarray | None = None) -> float:
    """``w`` verilmezse duz RMSLE ile birebir ayni sayiyi verir."""
    g = np.log1p(np.clip(np.asarray(gercek, dtype="float64"), 0.0, None))
    t = np.log1p(np.clip(np.asarray(tahmin, dtype="float64"), 0.0, None))
    kare = (g - t) ** 2
    if w is None:
        return float(np.sqrt(kare.mean()))
    return float(np.sqrt(np.dot(w, kare) / w.sum()))
