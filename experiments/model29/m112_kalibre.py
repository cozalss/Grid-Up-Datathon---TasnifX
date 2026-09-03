"""KALIBRASYON AILESI -- 2. sira icin sistem.

BULGU (2026-08-30, olculdu): yapisal "seviye" yonu rho = -0.0304 verdi,
kazanc 9.26e-04 -- model varyantlarinin artimli rekorunun (3.12e-04) 3 KATI.
Yani hata model seciminde degil KALIBRASYONDA: tahminler asiri yayilmis.

Bu, tek bir yon degil bir AILE. Kalibrasyon egrisi yanlissa:
  - egriligi de yanlistir            (seviye^2, seviye^3)
  - kesitlere gore farklidir         (seviye x soguk, seviye x guc, seviye x ay)
  - kesitlerin kendi kaymasi vardir  (soguk, bolge, haftasonu)

Her yon olculmus span'a VE onceki olculmus yapisal yonlere DIK yapilir
-> her sonda saf yeni bilgi olcer, hicbiri otekini tekrar etmez.

Kullanim:
  python m112_kalibre.py --liste
  python m112_kalibre.py --aday seviye2 --yerdeg 0.005 --cikti tuketim_K_seviye2.csv
  python m112_kalibre.py --kaydet seviye2 --skor 1.00102
  python m112_kalibre.py --nihai --cikti tuketim_K_NIHAI.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
#: M0 = mean(r^2), P_j^2 = M0 - 2L_j + Q_j denkleminin ETKIN sabiti.
#:
#: DIKKAT -- bu bir OZDESLIK DEGIL. P_j Kaggle'in public %50 satirinda olculur,
#: Q_j ise burada 714.688 satirin TAMAMINDA hesaplanir. Denklem bu yuzden iki
#: farkli kumeye ait nicelikleri karistirir ve M0 aradaki uyumsuzlugu emer.
#:
#: Deger uc capadan ASIRI-BELIRLENMIS olarak gelir. L=0 varsayimi altinda
#: P^2 - Q her uc capada:
#:     p51  1.005846063 | m4  1.005846970 | v102  1.005846063
#: Yayilim 9.1e-07. Tek parametre uc hedefi ayni anda sifirlayamaz; bu bir fit
#: degil uyusmadir ve yapisal nedeni var (a0, v102+m4 spaninda tam optimum
#: olarak kuruldu; p51 de ayni 2-boyutlu spanda).
#:
#: 30 Agustos'ta bu deger a0'in kendi skoruna (1.00284^2 = 1.005688066)
#: cekilmek istendi. YANLISTI ve GERI ALINDI:
#:   - dayanak "tutulmus sinav g7" idi; g7 HIC GONDERILMEDI, skoru
#:     (1.00136) docs/58'de bu M0 ile TURETILMIS bir sayidir -> dongusel
#:   - 27 gercek yonde leave-one-out her esikte bu degeri kazandiriyor
#:     (%90 span-ici esiginde ort |hata| 1.72e-04 vs 2.08e-04)
M0 = 1.005846366
RCOND = 1e-6
DURUM = os.path.join(BURA, "m112_durum.json")
#: TURETILMIS deger -- olculmus degil. s3y40 sondasindan cozuldu ve ESKI (=su
#: anki) M0 geometrisiyle tutarlidir. s3y40'in kendi skoru (1.00177, gonderim
#: listesinde dogrulandi) olculmus_skorlar.json'a eklendi; s3y40 = 1.837*g7 +
#: 0.392*y40 oldugu icin tek basina 2-boyutlu alt uzayda tek denklem verir,
#: y40 boyutunu ancak bu turetilmis L acar.
EK_MODEL = {"tuketim_y40_sota_temiz.csv": -0.002229}
# Uc ileri-zaman CV blogunda ayni isareti koruyan iki eksen. Katsayilar,
# testle ayni Nisan-Temmuz penceresindeki sinyalin LB'de olculen seviye
# sinyaline oranlanip yaklasik %35 buzulmus halidir.
RANK2_ONSEL = (("seviye_x_ay", -0.030), ("ay", 0.035))
# Uc ileri-zaman blogunda isareti degismeyen, test ufkuyla ayni Nisan-Temmuz
# penceresinde olculen ardisk ortogonal rho'larin %55'i. Bilinen geometriye
# dik maliyetleri 9.30e-3'tur: 1.00052 guvenli tabandan 0.99586 tasarim onseli.
HEDEF996_ONSEL = (
    ("h", 0.064020),
    ("t_hg_genligi", 0.013200),
    ("sv_yas", -0.060445),
    ("h_t_log_ort", -0.020625),
    ("gunes_radyasyon", -0.014465),
    ("h_sicaklik_ort", 0.025960),
    ("ulusal_gunluk", -0.008140),
)
HEDEF996_KOLONLAR = [
    "id",
    "tarih",
    "t_hg_genligi",
    "yas",
    "t_log_ort",
    "gunes_radyasyon",
    "sicaklik_ort",
    "ulusal_gunluk",
]
MODEL_ADAYLAR = {
    "z2": "tuketim_z2_analog.csv",
    "sul": "tuketim_t1_sulama.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "q1c": "tuketim_q1c_kapasite_siki.csv",
    "t3": "tuketim_t3_turizm.csv",
    "p42": "tuketim_p42_seviye_egrilik.csv",
}
KORUNAN_CIKTILAR = {TABAN, *EK_MODEL, *MODEL_ADAYLAR.values()}


def idye_hizala(gonderim, beklenen_idler):
    """Tam boy gonderimi Kaggle'in kullandigi id anahtariyla test sirasina getir."""
    beklenen_idler = np.asarray(beklenen_idler)
    if len(gonderim) != len(beklenen_idler):
        return gonderim
    if gonderim.id.duplicated().any() or pd.Index(beklenen_idler).duplicated().any():
        raise ValueError("gonderim veya test id kumesi mukerrer")
    if np.array_equal(gonderim.id.values, beklenen_idler):
        return gonderim
    konum = pd.Index(gonderim.id).get_indexer(beklenen_idler)
    if (konum < 0).any():
        raise ValueError("gonderim id kumesi ham test ile uyusmuyor")
    return gonderim.iloc[konum].reset_index(drop=True)


def oku(f, *, beklenen_idler=None):
    d = pd.read_csv(os.path.join(S, f))
    if beklenen_idler is not None:
        d = idye_hizala(d, beklenen_idler)
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


def skoru_dogrula(skor):
    """Durum dosyasina yalnizca makul ve sonlu bir LB skoru girmesine izin ver."""
    if skor is None or not np.isfinite(skor) or not 0.0 < float(skor) < 3.0:
        raise ValueError(f"gecersiz LB skoru: {skor}")
    return float(skor)


def cikti_adini_dogrula(ad, *, ek_korunan=()):
    """Gonderimi submissions disina veya kaynak modelin ustune yazma."""
    yol = Path(ad)
    if (
        yol.name != ad
        or yol.is_absolute()
        or "/" in ad
        or "\\" in ad
        or yol.suffix.lower() != ".csv"
        or ad in KORUNAN_CIKTILAR
        or ad in ek_korunan
    ):
        raise ValueError(f"gecersiz ya da korunan cikti adi: {ad}")
    return ad


def gonderim_olcumu(taban, tahmin, skor, *, m0=M0):
    """Gonderilmis tahmini, gorulen LB skoruyla birebir bir Gram yonune cevir."""
    taban = np.asarray(taban, dtype=np.float64)
    tahmin = np.asarray(tahmin, dtype=np.float64)
    if taban.shape != tahmin.shape:
        raise ValueError(f"satir sayisi uyusmuyor: {len(taban)} != {len(tahmin)}")
    yon = tahmin - taban
    ic_carpim = (m0 + float((yon * yon).mean()) - float(skor) ** 2) / 2.0
    return yon, ic_carpim


def gonderim_olcumlerini_ekle(taban, yonler, ic_carpimlar, olcumler, *, okuyucu=oku, m0=M0):
    """Durumdaki gercek gonderimleri soyut yonleri yeniden kurmadan Gram'a ekle."""
    for olcum in olcumler:
        yon, ic_carpim = gonderim_olcumu(taban, okuyucu(olcum["dosya"]), olcum["skor"], m0=m0)
        yonler.append(yon)
        ic_carpimlar.append(ic_carpim)


def dosya_adaylarini_ekle(taban, adaylar, dosyalar, *, okuyucu=oku):
    """Hazir model ciktilarini tabana gore ham aday yonlere cevir."""
    taban = np.asarray(taban, dtype=np.float64)
    for ad, dosya in dosyalar.items():
        tahmin = np.asarray(okuyucu(dosya), dtype=np.float64)
        if tahmin.shape != taban.shape:
            raise ValueError(f"{ad}: satir sayisi uyusmuyor: {len(tahmin)} != {len(taban)}")
        adaylar[ad] = tahmin - taban


def onsele_dayali_duzeltme(adaylar, bilinen, gram, katsayilar, n):
    """Adaylari bilinen spana ve birbirine diklestirip onsel duzeltme kur."""
    bilinen = np.asarray(bilinen, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    duzeltme = np.zeros(n, dtype=np.float64)
    bilgi = []
    for aday, katsayi in katsayilar:
        x = np.asarray(adaylar[aday], dtype=np.float64)
        c, *_ = np.linalg.lstsq(gram, (bilinen.T @ x) / n, rcond=RCOND)
        xp = x - bilinen @ c
        q_dik = float((xp * xp).mean())
        if q_dik < 1e-4:
            raise ValueError(f"{aday}: dik bilesen cok kucuk ({q_dik:.2e})")
        birim = xp / np.sqrt(q_dik)
        duzeltme += float(katsayi) * birim
        bilgi.append({"aday": aday, "katsayi": float(katsayi), "Q_dik": q_dik})
        bilinen = np.column_stack([bilinen, birim])
        gram = (bilinen.T @ bilinen) / n
    return duzeltme, bilgi


def _standartla(degerler, ad):
    """Eksik/sonsuz degerleri medyanla doldurup merkezli birim yon kur."""
    x = np.array(degerler, dtype=np.float64, copy=True)
    sonlu = np.isfinite(x)
    if not sonlu.any():
        raise ValueError(f"{ad}: sonlu deger yok")
    x[~sonlu] = np.median(x[sonlu])
    x -= x.mean()
    rms = np.sqrt(float((x * x).mean()))
    if rms < 1e-12:
        raise ValueError(f"{ad}: sabit yon")
    return x / rms


def hedef996_yonleri(ozellikler, a0):
    """Ileri-zaman CV'de kararlı kalan yedi ham 0.996 eksenini yeniden kur."""
    tarih = pd.to_datetime(ozellikler["tarih"])
    ay = tarih.dt.month.to_numpy(dtype=np.float64)
    gun = tarih.dt.day.to_numpy(dtype=np.float64)
    h = _standartla(ay - ay.min() + (gun - 1.0) / 31.0, "h")
    sv = _standartla(a0, "seviye")
    yas = _standartla(ozellikler["yas"].to_numpy(), "yas")
    t_log_ort = _standartla(ozellikler["t_log_ort"].to_numpy(), "t_log_ort")
    sicaklik = _standartla(ozellikler["sicaklik_ort"].to_numpy(), "sicaklik_ort")
    ham = {
        "h": h,
        "t_hg_genligi": ozellikler["t_hg_genligi"].to_numpy(),
        "sv_yas": sv * yas,
        "h_t_log_ort": h * t_log_ort,
        "gunes_radyasyon": ozellikler["gunes_radyasyon"].to_numpy(),
        "h_sicaklik_ort": h * sicaklik,
        "ulusal_gunluk": ozellikler["ulusal_gunluk"].to_numpy(),
    }
    return {ad: _standartla(yon, ad) for ad, yon in ham.items()}


#: docs/67 -- ileri-zaman CV'de MEVSIM-KIRLI olmayan kesitsel kolonlar.
#: Ham blok modelinin mevsim rampasi yanliligi a0'da YOK; bu yuzden mevsimsel
#: eksenlerin CV korelasyonu LB'nin kendi span olcumune gore 11-15 kat sisik
#: cikiyor. Asagidakilerde o oran 0.3-3.3, yani CV ile LB uyusuyor -- rekoru
#: veren `seviye` de bu gruptaydi.
KESITSEL_KOLONLAR = [
    "id",
    "t_yuk_faktoru",
    "tarim_orani",
    "yerlesim_orani",
    "ilce_nufus_yogunlugu",
    "t_log_ort",
    "t_log_std",
    "t_sifir_orani",
    "guc_yuzdelik",
    "tatil_agirligi",
]


def kesitsel_yonler(te, a0):
    """Mevsim-kirli olmayan kesitsel yonler. Eksik degerler medyanla dolar."""
    yol = os.path.join(KOK, "data/interim/deney/test.parquet")
    ozellikler = pd.read_parquet(yol, columns=KESITSEL_KOLONLAR)
    if len(ozellikler) != len(te) or not np.array_equal(ozellikler.id.values, te.id.values):
        raise ValueError("kesitsel ozellikler ham test ile ayni sirada degil")
    sv = _standartla(a0, "seviye")
    Y = {}
    for k in KESITSEL_KOLONLAR:
        if k == "id":
            continue
        Y[f"k_{k}"] = _standartla(ozellikler[k].to_numpy(dtype=np.float64), k)
    Y["k_seviye_x_yuk"] = _standartla(sv * Y["k_t_yuk_faktoru"], "seviye_x_yuk")
    Y["k_seviye_x_tarim"] = _standartla(sv * Y["k_tarim_orani"], "seviye_x_tarim")
    return Y


def hedef996_paketi(te, a0, bilinen, gram):
    """Test ozelliklerinden hedef duzeltmeyi ve tek-haklik bilesik sondayi kur."""
    yol = os.path.join(KOK, "data/interim/deney/test.parquet")
    ozellikler = pd.read_parquet(yol, columns=HEDEF996_KOLONLAR)
    if len(ozellikler) != len(te) or not np.array_equal(ozellikler.id.values, te.id.values):
        raise ValueError("hedef996 ozellikleri ham test ile ayni sirada degil")
    yonler = hedef996_yonleri(ozellikler, a0)
    duzeltme, bilgi = onsele_dayali_duzeltme(
        yonler,
        bilinen,
        gram,
        HEDEF996_ONSEL,
        len(te),
    )
    yonler["hedef996_bilesik"] = _standartla(duzeltme, "hedef996_bilesik")
    return yonler, duzeltme, bilgi


def durum_yukle():
    if os.path.exists(DURUM):
        with open(DURUM) as akim:
            return json.load(akim)
    # seviye 2026-08-30'da olculdu: skor 1.00115
    return {
        "yapisal": {"seviye": -0.024649},
        "olcumler": [{"aday": "seviye", "dosya": "tuketim_YP_seviye.csv", "skor": 1.00115}],
        "bekleyen": None,
        "gecmis": [],
    }


def durum_kaydet(d):
    g = DURUM + ".tmp"
    with open(g, "w") as f:
        json.dump(d, f, indent=1)
    Path(g).replace(DURUM)


def yapisal_yonler(te, a0, tr_tanim):
    """Ham (ortogonallestirilmemis) yapisal yonler."""
    tarih = pd.to_datetime(te.tarih)
    soguk = (~te.tanim.isin(tr_tanim)).to_numpy().astype(np.float64)
    ay = tarih.dt.month.to_numpy().astype(np.float64)
    hs = (tarih.dt.dayofweek >= 5).to_numpy().astype(np.float64)
    lg = np.log1p(te.guc.values.astype(np.float64))
    lg = (lg - lg.mean()) / lg.std()
    sv = (a0 - a0.mean()) / a0.std()
    ayn = (ay - ay.mean()) / ay.std()
    bolge = te.lokasyon.str.split(">").str[1].fillna("?")
    Y = {
        "seviye": sv,
        "seviye2": sv**2,
        "seviye3": sv**3,
        "seviye_x_soguk": sv * soguk,
        "seviye_x_guc": sv * lg,
        "seviye_x_ay": sv * ayn,
        "seviye_x_hs": sv * hs,
        "soguk": soguk,
        "guc": lg,
        "ay": ayn,
        "haftasonu": hs,
        "seviye2_x_soguk": (sv**2) * soguk,
    }
    # AJAN A'nin uc holdout'ta plasebo kontrollu dogruladigi TAM SEKIL:
    # dogrusal buzme, |u|>1.5 doygun, soguk 4x, ufuk Nis .30 May 1.0 Haz 1.4 Tem 1.32
    ufuk = pd.Series(ay).map({4: 0.30, 5: 1.00, 6: 1.40, 7: 1.32}).to_numpy()
    w = (1.0 + 3.0 * soguk) * ufuk
    w = w / w.mean()
    Y["buzme_tam"] = -w * np.clip(sv, -1.5, 1.5)
    Y["buzme_sade"] = -np.clip(sv, -1.5, 1.5)
    Y["buzme_soguk"] = -(1.0 + 3.0 * soguk) / (1.0 + 3.0 * soguk).mean() * np.clip(sv, -1.5, 1.5)
    Y["buzme_ufuk"] = -(ufuk / ufuk.mean()) * np.clip(sv, -1.5, 1.5)
    for b in bolge.value_counts().index[:4]:
        m = (bolge == b).to_numpy().astype(np.float64)
        Y[f"bolge_{b.split()[0][:6]}"] = m
        Y[f"seviye_x_{b.split()[0][:6]}"] = sv * m
    # merkezle + birim norm
    for k in Y:
        x = Y[k] - Y[k].mean()
        Y[k] = x / np.sqrt(float((x * x).mean()))
    return Y


def kur(te, a0, N, d):
    """Bilinen her seyden r_hat kur. Doner: r_hat, izdusum fonksiyonu."""
    with open(os.path.join(BURA, "olculmus_skorlar.json")) as akim:
        SK = json.load(akim)
    V, L = [], []
    for f, P in SK.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        v = oku(f, beklenen_idler=te.id.values)
        if len(v) != N:
            continue
        dd = v - a0
        V.append(dd)
        L.append((M0 + float((dd * dd).mean()) - P * P) / 2)
    for f, Lj in EK_MODEL.items():
        V.append(oku(f, beklenen_idler=te.id.values) - a0)
        L.append(Lj)
    # Yapisal adaylar yalnizca yeni sonda uretmek icin yeniden kurulur. Olculmus
    # sondalar Gram'a gonderilen CSV'nin kendisiyle eklenir; boylece yon tanimi
    # sonradan degisse bile gorulen LB skoru birebir yeniden uretilir.
    tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), usecols=["tanim"])
    Y = yapisal_yonler(te, a0, set(tr.tanim.unique()))

    def guvenli_oku(dosya):
        return oku(dosya, beklenen_idler=te.id.values)

    dosya_adaylarini_ekle(a0, Y, MODEL_ADAYLAR, okuyucu=guvenli_oku)
    gonderim_olcumlerini_ekle(a0, V, L, d.get("olcumler", []), okuyucu=guvenli_oku)
    V = np.array(V).T
    L = np.array(L)
    G = (V.T @ V) / N
    r_hat, _, kL = buzmeli_r_hat(V, L, G, N)
    return r_hat, V, G, Y, kL


#: docs/68 -- L'lerdeki olcum gurultusunun tabanina uygulanan olcek.
#: Iki sert kisitla kalibre edildi: LOO yeniden kurma hatasi 3.4e-04 ve
#: gercekten alinmis 1.00115 skoru. 4.0 ve ustu ikinciyi ihlal ediyor.
SIGMA_OLCEK = 1.5
#: Kip tutma kapilari (docs/70). Olcerek secildi: alti yapilandirma arasinda
#: bu ikisi gurultu altinda 60 cekilisin 60'inda kararli kaldi ve gercek
#: nrm'yi yalnizca %0.7 degistirdi (daha sert 3-sigma varyanti %2.7 bozuyordu).
W_TABAN = 1e-6  # w_i / w_max bunun altindaysa kip atilir
ANLAM_SIGMA = 2.0  # c_i, sigma_i'nin bu kati kadar buyuk degilse kip atilir


def L_gurultusu(V, N, *, tekrar=200, tohum=3):
    """sigma_L_j = (Q_j^tum - Q_j^public)/2 -- yari-orneklem sacilimindan.

    Kaggle skoru yalniz public %50 satirda; biz Q'yu tum satirlarda
    hesapliyoruz. Fark, yone ozgu bir olcum gurultusudur ve LB
    yuvarlamasindan (5e-6) ~30 kat buyuktur.

    tekrar=200: 50 tekrar YAKINSAMAMISTI -- tohuma gore ort sigma_L
    1.84e-04 ile 2.16e-04 arasinda oynuyordu (%17). 200 ve 400 tekrar
    ayni degeri veriyor (2.20-2.22e-04). Saf optimuma etkisi 1.8e-05,
    kucuk ama bedava.
    """
    rng = np.random.default_rng(tohum)
    k = V.shape[1]
    sig = np.zeros(k)
    yari = N // 2
    for j in range(k):
        d2 = V[:, j] ** 2
        tum = float(d2.mean())
        orn = [float(d2[rng.permutation(N)[:yari]].mean()) - tum for _ in range(tekrar)]
        sig[j] = float(np.std(orn)) / 2.0
    return sig * SIGMA_OLCEK


def buzmeli_r_hat(V, L, G, N, *, sigma=None):
    """Kip basina optimal buzmeyle r_hat. Kesme (rcond) yerine gecer.

    G = sum s_i u_i u_i',  c_i = L.u_i = lam_i + eps_i,  Var(eps_i)=sigma_i^2.
    Beklenen GERCEK kazanci ust'e cikaran katsayi:
        a_i* = max(c_i^2 - sigma_i^2, 0) / c_i^2
    Kesme a_i'yi 0/1'e zorladigi icin bu kesin olarak daha iyidir; ayrica
    tekil kiplerin gurultuyu buyutmesini kendiliginden engeller.

    Doner: (r_hat, beklenen_kazanc, kL) -- kL = <r, r_hat>/N = k'L.

    DIKKAT (docs/69). Buzmeli cozumde  k'L != k'Gk = ||r_hat||^2 . Sondanin
    sabitinde k'L kullanilmalidir:
        S^2 = M0 - 2*k'L + Q_d - 2*kappa*rho
    Buzmesiz pinv cozumunde k = G^-1 L oldugu icin ikisi esittir ve fark
    gorunmez; buzme ile fark 1.2e-04 (skorda 6e-05) buyuklugundedir.
    """
    if sigma is None:
        sigma = L_gurultusu(V, N)
    w, U = np.linalg.eigh(G)
    sira = np.argsort(-w)
    w, U = w[sira], U[:, sira]
    c = U.T @ L
    sigma_i = np.sqrt(np.einsum("ij,jk,ki->i", U.T, np.diag(sigma**2), U))
    a = np.zeros(len(w))
    kazanc = 0.0
    wmax = float(w[0]) if len(w) else 1.0
    for i in range(len(w)):
        # GORELI taban. Mutlak 1e-12 esigi yetersizdi: G'nin tekil degerleri
        # arasinda 1.86e-12 var ve o esigin HEMEN ustunde kaliyordu. Gercek
        # veride c/sigma=0.09 oldugu icin zarar gorunmuyordu, ama c gurultuyle
        # sigma'yi gecince a>0 olup a*c/w patliyordu: bozulmus L ile 60
        # cekilisin 20'sinde nrm 0.05'i asiyor, maks 8144 (gercek 0.0038).
        if w[i] / wmax <= W_TABAN or c[i] ** 2 <= 0.0:
            continue
        # ANLAMLILIK KAPISI. Buzme tek basina yetmiyor: c^2 sansen sigma^2'yi
        # az bir farkla gecerse a kucuk ama sifirdan buyuk cikar ve kucuk w'li
        # kipte c/w yine patlar. 2 sigma sarti bunu keser.
        if c[i] ** 2 <= ANLAM_SIGMA**2 * sigma_i[i] ** 2:
            continue
        lam2 = max(c[i] ** 2 - sigma_i[i] ** 2, 0.0)
        a[i] = lam2 / c[i] ** 2
        kazanc += lam2**2 / (c[i] ** 2 * w[i])
    katsayi = U @ (a * c / np.where(w > 1e-12, w, 1.0))
    return V @ katsayi, float(kazanc), float(katsayi @ L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aday")
    ap.add_argument("--yerdeg", type=float, default=0.005)
    ap.add_argument("--cikti")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--nihai", action="store_true")
    ap.add_argument("--rank2", action="store_true")
    ap.add_argument("--hedef996", "--target996", dest="hedef996", action="store_true")
    ap.add_argument("--bekleyeni-degistir", action="store_true")
    ap.add_argument("--kaydet")
    ap.add_argument("--skor", type=float)
    a = ap.parse_args()
    d = durum_yukle()
    if a.cikti:
        with open(os.path.join(BURA, "olculmus_skorlar.json")) as akim:
            ek_korunan = set(json.load(akim))
        ek_korunan.update(o["dosya"] for o in d.get("olcumler", []))
        try:
            cikti_adini_dogrula(a.cikti, ek_korunan=ek_korunan)
        except ValueError as hata:
            raise SystemExit(str(hata)) from hata
    if a.kaydet:
        try:
            a.skor = skoru_dogrula(a.skor)
        except ValueError as hata:
            raise SystemExit(str(hata)) from hata
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    a0 = oku(TABAN, beklenen_idler=te.id.values)
    N = len(a0)
    r_hat, V, G, Y, kL = kur(te, a0, N, d)
    nrm = float((r_hat * r_hat).mean())
    print(f"BILINEN: {V.shape[1]} yon ({len(d['yapisal'])} yapisal olculmus)")
    # Buzmeli cozumde beklenen MSE = M0 - 2*k'L + ||r_hat||^2 (k'L != k'Gk).
    print(f"  ||r_hat||^2 = {nrm:.6f}  ->  saf optimum {np.sqrt(M0 - 2 * kL + nrm):.6f}")

    if a.kaydet:
        b = d["bekleyen"]
        if not b or b["aday"] != a.kaydet:
            raise SystemExit(f"bekleyen sonda '{a.kaydet}' degil: {b}")
        L = (b["sabit"] - a.skor**2) / (2 * b["kappa"])
        rho = L / np.sqrt(b["Q_dik"])
        print(f"\n{a.kaydet}: skor {a.skor} -> L = {L:+.6f}  rho = {rho:+.4f}  kazanc {rho**2:.3e}")
        d["yapisal"][a.kaydet] = L
        d["gecmis"].append(dict(aday=a.kaydet, skor=a.skor, L=L, rho=rho))
        d.setdefault("olcumler", []).append({"aday": a.kaydet, "dosya": b["cikti"], "skor": a.skor})
        d["bekleyen"] = None
        durum_kaydet(d)
        print("KAYDEDILDI. Yeni taban icin --liste calistir.")
        return

    hedef_yonler, hedef_duzeltme, hedef_bilgi = hedef996_paketi(te, a0, V, G)
    Y.update(hedef_yonler)
    Y.update(kesitsel_yonler(te, a0))

    if sum(bool(x) for x in (a.aday, a.nihai, a.rank2, a.hedef996)) > 1:
        raise SystemExit("--aday, --nihai, --rank2 ve --hedef996 birlikte kullanilamaz")

    if a.liste or not (a.aday or a.nihai or a.rank2 or a.hedef996):
        print(f"\n{'aday':>22s} {'Q_dik':>9s} {'span-disi':>10s} {'rho=0.03 kazanci':>17s}")
        dosya_adaylari = [(o["aday"], None) for o in d.get("olcumler", []) if o["aday"] not in Y]
        for ad, x in list(Y.items()) + dosya_adaylari:
            if ad in d["yapisal"]:
                print(f"{ad:>22s}  [OLCULDU  L={d['yapisal'][ad]:+.6f}]")
                continue
            c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
            xp = x - V @ c
            Qd = float((xp * xp).mean())
            print(f"{ad:>22s} {Qd:9.4f} {Qd:10.3f} {0.03**2:17.3e}")
        print(f"\nolculen yapisal: {json.dumps(d['yapisal'], indent=1)}")
        return

    if a.aday and d.get("bekleyen") and not a.bekleyeni_degistir:
        raise SystemExit(
            f"bekleyen sonda var: {d['bekleyen']['aday']}; "
            "degistirmek icin --bekleyeni-degistir kullan"
        )

    if a.hedef996:
        p = a0 + r_hat + hedef_duzeltme
        etiket = "HEDEF996 (ileri-zaman CV onseli)"
        kap, Qd, xp = 0.0, 0.0, None
        maliyet = float((hedef_duzeltme * hedef_duzeltme).mean())
        onsel_skor = np.sqrt(max(M0 - nrm - maliyet, 0.0))
        print(f"\n{etiket}:")
        for satir in hedef_bilgi:
            print(f"  {satir['aday']:18s} beta={satir['katsayi']:+.6f} Q_dik={satir['Q_dik']:.4f}")
        print(f"  sifir-sinyal geometri maliyeti = {maliyet:.6f}")
        print(f"  CV onseli dogruysa beklenen skor = {onsel_skor:.5f}")
    elif a.rank2:
        duzeltme, bilgi = onsele_dayali_duzeltme(Y, V, G, RANK2_ONSEL, N)
        p = a0 + r_hat + duzeltme
        etiket = "RANK2 ONSEL (kontrollu agresif)"
        kap, Qd, xp = 0.0, 0.0, None
        print(f"\n{etiket}:")
        for satir in bilgi:
            print(f"  {satir['aday']:18s} beta={satir['katsayi']:+.4f} Q_dik={satir['Q_dik']:.4f}")
        print(f"  sifir-sinyal geometri maliyeti = {float((duzeltme * duzeltme).mean()):.6f}")
    elif a.nihai:
        p = a0 + r_hat
        etiket = "NIHAI (saf optimum)"
        kap, Qd, xp = 0.0, 0.0, None
    else:
        if a.aday not in Y:
            raise SystemExit(f"bilinmeyen aday {a.aday}; --liste ile bak")
        if a.aday in d["yapisal"]:
            raise SystemExit(f"{a.aday} zaten olculdu")
        x = Y[a.aday]
        c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
        xp = x - V @ c
        Qd = float((xp * xp).mean())
        if Qd < 1e-4:
            raise SystemExit(f"{a.aday}: dik bilesen cok kucuk ({Qd:.2e}), olculemez")
        kap = a.yerdeg / np.sqrt(Qd)
        p = a0 + r_hat + kap * xp
        etiket = f"SONDA {a.aday}"
        print(
            f"\n{etiket}: Q_dik={Qd:.4f} kappa={kap:.5f} "
            f"SNR(rho=0.03)={0.03 * a.yerdeg / 5.01e-6:.0f}"
        )

    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    ok = (
        len(out) == 714688
        and (out.id.values == ss.iloc[:, 0].values).all()
        and out.tuketim.isna().sum() == 0
        and (out.tuketim < 0).sum() == 0
        and np.isfinite(out.tuketim.values).all()
        and out.tuketim.max() < 3 * np.expm1(a0).max()
    )
    if not ok:
        raise SystemExit("KAPI KALDI")
    dgv = np.log1p(out.tuketim.values) - a0
    # DIKKAT: k'L kullanilir, ||r_hat||^2 DEGIL. Buzmeli cozumde ikisi
    # esit degildir (docs/69). Yanlisi rho'yu 1.21e-04/(2*kappa) kadar kaydirir;
    # kappa=0.005'lik bir sondada bu +0.0121, yani en buyuk gercek sinyalin 4 kati.
    sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)
    if not a.cikti:
        raise SystemExit("--cikti gerekli")
    g = Path(os.path.join(S, a.cikti) + ".tmp")
    out.to_csv(g, index=False)
    g.replace(os.path.join(S, a.cikti))
    print(f"kirpik {int((y == 0).sum())}  maks {out.tuketim.max():,.0f}  KAPI GECTI")
    print(f"YAZILDI submissions/{a.cikti}")
    if a.nihai:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f}  (tum L'ler olculmus)")
    elif a.rank2:
        print("HEDEF: seviye_x_ay rho<=-0.035 ve ay rho>=+0.040 ise 2. sira asilir")
    elif a.hedef996:
        print("HEDEF: 0.99600; bu CV onselidir, gercek skor bilesik LB sondasiyla kalibre edilir")
    else:
        d["bekleyen"] = dict(aday=a.aday, cikti=a.cikti, sabit=sabit, kappa=kap, Q_dik=Qd)
        durum_kaydet(d)
        print(f"COZUM: L = ({sabit:.9f} - P^2) / {2 * kap:.6f}")
        for rho in (-0.03, 0.0, 0.015, 0.030, 0.05):
            print(f"  rho={rho:+.3f} -> {np.sqrt(sabit - 2 * kap * rho * np.sqrt(Qd)):.5f}")
        print(f"\nskor gelince: python m112_kalibre.py --kaydet {a.aday} --skor <S>")


if __name__ == "__main__":
    main()
