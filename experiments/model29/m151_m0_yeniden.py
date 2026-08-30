"""M0'in SIFIRDAN, bagimsiz yeniden turetilmesi ve dongusellik denetimi.

Sorulan: M0 = 1.005846366 dogru mu, hata payi ne, kararlari etkiliyor mu?

CEVAP OZETI (ayrinti icin betigi calistir):

1. TANIM. Skor RMSLE oldugu icin, a0 = log1p(TABAN) ve r = log1p(y) - a0 iken
   Kaggle'in public satirlarinda  mean(r^2) = P_a0^2  BIR OZDESLIKTIR.
   Ama Q_j kodda 714.688 satirin TAMAMINDA hesaplaniyor. Denklem
       P_j^2 = M0 + Q_j - 2 L_j
   bu yuzden iki farkli kumeyi karistirir; M0 saf ozdeslik degil ETKIN bir
   sabittir ve yon basina  delta_j = (Q_j^tum - Q_j^public)/2  hatasini emmeye
   calisir. delta_j SABIT DEGIL, Q_j ile buyur (olculen oran ~0.0025*Q_j).
   Tek bir sabit ancak delta_j'nin ORTALAMASINI emebilir.

2. KISIT SAYIMI. n olculmus yon n denklem ama n+1 bilinmeyen (M0 + her yonun
   kendi L'si) verir. M0 ancak yonler ARASINDA tam dogrusal bagimlilik varsa
   belirlenir: V u = 0 ve u'1 != 0 olan her u, L'DEN BAGIMSIZ bir kisittir:
       M0 = u'(P^2 - Q) / (u'1)
   27 olculmus yonun rank'i 23; dort bagimli mod var, ikisinin u'1'i sifir
   (bilgi tasimaz). Yani L-bagimsiz kisit sayisi ikidir; a0'in kendi skoruyla
   birlikte UC kisit, iki serbestlik derecesi.

3. DONGUSELLIK. docs/69 par.1.2'nin "uc capa asiri-belirlenmis bicimde eski
   degerde anlasiyor, yayilim 9.1e-07" argumani GECERSIZDIR:
     (a) p51/m4/v102 yonleri rank 2'dir. Uc denklem, uc bilinmeyen
         (M0 + span uzerindeki iki L bileseni) -> sistem TAM BELIRLENMIS.
         Uc capa M0 hakkinda UC degil TEK sey soyler. "9.1e-07 yayilim"
         bagimsiz olcumlerin sacilimi degildir; o tek olcumun hata payi
         yuvarlamadan degil public/tum uyusmazligindan gelir: ~2.2e-04.
     (b) Sifira yakin yayilim bir uyusma degil INSAAT IZIDIR: a0, o span'da
         o zamanki M0 ile cozulmus normal denklemlerin optimumudur; bu
         durumda span'daki HER yon icin L = -(dM0)/2 cikar, yani uc capa
         hangi M0 kullanildiysa onu geri verir. Betik bunu sayiyla gosterir.
     (c) O tek kisit aslinda "a0'in kendi skoru 1.00292 olmali" ongorusudur;
         LB 1.00284 olctu. docs/69 OLCULENI degil ONGORULENI secmistir.
   Gercek bilgi tek bir kisittir: M0 = 1.005846 +- 2.2e-04 (model gurultusu).

4. HATA PAYI. Agirlikli en kucuk kareler (yuvarlama + public/tum model
   gurultusu) M0 = 1.0056885 +- 1.0e-05 verir; ki-kare 0.85/2 sd, yani uc
   kisit birbiriyle TUTARLI. Kullanilan 1.005846366 bu tahminden 1.6e-04
   uzaktir -- L-bagimsiz kisitlarin kendi model gurultusunun 0.6-0.8 sigmasi.
   Yani ISTATISTIKSEL OLARAK AYIRT EDILEMEZ.

5. KARAR ETKISI. Tam boru hattinda M0'i 1.005846366'dan 1.005688066'ya
   cekmek D1 sondasinin cozdugu rho_1'i yalnizca -7.9e-04 kaydirir (kL de
   birlikte kaydigi icin naif d_rho = dM0/(2*kappa) = 1.5e-03'un yarisi).
   2. sira icin gereken rho ~0.099. Kayma bunun %0.8'i, skorda 4e-05.
   HUKUM: M0 belirsizligi 2. sira kararini ETKILEMEZ.

Calistirma:
    ./.venv/Scripts/python.exe experiments/model29/m151_m0_yeniden.py
    ... --cevrimdisi   (kaggle CLI'yi cagirma)
    ... --hizli        (sigma_L tekrarini dusur)

HICBIR GONDERIM YAPMAZ, submissions/ altina YAZMAZ.
"""

import argparse
import io
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
S = os.path.join(KOK, "submissions")
CIKTI = os.path.join(BURA, "m151_m0_yeniden.json")
sys.path.insert(0, BURA)
from m112_kalibre import EK_MODEL, M0, TABAN, buzmeli_r_hat  # noqa: E402

#: Kaggle skorlari bes ondalikla verilir -> P +- 5e-06 (duzgun dagilim).
YUVARLAMA = 5e-6
#: docs/72 par.3 ve m148_demet.json: D1 sondasinin etkin kappa'si.
KAPPA_ETKIN = 0.0516962677376078
#: D1'in "hicbir sey tutmazsa" skoru; duyarlilik tablosunda referans P.
D1_REFERANS_P = 1.00235
#: 2. sira icin gereken rho (docs/72).
GEREKEN_RHO = 0.0991
#: a0'in olculmus LB skoru (ref 55859849).
P_A0 = 1.00284


def oku(dosya, idler):
    """Gonderimi test id sirasina hizala; hizalanamazsa None don (m148 ile ayni)."""
    yol = os.path.join(S, dosya)
    if not os.path.exists(yol):
        return None
    d = pd.read_csv(yol)
    if "id" not in d.columns or len(d) != len(idler):
        return None
    kolon = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, idler):
        if d.id.duplicated().any():
            return None
        konum = pd.Index(d.id).get_indexer(idler)
        if (konum < 0).any():
            return None
        d = d.iloc[konum].reset_index(drop=True)
    return np.log1p(d[kolon].values.astype(np.float64))


def kaggle_gonderimleri(cevrimdisi):
    """Gercekten gonderilmis dosya -> [(ref, publicScore)] eslemesi."""
    if cevrimdisi:
        return None
    komut = [
        sys.executable,
        "-m",
        "kaggle",
        "competitions",
        "submissions",
        "-c",
        "grid-up-datathon",
        "-v",
    ]
    try:
        ham = subprocess.run(
            komut, capture_output=True, encoding="utf-8", timeout=180, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError) as hata:
        print(f"  UYARI: kaggle CLI calismadi ({hata}); dogrulama atlandi")
        return None
    satirlar = [s for s in ham.splitlines() if s.strip()]
    tablo = pd.read_csv(io.StringIO("\n".join(satirlar)))
    esleme = {}
    for _, s in tablo.iterrows():
        esleme.setdefault(str(s["fileName"]), []).append((str(s["ref"]), s["publicScore"]))
    return esleme


def bolum1_tanim():
    print("=" * 78)
    print("1. TANIM -- M0 nedir, kodda tutarli mi")
    print("=" * 78)
    print(
        """
  Skor RMSLE:  P_j = sqrt( mean_public( (log1p(tahmin_j) - log1p(y))^2 ) )
  a0 = log1p(TABAN),  d_j = log1p(tahmin_j) - a0,  r = log1p(y) - a0
      P_j^2 = mean_public(d_j^2) - 2*mean_public(d_j r) + mean_public(r^2)
  Kod mean_public(d_j^2) yerine TUM satirlarda Q_j kullanir, dolayisiyla
      P_j^2 = M0 + Q_j - 2 L_j     (m112:362, m148:81)
  denkleminde M0, mean_public(r^2) = P_a0^2 OZDESLIGI DEGIL; aradaki
      delta_j = (Q_j^tum - Q_j^public)/2
  farkini emmesi beklenen ETKIN bir sabittir. delta_j sabit degildir:
  olculen sacilim  sd(delta_j) ~ 0.0025 * Q_j  (asagida yon yon basilir).
  Tek bir sabit ancak delta_j'nin ORTALAMASINI emebilir; Q'ya oranli
  bir sistematigi ememez."""
    )
    print(f"\n  TABAN                = {TABAN}")
    print(f"  P_a0 (olculen)       = {P_A0:.5f}   -> P_a0^2 = {P_A0**2:.9f}")
    print(f"  m112.M0 (kullanilan) = {M0:.9f}")
    print(f"  fark                 = {M0 - P_A0**2:+.3e}")
    print("  Kodda tek kaynak m112.M0; m148 onu import eder -> TUTARLI.")


def bolum2_kaynak(idler, cevrimdisi):
    print("\n" + "=" * 78)
    print("2. DONGUSELLIK DENETIMI -- her skor gercekten gonderilmis mi")
    print("=" * 78)
    with open(os.path.join(BURA, "olculmus_skorlar.json")) as akim:
        skorlar = json.load(akim)
    with open(os.path.join(BURA, "m112_durum.json")) as akim:
        durum = json.load(akim)
    kayitlar = [(f, p, "olculmus_skorlar") for f, p in skorlar.items()]
    kayitlar += [(o["dosya"], o["skor"], "m112_durum") for o in durum.get("olcumler", [])]
    esleme = kaggle_gonderimleri(cevrimdisi)
    rapor, temiz = [], []
    for dosya, skor, kaynak in kayitlar:
        vektor = oku(dosya, idler)
        if esleme is None:
            hal = "DOGRULANMADI(cevrimdisi)"
        elif dosya not in esleme:
            hal = "GONDERIM YOK -- KULLANMA"
        elif not any(abs(float(s) - float(skor)) < 1e-9 for _, s in esleme[dosya]):
            hal = "SKOR UYUSMUYOR -- KULLANMA"
        else:
            hal = "GECERLI"
        if vektor is None:
            hal += " | TEST ID'LERINE HIZALANMIYOR -- ATILDI"
        ref = esleme.get(dosya, [("-", "-")])[0][0] if esleme else "-"
        rapor.append({"dosya": dosya, "skor": skor, "kaynak": kaynak, "ref": ref, "durum": hal})
        print(f"  {dosya:38s} {skor:8.5f} {kaynak:17s} ref={ref:>9s}  {hal}")
        if vektor is not None and hal == "GECERLI":
            temiz.append((dosya, skor, vektor))
    for dosya in EK_MODEL:
        gonderilmis = esleme is not None and dosya in esleme
        etiket = "GONDERILMIS!" if gonderilmis else "TURETILMIS -- KANIT DEGIL"
        print(f"  {dosya:38s} {'--':>8s} {'EK_MODEL':17s} {'-':>13s}  {etiket}")
    print(f"\n  M0 turetiminde kullanilacak GERCEK olcum sayisi: {len(temiz)}")
    return temiz, rapor


def delta_sacilimi(V, tekrar, tohum=7):
    """sd(delta_j) = sd((Q_j^tum - Q_j^yari)/2) -- public %50'nin yerine yarim orneklem."""
    rng = np.random.default_rng(tohum)
    n = V.shape[0]
    yari = n // 2
    sd = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        kare = V[:, j] ** 2
        tum = float(kare.mean())
        ornek = [float(kare[rng.permutation(n)[:yari]].mean()) - tum for _ in range(tekrar)]
        sd[j] = float(np.std(ornek)) / 2.0
    return sd


def bolum3_kisitlar(ad, P, V, sd):
    print("\n" + "=" * 78)
    print("3. ASIRI BELIRLENMIS SISTEM -- kac BAGIMSIZ kisit var, M0 ne cikiyor")
    print("=" * 78)
    n = V.shape[0]
    q = (V * V).mean(0)
    p = P**2
    y = p - q  # = M0 - 2 L_j  (public/tum uyusmazligi disinda)
    G = (V.T @ V) / n
    w, U = np.linalg.eigh(G)
    wmax = float(w.max())
    rank = int((w / wmax > 1e-10).sum())
    print(f"\n  {len(ad)} denklem, {len(ad)} + 1 bilinmeyen (M0 + her yonun L'si).")
    print(f"  rank(V) = {rank}  ->  {len(ad) - rank} tam bagimli mod.")
    print("  Yalnizca  V u = 0  ve  u'1 != 0  olan modlar M0'i L'DEN BAGIMSIZ belirler.\n")
    print(f"  {'sd(delta_j)':>12s} {'Q_j':>10s} {'oran':>7s}  yon")
    for j in np.argsort(-q):
        print(f"  {sd[j]:12.3e} {q[j]:10.6f} {sd[j] / q[j]:7.4f}  {ad[j]}")
    kisitlar = []
    print("\n  --- L-BAGIMSIZ KISITLAR ---")
    for i in range(len(w)):
        if w[i] / wmax > 1e-10:
            continue
        u = U[:, i]
        toplam = float(u.sum())
        if abs(toplam) < 1e-4:
            print(f"  mod w={w[i]:+.1e}: u'1 = {toplam:+.1e}  ->  M0 bilgisi TASIMAZ")
            continue
        tahmin = float(u @ y / toplam)
        sd_yuv = float(np.sqrt((((2 * P * YUVARLAMA / np.sqrt(3)) * u) ** 2).sum())) / abs(toplam)
        sd_mod = 2 * float(np.sqrt(((u * sd) ** 2).sum())) / abs(toplam)
        kisitlar.append([f"null mod w={w[i]:.1e}", tahmin, sd_yuv, sd_mod])
        print(f"  mod w={w[i]:+.1e}: u'1 = {toplam:+.4f}   M0 = {tahmin:.9f}")
        print(f"        sd_yuvarlama = {sd_yuv:.2e}   sd_model(public/tum) = {sd_mod:.2e}")
        for b in np.argsort(-abs(u))[:4]:
            print(f"        {u[b]:+.4f}  {ad[b]}")
    ozdeslik = P_A0**2
    kisitlar.append(["a0 ozdesligi", ozdeslik, 2 * P_A0 * YUVARLAMA, 0.0])
    print("\n  a0 ozdesligi: Q_a0 = 0 oldugu icin model hatasi da SIFIRDIR.")
    print(f"        M0 = {ozdeslik:.9f}   sd_yuvarlama = {2 * P_A0 * YUVARLAMA:.2e}   sd_model = 0")
    m = np.array([k[1] for k in kisitlar])
    s = np.array([float(np.hypot(k[2], k[3])) for k in kisitlar])
    agirlik = 1.0 / s**2
    m0h = float(agirlik @ m / agirlik.sum())
    sdh = float(1.0 / np.sqrt(agirlik.sum()))
    ki2 = float((agirlik * (m - m0h) ** 2).sum())
    print(f"\n  AGIRLIKLI EN KUCUK KARELER ({len(m)} kisit, {len(m) - 1} serbestlik):")
    print(f"      M0 = {m0h:.9f}  +-  {sdh:.2e}")
    tutarli = "TUTARLI" if ki2 < 2 * len(m) else "CELISKILI"
    print(f"      ki-kare = {ki2:.2f} / {len(m) - 1}  ->  kisitlar birbiriyle {tutarli}")
    print(f"      kullanilan {M0:.9f} sapmasi = {M0 - m0h:+.3e}")
    print("\n      NEDEN BU SAPMA ANLAMSIZ: EKU'yu a0 ozdesligi surukluyor (tek")
    print("      model-hatasiz kisit). Kullanilan deger ise null modlarin verdigi")
    print(f"      degere yapisiyor. Iki grup arasindaki fark {M0 - m0h:.2e},")
    print(f"      null modlarin KENDI model gurultusu ise {s[0]:.1e} ve {s[1]:.1e}:")
    print(f"          ozdeslik-null1 farki = {abs(m[0] - m[-1]) / s[0]:.2f} sigma")
    print(f"          ozdeslik-null2 farki = {abs(m[1] - m[-1]) / s[1]:.2f} sigma")
    print("      -> iki taraf AYIRT EDILEMEZ; M0 secimi bir olcum degil, konvansiyon.")
    return {"kisitlar": kisitlar, "eku": m0h, "eku_sd": sdh, "ki2": ki2}


def bolum4_dongusellik(ad, P, V):
    print("\n" + "=" * 78)
    print("4. UC CAPA ARGUMANI GECERSIZ -- neden dongusel")
    print("=" * 78)
    capalar = [
        "tuketim_p51_sicak05.csv",
        "tuketim_m4_hava_capali.csv",
        "tuketim_v102_kappa_optimum.csv",
    ]
    idx = [ad.index(c) for c in capalar if c in ad]
    if len(idx) < 3:
        print("  capalar bulunamadi, atlandi")
        return {}
    n = V.shape[0]
    q = (V * V).mean(0)
    p = P**2
    G3 = (V[:, idx].T @ V[:, idx]) / n
    w3 = np.linalg.eigvalsh(G3)
    u = np.linalg.eigh(G3)[1][:, 0]
    toplam = float(u.sum())
    tek = float(u @ (p[idx] - q[idx]) / toplam)
    print(f"\n  (a) Uc capanin Gram ozdegerleri: {np.array2string(w3, precision=3)}")
    print("      -> rank 2. Uc denklem, UC bilinmeyen (M0 + span'da iki L bileseni).")
    print("      Sistem TAM BELIRLENMISTIR: bir tek cozum, sifir artik. Yani uc capa")
    print(f"      M0 hakkinda UC degil TEK bir sey soyler: M0 = {tek:.9f}.")
    print(
        f"\n      {'M0 varsayimi':>16s} {'L_p51':>12s} {'L_m4':>12s} {'L_v102':>12s} {'artik':>11s}"
    )
    izdusum = np.linalg.pinv(G3[:2, :2]) @ G3[:2, 2]
    for deneme in (P_A0**2, M0, tek, 1.00700):
        L = (deneme + q[idx] - p[idx]) / 2
        artik = float(L[2] - izdusum @ L[:2])
        print(f"      {deneme:16.9f} {L[0]:+12.3e} {L[1]:+12.3e} {L[2]:+12.3e} {artik:11.3e}")
    print("\n      docs/69'un 'uc capa 9.1e-07 yayilimla anlasiyor, a0 ozdesligi ise")
    print("      174 KAT uzakta' argumani BU YUZDEN GECERSIZ: 9.1e-07 uc bagimsiz")
    print("      olcumun sacilimi DEGIL; rank 2 oldugu icin bagimsiz olcum sayisi")
    print("      birdir ve o tek olcumun hata payi yuvarlamadan degil public/tum")
    print("      uyusmazligindan gelir (bolum 3: ~2.2e-04, yani 9.1e-07'nin 240 kati).")
    print("      Sifira yakin yayilim bir 'uyusma' degil, (b)'deki insaat izidir.")
    print("\n  (b) L uc capada AYNI SABIT cikiyor -- bu M0'in degil, a0'in insaatinin izi.")
    for deneme, etiket in ((P_A0**2, "a0 ozdesligi"), (M0, "KULLANILAN")):
        L = (deneme + q[idx] - p[idx]) / 2
        print(
            f"      {etiket:16s}: L = {np.array2string(L, precision=7)}"
            f"   yayilim {L.max() - L.min():.1e}"
        )
    print(
        """
      a0, bu span'da o zamanki M0 ile cozulmus NORMAL DENKLEMLERIN optimumudur:
          k = G^-1 L_tahmin,   L_tahmin = L_gercek + (dM0/2)*1
      Bu durumda yeni artiga gore span'daki HER x icin
          <x, r_yeni>/N = L_x - (G G^-1 L_tahmin)_x = -(dM0)/2
      yani BUYUKLUKTEN BAGIMSIZ ayni sabit. Q uc capada 16 kat degisirken L'nin
      birebir ayni cikmasi tam olarak bu imzadir. Sonuc: 'capalar eski M0'da
      anlasiyor' ifadesi, a0'in o M0 ile kurulmus olmasinin TEKRARIDIR.
      git 8821cc1 bunu acikca yaziyor: 'uc yon eski m0 altinda TAM olarak ayni
      -0,000079'u veriyordu -> m0 sistematik eksikti'.  -0,000079 = -(dM0)/2."""
    )
    print("\n  (c) O TEK kisit aslinda su ongorudur: 'uc capanin skorundan a0'in")
    print(f"      kendi skoru {np.sqrt(tek):.5f} cikar'. LB'nin olctugu ise {P_A0:.5f}.")
    print("      git 9dd219c: 'ongoru 1,00292 vs gerceklesen 1,00284'. Yani iki")
    print("      DURUST belirleme var ve aralarindaki 1.6e-04, ikisinin ortak model")
    print("      hatasi olan public/tum uyusmazliginin (2.2e-04) 0.7 sigmasidir.")
    print("      docs/69 OLCULENI degil ONGORULENI secmistir; bu savunulabilir bir")
    print("      konvansiyondur ama 'asiri-belirlenmis kanit' DEGILDIR.")
    return {"capa_gram": w3.tolist(), "tek_kisit": tek}


def bolum5_loo(P, V, sig):
    print("\n" + "=" * 78)
    print("5. LEAVE-ONE-OUT -- hangi M0 yeni bir yonun skorunu daha iyi ongoruyor")
    print("=" * 78)
    n = V.shape[0]
    q = (V * V).mean(0)
    p = P**2
    G = (V.T @ V) / n
    k = V.shape[1]

    def loo(m0):
        L = (m0 + q - p) / 2
        hata = []
        for j in range(k):
            digerleri = [i for i in range(k) if i != j]
            r_hat, _, _ = buzmeli_r_hat(
                V[:, digerleri],
                L[digerleri],
                G[np.ix_(digerleri, digerleri)],
                n,
                sigma=sig[digerleri],
            )
            hata.append((m0 + q[j] - 2 * float((r_hat * V[:, j]).mean())) - p[j])
        return np.array(hata)

    print(f"\n  {'M0':>12s} {'ort|dP^2|':>12s} {'medyan':>12s} {'rms':>12s}")
    en_iyi = (np.inf, None)
    for m0 in np.arange(1.00550, 1.00655, 0.00015):
        e = loo(float(m0))
        ort = float(np.abs(e).mean())
        print(
            f"  {m0:12.6f} {ort:12.4e} {np.median(np.abs(e)):12.4e} {np.sqrt((e * e).mean()):12.4e}"
        )
        if ort < en_iyi[0]:
            en_iyi = (ort, float(m0))
    e_ozdeslik, e_kullanilan = np.abs(loo(P_A0**2)), np.abs(loo(M0))
    fark = e_ozdeslik - e_kullanilan
    t = float(fark.mean() / (fark.std(ddof=1) / np.sqrt(len(fark))))
    print(f"\n  LOO en iyi M0 ~ {en_iyi[1]:.5f}   (egri COK YAYVAN: 1.0055-1.0065")
    print("  arasinda toplam degisim ~%35; LOO M0'i +-5e-04'ten iyi konumlandiramaz)")
    print(f"  esli karsilastirma: {int((fark > 0).sum())}/{len(fark)} yonde kullanilan daha iyi,")
    anlam = "ANLAMLI" if abs(t) > 2 else "ANLAMSIZ (karar veremez)"
    print(f"  t = {t:+.2f}  ->  {anlam}")
    print("  NOT: docs/69'daki 1.72e-04 / 2.08e-04 LOO degerleri BU DEPODA")
    print("  YENIDEN URETILEMIYOR; buradaki hatalar ~6e-04, uc kat buyuk.")
    return {"loo_en_iyi": en_iyi[1], "loo_t": t}


def bolum6_duyarlilik(P, V, sig, idler):
    print("\n" + "=" * 78)
    print("6. DUYARLILIK -- M0 kaymasi rho'yu ve karari ne kadar oynatiyor")
    print("=" * 78)
    n = V.shape[0]
    q = (V * V).mean(0)
    p = P**2
    a0 = oku(TABAN, idler)
    ek_L, ek_V = [], []
    for dosya, Lj in EK_MODEL.items():
        v = oku(dosya, idler)
        if v is not None:
            ek_V.append(v - a0)
            ek_L.append(Lj)
    W = np.column_stack([V] + ek_V) if ek_V else V
    sg = np.concatenate([sig, np.full(len(ek_V), float(np.median(sig)))]) if ek_V else sig
    G = (W.T @ W) / n
    d1 = oku("tuketim_D1_demet.csv", idler)
    if d1 is None:
        print("  D1 dosyasi yok, duyarlilik atlandi")
        return {}
    yon = d1 - a0
    Qd = float((yon * yon).mean())
    print(f"\n  D1: Q_d = {Qd:.9f}   kappa_etkin = {KAPPA_ETKIN:.8f}")
    print("  sabit = M0 - 2 k'L + Q_d ;  rho_1 = (sabit - P^2) / (2 kappa_etkin)\n")
    print(f"  {'M0':>13s} {'k^T L':>11s} {'sabit':>14s} {'rho_1':>12s} {'d_rho':>10s}")
    taban = None
    sonuc = {}
    for m0 in (M0, P_A0**2, M0 - 1e-5, M0 + 1e-5):
        L = (m0 + q - p) / 2
        if ek_V:
            L = np.concatenate([L, np.array(ek_L)])
        _, _, kL = buzmeli_r_hat(W, L, G, n, sigma=sg)
        sabit = m0 - 2 * kL + Qd
        rho = (sabit - D1_REFERANS_P**2) / (2 * KAPPA_ETKIN)
        if taban is None:
            taban = rho
        print(f"  {m0:13.9f} {kL:11.7f} {sabit:14.10f} {rho:12.6f} {rho - taban:+10.6f}")
        sonuc[f"{m0:.9f}"] = {"kL": kL, "sabit": sabit, "d_rho": rho - taban}
    naif = (M0 - P_A0**2) / (2 * KAPPA_ETKIN)
    print(
        f"\n  naif d_rho = d_M0/(2*kappa) = {M0 - P_A0**2:.3e} / {2 * KAPPA_ETKIN:.5f} = {naif:.6f}"
    )
    print("  gercek kayma bunun YARISI: L'ler de dM0/2 kaydigi icin k'L kismen goturuyor.")
    print("  +-1e-05'lik M0 belirsizligi -> rho'da +-1.3e-05.")
    en_kotu = abs(sonuc[f"{P_A0**2:.9f}"]["d_rho"])
    print(f"\n  KARAR OLCEGI: 2. sira icin gereken rho ~ {GEREKEN_RHO} (docs/72).")
    print(
        f"  En uc M0 secimi rho'yu {en_kotu:.2e} oynatiyor = gerekenin "
        f"%{100 * en_kotu / GEREKEN_RHO:.1f}'i, skorda ~{KAPPA_ETKIN * en_kotu:.1e}."
    )
    print("  1.00115 -> 0.99614 mesafesi 5e-03. ETKISIZ.")
    sonuc["en_kotu_d_rho"] = en_kotu
    return sonuc


def hukum(kisit, loo):
    print("\n" + "=" * 78)
    print("HUKUM")
    print("=" * 78)
    print(
        f"""
  1. M0 = {M0:.9f} YANLIS DEGIL, ama docs/69'un ONU SAVUNAN ARGUMANI
     GECERSIZ. "Uc capa 9.1e-07 yayilimla anlasiyor, a0 ozdesligi 174 KAT
     uzakta" ifadesi bir anlamlilik olcusu DEGILDIR: capalar rank 2'dir, uc
     bagimsiz olcum degil TEK bir olcum verirler ve o olcumun hata payi
     yuvarlamadan degil public/tum uyusmazligindan gelir (~2.2e-04, yani
     9.1e-07'nin 240 kati). Yayilimin sifira yakinligi ise DONGUSELDIR:
     a0 tam o span'da o M0 ile cozulmustur, dolayisiyla span'daki her yonun
     L'si zorunlu olarak -(dM0)/2 sabitine esittir.

  2. GERCEKTEN BAGIMSIZ kisit sayisi UCTUR (iki null mod + a0 ozdesligi):
         a0 ozdesligi  {kisit["kisitlar"][-1][1]:.9f} +- 1.0e-05  (model hatasi YOK)
         null mod 1    {kisit["kisitlar"][0][1]:.9f} +- {float(np.hypot(*kisit["kisitlar"][0][2:])):.1e}
         null mod 2    {kisit["kisitlar"][1][1]:.9f} +- {float(np.hypot(*kisit["kisitlar"][1][2:])):.1e}
     Ucu de birbiriyle TUTARLI (ki-kare {kisit["ki2"]:.2f}/2). Agirlikli en kucuk kareler
     a0 ozdesligine yapisir ({kisit["eku"]:.9f}), kullanilan deger ise null
     modlara. Aradaki {M0 - kisit["eku"]:.1e} fark, null modlarin kendi gurultusunun
     0.7 sigmasi -> AYIRT EDILEMEZ. LOO da karar veremiyor
     (t = {loo.get("loo_t", float("nan")):+.2f}, egri +-5e-04 genisliginde yayvan).

  3. HATA PAYI: M0'in operasyonel belirsizligi ~2.2e-04'tur (public %50 ile
     tum satir Q'su arasindaki uyusmazlik). Yuvarlama katkisi yalnizca 1e-05.
     Bu belirsizlik SABIT BIR KAYMAYLA GIDERILEMEZ: delta_j, Q_j ile orantili
     buyudugu icin tek bir M0 ancak ortalamasini emebilir.

  4. KARAR ETKISI YOK. En uc M0 secimi bile D1'in cozdugu rho_1'i ~7.9e-04
     kaydiriyor; 2. sira icin gereken 0.0991'in %0.8'i, skorda ~4e-05.
     M0 tartismasi 2. sira kararini ETKILEMEZ.

  5. TAVSIYE: M0 DEGISTIRILMESIN (kayit tutarliligi icin), ama docs/69 par.1.2
     ve Kural 65'teki "uc capa" gerekcesi DUZELTILSIN. Dogru gerekce:
     "M0 etkin bir sabittir, belirsizligi ~2.2e-04'tur, ve bu belirsizlik
     hicbir kararimizi degistirmiyor."

  6. YAN BULGU (dongusellik riski): experiments/model29/m148_olcumler.json
     icinde iki 'olcum' var ama Kaggle gonderim listesinde D1/D2 dosyalari
     YOK. Bunlar GERCEK SKOR DEGIL. m148 bu dosyayi okuyup rho cozdugu icin,
     temizlenmezse turetilmis sayilar olcum gibi zincire girer -- g7 olayinin
     aynisi (Kural 66).

  7. YAN BULGU (veri): submissions/gun1_baseline.csv 80 satirlik ve 'hedef'
     sutunlu; test id'lerine hizalanmiyor. olculmus_skorlar.json'da yer
     aliyor ama boru hatti onu sessizce ATIYOR (dogru davranis). Naif bir
     okuyucu dahil ederse Gram patliyor: ||r_hat|| 0.061 -> 6.7, k'L 44.
     Bu satir olculmus_skorlar.json'dan cikarilmali ya da isaretlenmeli."""
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cevrimdisi", action="store_true", help="kaggle CLI'yi cagirma")
    ap.add_argument("--hizli", action="store_true", help="sigma_L tekrarini dusur")
    a = ap.parse_args()
    idler = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), usecols=["id"]).id.values

    bolum1_tanim()
    temiz, rapor = bolum2_kaynak(idler, a.cevrimdisi)
    a0 = oku(TABAN, idler)
    ad = [f for f, _, _ in temiz if f != TABAN]
    P = np.array([p for f, p, _ in temiz if f != TABAN])
    V = np.column_stack([v - a0 for f, _, v in temiz if f != TABAN])
    sd = delta_sacilimi(V, 40 if a.hizli else 200)
    kisit = bolum3_kisitlar(ad, P, V, sd)
    dongu = bolum4_dongusellik(ad, P, V)
    sig = sd * 1.5  # m112.SIGMA_OLCEK
    loo = bolum5_loo(P, V, sig)
    duyarlilik = bolum6_duyarlilik(P, V, sig, idler)
    hukum(kisit, loo)

    with open(CIKTI, "w", encoding="utf-8") as akim:
        json.dump(
            {
                "M0_kullanilan": M0,
                "M0_a0_ozdesligi": P_A0**2,
                "kisitlar": kisit,
                "capa": dongu,
                "loo": loo,
                "duyarlilik": duyarlilik,
                "gonderim_dogrulamasi": rapor,
            },
            akim,
            indent=1,
        )
    print(f"\n  yazildi: {CIKTI}")


if __name__ == "__main__":
    main()
