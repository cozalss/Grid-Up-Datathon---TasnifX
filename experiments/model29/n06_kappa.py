"""KAPPA'yi blok basina yeniden optimize eder.

Mevcut deger 0.0517 TEKDUZE ve DAR SPAN doneminde (ongorulen toplam rho
0.2774) secilmisti. Genis spanda ongorulen toplam 0.4832 -- blok basina
ongorulen rho neredeyse IKI KAT. Bu kappa'yi iki yerde bozuyor:

  1) OLCUM GURULTUSU. rho_k = (sabit - 2*CAPRAZ - P^2) / (2*kappa_k), yani
     sabitin kendi hatasi (m112 LOO: 1.72e-04) 1/(2*kappa) ile buyur.
     kappa 0.0517 -> sigma(rho) = 1.72e-4/0.1034 = 1.66e-3. rho ~ 0.11'e
     gore %1.5 goreli hata; nihai rho^2 kaybi blok basina sigma^2.
     Yani olcum gurultusu ZATEN kucuk -- asil kazanc (2)'de.
  2) SONDANIN KENDI DEGERI. Sonda k'nin dosyasi, hak biterse elde kalan
     dosyadir. O dosyanin kazanci 2*kappa*rho_k - kappa^2, en iyisi
     kappa = rho_k. 0.0517 << 0.11 iken masada para birakiyoruz.

Karsi basinc (kirmizi takim K1): CAPRAZ terimi kappa'ya BOLUNMEZ ama
onceki bloklarin OLCUM HATASI sonraki cozumlere r_j/kappa_k kazanciyla
sizar. Yani kappa_k'yi KUCULTMEK zincir hatasini buyutur; BUYUTMEK azaltir.
Iki etki de buyuk kappa'yi destekliyor -- tek fren, rho_k'nin ongorulenden
COK kucuk cikmasi ihtimali (o zaman -kappa^2 cezasi odenir, ama YALNIZCA
hak biterse, cunku nihai dosya olculen rho'yu kullanir).

Belirsizlik: rho_k = |c| * ongorulen_k / 1.95, |c| ~ log-normal
(medyan 0.57, %90 GA [0.17, 1.26]).
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
SABIT_HATA = 1.72e-4
YUV = 5e-6 / np.sqrt(3.0)
TAVAN = 1.95

with open(os.path.join(M29, "m148_demet.json"), encoding="utf-8") as fh:
    _D = json.load(fh)
ONG = np.array(_D["rho_k_tahmin"], dtype=float)
B = len(ONG)

# ZATEN URETILMIS sondalarin kappa'si DONDURULUR. Dosya diskte duruyor ve
# sabit/kappa_etkin m148_demet.json'a yazildi; buradan farkli bir kappa
# yazarsak cozum formulu dosyayla UYUSMAZ ve olculen rho YANLIS cikar.
DONUK = {}
for _s in _D.get("sondalar", []):
    DONUK[int(_s["sonda"]) - 1] = float(_s["kappa"])
if DONUK:
    print(
        f"donduruldu (zaten uretilmis): {sorted(DONUK)} -> "
        f"{[round(DONUK[i], 4) for i in sorted(DONUK)]}"
    )

rng = np.random.default_rng(23)

# --- |c| ONSELI, VARSA OLCUMLE GUNCELLENIR --------------------------------
# ONEMLI: 1. sonda yalnizca blok 1'in rho'sunu olcmez, |c|'YI DE OLCER:
#     |c| = 1.95 * rho_1_olculen / ongorulen_1
# |c| tum bloklari AYNI oranda olcekledigi icin, D1 geldikten sonra kalan
# bloklarin kappa'si COK daha dar bir belirsizlikle secilebilir. Onsel
# %90 GA [0.17, 1.26] -- yedi kat genislik; olcumden sonra bu daralir.
#
# Kullanim: D1'in LB skoru gelince m148 rho_1'i cozer; asagidaki degeri
# elle gir (ya da C_OLCULEN ortam degiskeniyle ver) ve bu betigi TEKRAR kos,
# sonra m148'i tekrar kos. Kalan bloklarin kappa'si guncellenir.
_c_olc = os.environ.get("C_OLCULEN")
if _c_olc:
    c_mid = float(_c_olc)
    sd = 0.15  # olcumden sonra kalan belirsizlik (LB yuvarlamasi + sabit hatasi)
    print(f"|c| OLCULDU: {c_mid:.3f} (sigma_log {sd})")
else:
    c_mid, sd = 0.57, (np.log(1.26) - np.log(0.17)) / (2 * 1.6449)
    print(f"|c| ONSEL: medyan {c_mid}, %90 GA [0.17, 1.26]")
mu = np.log(c_mid)
c = np.exp(rng.normal(mu, sd, 200000))  # (S,)
RHO = np.outer(c, ONG) / TAVAN  # (S, B) gercek rho ornekleri

ADAY = np.round(np.arange(0.02, 0.301, 0.005), 4)

print(f"ongorulen bloklar {np.round(ONG, 4).tolist()}")
print(f"beklenen gercek rho_k (medyan) {np.round(np.median(RHO, 0), 4).tolist()}")
print()
print(
    f"{'blok':>5s} {'ongorulen':>10s} {'medyan rho':>11s} {'kappa*':>8s} "
    f"{'sigma(rho)':>11s} {'nihai kayip':>12s} {'yedek kazanc':>13s}"
)

SEC = []
for k in range(B):
    r = RHO[:, k]
    # NIHAI dosyaya maliyet: olcum hatasi rho^2'den dusulur -> sigma^2
    sig = np.sqrt((SABIT_HATA / (2 * ADAY)) ** 2 + (YUV / (2 * ADAY)) ** 2)
    nihai_kayip = sig**2
    # SONDA dosyasinin kendi degeri (hak biterse elde kalan): 2*k*rho - k^2
    yedek = 2 * ADAY[None, :] * r[:, None] - ADAY[None, :] ** 2
    yedek_ort = yedek.mean(0)
    # Agirlik: hak bitme olasiligi. 6 hak, 4 sonda + nihai + yedek ->
    # plan yurursey nihai HER ZAMAN gonderilir; sonda dosyasi ancak
    # yurutme kazasi olursa devreye girer. p = 0.10 muhafazakar.
    P_KAZA = 0.10
    amac = -nihai_kayip + P_KAZA * yedek_ort
    i = int(np.argmax(amac))
    if k in DONUK:  # uretilmis sonda: kappa'sina DOKUNMA
        SEC.append(DONUK[k])
        print(
            f"{k + 1:5d} {ONG[k]:10.4f} {np.median(r):11.4f} {DONUK[k]:8.3f} "
            f"{'':>11s} {'':>12s} {'DONUK (uretilmis)':>13s}"
        )
        continue
    SEC.append(float(ADAY[i]))
    print(
        f"{k + 1:5d} {ONG[k]:10.4f} {np.median(r):11.4f} {ADAY[i]:8.3f} "
        f"{sig[i]:11.5f} {nihai_kayip[i]:12.2e} {yedek_ort[i]:+13.5f}"
    )

SEC = np.array(SEC)
print()
# ZINCIR KARARLILIGI: onceki bloklarin hatasi sonrakine r_j/kappa_k ile sizar
print("zincir kontrolu (kirmizi takim K1): en buyuk buyutme carpani")
rmed = np.median(RHO, 0)
for k in range(1, B):
    kaz = float(np.abs(rmed[:k]).max() / SEC[k])
    print(f"  blok {k + 1}: max|r_j|/kappa_k = {kaz:.2f}")

eski = 0.05174190699701174
sig_eski = np.sqrt((SABIT_HATA / (2 * eski)) ** 2 + (YUV / (2 * eski)) ** 2)
sig_yeni = np.sqrt((SABIT_HATA / (2 * SEC)) ** 2 + (YUV / (2 * SEC)) ** 2)
print()
print(f"ESKI tekduze kappa {eski:.5f}: toplam olcum kaybi {B * sig_eski**2:.2e} rho^2")
print(
    f"YENI kappa {np.round(SEC, 3).tolist()}: toplam olcum kaybi "
    f"{float((sig_yeni**2).sum()):.2e} rho^2"
)
print(
    f"gereken toplam rho^2 ~ 0.0225 -- kayip orani "
    f"{100 * B * sig_eski**2 / 0.0225:.2f}% -> "
    f"{100 * float((sig_yeni**2).sum()) / 0.0225:.2f}%"
)

with open(os.path.join(M29, "n06_kappa.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "ongorulen": ONG.tolist(),
            "kappa_yeni": SEC.tolist(),
            "kappa_eski": eski,
            "olcum_kaybi_eski": float(B * sig_eski**2),
            "olcum_kaybi_yeni": float((sig_yeni**2).sum()),
        },
        fh,
        indent=1,
    )
print("\n-> n06_kappa.json")
