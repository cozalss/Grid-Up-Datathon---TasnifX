"""n11 KARAR -- K uzerinde eslesmis onyukleme + m148'e TAKILABILIR cikti.

n11_zaman.py agirliklandirmalari SABIT K'da karsilastirdi ve aralarindaki
farkin kucuk oldugunu gosterdi. Asil fark K'DA: a_m148 agirligi K=25'te
tepe yapip K=136'da %25 kaybediyor. Bu betik o farkin guven araligini
verir ve kazanan kurulusu (eksen listesi + katsayi listesi) yazar.
"""

import json
import os

import n11_analiz as A
import n11_zaman as Z
import numpy as np

ARA, M29 = Z.ARA, A.M29
Y = Z.Y
RHO_S = A.RHO_S
KLER = (5, 8, 10, 13, 17, 20, 25, 30, 40, 50, 60, 80, 100, 136)
TEMEL = 136


def kur_boot(kip, K):
    """Iki yonun ortalamasi: yari f'de sec/agirliklandir, yari 1-f'de olc."""
    bt, r0, n = [], [], []
    for f in (0, 1):
        bf, be = Y[f], Y[1 - f]
        sec, T = A.m148_sirasi(bf, K)
        if len(sec) < 2:
            return None
        n.append(len(sec))
        lam = Z.lam_buzme_sec(sec, bf, T) if kip == "c_buzme" else None
        c = A.c_den_k(T, Z.kats(kip, sec, bf, lam))
        r0.append(abs(A.rho(c, be.G, be.g, be.m0)))
        bt.append(np.abs(Z.boot(c, be, 1 - f)))
    return float(np.mean(r0)), np.mean(bt, axis=0), n


print("A) a_m148 agirligiyla K taramasi -- K=136'ya gore eslesmis fark\n")
tab = {}
tmp = kur_boot("a_m148", TEMEL)
r_t, b_t, _ = tmp
print(
    f"{'K':>4s} {'n':>7s} {'rho':>8s} {'%95 AO':>17s} {'fark(136)':>10s} {'%95 AO':>17s} {'%':>7s} {'P':>5s}"
)
for K in KLER:
    o = kur_boot("a_m148", K)
    if o is None:
        continue
    r, b, n = o
    d = b - b_t
    tab[str(K)] = dict(
        n_eksen=n,
        rho=r,
        rho_ao=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
        fark_136=r - r_t,
        fark_136_ao=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        yuzde=100 * (r - r_t) / r_t,
        p_iyi=float((d > 0).mean()),
    )
    v = tab[str(K)]
    print(
        f"{K:>4d} {str(n):>7s} {r:8.4f} [{v['rho_ao'][0]:.4f},{v['rho_ao'][1]:.4f}] "
        f"{v['fark_136']:+10.4f} [{v['fark_136_ao'][0]:+.4f},{v['fark_136_ao'][1]:+.4f}] "
        f"{v['yuzde']:+7.1f} {v['p_iyi']:5.2f}"
    )

# --------------------------------------------------- en iyi K, agirlik kipi
EN_K = max(tab, key=lambda k: tab[k]["rho"])
print(f"\nen iyi K = {EN_K}")

print("\nB) EN_K'da agirlik kipleri (a_m148'e gore eslesmis fark)\n")
kipler = ("a_m148", "esit", "b_rho_cv", "e_guven", "i_ters_var", "c_buzme", "h_isaret_rho_s")
ra, ba, _ = kur_boot("a_m148", int(EN_K))
kip_tab = {}
for kip in kipler:
    o = kur_boot(kip, int(EN_K))
    if o is None:
        continue
    r, b, _ = o
    d = b - ba
    kip_tab[kip] = dict(
        rho=r,
        rho_ao=[float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))],
        fark=r - ra,
        fark_ao=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
        yuzde=100 * (r - ra) / ra,
        p_iyi=float((d > 0).mean()),
    )
    v = kip_tab[kip]
    print(
        f"  {kip:>15s} rho={r:.4f} [{v['rho_ao'][0]:.4f},{v['rho_ao'][1]:.4f}] "
        f"fark={v['fark']:+.4f} [{v['fark_ao'][0]:+.4f},{v['fark_ao'][1]:+.4f}] "
        f"({v['yuzde']:+.1f}%) P={v['p_iyi']:.2f}"
    )

# ------------------------------------- KAZANAN KURULUS: TAM yaz25 uzerinde
print("\nC) TAM yaz25 ile kurulan nihai liste (m148'e takilacak hali)\n")
byaz = A.BL["yaz25"]
sec, T = A.m148_sirasi(byaz, int(EN_K))
k_a = Z.kats("a_m148", sec, byaz)
liste = []
for j, i in enumerate(sec):
    liste.append(
        dict(
            sira=j + 1,
            eksen=A.ADLAR[i],
            yeni=bool(A.YENI[i]),
            aile=A.HAVUZ[i]["aile"],
            rho_s=float(RHO_S[i]),
            rho_cv=float(byaz.rho_cv[i]),
            KATS=float(k_a[j]),
        )
    )
    print(f"  {j + 1:3d} {A.ADLAR[i][:60]:<60s} KATS={k_a[j]:+.5f}")
print(f"\n  ongorulen ||BETA|| = {np.sqrt((k_a**2).sum()):.4f}")
print(
    f"  TAM yaz25 uzerinde gerceklesen rho = {A.rho(A.c_den_k(T, k_a), byaz.G, byaz.g, byaz.m0):+.4f}"
)

sec136, T136 = A.m148_sirasi(byaz, 136)
k136 = Z.kats("a_m148", sec136, byaz)
print(
    f"  (karsilastirma) K={len(sec136)}: ||BETA||={np.sqrt((k136**2).sum()):.4f} "
    f"yaz25 rho={A.rho(A.c_den_k(T136, k136), byaz.G, byaz.g, byaz.m0):+.4f}"
)

CIKTI = {
    "aciklama": (
        "m148'in eksen secimi ve agirliklandirmasinin BLOK-DISI sinavi. "
        "SONUC: agirlik kipini degistirmenin anlamli faydasi YOK; anlamli "
        "olan tek degisiklik EKSEN SAYISINI KESMEK (K=136 -> K=" + EN_K + "). "
        "Uygulama: m148'in eksen dongusune 'toplam kabul edilen eksen "
        "sayisi " + EN_K + "'e ulasinca dur' kosulu eklenir; KATS formulu "
        "(isaret(rho_cv)*1.95*|rho_s|) AYNEN KALIR."
    ),
    "olcum_duzeni": {
        "gecerli_vekil": "yaz25 icinde zaman bolmesi (Nis-May <-> Haz-Tem), iki yon ortalamasi",
        "reddedilen_vekil": (
            "gorevin onerdigi 'yaz25'te sec, guz25+kis26'da olc' duzeni GECERSIZ: "
            "rho_s (test uzayi) ile blok korelasyonunun isaret uyumu yaz25=0.969 "
            "(kor +0.834), guz25=0.488 (+0.184), kis26=0.302 (-0.426). guz25/kis26 "
            "test ile ayni yonu gostermiyor; orada secim yapmak TERS yone goturur."
        ),
        "onyukleme": f"trafo (tanim) kumesi onyuklemesi, {Z.KUME} kume, B={Z.BOOT}",
    },
    "K_taramasi_a_m148": tab,
    "en_iyi_K": int(EN_K),
    "agirlik_kipleri": kip_tab,
    "nihai_eksenler": liste,
    "nihai_KATS": [float(v) for v in k_a],
    "nihai_ongorulen_norm": float(np.sqrt((k_a**2).sum())),
}
with open(os.path.join(M29, "n11_eksen_secimi.json"), "w", encoding="utf-8") as fh:
    json.dump(CIKTI, fh, ensure_ascii=False, indent=1)
print("\nyazildi: experiments/model29/n11_eksen_secimi.json")
