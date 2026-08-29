"""DUSMANCA DENETIM -- her iddiayi sifirdan yeniden turet ve olc.

Hicbir mevcut dosyayi degistirmez. Sadece okur. Kaggle'a hicbir sey gondermez.
Cikti: d1_denetim.json
"""

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = {}
BULGU = []


def gecti(ad, kosul, **sayilar):
    R[ad] = dict(hukum="GECTI" if kosul else "KALDI", **sayilar)
    print(f"[{'GECTI' if kosul else 'KALDI '}] {ad}: {sayilar}")
    return kosul


# ------------------------------------------------------------------ 0. YUKLE
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
S = lambda n: pd.read_csv(os.path.join(KOK, "submissions", n))
V = S("tuketim_v102_kappa_optimum.csv")
M4 = S("tuketim_m4_hava_capali.csv")
P51 = S("tuketim_p51_sicak05.csv")
M6 = S("tuketim_m6_ikiyon.csv")

N = len(te)
hiz = dict(
    N=int(N),
    ss=int(len(ss)),
    v102=int(len(V)),
    m4=int(len(M4)),
    p51=int(len(P51)),
    m6=int(len(M6)),
)
id_ok = (
    all((df.id.values == ss.id.values).all() for df in (V, M4, P51, M6))
    and (te.id.values == ss.id.values).all()
)
gecti("0_hizalama", id_ok and len(set(hiz.values())) == 1, **hiz, id_birebir=bool(id_ok))

a = np.log1p(V.tuketim.values.astype(np.float64))
b = np.log1p(M4.tuketim.values.astype(np.float64))
d = b - a
p51 = np.log1p(P51.tuketim.values.astype(np.float64))
m6 = np.log1p(M6.tuketim.values.astype(np.float64))

# ------------------------------------------------------------- 1. OZDESLIK
# p = a + k*d ;  y = gercek log1p hedef
# MSE(k) = <a + k d - y, a + k d - y>/N
#        = m0 + 2k<a-y,d>/N + k^2 <d,d>/N
# L := -<a-y,d>/N   =>  MSE(k) = m0 - 2kL + k^2 Q
# k* = L/Q ,  MSE* = m0 - L^2/Q
# k=1 (yani m4):  M4^2 = m0 - 2L + Q  =>  L = (m0 + Q - M4^2)/2   [IDDIA 2 imzasi]
rng = np.random.default_rng(0)
y_sahte = a + rng.normal(0, 0.7, N)  # keyfi "gercek" ile ozdesligi sayisal dogrula
m0_s = float(((a - y_sahte) ** 2).mean())
Q_s = float((d**2).mean())
L_s = float(-((a - y_sahte) * d).mean())
hata = 0.0
for k in (-0.7, 0.0, 0.13, 0.5, 1.0, 2.3):
    sol = float(((a + k * d - y_sahte) ** 2).mean())
    sag = m0_s - 2 * k * L_s + k * k * Q_s
    hata = max(hata, abs(sol - sag))
k_yildiz = L_s / Q_s
mse_yildiz = m0_s - L_s**2 / Q_s
kaba = min(
    float(((a + k * d - y_sahte) ** 2).mean())
    for k in np.linspace(k_yildiz - 0.3, k_yildiz + 0.3, 2001)
)
gecti(
    "1_ozdeslik",
    hata < 1e-12 and abs(kaba - mse_yildiz) < 1e-9,
    maks_hata=hata,
    optimum_dogrulama=abs(kaba - mse_yildiz),
    L_tanimi="L = -<a-y,d>/N ; MSE(k)=m0-2kL+k^2 Q ; k*=L/Q",
)

# --------------------------------------------------- 2. Q ve SOGUK MASKESI
Q_top = float((d**2).mean())
tanim_te = te.tanim
nan_tanim = int(tanim_te.isna().sum())
sicak_kume = set(tr.tanim.dropna())
soguk = (~tanim_te.isin(sicak_kume)).values
n_sog = int(soguk.sum())
# tip/metin saglamasi
te_saf = tanim_te.astype(str)
metinli = te_saf[~te_saf.str.fullmatch(r"\d+")].unique().tolist()
tr_saf = tr.tanim.astype(str)
tr_metinli = tr_saf[~tr_saf.str.fullmatch(r"\d+")].unique().tolist()
bosluklu = int(te_saf.str.strip().ne(te_saf).sum() + tr_saf.str.strip().ne(tr_saf).sum())
# alternatif maske: strip + sadece rakam kismi
soguk_alt = (~te_saf.str.strip().isin(set(tr_saf.str.strip()))).values
soguk_num = (
    ~te_saf.str.extract(r"^(\d+)", expand=False)
    .fillna("?")
    .isin(set(tr_saf.str.extract(r"^(\d+)", expand=False).fillna("!")))
).values
Qw = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
capraz = float((d * soguk * d * (~soguk)).sum() / N)
gecti(
    "2_maske",
    n_sog == 158369,
    soguk_satir=n_sog,
    iddia=158369,
    nan_tanim=nan_tanim,
    test_metinli_tanim=len(metinli),
    train_metinli_tanim=len(tr_metinli),
    ornek_metinli=metinli[:5],
    bosluk_farki=bosluklu,
    strip_maske_farki=int((soguk != soguk_alt).sum()),
    sayisal_on_ek_maske_farki=int((soguk != soguk_num).sum()),
)
gecti(
    "3_diklik",
    capraz == 0.0,
    capraz_ic_carpim=capraz,
    Q_sicak=Qw,
    Q_soguk=Qc,
    toplam=Qw + Qc,
    Q_dogrudan=Q_top,
    fark=abs(Qw + Qc - Q_top),
)
gecti(
    "3b_Q_iddia",
    abs(Q_top - 0.121396) < 5e-7 and abs(Qw - 0.086624) < 5e-7 and abs(Qc - 0.034772) < 5e-7,
    Q_toplam=Q_top,
    iddia_Q_toplam=0.121396,
    Q_sicak=Qw,
    iddia_Q_sicak=0.086624,
    Q_soguk=Qc,
    iddia_Q_soguk=0.034772,
)

# ------------------------------------------------------------- 4. L_toplam
m0 = 1.00553**2
M4S = 1.04300
PS = 1.00946
M6S = 1.00284
Ltot = (m0 + Q_top - M4S**2) / 2
gecti(
    "4_L_toplam",
    abs(Ltot - 0.022319) < 5e-7,
    L_toplam=Ltot,
    iddia=0.022319,
    m0=m0,
    tek_kappa=float(Ltot / Q_top),
    tek_kappa_optimum=float(np.sqrt(m0 - Ltot**2 / Q_top)),
)

# ------------------------------------------- 5. p51 ve m6'yi yeniden kur
TW = 0.50
TC = Ltot / (Qw + Qc)
p51_yeni = np.clip(np.expm1(a + TW * d * (~soguk) + TC * d * soguk), 0.0, None)
f_p51 = float(np.abs(np.log1p(p51_yeni) - p51).max())
gecti(
    "5_p51_yeniden",
    f_p51 < 1e-12,
    tw=TW,
    tc=TC,
    iddia_tc=0.18385,
    maks_log_fark=f_p51,
)

Lw = (m0 + TW * TW * Qw + TC * TC * Qc - 2 * TC * Ltot - PS**2) / (2 * (TW - TC))
Lc = Ltot - Lw
kw, kc = Lw / Qw, Lc / Qc
opt = float(np.sqrt(m0 - Lw**2 / Qw - Lc**2 / Qc))
m6_yeni = np.clip(np.expm1(a + kw * d * (~soguk) + kc * d * soguk), 0.0, None)
f_m6 = float(np.abs(np.log1p(m6_yeni) - m6).max())
gecti(
    "5b_m6_yeniden",
    f_m6 < 1e-12,
    kw=kw,
    kc=kc,
    iddia_kw=0.12243,
    iddia_kc=0.33688,
    maks_log_fark=f_m6,
)
gecti(
    "6_L_ayrisma",
    abs(Lw - 0.010605) < 5e-7 and abs(Lc - 0.011714) < 5e-7,
    L_sicak=Lw,
    iddia=0.010605,
    L_soguk=Lc,
    iddia_c=0.011714,
)
gecti(
    "7_ongoru",
    abs(opt - 1.00292) < 5e-6,
    ongoru=opt,
    gerceklesen=M6S,
    sapma=float(M6S - opt),
)

# ------------------------------- 8. KIRPMA / expm1 KAYBI (kac satir bozuldu?)
lp_p51 = a + TW * d * (~soguk) + TC * d * soguk
lp_m6 = a + kw * d * (~soguk) + kc * d * soguk
kirpilan_m6 = int((lp_m6 < 0).sum())
kirpilan_p51 = int((lp_p51 < 0).sum())
sifir_v102 = int((V.tuketim.values <= 0).sum())
sifir_m6 = int((M6.tuketim.values <= 0).sum())
# CSV yuvarlama kaybi
rt = float(np.abs(np.log1p(M6.tuketim.values) - lp_m6).max())
gecti(
    "8_kirpma",
    kirpilan_m6 == 0 and kirpilan_p51 == 0,
    m6_negatif_log=kirpilan_m6,
    p51_negatif_log=kirpilan_p51,
    v102_sifir=sifir_v102,
    m6_sifir=sifir_m6,
    csv_gidis_donus_maks_log_hata=rt,
    a_min=float(a.min()),
    b_min=float(b.min()),
)


# ------------------------------------------- 9. YUVARLAMA BANDI (Monte Carlo)
def coz(m0_, M4_, P_, Qw_, Qc_):
    Qt = Qw_ + Qc_
    Lt = (m0_ + Qt - M4_**2) / 2
    tc = Lt / Qt  # NOT: dosyada donmus TC degil, ama m94 boyle uretti
    return Lt, tc


def ongor(m0_, M4_, P_, Qw_, Qc_, tc_sabit=None, tw=TW):
    Qt = Qw_ + Qc_
    Lt = (m0_ + Qt - M4_**2) / 2
    tc = TC if tc_sabit is None else tc_sabit
    Lw_ = (m0_ + tw * tw * Qw_ + tc * tc * Qc_ - 2 * tc * Lt - P_**2) / (2 * (tw - tc))
    Lc_ = Lt - Lw_
    return float(np.sqrt(max(m0_ - Lw_**2 / Qw_ - Lc_**2 / Qc_, 0))), Lw_, Lc_


rg = np.random.default_rng(7)
K = 200000
e0 = rg.uniform(-5e-6, 5e-6, K)
e4 = rg.uniform(-5e-6, 5e-6, K)
ep = rg.uniform(-5e-6, 5e-6, K)
ons = np.array(
    [ongor((1.00553 + x) ** 2, 1.04300 + u, 1.00946 + v, Qw, Qc)[0] for x, u, v in zip(e0, e4, ep)]
)
alt, ust = float(ons.min()), float(ons.max())
gecti(
    "9_yuvarlama_bandi",
    alt <= M6S <= ust,
    bant=[alt, ust],
    genislik=ust - alt,
    merkez=float(ons.mean()),
    gerceklesen=M6S,
    sapma=float(M6S - opt),
    sapma_bant_genisliginin_kati=float(abs(M6S - opt) / ((ust - alt) / 2)),
)

# --------------------------- 10. PUBLIC/PRIVATE: Q tam veri, skor yarim veri
uw = (d**2) * (~soguk)
uc = (d**2) * soguk
rg2 = np.random.default_rng(11)
T = 600
sap_opt = np.empty(T)
sap_ger = np.empty(T)
for t in range(T):
    m = rg2.random(N) < 0.5
    Qw_S = float(uw[m].mean())
    Qc_S = float(uc[m].mean())
    # gercek public denklemleri Q_S ile kurulur; biz Q_all kullandik ->
    # olculen skorlar Q_S'ye gore uretilmis gibi davran, cozumu Q_all ile yap
    o_S, _, _ = ongor(m0, M4S, PS, Qw_S, Qc_S)
    o_all, _, _ = ongor(m0, M4S, PS, Qw, Qc)
    sap_opt[t] = o_all - o_S
sQ = dict(
    sd_Q_sicak_yari=float(np.std([float(uw[rg2.random(N) < 0.5].mean()) for _ in range(200)])),
    sd_Q_soguk_yari=float(np.std([float(uc[rg2.random(N) < 0.5].mean()) for _ in range(200)])),
)
gecti(
    "10_Q_public_yanlisi",
    float(np.std(sap_opt)) < 5e-5,
    sd_ongoru_sapmasi=float(np.std(sap_opt)),
    ortalama=float(np.mean(sap_opt)),
    p05_p95=[float(np.percentile(sap_opt, 5)), float(np.percentile(sap_opt, 95))],
    gozlenen_sapma=float(opt - M6S),
    **sQ,
)

# --------------------------- 11. ASIRI UYUM: public'e fit edilen kappa'lar
# k_j public uzerinde L_j^pub/Q_j ile secildi. private'te L_j^priv farkli.
# Beklenen private kaybi ~ sum_j Var(L_j^pub - L_j^priv) / Q_j
# L_j = -<a-y,d_j>/n : orneklem hatasi. r_i = -(a_i-y_i) d_i bilinmiyor ama
# |r_i| <= |d_i| * |a_i-y_i| ; Var(r) ustten sinirlanabilir.
# Ust sinir: Var(r_i) <= E[d_i^2 (a_i-y_i)^2] <= sqrt(E d^4 * E (a-y)^4)  (Cauchy-Schwarz)
n_yari = N / 2
Ed4_w = float((uw**2).mean())
Ed4_c = float((uc**2).mean())
# (a-y)^2 ortalamasi m0; dorduncu momenti bilinmiyor -> kurtoz varsayimi 3..10
sinirlar = {}
for kurt in (3.0, 6.0, 10.0):
    E4 = kurt * m0**2
    for ad, Ed4, Qj in (("sicak", Ed4_w, Qw), ("soguk", Ed4_c, Qc)):
        Vr = np.sqrt(Ed4 * E4)
        # L^pub - L^priv iki bagimsiz yarinin farki: var = 2*Vr/n_yari  (N*Q_j olcegi ile)
        varL = 2 * Vr / n_yari
        sinirlar[f"kurt{kurt:g}_{ad}_sd_dL"] = float(np.sqrt(varL))
        sinirlar[f"kurt{kurt:g}_{ad}_beklenen_MSE_kaybi"] = float(varL / Qj)
kayip_mse = sum(v for k, v in sinirlar.items() if k.startswith("kurt6") and "kaybi" in k)
kayip_rmsle = float(np.sqrt(m0 - Ltot**2 / Q_top)) * 0  # yer tutucu
priv_tahmin = float(np.sqrt(opt**2 + kayip_mse))
gecti(
    "11_asiri_uyum",
    priv_tahmin - opt < 5e-4,
    ust_sinir_MSE_kaybi_kurt6=kayip_mse,
    private_ust_tahmin=priv_tahmin,
    rmsle_kaybi=priv_tahmin - opt,
    **sinirlar,
)
# parca sayisina gore olcek: J parca -> kayip ~ J * ortalama
per_parca = kayip_mse / 2
esikler = {}
for J in (2, 3, 4, 6, 8, 12, 20, 50):
    kayip = per_parca * J
    esikler[f"J{J}"] = float(np.sqrt(opt**2 + kayip) - opt)
R["11b_parca_olcegi"] = dict(hukum="BILGI", parca_basina_MSE=per_parca, rmsle_kaybi=esikler)
print("11b parca olcegi:", json.dumps(esikler, indent=1))

# --------------------------------------------------- 12. DAGILIM SAGLAMASI
te2 = te.copy()
te2["tarih"] = pd.to_datetime(te2.tarih)
ilk_te = te2.groupby("tanim").tarih.min()
dalga = te2.tanim.map(ilk_te).eq(pd.Timestamp("2026-05-11")).values
vv = V.tuketim.values
mm = M6.tuketim.values
oran = np.where(vv > 1e-9, mm / np.maximum(vv, 1e-9), np.nan)
dag = dict(
    v102=dict(
        min=float(vv.min()),
        maks=float(vv.max()),
        ort=float(vv.mean()),
        medyan=float(np.median(vv)),
        sifira_yakin=int((vv < 1e-6).sum()),
        alti_1=int((vv < 1).sum()),
    ),
    m6=dict(
        min=float(mm.min()),
        maks=float(mm.max()),
        ort=float(mm.mean()),
        medyan=float(np.median(mm)),
        sifira_yakin=int((mm < 1e-6).sum()),
        alti_1=int((mm < 1).sum()),
    ),
    oran=dict(
        min=float(np.nanmin(oran)),
        maks=float(np.nanmax(oran)),
        p01=float(np.nanpercentile(oran, 0.1)),
        p999=float(np.nanpercentile(oran, 99.9)),
        birden_buyuk=int(np.nansum(oran > 1)),
    ),
    dalga_kohortu=dict(
        satir=int(dalga.sum()),
        v102_ort=float(vv[dalga].mean()),
        m6_ort=float(mm[dalga].mean()),
        d_ort=float(d[dalga].mean()),
        Q_payi=float((d[dalga] ** 2).sum() / N / Q_top),
        soguk_orani=float(soguk[dalga].mean()),
    ),
    m4_maks=float(M4.tuketim.values.max()),
)
gecti(
    "12_dagilim",
    mm.min() >= 0 and np.isfinite(mm).all() and mm.max() <= max(vv.max(), M4.tuketim.values.max()),
    **dag,
)

# ------------------------------- 13. p51 gercekten iddia edilen katsayilar mi
gecti(
    "13_p51_katsayi",
    abs(TC - 0.18385) < 5e-6,
    tc_hesaplanan=TC,
    iddia=0.18385,
)


# --------------------- 14. gerceklesen skoru aciklayan Lw ne olurdu (ters coz)
def m6_skoru(Lw_, Lc_):
    return float(np.sqrt(m0 - Lw_**2 / Qw - Lc_**2 / Qc))


# gerceklesen 1.00284 -> Lw,Lc cifti (P denklemi ile birlikte) tutarli mi?
from scipy.optimize import brentq  # noqa: E402


def f(x):
    Lw_ = x
    Lc_ = Ltot - Lw_
    return m6_skoru(Lw_, Lc_) - M6S


try:
    kok1 = brentq(f, Lw, Ltot - 1e-9)
except Exception:
    kok1 = float("nan")
try:
    kok2 = brentq(f, 1e-9, Lw)
except Exception:
    kok2 = float("nan")
R["14_ters_cozum"] = dict(
    hukum="BILGI",
    Lw_P_denkleminden=Lw,
    Lw_gerceklesen_ile_uyusan=[kok1, kok2],
    gereken_Lw_kaymasi=[float(kok1 - Lw), float(kok2 - Lw)],
    aciklama="P denklemi ile m6 denklemi ayni Lw'yi vermiyorsa asiri belirlenmis sistem tutarsiz",
)
print("14 ters cozum:", json.dumps(R["14_ters_cozum"], indent=1))

# --------------------------------------- 15. m0 tutarliligi: v102 gecmis ongoru
R["15_m0"] = dict(hukum="BILGI", m0=m0, kaynak="1.00553^2, olculmus LB")

json.dump(
    R,
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "d1_denetim.json"), "w"),
    indent=1,
)
print("\nYAZILDI d1_denetim.json")
