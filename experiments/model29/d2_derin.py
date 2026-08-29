"""DUSMANCA DENETIM -- 2. asama: sapmanin kaynagi, maske kenar durumlari,
asiri uyum bedelinin turetilmesi, dagilim uc degerleri.  Sadece OKUR.
Cikti: d2_derin.json  (d1_denetim.json'a eklenir)
"""

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
R = {}
oku = lambda n: pd.read_csv(os.path.join(KOK, "submissions", n))

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
V, M4 = oku("tuketim_v102_kappa_optimum.csv"), oku("tuketim_m4_hava_capali.csv")
P51, M6 = oku("tuketim_p51_sicak05.csv"), oku("tuketim_m6_ikiyon.csv")
a = np.log1p(V.tuketim.values)
b = np.log1p(M4.tuketim.values)
d = b - a
N = len(d)
soguk = (~te.tanim.isin(set(tr.tanim))).values
Qw = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
Qt = Qw + Qc
m0 = 1.00553**2
M4S, PS, M6S = 1.04300, 1.00946, 1.00284

# =============== A. DOSYALARI, URETIM ZINCIRININ SABITLERIYLE YENIDEN KUR
LTOT_KOD = 0.022319  # m94/m95'te ELLE YAZILMIS (yuvarlanmis) deger
LTOT_TAM = (m0 + Qt - M4S**2) / 2
TC_KOD = LTOT_KOD / Qt
TW = 0.50
p51_kod = np.clip(np.expm1(a + TW * d * (~soguk) + TC_KOD * d * soguk), 0.0, None)
f51 = float(np.abs(np.log1p(p51_kod) - np.log1p(P51.tuketim.values)).max())
Lw_kod = (m0 + TW * TW * Qw + TC_KOD * TC_KOD * Qc - 2 * TC_KOD * LTOT_KOD - PS**2) / (
    2 * (TW - TC_KOD)
)
Lc_kod = LTOT_KOD - Lw_kod
kw_kod, kc_kod = Lw_kod / Qw, Lc_kod / Qc
m6_kod = np.clip(np.expm1(a + kw_kod * d * (~soguk) + kc_kod * d * soguk), 0.0, None)
f6 = float(np.abs(np.log1p(m6_kod) - np.log1p(M6.tuketim.values)).max())
opt_kod = float(np.sqrt(m0 - Lw_kod**2 / Qw - Lc_kod**2 / Qc))
# tam LTOT ile yapilsaydi
TC_TAM = LTOT_TAM / Qt
Lw_tam = (m0 + TW * TW * Qw + TC_TAM * TC_TAM * Qc - 2 * TC_TAM * LTOT_TAM - PS**2) / (
    2 * (TW - TC_TAM)
)
Lc_tam = LTOT_TAM - Lw_tam
opt_tam = float(np.sqrt(m0 - Lw_tam**2 / Qw - Lc_tam**2 / Qc))
# kod-kappa'lari ile GERCEK ulasilan MSE, tam-L dunyasinda
mse_kod_gercek = m0 - 2 * kw_kod * Lw_tam - 2 * kc_kod * Lc_tam + kw_kod**2 * Qw + kc_kod**2 * Qc
R["A_dosya_kimligi"] = dict(
    hukum="GECTI" if max(f51, f6) < 1e-12 else "KALDI",
    p51_maks_log_fark=f51,
    m6_maks_log_fark=f6,
    LTOT_kodda=LTOT_KOD,
    LTOT_tam=LTOT_TAM,
    LTOT_yuvarlama_hatasi=LTOT_KOD - LTOT_TAM,
    kw=kw_kod,
    kc=kc_kod,
    kw_LTOT_tam_olsa=Lw_tam / Qw,
    kc_LTOT_tam_olsa=Lc_tam / Qc,
    ongoru_kod=opt_kod,
    ongoru_LTOT_tam=opt_tam,
    yuvarlamanin_skor_bedeli_RMSLE=float(np.sqrt(max(mse_kod_gercek, 0)) - opt_tam),
)

# =============== B. v102'nin GECMIS ONGORUSU: ayni araci geriye uygula
try:
    A83 = oku("tuketim_v83_sicak_optimum.csv")
    A101 = oku("tuketim_v101_hepsi.csv")
    a83 = np.log1p(A83.tuketim.values)
    d83 = np.log1p(A101.tuketim.values) - a83
    Q83 = float((d83**2).mean())
    m083 = 1.01318**2
    L83 = (m083 + Q83 - 1.01614**2) / 2
    k83 = L83 / Q83
    ong83 = float(np.sqrt(m083 - L83**2 / Q83))
    v102_yeni = np.clip(np.expm1(a83 + k83 * d83), 0.0, None)
    f83 = float(np.abs(np.log1p(v102_yeni) - a).max())
    R["B_v102_gecmis"] = dict(
        hukum="BILGI",
        taban="v83 1.01318",
        yon="v101 1.01614",
        Q=Q83,
        L=L83,
        kappa=k83,
        ongoru=ong83,
        gerceklesen=1.00553,
        sapma=1.00553 - ong83,
        v102_yeniden_kurma_maks_log_fark=f83,
        NOT="tek yon, tek kappa -> 1 parametre; m6 iki parametre",
    )
except Exception as e:  # pragma: no cover
    R["B_v102_gecmis"] = dict(hukum="HATA", mesaj=str(e))

# =============== C. PUBLIC/PRIVATE: TAM SIMULASYON
# Gercek dunya: skorlar PUBLIC yarida olculuyor, biz Q'yu TUM satirlarda hesapliyoruz.
# Simulasyon: gercek Lw_S, Lc_S sabit varsayilir; S rastgele yarim; olculen skorlar
# uretilip 5 haneye yuvarlanir; prosedurumuz calistirilir; ongoru vs gercek karsilastirilir.
uw = (d**2) * (~soguk)
uc = (d**2) * soguk
# rows-level "r" bilinmiyor; L_S dalgalanmasini da modellemek icin bagimsizlik
# varsayimiyla r_i ~ e_i * d_i, e_i ~ N(0, m0) uret (yalnizca DUYARLILIK amacli)
rg = np.random.default_rng(4242)
T = 3000
sap = np.empty(T)
sap_yuv = np.empty(T)
Lw0, Lc0 = 0.010605488716869239, 0.011713057762842748  # d1'den (tam LTOT)


def prosedur(m0_, M4_, P_, Qw_, Qc_):
    Qt_ = Qw_ + Qc_
    Lt = (m0_ + Qt_ - M4_**2) / 2
    tc = Lt / Qt_
    Lw_ = (m0_ + TW * TW * Qw_ + tc * tc * Qc_ - 2 * tc * Lt - P_**2) / (2 * (TW - tc))
    Lc_ = Lt - Lw_
    return Lw_ / Qw_, Lc_ / Qc_, float(np.sqrt(max(m0_ - Lw_**2 / Qw_ - Lc_**2 / Qc_, 0))), tc


for t in range(T):
    m = rg.random(N) < 0.5
    QwS = float(uw[m].mean())
    QcS = float(uc[m].mean())
    # gercek public L'ler = referans (Q'nun disindaki tek belirsizligi izole et)
    LwS, LcS = Lw0, Lc0
    tcS = (m0 + QwS + QcS - M4S**2) / 2 / (QwS + QcS)
    M4_S = np.sqrt(m0 - 2 * (LwS + LcS) + QwS + QcS)
    P_S = np.sqrt(m0 - 2 * TW * LwS - 2 * TC_KOD * LcS + TW**2 * QwS + TC_KOD**2 * QcS)
    for yuv, hedef in ((False, sap), (True, sap_yuv)):
        M4o = round(M4_S, 5) if yuv else M4_S
        Po = round(P_S, 5) if yuv else P_S
        kwh, kch, ongh, _ = prosedur(m0, M4o, Po, Qw, Qc)
        gerc = np.sqrt(max(m0 - 2 * kwh * LwS - 2 * kch * LcS + kwh**2 * QwS + kch**2 * QcS, 0))
        hedef[t] = ongh - gerc
R["C_public_Q_yanlisi"] = dict(
    hukum="BILGI",
    aciklama="ongoru - gerceklesen; Q TUM satirda, skorlar PUBLIC yarida",
    sd_yuvarlamasiz=float(sap.std()),
    ort_yuvarlamasiz=float(sap.mean()),
    sd_yuvarlamali=float(sap_yuv.std()),
    ort_yuvarlamali=float(sap_yuv.mean()),
    p05_p95=[float(np.percentile(sap_yuv, 5)), float(np.percentile(sap_yuv, 95))],
    gozlenen_sapma=float(1.0029189941962997 - M6S),
    kac_sd=float((1.0029189941962997 - M6S) / sap_yuv.std()),
    P_gozlenen_yuzde=float((sap_yuv >= (1.0029189941962997 - M6S)).mean()),
)

# =============== D. ASIRI UYUM BEDELI -- kapali form + sayisal
# L_j^S = (1/n) sum_{i in S cap j} r_i ,  r_i = -(a_i-y_i) d_i , n = N/2
# Bagimsizlik varsayimi: Var(r_i) = E[d_i^2] * m0
# => Var(L_j^S - L_j^priv) = 2 Var(L_j^S) = 2 * (N Q_j / 2) * m0 / (N/2)^2 = 4 m0 Q_j / N
# => beklenen ek MSE = Var / Q_j = 4 m0 / N   (PARCA BASINA, Q'dan bagimsiz!)
parca_bagimsiz = 4 * m0 / N
Ed4 = float((d**4).mean())
Ed2 = float((d**2).mean())
kurt_d = Ed4 / Ed2**2
# kotu durum (Cauchy-Schwarz, e^4 icin kurtoz 6 varsayimi ile)
parca_kotu_w = 2 * np.sqrt(float((uw**2).mean()) * 6 * m0**2) / (N / 2) / Qw
parca_kotu_c = 2 * np.sqrt(float((uc**2).mean()) * 6 * m0**2) / (N / 2) / Qc
parca_kotu = float((parca_kotu_w + parca_kotu_c) / 2)
rmsle = lambda dmse: dmse / (2 * M6S)
R["D_asiri_uyum"] = dict(
    hukum="BILGI",
    N=int(N),
    parca_basina_MSE_bagimsiz=float(parca_bagimsiz),
    parca_basina_RMSLE_bagimsiz=float(rmsle(parca_bagimsiz)),
    parca_basina_MSE_kotu_durum=parca_kotu,
    parca_basina_RMSLE_kotu_durum=float(rmsle(parca_kotu)),
    d_kurtozu=kurt_d,
    mevcut_2_parca_beklenen_private_kaybi_RMSLE=[
        float(2 * rmsle(parca_bagimsiz)),
        float(2 * rmsle(parca_kotu)),
    ],
    private_m6_tahmini=[
        float(M6S + 2 * rmsle(parca_bagimsiz)),
        float(M6S + 2 * rmsle(parca_kotu)),
    ],
    J_bedeli_RMSLE={
        str(J): [float(J * rmsle(parca_bagimsiz)), float(J * rmsle(parca_kotu))]
        for J in (2, 3, 4, 6, 8, 12, 16, 24, 40, 80)
    },
    esik_marjinal_kazanc=dict(
        aciklama="yeni bir parca ancak LB kazanci bu esigi asarsa karli",
        bagimsiz=float(rmsle(parca_bagimsiz)),
        kotu_durum=float(rmsle(parca_kotu)),
    ),
    esik_4_siraya_dusme=dict(
        aciklama="4. sira 1.00480; private kayip bu farki yerse sira kaybi",
        fark=float(1.00480 - M6S),
        J_bagimsiz=float((1.00480 - M6S) / rmsle(parca_bagimsiz)),
        J_kotu_durum=float((1.00480 - M6S) / rmsle(parca_kotu)),
    ),
)

# =============== E. MASKE KENAR DURUMLARI
te_s = te.tanim.astype(str)
tr_s = tr.tanim.astype(str)
te_metin = sorted(set(te_s[~te_s.str.fullmatch(r"\d+")]))
tr_metin = sorted(set(tr_s[~tr_s.str.fullmatch(r"\d+")]))
say = te_s.value_counts()
te_metin_satir = {t: int(say.get(t, 0)) for t in te_metin}
# on-ek maskesi ile ayrilan 490 satir kim?
on_te = te_s.str.extract(r"^(\d+)", expand=False).fillna("?")
on_tr = set(tr_s.str.extract(r"^(\d+)", expand=False).fillna("!"))
soguk_on = (~on_te.isin(on_tr)).values
farkli = soguk != soguk_on
kimler = te_s[farkli].value_counts().to_dict()
# id'deki tanim ile sutundaki tanim tutuyor mu?
id_tanim = te.id.str.rsplit("_", n=1).str[0]
id_uyum = int((id_tanim.values != te_s.values).sum())
R["E_maske"] = dict(
    hukum="UYARI" if farkli.sum() else "GECTI",
    test_metinli=te_metin,
    test_metinli_satir=te_metin_satir,
    train_metinli=tr_metin,
    on_ek_maskesiyle_degisen_satir=int(farkli.sum()),
    degisen_tanimlar=kimler,
    id_sutunu_tanim_uyusmazligi=id_uyum,
    NOT="tam-string maskesi kullanildi; on-ek maskesi 490 satiri SICAK'a alirdi",
)
# 490 satir yanlis tarafta olsaydi skor etkisi
if farkli.sum():
    dQ = float((d[farkli] ** 2).sum() / N)
    R["E_maske"]["degisen_satirlarin_Q_payi"] = dQ
    R["E_maske"]["Q_payi_yuzde"] = float(100 * dQ / Qt)
    R["E_maske"]["kappa_farki_ile_MSE_etkisi_ust_sinir"] = float(dQ * (kc_kod - kw_kod) ** 2)

# =============== F. d'NIN UC DEGERLERI / Q YOGUNLASMASI
d2 = d**2
sira = np.argsort(-d2)
kum = np.cumsum(d2[sira]) / d2.sum()
R["F_uc_degerler"] = dict(
    hukum="UYARI",
    d_maks=float(np.abs(d).max()),
    d_p999=float(np.percentile(np.abs(d), 99.9)),
    Q_ust_10_satir=float(kum[9]),
    Q_ust_100_satir=float(kum[99]),
    Q_ust_1000_satir=float(kum[999]),
    Q_ust_binde1=float(kum[N // 1000]),
    Q_ust_yuzde1=float(kum[N // 100]),
    d_kurtozu=float(kurt_d),
    m4_sifir_satir=int((M4.tuketim.values <= 0).sum()),
    m4_sifir_v102_ortalamasi=float(V.tuketim.values[M4.tuketim.values <= 0].mean())
    if (M4.tuketim.values <= 0).any()
    else None,
    m6_orani_maks=float(np.exp(np.log1p(M6.tuketim.values) - a).max()),
    NOT="Q agir kuyruklu ise Q_public ile Q_all farki buyur -> ongoru bandi genisler",
)

# =============== G. m6 DAGILIM SAGLAMASI (kohortlar)
te2 = te.copy()
te2["tarih"] = pd.to_datetime(te2.tarih)
ilk = te2.groupby("tanim").tarih.min()
dalga = te2.tanim.map(ilk).eq(pd.Timestamp("2026-05-11")).values
vv, mm = V.tuketim.values, M6.tuketim.values
tr_ort = tr.groupby("tanim").tuketim.mean()
gec = te2.tanim.map(tr_ort)
koh = {}
for ad, msk in (
    ("hepsi", np.ones(N, bool)),
    ("soguk", soguk),
    ("sicak", ~soguk),
    ("dalga_2026-05-11", dalga),
):
    koh[ad] = dict(
        satir=int(msk.sum()),
        v102_ort=float(vv[msk].mean()),
        m6_ort=float(mm[msk].mean()),
        oran_ort=float(mm[msk].mean() / vv[msk].mean()),
        v102_medyan=float(np.median(vv[msk])),
        m6_medyan=float(np.median(mm[msk])),
        d_ort=float(d[msk].mean()),
    )
gec_ok = gec.notna().values
koh["sicak_train_ort_kiyas"] = dict(
    satir=int(gec_ok.sum()),
    train_ort=float(gec[gec_ok].mean()),
    v102_ort=float(vv[gec_ok].mean()),
    m6_ort=float(mm[gec_ok].mean()),
)
R["G_kohort"] = dict(hukum="BILGI", **koh)

json.dump(R, open(os.path.join(BURA, "d2_derin.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
print("\nYAZILDI d2_derin.json")
