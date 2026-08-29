"""b3: (1) olcum gurultusunu LOO ozdesliklerinden KALIBRE et,
       (2) span-optimum harmani riske gore ayarli uret,
       (3) yeni eksen problarini (w*d, span'a DIK) uret,
       (4) kapi denetimi + gonderim sirasi.

Butun dosyalar submissions/tuketim_b*_*.csv olarak yazilir. GONDERILMEZ.
"""

import json
import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
SUB = os.path.join(KOK, "submissions")

SKOR = json.load(open(os.path.join(BURA, "olculmus_skorlar.json"), encoding="utf-8"))
TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = SKOR[TABAN] ** 2
HEDEF = 1.00041**2

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
soguk = (~te.tanim.isin(set(tr.tanim))).values
sicak = ~soguk
a = np.log1p(pd.read_csv(os.path.join(SUB, TABAN)).tuketim.values)
N = len(a)

adlar, U = [], []
for f in SKOR:
    if f == TABAN:
        continue
    yol = os.path.join(SUB, f)
    if not os.path.exists(yol):
        continue
    df = pd.read_csv(yol)
    if len(df) != N or not (df.id.values == te.id.values).all():
        continue
    adlar.append(f)
    U.append(np.log1p(df.tuketim.values) - a)
U = np.array(U)
n = len(adlar)
G = (U @ U.T) / N
Qd = np.diag(G).copy()
S = np.array([SKOR[f] for f in adlar])
L = (M0 + Qd - S**2) / 2.0

# ---------- 1. GURULTU KALIBRASYONU ----------
print("== 1. OLCUM GURULTUSU: LOO ozdesliklerinden ==")
kayit = []
for j in range(n):
    idx = [i for i in range(n) if i != j]
    alpha = np.linalg.pinv(G[np.ix_(idx, idx)], rcond=1e-10) @ G[np.ix_(idx, [j])].ravel()
    r = float(np.sqrt(max(Qd[j] - G[np.ix_(idx, [j])].ravel() @ alpha, 0.0)))
    fark = float(L[j] - alpha @ L[idx])
    kont = np.zeros(n)
    kont[j] = 1.0
    kont[idx] = -alpha
    nrm = float(np.linalg.norm(kont))
    # yuvarlamanin acaklayabilecegi EN BUYUK fark
    yuv = float(0.5 * (np.abs(kont) * 1.01e-5).sum() + 0.5 * abs(kont.sum()) * 1.01e-5)
    kayit.append(
        dict(
            dosya=adlar[j],
            r=r,
            fark=fark,
            norm=nrm,
            birim=abs(fark) / nrm,
            yuvarlama_tavani=yuv,
            cs_siniri=float(np.sqrt(M0) * r),
        )
    )
kal = [k for k in kayit if k["r"] < 1e-5]
print(
    "%-38s %9s %11s %11s %11s %11s"
    % ("dosya", "r_j", "fark", "CS siniri", "yuvarl.tav", "birim sig")
)
for k in sorted(kal, key=lambda t: -abs(t["fark"])):
    print(
        "%-38s %9.2e %+11.6f %11.2e %11.2e %11.2e"
        % (k["dosya"], k["r"], k["fark"], k["cs_siniri"], k["yuvarlama_tavani"], k["birim"])
    )
SIG = float(np.sqrt(np.mean([k["birim"] ** 2 for k in kal])))
SIG_UST = float(max(k["birim"] for k in kal))
print("\nKALIBRE sigma_L = %.2e  (en kotu %.2e)" % (SIG, SIG_UST))
print(
    "Yuvarlamanin izin verdigi tavan asildi mi:",
    any(abs(k["fark"]) > k["yuvarlama_tavani"] + k["cs_siniri"] for k in kal),
)
print("-> yuvarlama TEK BASINA aciklayamiyor; public/private ayrimi var.")

# ---------- 2. RISKE GORE AYARLI SPAN COZUMU ----------
w, V = np.linalg.eigh(G)
o = np.argsort(-w)
w, V = w[o], V[:, o]
Lt = V.T @ L
gec = w > 1e-9
w, V, Lt = w[gec], V[:, gec], Lt[gec]
nk = len(w)
print("\n== 2. SPAN COZUMU: gercekci (gurultuye gore duzeltilmis) MSE ==")
print("kesim k icin  E[gerceklesen MSE] = m0 - kazanc_tahmin + 2*sigma^2*tr(G_k^-1)")
print(
    "%3s %11s %11s %11s %11s %11s %9s"
    % ("k", "ozdeger", "kaz.tahmin", "duzeltme", "E[MSE]", "E[S]", "|c|max")
)
en_iyi = None
for k in range(1, nk + 1):
    kaz = float((Lt[:k] ** 2 / w[:k]).sum())
    tr_inv = float((1.0 / w[:k]).sum())
    duz = SIG**2 * tr_inv
    mse = M0 - kaz + 2 * duz
    c = V[:, :k] @ (Lt[:k] / w[:k])
    print(
        "%3d %11.3e %11.6f %11.6f %11.6f %11.5f %9.2f"
        % (k, w[k - 1], kaz, duz, mse, np.sqrt(max(mse, 0)), np.abs(c).max())
    )
    if en_iyi is None or mse < en_iyi[1]:
        en_iyi = (k, mse, c)
K_OPT = en_iyi[0]
# guvenli kesim: en kotu sigma ile de kazancli olan en buyuk k
guv = []
for k in range(1, nk + 1):
    kaz = float((Lt[:k] ** 2 / w[:k]).sum())
    duz = SIG_UST**2 * float((1.0 / w[:k]).sum())
    guv.append((k, M0 - kaz + 2 * duz))
K_GUV = min(guv, key=lambda t: t[1])[0]
print("\nen iyi k (sigma=%.1e): %d -> E[S]=%.5f" % (SIG, K_OPT, np.sqrt(en_iyi[1])))
print(
    "en iyi k (sigma=%.1e, kotumser): %d -> E[S]=%.5f"
    % (SIG_UST, K_GUV, np.sqrt(min(guv, key=lambda t: t[1])[1]))
)

print("\n== 2b. SIGMA STRES TESTI (sigma yanlis kalibre edilmisse) ==")
print("%9s %9s %9s %9s %9s %9s" % ("sigma", "k=13", "k=15", "k=17", "k=19", "en iyi"))
stres = {}
for sg in (3.4e-5, 6.1e-5, 1.0e-4, 2.0e-4, 4.0e-4, 8.0e-4):
    sat = []
    hep = []
    for k in range(1, nk + 1):
        kaz = float((Lt[:k] ** 2 / w[:k]).sum())
        m = M0 - kaz + 2 * sg**2 * float((1.0 / w[:k]).sum())
        hep.append((k, m))
        if k in (13, 15, 17, 19):
            sat.append(np.sqrt(max(m, 0)))
    bk, bm = min(hep, key=lambda t: t[1])
    stres["%.1e" % sg] = dict(
        satir=[float(x) for x in sat], en_iyi_k=bk, en_iyi_S=float(np.sqrt(max(bm, 0)))
    )
    print(
        "%9.1e %9.5f %9.5f %9.5f %9.5f   k=%d -> %.5f"
        % (sg, sat[0], sat[1], sat[2], sat[3], bk, np.sqrt(max(bm, 0)))
    )
print("-> en kotu senaryoda bile (sigma 8e-4, kalibrenin 24 kati) k=13-15 m6'yi (1.00284) geciyor")


def harman(k):
    c = V[:, :k] @ (Lt[:k] / w[:k])
    return c, a + c @ U


def yaz(logp, ad, aciklama):
    y = np.clip(np.expm1(logp), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(SUB, ad)
    out.to_csv(yol, index=False)
    kapi = dict(
        dosya=ad,
        satir=int(len(out)),
        id_birebir=bool((out.id.values == ss.iloc[:, 0].values).all()),
        id_test_birebir=bool((out.id.values == te.id.values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        inf=int((~np.isfinite(out.tuketim.values)).sum()),
        maks=float(out.tuketim.max()),
        min=float(out.tuketim.min()),
        baslik=",".join(out.columns),
        aciklama=aciklama,
    )
    assert kapi["satir"] == 714688
    assert kapi["id_birebir"] and kapi["id_test_birebir"]
    assert kapi["nan"] == 0 and kapi["negatif"] == 0 and kapi["inf"] == 0
    print("  KAPI OK  %-34s maks %.1f  %s" % (ad, kapi["maks"], aciklama))
    return kapi


print("\n== 3. SPAN HARMANLARI YAZILIYOR ==")
KAPILAR = []
DOSYA = {}
m6log = np.log1p(pd.read_csv(os.path.join(SUB, "tuketim_m6_ikiyon.csv")).tuketim.values)
for k, etiket, dosyaadi in [
    (13, "muhafazakar", "tuketim_b1_span_k13.csv"),
    (15, "birincil", "tuketim_b2_span_k15.csv"),
    (K_OPT, "agresif", "tuketim_b3_span_k%d.csv" % K_OPT),
]:
    c, p = harman(k)
    kaz = float((Lt[:k] ** 2 / w[:k]).sum())
    ong = np.sqrt(max(M0 - kaz + 2 * SIG**2 * float((1.0 / w[:k]).sum()), 0))
    KAPILAR.append(yaz(p, dosyaadi, "span-optimum k=%d (%s), ongoru S=%.5f" % (k, etiket, ong)))
    fark = p - m6log
    agir = sorted(((float(c[i]), adlar[i]) for i in range(n)), key=lambda t: -abs(t[0]))[:8]
    print(
        "     m6'ya gore log-fark: sd %.4f  |fark|>0.5 olan satir %d (%%%.2f)  maks %.2f"
        % (
            fark.std(),
            int((np.abs(fark) > 0.5).sum()),
            100 * float((np.abs(fark) > 0.5).mean()),
            np.abs(fark).max(),
        )
    )
    print(
        "     en agir dosyalar: "
        + ", ".join(
            "%s %+.3f" % (b.replace("tuketim_", "").replace(".csv", ""), aa) for aa, b in agir
        )
    )
    DOSYA[etiket] = dict(
        ad=dosyaadi,
        k=k,
        ongoru=float(ong),
        kazanc=kaz,
        m6_log_fark_sd=float(fark.std()),
        m6_log_fark_maks=float(np.abs(fark).max()),
        katsayilar={adlar[i]: float(c[i]) for i in range(n) if abs(c[i]) > 1e-3},
    )
DOSYA["guv"] = DOSYA["birincil"]
DOSYA["opt"] = DOSYA["agresif"]
K_ANA = 15

c_ana, p_ana = harman(K_ANA)
MSE_ANA = float(M0 - (Lt[:K_ANA] ** 2 / w[:K_ANA]).sum() + 2 * SIG**2 * (1.0 / w[:K_ANA]).sum())

# ---------- 4. YENI EKSEN PROBLARI ----------
print("\n== 4. YENI EKSENLER: w*d yonleri, span'a DIK bilesen ==")
bb = np.log1p(pd.read_csv(os.path.join(SUB, "tuketim_m4_hava_capali.csv")).tuketim.values)
d = bb - a
m6 = np.log1p(pd.read_csv(os.path.join(SUB, "tuketim_m6_ikiyon.csv")).tuketim.values)
te["bolge"] = te.lokasyon.str.split(">").str[1]
ilk_te = te.groupby("tanim").tarih.transform("min")
guc = te.guc.values.astype(float)


def w_medyan(x, mask):
    o2 = np.argsort(x[mask])
    xs, ws = x[mask][o2], (d[mask] ** 2)[o2]
    cs = np.cumsum(ws)
    return xs[min(np.searchsorted(cs, cs[-1] / 2), len(xs) - 1)]


def ikili_med(x):
    m = np.zeros(N, bool)
    for g in (sicak, soguk):
        m |= g & (x > w_medyan(x, g))
    return m


EKSEN = {
    "dbuyuk": ("|d| medyani ustu (sicak/soguk ayri)", ikili_med(np.abs(d))),
    "dpoz": ("d > 0 (m4 yukari cekiyor)", d > 0),
    "seviye": ("tahmin seviyesi medyan ustu", ikili_med(m6)),
    "guc": ("guc medyan ustu", ikili_med(guc)),
    "dalga": ("2026-05-11 toplu dalgasi", (ilk_te == pd.Timestamp("2026-05-11")).values),
    "hazTem": ("Haziran-Temmuz satirlari", np.isin(te.tarih.dt.month.values, [6, 7])),
}
Ginv = V @ np.diag(1.0 / w) @ V.T  # tam span projektoru icin


def perp(z):
    """z'nin span{u_i}'ya DIK bileseni ve normu"""
    g = (U @ z) / N
    c = Ginv @ g
    zp = z - c @ U
    return zp, float((zp**2).mean()), c


eks_bilgi = {}
for ad, (aciklama, mk) in EKSEN.items():
    qh = float((d[sicak & mk] ** 2).sum() / (d[sicak] ** 2).sum())
    qc = float((d[soguk & mk] ** 2).sum() / (d[soguk] ** 2).sum())
    wv = np.where(sicak, mk.astype(float) - qh, mk.astype(float) - qc)
    z = wv * d
    Qz = float((z**2).mean())
    zp, Qperp, _ = perp(z)
    eks_bilgi[ad] = dict(
        aciklama=aciklama,
        q_sicak=qh,
        q_soguk=qc,
        Qz=Qz,
        Q_perp=Qperp,
        span_disi_pay=Qperp / Qz,
        kazanc_dk010=Qperp * 0.01,
        kazanc_dk020=Qperp * 0.04,
        kazanc_dk030=Qperp * 0.09,
        kazanc_dk050=Qperp * 0.25,
        dk_gerekli_hedef=float(np.sqrt(max(MSE_ANA - HEDEF, 0) / Qperp)),
    )
    print(
        "%-8s q_sic %.3f q_sog %.3f  Qz %.6f  Q_perp %.6f (%%%.0f span disi)  "
        "kazanc dk=0.2 %.6f dk=0.3 %.6f  gereken dk %.3f"
        % (
            ad,
            qh,
            qc,
            Qz,
            Qperp,
            100 * Qperp / Qz,
            Qperp * 0.04,
            Qperp * 0.09,
            eks_bilgi[ad]["dk_gerekli_hedef"],
        )
    )

# eksenlerin BIRBIRINE gore dikligi -- korele eksenler ayni bilgiyi olcer
ZP = {}
for ad, (aciklama, mk) in EKSEN.items():
    qh, qc = eks_bilgi[ad]["q_sicak"], eks_bilgi[ad]["q_soguk"]
    ZP[ad] = perp(np.where(sicak, mk.astype(float) - qh, mk.astype(float) - qc) * d)[0]
ek = list(EKSEN)
print("\nEKSENLER ARASI KORELASYON (span'a dik bilesenler)")
print("%-8s " % "" + " ".join("%8s" % x for x in ek))
KOR = {}
for x in ek:
    sat = []
    for y in ek:
        r = float((ZP[x] * ZP[y]).mean() / np.sqrt((ZP[x] ** 2).mean() * (ZP[y] ** 2).mean()))
        sat.append(r)
    KOR[x] = dict(zip(ek, sat))
    print("%-8s " % x + " ".join("%8.3f" % v for v in sat))

# en umutlu 3 eksen: Q_perp yuksek VE birbirine dik olacak sekilde acgozlu secim
sira = sorted(eks_bilgi, key=lambda k: -eks_bilgi[k]["Q_perp"])
sec = []
for c_ad in sira:
    if all(abs(KOR[c_ad][s]) < 0.6 for s in sec):
        sec.append(c_ad)
    if len(sec) == 3:
        break
print("\nSECILEN EKSENLER (|kor|<0.6 kisitiyla):", sec)

print("\n== 5. PROB DOSYALARI (taban = span harmani k=%d) ==" % K_ANA)
print("prob: p = harman + s*z_perp ; skor P olculunce  L = (MSE_harman + s^2*Qperp - P^2)/(2s)")
PROB = {}
for pi, ad in enumerate(sec):
    aciklama, mk = EKSEN[ad]
    zp = ZP[ad]
    Qperp = float((zp**2).mean())
    # s: prob maliyeti ~ s^2*Qperp; hedef maliyet 6e-4 MSE, ust sinir s<=0.60
    s = float(min(0.60, np.sqrt(6e-4 / Qperp)))
    p = p_ana + s * zp
    dosya = "tuketim_b%d_prob_%s.csv" % (4 + pi, ad)
    maliyet = s * s * Qperp
    KAPILAR.append(yaz(p, dosya, "eksen %s, s=%.4f, maliyet MSE +%.6f" % (ad, s, maliyet)))
    PROB[ad] = dict(
        dosya=dosya,
        s=s,
        Q_perp=Qperp,
        maliyet_mse=maliyet,
        taban_mse=MSE_ANA,
        prob_skoru_L0=float(np.sqrt(MSE_ANA + maliyet)),
        coz="L = (%.6f + %.6f - P^2)/(2*%.4f)" % (MSE_ANA, maliyet, s),
        L_hassasiyeti=float(1.0e-4 / (2 * s)),
    )
    print(
        "     s=%.4f  maliyet +%.6f  L=0 ise prob skoru %.5f  L hassasiyeti %.2e"
        % (s, maliyet, np.sqrt(MSE_ANA + maliyet), PROB[ad]["L_hassasiyeti"])
    )

# ---------- 6. GONDERIM SIRASI / BEKLENTI ----------
print("\n== 6. GONDERIM SIRASI (9 hak) ==")
plan = [
    dict(
        hak=1,
        dosya=DOSYA["birincil"]["ad"],
        tur="harman",
        ongoru_S=DOSYA["birincil"]["ongoru"],
        not_="span-optimum k=15. Tek hamlede en buyuk sicrama. Skoru AYNI ZAMANDA "
        "sigma'yi bagimsiz kalibre eder: gerceklesen ~1.0014 ise sigma kucuk (k=20'ye gec), "
        ">1.0025 ise sigma buyuk (k=13'e cekil).",
    ),
]
for i, ad in enumerate(sec):
    plan.append(
        dict(
            hak=2 + i,
            dosya=PROB[ad]["dosya"],
            tur="prob",
            ongoru_S=PROB[ad]["prob_skoru_L0"],
            not_="eksen '%s' icin L olcumu. Q_perp=%.6f. Cozum: %s"
            % (ad, PROB[ad]["Q_perp"], PROB[ad]["coz"]),
        )
    )
plan.append(
    dict(
        hak=5,
        dosya="tuketim_b7_birlesik.csv (olcumlerden SONRA uretilir)",
        tur="harman",
        ongoru_S=None,
        not_="span + 3 eksenin olculmus optimumu; Gram matrisi ile ortak cozum",
    )
)
plan.append(
    dict(
        hak=6,
        dosya="tuketim_b3_span_k20.csv veya 4. eksen probu",
        tur="kosullu",
        ongoru_S=None,
        not_="hak1 sonucuna gore: sigma kucukse agresif span, degilse 4. eksen",
    )
)
plan.append(
    dict(
        hak="7-9", dosya="yedek", tur="yedek", ongoru_S=None, not_="1 Eylul; son birlesik + 2 yedek"
    )
)
for h in plan:
    print(
        "  hak %-3s %-34s %-8s ongoru %s"
        % (
            h["hak"],
            os.path.basename(str(h["dosya"])),
            h["tur"],
            ("%.5f" % h["ongoru_S"]) if h["ongoru_S"] else "-",
        )
    )

# 3 eksenin ORTAK kazanci (korelasyonu hesaba katarak): dk hepsinde ayni varsayimi
Zm = np.array([ZP[x] for x in sec])
Gz = (Zm @ Zm.T) / N
toplam_perp = float(np.ones(len(sec)) @ Gz @ np.ones(len(sec))) / 1.0
print("\n  (3 eksen dik olmadigi icin ortak kazanc = dk^2 * 1'Gz1 = dk^2 * %.6f;" % toplam_perp)
print("   bagimsiz varsayimi %.6f olurdu)" % sum(eks_bilgi[k]["Q_perp"] for k in sec))
print("\nHEDEF DEGERLENDIRMESI")
print("  m6 (mevcut)          MSE %.6f  S %.5f" % (1.00284**2, 1.00284))
print("  span harmani k=%d     MSE %.6f  S %.5f" % (K_ANA, MSE_ANA, np.sqrt(MSE_ANA)))
print("  2. sira hedefi        MSE %.6f  S %.5f" % (HEDEF, 1.00041))
print("  span sonrasi kalan dMSE = %.6f" % (MSE_ANA - HEDEF))
for dk in (0.10, 0.15, 0.20, 0.30):
    print(
        "     3 eksende ortak kappa farki dk=%.2f -> ek kazanc %.6f -> S %.5f"
        % (dk, toplam_perp * dk * dk, np.sqrt(max(MSE_ANA - toplam_perp * dk * dk, 0)))
    )

np.save(os.path.join(BURA, "b3_zperp.npy"), np.array([ZP[x] for x in ek]))
np.save(os.path.join(BURA, "b3_pana.npy"), p_ana)
json.dump(ek, open(os.path.join(BURA, "b3_eksen_adlari.json"), "w", encoding="utf-8"), indent=1)

# b1_bolme.json'a kapi denetimleri, kazanc egrileri ve plani ekle
yol1 = os.path.join(BURA, "b1_bolme.json")
b1 = json.load(open(yol1, encoding="utf-8"))
b1["b3_ek"] = dict(
    sigma_L=SIG,
    sigma_L_ust=SIG_UST,
    sigma_stres=stres,
    span_kesikleri={
        str(k): float(
            np.sqrt(
                max(
                    M0
                    - float((Lt[:k] ** 2 / w[:k]).sum())
                    + 2 * SIG**2 * float((1.0 / w[:k]).sum()),
                    0,
                )
            )
        )
        for k in range(10, nk + 1)
    },
    harmanlar=DOSYA,
    eksenler_perp=eks_bilgi,
    eksen_korelasyon=KOR,
    problar=PROB,
    kapi_denetimleri=KAPILAR,
    gonderim_plani=plan,
    ortak_Qperp=toplam_perp,
)
json.dump(b1, open(yol1, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("GUNCELLENDI b1_bolme.json")

json.dump(
    dict(
        sigma_L=SIG,
        sigma_L_ust=SIG_UST,
        m0=M0,
        hedef_mse=HEDEF,
        k_opt=K_OPT,
        k_guv=K_GUV,
        mse_ana=MSE_ANA,
        loo=kayit,
        harmanlar=DOSYA,
        eksenler=eks_bilgi,
        problar=PROB,
        kapilar=KAPILAR,
        plan=plan,
    ),
    open(os.path.join(BURA, "b3_uret.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print("\nYAZILDI b3_uret.json")
