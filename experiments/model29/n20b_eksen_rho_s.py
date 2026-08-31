"""n20b -- |c| = |rho_dik| / |rho_s| DOGRU NESNEDE: OZNITELIK EKSENI.

HICBIR GONDERIM. submissions/ ALTINA YAZMA YOK. m148 SALT OKUNDU (kopyalanmadi).

=== n20'NIN KUSURU (bu betik onu duzeltiyor) ===
n20 rho_s'i GONDERIM FARKI d_j = (gonderim - taban) uzerinden hesapladi.
Ama m113/m148/n18'in "rho_s"i BASKA bir nesnedir:
    x       = OZNITELIK EKSENI (test tablosundan kurulmus birim vektor)
    x       = x_span + x_dik
    rho_s   = L(x_span) / sqrt(Q_span)      <- EKSENIN span parcasi
    rho_dik = L(x_dik)  / sqrt(Q_dik)       <- EKSENIN dik parcasi (LB'den)
    |c|     = |rho_dik| / |rho_s|
d_j'nin span parcasi ise r_hat'in TAMAMIDIR (tam-span hamlesi), eksenin
span parcasi DEGIL. n20'nin +0.0622'si iste odur -- YANLIS PAYDA.

=== TUZAK: KENDI OLCUMUNU ACIKLAMA ===
n18 rho_s(seviye) = -0.0153'u 29 yonluk TAM span ile olcmus; ama o span
YP_seviye'yi ICERIR ve YP_seviye tam olarak seviye'nin DIK parcasi
yonunde hareket eder. O zaman x_span, x_dik'i de yutar; rho_s kendi
rho_dik'ini aciklar ve ISARETI DONER. Belgelerdeki +0.0156 -> -0.0153
isaret donmesinin sebebi budur (span'in "buyumesi" degil, KIRLENMESI).
Bu betik her eksen icin span'dan o eksenin KENDI sondasini CIKARIR.

=== NOKTALAR ===
  1. seviye        LB sondasi tuketim_YP_seviye.csv  (P=1.00115)
  2. yenibaslangic LB sondasi tuketim_K_yenibas.csv  (P=1.00191)
Baska oznitelik ekseni sondasi LB'ye HIC GONDERILMEDI (n20 B0 kaniti).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
DN = os.path.join(KOK, "data/interim/deney")
M29 = os.path.join(KOK, "experiments/model29")
SCR = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
TABAN = "tuketim_m6_ikiyon.csv"
RCOND = 1e-6
SIGMA_L = 2.9377803611172106e-06
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0  # noqa: E402
from m113_yon_kurucu import yonler  # noqa: E402  (yalniz fonksiyon; m113 kendi cikti verir)

np.seterr(all="ignore")
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    x = np.log1p(d[k].values.astype(np.float64))
    assert np.isfinite(x).all()
    return x


print("=" * 78)
print("n20b  |c| OZNITELIK EKSENINDE -- rho_s eksenin KENDI span parcasindan")
print("=" * 78)

with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)

a0 = oku(TABAN)
N = len(a0)
AD, VV, LL = [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        print(f"ATLANDI (id kumesi uyusmuyor): {f}")
        continue
    d = v - a0
    AD.append(f)
    VV.append(d)
    LL.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
for f, Lj in EK_MODEL.items():
    AD.append(f)
    VV.append(oku(f) - a0)
    LL.append(float(Lj))
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    VV.append(d)
    LL.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
V = np.array(VV).T  # N x K
del VV
L = np.array(LL)
K = V.shape[1]
IX = {a: i for i, a in enumerate(AD)}
G = (V.T @ V) / N
QT = np.diag(G).copy()
print(f"N = {N}  K = {K} yon.  M0 = {M0}")

J_SEV, J_YEN = IX["tuketim_YP_seviye.csv"], IX["tuketim_K_yenibas.csv"]


# ---------------------------------------------------------------------------
def span_coz(ix, x):
    """x'i V[:,ix] span'ina ayristir. x BIRIM DEGIL de olabilir; rho olcek-bagimsiz."""
    ix = np.asarray(ix, int)
    Gr = G[np.ix_(ix, ix)]
    b = (V[:, ix].T @ x) / N
    cc = np.linalg.pinv(Gr, rcond=RCOND) @ b
    Qsp = float(cc @ Gr @ cc)
    Qx = float((x * x).mean())
    Qdk = Qx - Qsp
    Lsp = float(cc @ L[ix])
    cn2 = float(cc @ cc)
    return dict(Qx=Qx, Qsp=Qsp, Qdk=Qdk, Lsp=Lsp, cn2=cn2, cc=cc, ix=ix)


def dik_yon_coz(ix, j):
    """Gonderim j'nin ix span'ina dik parcasinin rho'su -- LB'den, EXACT."""
    ix = np.asarray(ix, int)
    Gr = G[np.ix_(ix, ix)]
    gj = G[ix, j]
    cc = np.linalg.pinv(Gr, rcond=RCOND) @ gj
    Qsp = float(cc @ Gr @ cc)
    Qdk = float(G[j, j] - 2 * cc @ gj + Qsp)
    Lsp = float(cc @ L[ix])
    rd = (L[j] - Lsp) / np.sqrt(Qdk)
    sd = np.sqrt(SIGMA_L**2 * (1.0 + float(cc @ cc)) / Qdk)
    return rd, sd, Qdk, Qdk / G[j, j]


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 1  EKSENLERI KUR (m113.yonler, test tablosu uzerinde)")
print("=" * 78)
tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(tp.id.values, IDS), "test.parquet sirasi ham test ile ayni degil"
Y = yonler(tp, a0)
print(f"{len(Y)} oznitelik ekseni kuruldu (hepsi birim: <x^2>=1).")

# yenibaslangic ekseni. n20c 16 aday yapisal vektoru taradi; K_yenibas'in
# LB'de olculen DIK yonuyle kosinusu 1 olan TEK aday tuketim_KES_yenibaslangic.csv
# (kos = +0.9982). ze_duzeltme.npy DEGIL (kos = -0.055) -- n20c kaniti.
YBF = "tuketim_KES_yenibaslangic.csv"
xyb = oku(YBF) - a0
assert xyb is not None and len(xyb) == N
_s = np.sqrt(float((xyb * xyb).mean()))
print(f"yenibaslangic ekseni = {YBF} - taban;  RMS {_s:.5f}  (n20c: kos = +0.9982)")
Y["yenibaslangic"] = xyb / _s  # birimlestir (rho olcek-bagimsizdir, yine de)

# K_yenibas gonderiminin GERCEK dik yonu ile bu eksenin ortusmesi -- DENETIM
d_yb = V[:, J_YEN]
_ixp = [i for i in range(K) if i not in (J_SEV, J_YEN)] + [J_SEV]
_sp = span_coz(_ixp, d_yb)
_perp = d_yb - V[:, _ixp] @ _sp["cc"]
_spx = span_coz(_ixp, Y["yenibaslangic"])
_perpx = Y["yenibaslangic"] - V[:, _ixp] @ _spx["cc"]
_kos = float((_perp * _perpx).mean()) / np.sqrt(
    float((_perp**2).mean()) * float((_perpx**2).mean())
)
print(
    f"DENETIM: K_yenibas'in dik yonu ile 'yenibaslangic' ekseninin dik yonu kosinusu = {_kos:+.4f}"
)
d_sv = V[:, J_SEV]
_ixs = [i for i in range(K) if i not in (J_SEV, J_YEN)]
_sp2 = span_coz(_ixs, d_sv)
_p2 = d_sv - V[:, _ixs] @ _sp2["cc"]
_sx2 = span_coz(_ixs, Y["seviye"])
_px2 = Y["seviye"] - V[:, _ixs] @ _sx2["cc"]
_kos2 = float((_p2 * _px2).mean()) / np.sqrt(float((_p2**2).mean()) * float((_px2**2).mean()))
print(f"DENETIM: YP_seviye'nin dik yonu ile 'seviye' ekseninin dik yonu kosinusu   = {_kos2:+.4f}")
print("  (|kosinus| ~ 1 ise gonderim GERCEKTEN o eksenin dik parcasi boyunca hareket etmis;")
print("   kucukse gonderim baska bir sey olcmus ve |c| noktasi GECERSIZDIR.)")


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 2  KIRLENME KANITI -- n18'in -0.0153'u nereden geliyor?")
print("=" * 78)
print(f"{'span':>34s} {'Q_span':>8s} {'Q_dik':>8s} {'rho_s(seviye)':>14s}")
for ad, ix in (
    ("TAM 29 yon (n18/m148'in kullandigi)", list(range(K))),
    ("YP_seviye CIKARILMIS (28)", [i for i in range(K) if i != J_SEV]),
    ("YP_seviye + K_yenibas CIKARILMIS (27)", _ixs),
):
    r = span_coz(ix, Y["seviye"])
    print(f"{ad:>34s} {r['Qsp']:8.4f} {r['Qdk']:8.4f} {r['Lsp'] / np.sqrt(r['Qsp']):+14.5f}")
print("""
HUKUM. rho_s(seviye) span'a YP_seviye girer girmez isaret degistiriyor.
  Sebep "span buyudu" degil; YP_seviye'nin span'a kattigi tek YENI yon,
  seviye ekseninin DIK parcasidir. O yon span'a girince x_span artik
  x_dik'i de kapsar ve rho_s KENDI rho_dik'ini olcmeye baslar.
  Dolayisiyla |c| icin payda TEMIZ span'dan (27 yon) okunmalidir.
""")


# ---------------------------------------------------------------------------
print("=" * 78)
print("BOLUM 3  IKI NOKTA")
print("=" * 78)
NOKTA = []
TANIM = (
    ("seviye", "tuketim_YP_seviye.csv", _ixs),  # ikizi de disarida
    ("yenibaslangic", "tuketim_K_yenibas.csv", _ixp),  # S0 + seviye (kronolojik)
)
print(
    f"{'eksen':>15s} {'|S|':>4s} {'Q_span':>8s} {'Q_dik':>8s} {'dik pay':>8s} "
    f"{'rho_s':>9s} {'sd(rs)':>8s} {'rho_dik':>9s} {'sd(rd)':>8s} {'|c|':>8s}"
)
for ad, dosya, ix in TANIM:
    x = Y[ad]
    r = span_coz(ix, x)
    rs = r["Lsp"] / np.sqrt(r["Qsp"])
    sd_rs = np.sqrt(SIGMA_L**2 * r["cn2"] / r["Qsp"])
    j = IX[dosya]
    rd, sd_rd, Qdk_j, dp_j = dik_yon_coz(ix, j)
    c = abs(rd) / abs(rs)
    # olcum hatasi duzeltmesi (ikisi de bagimsiz gurultulu)
    c_duz = np.sqrt(max(rd**2 - sd_rd**2, 0.0) / max(rs**2 - sd_rs**2, 1e-30))
    NOKTA.append(
        dict(
            eksen=ad,
            dosya=dosya,
            n_span=len(ix),
            Qsp=r["Qsp"],
            Qdk=r["Qdk"],
            dikpay=r["Qdk"] / r["Qx"],
            rho_s=rs,
            sd_rho_s=sd_rs,
            rho_dik=rd,
            sd_rho_dik=sd_rd,
            c=c,
            c_duz=float(c_duz),
            snr_rs=abs(rs) / sd_rs,
            snr_rd=abs(rd) / sd_rd,
        )
    )
    print(
        f"{ad:>15s} {len(ix):4d} {r['Qsp']:8.4f} {r['Qdk']:8.4f} {r['Qdk'] / r['Qx']:8.3f} "
        f"{rs:+9.4f} {sd_rs:8.1e} {rd:+9.4f} {sd_rd:8.1e} {c:8.3f}"
    )
print("\nSNR (gurultuye gore):")
for p in NOKTA:
    print(
        f"  {p['eksen']:>15s}  rho_s SNR {p['snr_rs']:7.1f}   rho_dik SNR {p['snr_rd']:7.1f}   "
        f"|c| gurultu-duzeltmeli {p['c_duz']:.3f}"
    )

print("\nBELGE KARSILASTIRMASI (seviye):")
p0 = NOKTA[0]
print("  docs/68-69: rho_s = +0.0156, rho_dik(LB) = -0.0304, |c| = 1.95")
print("  n18 (KIRLI span, 29 yon): rho_s = -0.0153  -> |c| = 1.987")
print(
    f"  n20b (TEMIZ span, 27 yon): rho_s = {p0['rho_s']:+.5f}, rho_dik = {p0['rho_dik']:+.5f}"
    f"  -> |c| = {p0['c']:.3f}"
)


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 4  SPAN SECIMINE DUYARLILIK (her iki nokta icin)")
print("=" * 78)
TARIH = (
    json.load(open(os.path.join(M29, "n20_c_oznitelik.json"), encoding="utf-8")) if False else None
)
# kronolojik sira: olculmus_skorlar.json + EK + olcumler sirasini tarihe cevir
SIRALI = [
    "gun1_baseline.csv",
    "tuketim_v2.csv",
    "tuketim_v7.csv",
    "tuketim_v15.csv",
    "tuketim_v16.csv",
    "tuketim_v18.csv",
    "tuketim_v25_hedge.csv",
    "tuketim_v27_v18hedge.csv",
    "tuketim_v30_buzme.csv",
    "tuketim_v46_gun.csv",
    "tuketim_v44_v27yeni.csv",
    "tuketim_v47_eskison.csv",
    "tuketim_v50_nihai30.csv",
    "tuketim_v55_gunolcek.csv",
    "tuketim_v67_c1335_olay.csv",
    "tuketim_v73_soguk_gun160.csv",
    "tuketim_v79_S3.csv",
    "tuketim_v80_optimum.csv",
    "tuketim_v81_sicak08.csv",
    "tuketim_v83_sicak_optimum.csv",
    "tuketim_v101_hepsi.csv",
    "tuketim_v102_kappa_optimum.csv",
    "tuketim_v109_birlesik.csv",
    "tuketim_m4_hava_capali.csv",
    "tuketim_p51_sicak05.csv",
    "tuketim_s3y40.csv",
    "tuketim_y40_sota_temiz.csv",
]
SIRALI = [f for f in SIRALI if f in IX]
assert len(SIRALI) == K - 2, f"kronolojik liste eksik: {len(SIRALI)} != {K - 2}"
SI = [IX[f] for f in SIRALI]

DUY = {}
for ad, dosya, _ in TANIM:
    x = Y[ad]
    j = IX[dosya]
    ek = [J_SEV] if ad == "yenibaslangic" else []
    satir = []
    print(f"\n{ad}:  span = ilk m kronolojik" + (" + YP_seviye" if ek else ""))
    print(f"{'m':>4s} {'dik pay(x)':>11s} {'rho_s':>9s} {'rho_dik':>9s} {'sd(rd)':>9s} {'|c|':>8s}")
    for m in range(4, len(SI) + 1):
        ix = SI[:m] + ek
        r = span_coz(ix, x)
        if r["Qsp"] <= 1e-10 or r["Qdk"] <= 1e-10:
            continue
        rs = r["Lsp"] / np.sqrt(r["Qsp"])
        sd_rs = np.sqrt(SIGMA_L**2 * r["cn2"] / r["Qsp"])
        rd, sd_rd, _, _ = dik_yon_coz(ix, j)
        if sd_rd > 0.01 or abs(rs) < 3 * sd_rs or abs(rd) < 3 * sd_rd:
            gecer = False
        else:
            gecer = True
        satir.append((m, r["Qdk"] / r["Qx"], rs, rd, sd_rd, abs(rd / rs), gecer))
        if m % 3 == 0 or m == len(SI):
            print(
                f"{m:4d} {r['Qdk'] / r['Qx']:11.3f} {rs:+9.4f} {rd:+9.4f} {sd_rd:9.1e} "
                f"{abs(rd / rs):8.3f} {'' if gecer else '(kapi X)'}"
            )
    DUY[ad] = satir
    ok = [s[5] for s in satir if s[6] and s[0] >= 12]
    if ok:
        print(
            f"  m>=12 & kapi gecen: n={len(ok)}  medyan |c| = {np.median(ok):.3f}  "
            f"aralik [{min(ok):.3f}, {max(ok):.3f}]"
        )


# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("BOLUM 5  TOPLAMA")
print("=" * 78)
cs = np.array([p["c"] for p in NOKTA])
lg = np.log(cs)
gm = float(np.exp(lg.mean()))
if len(cs) >= 2:
    from scipy import stats

    sd = float(np.std(lg, ddof=1))
    t = stats.t.ppf(0.95, len(cs) - 1)
    alt, ust = (
        float(np.exp(lg.mean() - t * sd / np.sqrt(len(cs)))),
        float(np.exp(lg.mean() + t * sd / np.sqrt(len(cs)))),
    )
else:
    sd = alt = ust = float("nan")
for p in NOKTA:
    print(
        f"  {p['eksen']:>15s}  |c| = {p['c']:6.3f}   (dik pay {p['dikpay']:.3f}, "
        f"rho_s {p['rho_s']:+.4f}, rho_dik {p['rho_dik']:+.4f})"
    )
print(f"\n  n = {len(cs)}   geometrik ortalama |c| = {gm:.3f}   log-sd = {sd:.3f}")
print(f"  %90 aralik (t, n-1 sd) = [{alt:.3f}, {ust:.3f}]")
print("\n  m148/m113 capasi 1.986  |  n10 (gonderim farklari) 0.434")

# Duyarliliktan gelen genis aralik: iki eksenin butun gecerli span secimleri
HAVUZ = []
for ad in DUY:
    HAVUZ += [s[5] for s in DUY[ad] if s[6] and s[0] >= 12]
if HAVUZ:
    HAVUZ = np.array(HAVUZ)
    print(f"\n  SPAN-SECIMI HAVUZU (iki eksen, m>=12, kapi gecen): n = {len(HAVUZ)}")
    print(
        f"    medyan {np.median(HAVUZ):.3f}  %5-%95 [{np.quantile(HAVUZ, 0.05):.3f}, "
        f"{np.quantile(HAVUZ, 0.95):.3f}]  min-maks [{HAVUZ.min():.3f}, {HAVUZ.max():.3f}]"
    )

cikti = {
    "aciklama": "n20b -- |c| = |rho_dik|/|rho_s|, rho_s EKSENIN kendi span parcasindan",
    "sigma_L": SIGMA_L,
    "K_yon": int(K),
    "kosinus_denetimi": {
        "K_yenibas_vs_yenibaslangic_ekseni": _kos,
        "YP_seviye_vs_seviye_ekseni": _kos2,
    },
    "kirlenme": {
        "rho_s_seviye_tam29": float(
            span_coz(list(range(K)), Y["seviye"])["Lsp"]
            / np.sqrt(span_coz(list(range(K)), Y["seviye"])["Qsp"])
        ),
        "rho_s_seviye_temiz27": p0["rho_s"],
    },
    "noktalar": NOKTA,
    "n": int(len(cs)),
    "c_geometrik_ortalama": gm,
    "c_log_sd": sd,
    "c_90_alt": alt,
    "c_90_ust": ust,
    "span_havuzu": {
        "n": int(len(HAVUZ)) if len(HAVUZ) else 0,
        "medyan": float(np.median(HAVUZ)) if len(HAVUZ) else None,
        "q05": float(np.quantile(HAVUZ, 0.05)) if len(HAVUZ) else None,
        "q95": float(np.quantile(HAVUZ, 0.95)) if len(HAVUZ) else None,
    },
    "duyarlilik": {
        ad: [
            {
                "m": s[0],
                "dikpay": s[1],
                "rho_s": s[2],
                "rho_dik": s[3],
                "c": s[5],
                "kapi": bool(s[6]),
            }
            for s in DUY[ad]
        ]
        for ad in DUY
    },
}
YOL = os.path.join(M29, "n20b_eksen_rho_s.json")
with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(cikti, fh, ensure_ascii=False, indent=1, default=float)
print(f"\nYAZILDI {YOL}")
