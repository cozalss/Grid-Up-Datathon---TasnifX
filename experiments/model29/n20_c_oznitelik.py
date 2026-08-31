"""n20 -- |c| carpanini OZNITELIK EKSENLERINDE n=1'den yukari cikarma denemesi.

HICBIR GONDERIM YAPILMAZ. submissions/ ALTINA YAZILMAZ. Kaggle'a BAGLANILMAZ.
m148_demet_plani.py OKUNMAZ BILE -- yalniz cebiri taklit edilir.

=== TANIM (tek, degismez) ===
Bir yon d_j ve onu KAPSAMAYAN bir span S icin:
    cc     = pinv(G_S) g_j
    Q_sp   = cc' G_S cc         L_sp = cc . L_S       rho_s   = L_sp/sqrt(Q_sp)
    Q_dk   = G_jj - 2 cc.g_j + Q_sp                   rho_dik = (L_j-L_sp)/sqrt(Q_dk)
    |c|_j  = |rho_dik| / |rho_s|
L_j LB skorundan cebirsel gelir: L = (M0 + Q - P^2)/2. r_hat KULLANILMAZ.

=== BOLUMLER ===
B0  Envanter: LB'de GERCEKTEN olculmus dosyalar; hangileri oznitelik ekseni?
B1  LOO (n10 tekrari) -- referans.
B2  KRONOLOJIK ILERI DIKLESTIRME: her gonderim, KENDINDEN ONCEKILERIN
    span'ina gore diklestirilir. Dik pay boylece cok daha buyuk olur.
B3  HEDEFLI: seviye ve yenibaslangic -- ic ice (nested) kurulus, ikizi
    span'dan CIKARARAK. Oznitelik ekseninde gercek |c| noktalari.
B4  Toplama: kac gecerli nokta, medyan, %90 aralik, 1.986 ve 0.434 nerede.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
RCOND = 1e-6
SIGMA_L = 2.9377803611172106e-06  # n10 B0, G'nin tam sifir kiplerinden OLCULDU
SD_KAPI = 0.01  # n10 ile ayni geometrik kapi
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0  # noqa: E402

np.seterr(all="ignore")
rng = np.random.default_rng(20260831)

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
    assert np.isfinite(x).all(), f"{f}: sonlu olmayan deger"
    return x


# --- Kaggle gonderim tarihleri (SALT OKUMA ile alinan listeden, sabitlendi) ---
TARIH = {
    "gun1_baseline.csv": "2026-08-21 11:54",
    "tuketim_v2.csv": "2026-08-21 12:08",
    "tuketim_v7.csv": "2026-08-21 12:29",
    "tuketim_v15.csv": "2026-08-22 05:32",
    "tuketim_v16.csv": "2026-08-22 05:35",
    "tuketim_v18.csv": "2026-08-22 10:26",
    "tuketim_v25_hedge.csv": "2026-08-23 06:07",
    "tuketim_v27_v18hedge.csv": "2026-08-23 06:13",
    "tuketim_v30_buzme.csv": "2026-08-23 14:26",
    "tuketim_v46_gun.csv": "2026-08-24 04:19",
    "tuketim_v44_v27yeni.csv": "2026-08-24 04:27",
    "tuketim_v47_eskison.csv": "2026-08-24 04:30",
    "tuketim_v50_nihai30.csv": "2026-08-25 00:00",
    "tuketim_v55_gunolcek.csv": "2026-08-25 00:01",
    "tuketim_v67_c1335_olay.csv": "2026-08-26 00:02",
    "tuketim_v73_soguk_gun160.csv": "2026-08-26 00:03",
    "tuketim_v79_S3.csv": "2026-08-26 00:06",
    "tuketim_v80_optimum.csv": "2026-08-27 06:02",
    "tuketim_v81_sicak08.csv": "2026-08-27 06:02",
    "tuketim_v83_sicak_optimum.csv": "2026-08-27 06:09",
    "tuketim_v101_hepsi.csv": "2026-08-28 04:16",
    "tuketim_v102_kappa_optimum.csv": "2026-08-28 04:19",
    "tuketim_v109_birlesik.csv": "2026-08-28 07:08",
    "tuketim_m4_hava_capali.csv": "2026-08-29 04:46",
    "tuketim_p51_sicak05.csv": "2026-08-29 04:56",
    "tuketim_m6_ikiyon.csv": "2026-08-29 04:58",
    "tuketim_s3y40.csv": "2026-08-30 03:37",
    "tuketim_YP_seviye.csv": "2026-08-30 04:31",
    "tuketim_K_yenibas.csv": "2026-08-30 05:07",
    "tuketim_y40_sota_temiz.csv": "9999",  # LB'de olculmedi (turetilmis)
}
#: Yonun NE OLDUGU: "oznitelik" = test tablosundan kurulmus bir oznitelik
#: ekseninin dik bileseni; "model" = iki modelin/gonderimin farki.
CINS = {
    "tuketim_YP_seviye.csv": "oznitelik",  # tam-span taban + seviye dik bileseni
    "tuketim_K_yenibas.csv": "oznitelik",  # + yeni-trafo baslangic yanliligi
}

print("=" * 78)
print("n20  |c| ICIN OZNITELIK EKSENI NOKTALARI  (yalniz OKUMA)")
print("=" * 78)

with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)

a0 = oku(TABAN)
N = len(a0)
AD, D, L, P, OLC = [], [], [], [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    AD.append(f)
    D.append(d)
    L.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
    P.append(Pj)
    OLC.append(True)
for o in DUR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    D.append(d)
    L.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
    P.append(o["skor"])
    OLC.append(True)
for f, Lj in EK_MODEL.items():
    d = oku(f) - a0
    AD.append(f)
    D.append(d)
    L.append(float(Lj))
    P.append(np.nan)
    OLC.append(False)

D = np.array(D)
K = len(AD)
G = (D @ D.T) / N
del D
L = np.array(L)
P = np.array(P)
OLC = np.array(OLC)
QTAM = np.diag(G).copy()
assert len(set(AD)) == K, "yon adlari mukerrer"


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 0  ENVANTER -- LB'de gercekten olculmus yonler ve cinsleri")
print("=" * 78)
print(
    f"N = {N} satir | K = {K} yon | LB'de OLCULMUS = {int(OLC.sum())} | turetilmis = {K - int(OLC.sum())}"
)
print(
    "\nKaggle gonderim gecmisi (salt okuma) 30 satir dondurdu; 29 AYRI dosya\n"
    "(v55 iki kez gonderilmis, ayni skor). olculmus_skorlar.json + m112_durum\n"
    "'olcumler' bunlarin TAMAMINI kapsiyor. YANI:"
)
YEREL = sorted(x for x in os.listdir(S) if x.endswith(".csv"))
GONDERILEN = set(TARIH) - {"tuketim_y40_sota_temiz.csv"}
print(
    f"  submissions/ altinda {len(YEREL)} CSV var, bunlarin {len(GONDERILEN)} tanesi LB'ye GITMIS."
)
print(
    f"  {len(YEREL) - len(GONDERILEN & set(YEREL))} dosya HIC GONDERILMEMIS -> LB skoru YOK -> rho_dik OLCULEMEZ."
)
print("  Gorevde adi gecen b1/b2/b3 (span_k13/k15/k20), b4/b5/b6 (prob_*),")
print("  g7_span_tau3, q1a-q1d, YP_guc, YP_haftasonu, YP_seviye2, YP_bolge_*,")
print("  K_PROBE_* dosyalarinin HICBIRI Kaggle'a gonderilmemis:")
_ornek = [
    x
    for x in YEREL
    if x not in GONDERILEN and any(t in x for t in ("prob", "span_k", "YP_", "PROBE", "g7_", "q1"))
]
print("   ", ", ".join(_ornek))
print("  -> Bu dosyalar |c| icin NOKTA URETEMEZ. Sebep tekniktir: |c| tanimi")
print("     rho_dik'i icerir, rho_dik ise SADECE LB skorundan cozulur.")
print("\nLB'de olculmus 29 yonun cinsi:")
print(
    f"  oznitelik ekseni sondasi : {sum(1 for a in AD if CINS.get(a) == 'oznitelik')}  "
    f"({', '.join(a for a in AD if CINS.get(a) == 'oznitelik')})"
)
print(f"  model/gonderim farki     : {sum(1 for a in AD if CINS.get(a) != 'oznitelik')}")


# ===========================================================================
def coz(span_ix, j, etiket=""):
    """Verilen span'a gore j'nin rho_s, rho_dik, |c| ve tanilarini dondur."""
    ix = np.asarray(span_ix, dtype=int)
    Gr = G[np.ix_(ix, ix)]
    gj = G[ix, j]
    cc = np.linalg.pinv(Gr, rcond=RCOND) @ gj
    Qsp = float(cc @ Gr @ cc)
    Qdk = float(G[j, j] - 2 * cc @ gj + Qsp)
    Lsp = float(cc @ L[ix])
    cn2 = float(cc @ cc)
    if Qsp <= 1e-12 or Qdk <= 1e-12:
        return None
    rs = Lsp / np.sqrt(Qsp)
    rd = (L[j] - Lsp) / np.sqrt(Qdk)
    vrd = SIGMA_L**2 * (1.0 + cn2) / Qdk
    vrs = SIGMA_L**2 * cn2 / Qsp
    return dict(
        ad=AD[j],
        etiket=etiket,
        n_span=len(ix),
        Qdk=Qdk,
        dikpay=Qdk / QTAM[j],
        rho_s=rs,
        rho_dik=rd,
        sd_rs=np.sqrt(vrs),
        sd_rd=np.sqrt(vrd),
        c=abs(rd) / abs(rs) if rs != 0 else np.nan,
        cn2=cn2,
        # olcum hatasi duzeltilmis c^2 (Ec2 = (rd^2-vrd)/(rs^2-vrs))
        c_duz=np.sqrt(max((rd**2 - vrd) / (rs**2 - vrs), 0.0)) if (rs**2 - vrs) > 0 else np.nan,
    )


def yaz(r):
    print(
        f"{r['ad'].replace('tuketim_', '')[:26]:>26s} {r['etiket'][:12]:>12s} {r['n_span']:>3d} "
        f"{r['dikpay']:8.3f} {r['rho_s']:+9.4f} {r['sd_rs']:8.1e} {r['rho_dik']:+9.4f} "
        f"{r['sd_rd']:8.1e} {r['c']:8.3f} {r['c_duz']:8.3f}"
    )


BAS = (
    f"{'yon':>26s} {'etiket':>12s} {'|S|':>3s} {'dik pay':>8s} {'rho_s':>9s} {'sd(rs)':>8s} "
    f"{'rho_dik':>9s} {'sd(rd)':>8s} {'|c|':>8s} {'|c|duz':>8s}"
)


def gecerli(r, rs_snr=3.0):
    """Nokta kabul kapisi. rho_s ANLAMLI olmali, rho_dik ayirt edilebilir olmali."""
    if r is None:
        return False, "cozulemedi"
    if r["sd_rd"] > SD_KAPI:
        return False, f"sd(rho_dik)={r['sd_rd']:.1e} > {SD_KAPI}"
    if abs(r["rho_s"]) < rs_snr * r["sd_rs"]:
        return False, f"rho_s SNR={abs(r['rho_s']) / r['sd_rs']:.1f} < {rs_snr}"
    if abs(r["rho_dik"]) < rs_snr * r["sd_rd"]:
        return False, f"rho_dik SNR={abs(r['rho_dik']) / r['sd_rd']:.1f} < {rs_snr}"
    if r["dikpay"] < 0.01:
        return False, f"dik pay={r['dikpay']:.4f} < 0.01 (yon span ICINDE)"
    return True, ""


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 1  LOO (n10 tekrari) -- referans, BUGUNKU span ile")
print("=" * 78)
print(BAS)
LOO = {}
for j in range(K):
    if not OLC[j]:
        continue
    r = coz(list(range(K - 1 + 1)) and [i for i in range(K) if i != j], j, "LOO")
    LOO[AD[j]] = r
for a, r in sorted(LOO.items(), key=lambda t: -(t[1]["dikpay"] if t[1] else -1)):
    if r:
        yaz(r)
LOO_OK = [r for r in LOO.values() if gecerli(r)[0]]
print(f"\nLOO'da kapi gecen nokta sayisi: {len(LOO_OK)} / {int(OLC.sum())}")
if LOO_OK:
    _c = np.array([r["c_duz"] for r in LOO_OK])
    print(f"  |c| medyan {np.median(_c):.3f}  aralik [{_c.min():.3f}, {_c.max():.3f}]")
    print(
        f"  dik pay araligi: [{min(r['dikpay'] for r in LOO_OK):.3f}, {max(r['dikpay'] for r in LOO_OK):.3f}]"
    )
print("""
NOT. LOO'nun yapisal kusuru: bir yon, KENDISINDEN SONRA gonderilmis ve onu
  ICEREN gonderimlerle ayni span'da bulunuyor. Ornegin YP_seviye'nin LOO
  span'i K_yenibas'i icerir ve K_yenibas ZATEN seviye bilesenini tasir;
  boylece YP_seviye'nin dik payi yapay olarak SIFIRA yakin cikar ve nokta
  YOK OLUR. B2 ve B3 bu kusuru gideriyor.
""")


# ===========================================================================
print("=" * 78)
print("BOLUM 2  KRONOLOJIK ILERI DIKLESTIRME")
print("=" * 78)
print("Her gonderim yalniz KENDINDEN ONCE gonderilmislerin span'ina diklestirilir.")
print("Bu, gonderim aninda gercekten YENI olan yonu olcer; dik pay buyuktur.\n")
SIRA = sorted(range(K), key=lambda i: TARIH.get(AD[i], "9999"))
print(BAS)
KRON = []
for pos, j in enumerate(SIRA):
    if pos == 0 or not OLC[j]:
        continue
    span = SIRA[:pos]
    r = coz(span, j, "kron")
    if r is None:
        continue
    KRON.append(r)
    yaz(r)
KRON_OK = [r for r in KRON if gecerli(r)[0]]
print("\nKapiyi GECEMEYENLER ve nedeni:")
for r in KRON:
    ok, nd = gecerli(r)
    if not ok:
        print(f"  {r['ad'].replace('tuketim_', '')[:34]:>34s}  {nd}")
print(f"\nKRONOLOJIK kapi gecen nokta sayisi: {len(KRON_OK)}")


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 3  HEDEFLI -- OZNITELIK EKSENLERI (seviye, yenibaslangic)")
print("=" * 78)
IX = {a: i for i, a in enumerate(AD)}
J_SEV, J_YEN = IX["tuketim_YP_seviye.csv"], IX["tuketim_K_yenibas.csv"]
TEMEL = [i for i in range(K) if i not in (J_SEV, J_YEN)]  # 27 yon: "tam-span" cagi
print(f"TEMEL span = {len(TEMEL)} yon (iki oznitelik sondasi DISARIDA).")
print(BAS)
R_SEV = coz(TEMEL, J_SEV, "S0")
R_YEN0 = coz(TEMEL, J_YEN, "S0")
R_YEN1 = coz(TEMEL + [J_SEV], J_YEN, "S0+seviye")
R_SEV1 = coz(TEMEL + [J_YEN], J_SEV, "S0+yenibas")
for r in (R_SEV, R_SEV1, R_YEN0, R_YEN1):
    if r:
        yaz(r)
print("""
OKUMA.
  * "S0" satiri: sonda, IKIZI span'da YOKKEN. Dik payi buyuk -> saglam.
  * "S0+ikiz" satiri: ikizi span'a girince dik pay cokerse iki sonda AYNI
    yonu tasiyor demektir; o satir BAGIMSIZ bir nokta DEGILDIR.
DOGRU ic-ice kurulus (Gram-Schmidt, gonderim sirasina uygun):
  nokta 1 = seviye  | S0
  nokta 2 = yenibas | S0 + seviye     <-- seviyeden ARTAKALAN yeni bilgi
""")
HEDEFLI = []
for ad, r in (("seviye | S0", R_SEV), ("yenibas | S0+seviye", R_YEN1)):
    ok, nd = gecerli(r)
    print(
        f"  {ad:>22s}: |c| = {r['c_duz']:.3f}  dik pay {r['dikpay']:.3f}  "
        f"rho_s {r['rho_s']:+.4f} (SNR {abs(r['rho_s']) / r['sd_rs']:.0f})  "
        f"rho_dik {r['rho_dik']:+.4f} (SNR {abs(r['rho_dik']) / r['sd_rd']:.0f})  "
        f"{'GECERLI' if ok else 'RED: ' + nd}"
    )
    if ok:
        HEDEFLI.append(r)

# rho_s ISARETI ve BUYUKLUK denetimi (m140 tuzagi)
print("\nISARET DENETIMI (m140 tuzagi -- isaretli egimi buyuklukle karsilastirma):")
for ad, r in (("seviye", R_SEV), ("yenibas", R_YEN1)):
    print(
        f"  {ad:>10s}: isaret(rho_s)={np.sign(r['rho_s']):+.0f}  isaret(rho_dik)={np.sign(r['rho_dik']):+.0f}  "
        f"{'AYNI' if np.sign(r['rho_s']) == np.sign(r['rho_dik']) else 'TERS'}"
    )
print("  |c| BUYUKLUK oranidir; iki taraf da mutlak deger alinarak kullanildi.")

# Belgelerdeki bayat degerle karsilastirma
print("\nBELGE vs BUGUN (gorevin uyardigi tuzak):")
print(f"  seviye rho_s: belgede +0.0156 / n18 -0.0153 / n20 BUGUN {R_SEV['rho_s']:+.4f}")
print(f"  seviye rho_dik(LB): belgede -0.0304 / n20 BUGUN {R_SEV['rho_dik']:+.4f}")
print(f"  -> n20 |c|(seviye) = {R_SEV['c_duz']:.3f}   (belge/n18 orani 0.0304/0.0153 = 1.987)")

# Span buyuklugune duyarlilik: seviye icin span'i kucultup buyutelim
print("\nSPAN SECIMINE DUYARLILIK -- |c|(seviye), span kronolojik olarak buyurken:")
SIRA_T = [i for i in SIRA if i in TEMEL]
print(f"{'|S|':>4s} {'dik pay':>8s} {'rho_s':>9s} {'rho_dik':>9s} {'|c|':>8s}")
DUY_S = []
for m in range(3, len(SIRA_T) + 1):
    r = coz(SIRA_T[:m], J_SEV, "")
    if r is None:
        continue
    DUY_S.append((m, r))
    if m % 3 == 0 or m == len(SIRA_T):
        print(
            f"{m:4d} {r['dikpay']:8.3f} {r['rho_s']:+9.4f} {r['rho_dik']:+9.4f} {r['c_duz']:8.3f}"
        )
_cs = [r["c_duz"] for m, r in DUY_S if m >= 10 and gecerli(r)[0]]
if _cs:
    print(
        f"  |S|>=10 icin |c|(seviye): medyan {np.median(_cs):.3f}  "
        f"aralik [{min(_cs):.3f}, {max(_cs):.3f}]  n={len(_cs)}"
    )

print("\nSPAN SECIMINE DUYARLILIK -- |c|(yenibas), span = ilk m kronolojik + seviye:")
print(f"{'|S|':>4s} {'dik pay':>8s} {'rho_s':>9s} {'rho_dik':>9s} {'|c|':>8s}")
DUY_Y = []
for m in range(3, len(SIRA_T) + 1):
    r = coz(SIRA_T[:m] + [J_SEV], J_YEN, "")
    if r is None:
        continue
    DUY_Y.append((m, r))
    if m % 3 == 0 or m == len(SIRA_T):
        print(
            f"{m:4d} {r['dikpay']:8.3f} {r['rho_s']:+9.4f} {r['rho_dik']:+9.4f} {r['c_duz']:8.3f}"
        )
_cy = [r["c_duz"] for m, r in DUY_Y if m >= 10 and gecerli(r)[0]]
if _cy:
    print(
        f"  |S|>=10 icin |c|(yenibas): medyan {np.median(_cy):.3f}  "
        f"aralik [{min(_cy):.3f}, {max(_cy):.3f}]  n={len(_cy)}"
    )


# ===========================================================================
print("\n" + "=" * 78)
print("BOLUM 4  TOPLAMA")
print("=" * 78)
TUM = {"hedefli(oznitelik)": HEDEFLI, "kronolojik(model farklari)": KRON_OK, "LOO": LOO_OK}
for ad, grp in TUM.items():
    if not grp:
        print(f"{ad:>28s}: n = 0")
        continue
    c = np.array([r["c_duz"] for r in grp])
    print(
        f"{ad:>28s}: n = {len(c):2d}  medyan {np.median(c):.3f}  "
        f"aralik [{c.min():.3f}, {c.max():.3f}]  dik pay [{min(r['dikpay'] for r in grp):.3f},"
        f"{max(r['dikpay'] for r in grp):.3f}]"
    )

# dik pay ile |c| iliskisi -- KRONOLOJIK kumede dik paylar buyuk
HEP = KRON_OK + HEDEFLI
if len(HEP) >= 4:
    x = np.array([r["dikpay"] for r in HEP])
    y = np.log(np.clip([r["c_duz"] for r in HEP], 1e-3, None))
    ok = np.isfinite(y)
    rr = float(np.corrcoef(x[ok], y[ok])[0, 1])
    prm = [float(np.corrcoef(rng.permutation(x[ok]), y[ok])[0, 1]) for _ in range(8000)]
    pp = float((np.abs(prm) >= abs(rr)).mean())
    print(f"\nkor(dik pay, log|c|) = {rr:+.3f}  n = {int(ok.sum())}  permutasyon p = {pp:.3f}")
else:
    rr, pp = np.nan, np.nan
    print("\n(n<4, dik pay egilimi test edilemedi)")

# Nihai: oznitelik ekseni noktalari
if HEDEFLI:
    cH = np.array([r["c_duz"] for r in HEDEFLI])
    C_NOKTA = float(np.exp(np.mean(np.log(cH))))  # geometrik ortalama
    if len(cH) >= 2:
        sl = float(np.std(np.log(cH), ddof=1))
        # t dagilimi, n-1 sd; %90 aralik
        from scipy import stats

        t = stats.t.ppf(0.95, len(cH) - 1)
        C_ALT = float(np.exp(np.mean(np.log(cH)) - t * sl / np.sqrt(len(cH))))
        C_UST = float(np.exp(np.mean(np.log(cH)) + t * sl / np.sqrt(len(cH))))
    else:
        C_ALT = C_UST = float("nan")
else:
    cH, C_NOKTA, C_ALT, C_UST = np.array([]), float("nan"), float("nan"), float("nan")

print("\n" + "-" * 78)
print("OZNITELIK EKSENI NOKTALARI:")
for r in HEDEFLI:
    print(
        f"  {r['ad'].replace('tuketim_', ''):>24s}  |c| = {r['c_duz']:.3f}  dik pay {r['dikpay']:.3f}"
    )
print(
    f"\n  n = {len(cH)}   geometrik ortalama |c| = {C_NOKTA:.3f}   %90 [{C_ALT:.3f}, {C_UST:.3f}]"
)
print("  KARSILASTIRMA: m148/m113 capasi 1.986 | n10 (model farklari) 0.434")

cikti = {
    "aciklama": "n20 -- |c| = |rho_dik|/|rho_s|, OZNITELIK EKSENLERINDE olculdu",
    "sigma_L": SIGMA_L,
    "K_yon": K,
    "LB_olculmus": int(OLC.sum()),
    "gonderilmemis_sonda_sayisi": len(YEREL) - len(GONDERILEN & set(YEREL)),
    "oznitelik_noktalari": [
        {
            "ad": r["ad"],
            "etiket": r["etiket"],
            "c": r["c_duz"],
            "c_ham": r["c"],
            "dikpay": r["dikpay"],
            "rho_s": r["rho_s"],
            "rho_dik": r["rho_dik"],
            "sd_rho_s": r["sd_rs"],
            "sd_rho_dik": r["sd_rd"],
            "n_span": r["n_span"],
        }
        for r in HEDEFLI
    ],
    "kronolojik_noktalari": [
        {
            "ad": r["ad"],
            "c": r["c_duz"],
            "dikpay": r["dikpay"],
            "rho_s": r["rho_s"],
            "rho_dik": r["rho_dik"],
            "n_span": r["n_span"],
        }
        for r in KRON_OK
    ],
    "loo_noktalari": [{"ad": r["ad"], "c": r["c_duz"], "dikpay": r["dikpay"]} for r in LOO_OK],
    "duyarlilik_seviye": [{"n_span": m, "c": r["c_duz"], "dikpay": r["dikpay"]} for m, r in DUY_S],
    "duyarlilik_yenibas": [{"n_span": m, "c": r["c_duz"], "dikpay": r["dikpay"]} for m, r in DUY_Y],
    "n_oznitelik": int(len(cH)),
    "c_nokta": C_NOKTA,
    "c_90_alt": C_ALT,
    "c_90_ust": C_UST,
    "kor_dikpay_logc": rr,
    "p_perm": pp,
}
YOL = os.path.join(M29, "n20_c_oznitelik.json")
with open(YOL, "w", encoding="utf-8") as fh:
    json.dump(cikti, fh, ensure_ascii=False, indent=1)
print(f"\nYAZILDI {YOL}")
