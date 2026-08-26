"""H4 derinlestirme: (A) kirpma kaybi, (B) DENGESIZ PANEL yanliligi gun ekseninde.

A -- np.clip(...,0,None) uc son-islem betiginde de var. Kac satir SIFIRA
    kirpildi, kirpma ne kadar MSE bozuyor?
B -- son_islem_gunolcek.py'nin `oran`i, CAPA'yi DENGELI panelde (>=%90 gun),
    TEST'i ise DENGESIZ panelde olcuyor. Test sicak trafolarinin 1.469'u
    kismi. Iki yonlu "satir ort cikar -> gun ort al" kestiricisi dengesiz
    panelde YANLI. Ayni dengeleme kurali TEST'e de uygulanirsa c ne olur?

    uv run python scripts/h4_derin_kirpma_panel.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]


def basli(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78, flush=True)


te = pd.read_csv(KOK / "data/raw/test.csv", dtype={"tanim": str})
tr = pd.read_csv(KOK / "data/raw/train.csv", dtype={"tanim": str})
te["tarih"] = pd.to_datetime(te["tarih"])
tr["tarih"] = pd.to_datetime(tr["tarih"])
tr_tanim = set(tr["tanim"].unique())
sicak = te["tanim"].isin(tr_tanim).to_numpy()
log_guc = np.log1p(te["guc"].to_numpy(dtype="float64"))


def oku(ad: str) -> np.ndarray:
    return pd.read_csv(KOK / "submissions" / ad)["tuketim"].to_numpy(dtype="float64")


v50 = oku("tuketim_v50_nihai30.csv")
v66 = oku("tuketim_v66_c1335.csv")
v67 = oku("tuketim_v67_c1335_olay.csv")

# ============================================================ A. KIRPMA KAYBI
basli("A. KIRPMA KAYBI -- v50 -> v66 (gun ekseni, c=1,335)")

# Uretim cebiri: yeni = clip(expm1(r + (c-1)*etki + log_guc), 0, None)
# Kirpilan satirlari kirpmasiz yeniden uret.
r50 = np.log1p(v50) - log_guc
r66 = np.log1p(np.maximum(v66, 0.0)) - log_guc
# etki'yi geri coz: sadece SICAK satirlarda uygulandi
d_log = np.log1p(v66) - np.log1p(v50)  # = (c-1)*etki, kirpilmayan satirlarda
gun = te["tarih"].to_numpy()

# gun basina medyan d_log -> gercek (c-1)*etki (kirpma bunu bozmaz cunku
# gunde binlerce satir var ve kirpilan cok az)
gd = pd.Series(d_log).groupby(gun).median()
d_ger = np.array(pd.Series(gun).map(gd).to_numpy(dtype="float64"), copy=True)
d_ger[~sicak] = 0.0

kirpmasiz = np.expm1(np.log1p(v50) + d_ger)
kirpildi = kirpmasiz < 0
print(f"kirpmasiz deger < 0 olan satir  = {int(kirpildi.sum())}")
print(f"v50 sifir sayisi = {int((v50 == 0).sum())}")
print(f"v66 sifir sayisi = {int((v66 == 0).sum())}")
print(f"v67 sifir sayisi = {int((v67 == 0).sum())}")
yeni_sifir = (v66 == 0) & (v50 > 0)
print(f"v50'de POZITIF iken v66'da SIFIR olan = {int(yeni_sifir.sum())}")
print(f"v50'de SIFIR iken v66'da pozitif olan  = {int(((v50 == 0) & (v66 > 0)).sum())}")
if yeni_sifir.sum():
    kv = v50[yeni_sifir]
    print(
        f"  bu satirlarin v50 degeri: min={kv.min():.6g} med={np.median(kv):.6g} "
        f"max={kv.max():.6g} ort={kv.mean():.6g}"
    )
    print(f"  hepsi sicak mi? sicak={int(sicak[yeni_sifir].sum())}/{int(yeni_sifir.sum())}")
    print(
        f"  kirpmasiz olsalardi: min={kirpmasiz[yeni_sifir].min():.6g} "
        f"max={kirpmasiz[yeni_sifir].max():.6g}"
    )

# Kirpmanin MSE maliyeti: gercek y bilinmiyor. Iki senaryo ile SINIRLA.
# Senaryo 1: gercek y = 0  -> kirpma FAYDALI (kayip 0)
# Senaryo 2: gercek y, bu trafolarin train'deki AYNI MEVSIM ortalamasi
basli("A2. KIRPILAN SATIRLARIN GERCEK DEGERI NE OLABILIR? (train kanidi)")
kirp_tanim = te.loc[yeni_sifir, "tanim"]
print(f"etkilenen essiz trafo = {kirp_tanim.nunique()}")
sayim = kirp_tanim.value_counts()
print(
    f"trafo basina satir: med={sayim.median():.0f} max={sayim.max()} "
    f"(ilk 5: {sayim.head(5).to_dict()})"
)

# bu trafolarin train'deki SON 60 gunu
son60 = tr[tr["tarih"] >= tr["tarih"].max() - pd.Timedelta(days=59)]
alt = son60[son60["tanim"].isin(set(kirp_tanim))]
print(
    f"train son 60 gun, bu trafolarin satirlari = {len(alt)}, essiz trafo={alt['tanim'].nunique()}"
)
if len(alt):
    print(
        f"  tuketim: sifir orani={float((alt['tuketim'] == 0).mean()):.4f} "
        f"med={alt['tuketim'].median():.4g} ort={alt['tuketim'].mean():.4g}"
    )
    print(f"  log1p(tuketim) ort = {np.log1p(alt['tuketim']).mean():.4f}")
# tum train'de bu trafolarin sifir orani
alt_tum = tr[tr["tanim"].isin(set(kirp_tanim))]
print(
    f"train TUMU: satir={len(alt_tum)} sifir orani="
    f"{float((alt_tum['tuketim'] == 0).mean()):.4f} "
    f"log1p ort={np.log1p(alt_tum['tuketim']).mean():.4f}"
)

n = len(v66)
for etiket, ly in [
    ("y=0 (olu)", 0.0),
    ("log1p(y)=0.10", 0.10),
    ("log1p(y)=0.50", 0.50),
    ("log1p(y)=1.00", 1.00),
    ("train son60 ort", float(np.log1p(alt["tuketim"]).mean()) if len(alt) else 0.0),
]:
    lp_kirp = np.zeros(int(yeni_sifir.sum()))  # log1p(0)=0
    lp_ham = np.log1p(np.maximum(kirpmasiz[yeni_sifir], 0.0))
    # kirpmasiz negatif -> MSLE tanimsiz; en yakin gecerli deger 0. Yani
    # ALTERNATIF, kirpma yerine kucuk bir TABAN (v50 degerini korumak).
    lp_alt = np.log1p(v50[yeni_sifir])
    d1 = ((ly - lp_kirp) ** 2 - (ly - lp_alt) ** 2).sum() / n
    print(f"  {etiket:22s}: kirpma yerine v50 degerini KORUSAK dMSE = {-d1:+.6f}")

# ================================================== B. DENGESIZ PANEL YANLILIGI
basli("B. GUN EKSENI KESTIRICISI -- DENGELI vs DENGESIZ PANEL")

gun_say = te.groupby("tanim")["tarih"].nunique()
n_gun = int(te["tarih"].nunique())
sicak_tanim = sorted(set(te.loc[sicak, "tanim"]))
print(f"test essiz gun = {n_gun}")
print(f"SICAK trafo = {len(sicak_tanim)}")
tam_sicak = [t for t in sicak_tanim if gun_say[t] == n_gun]
d90_sicak = [t for t in sicak_tanim if gun_say[t] >= 0.9 * n_gun]
print(f"  TAM (122 gun)   = {len(tam_sicak)}")
print(f"  >=%90 (>=110 g) = {len(d90_sicak)}")
print(f"  KISMI           = {len(sicak_tanim) - len(tam_sicak)}")
print(f"sicak satir = {int(sicak.sum())}, gun basina ort = {sicak.sum() / n_gun:.1f}")
gs = pd.Series(gun[sicak]).value_counts().sort_index()
print(
    f"gun basina sicak satir: min={gs.min()} max={gs.max()} "
    f"ilk gun={gs.iloc[0]} son gun={gs.iloc[-1]}"
)


def gun_etkisi(tanim, g, r):
    x = pd.DataFrame({"t": tanim, "g": g, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def gun_etkisi_ols(tanim, g, r, tur=60):
    """Alternating projections -- gercek iki yonlu sabit etki."""
    t = pd.Series(tanim)
    d = pd.Series(g)
    y = pd.Series(r, dtype="float64").copy()
    at = np.zeros(len(y))
    ag = np.zeros(len(y))
    for _ in range(tur):
        kal = y - ag
        mt = kal.groupby(t).transform("mean")
        at = mt.to_numpy()
        kal2 = y - at
        mg = kal2.groupby(d).transform("mean")
        ag = mg.to_numpy()
    b = pd.Series(ag).groupby(d).first()
    return b - b.mean()


# --- CAPA (uretimdeki gibi: train 2025-04-01..07-31, >=%90 gun, tuketim>0)
g = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & (tr["tuketim"] > 0)]
rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(g["guc"].to_numpy(dtype="float64"))
xg = pd.DataFrame({"t": g["tanim"].to_numpy(), "gg": g["tarih"].to_numpy()})
tamg = xg.groupby("t")["gg"].nunique()
tamg = set(tamg[tamg >= 0.9 * xg["gg"].nunique()].index)
secg = np.isin(g["tanim"].to_numpy(), list(tamg))
print(
    f"\nCAPA: 2025-04..07, >=%90 gunu olan trafo = {len(tamg)}, "
    f"satir = {int(secg.sum())} / {len(g)}"
)
b_gecen = gun_etkisi(g["tanim"].to_numpy()[secg], g["tarih"].to_numpy()[secg], rg[secg])
b_gecen_ols = gun_etkisi_ols(g["tanim"].to_numpy()[secg], g["tarih"].to_numpy()[secg], rg[secg])
print(f"  b_gecen std (basit) = {b_gecen.std():.6f}")
print(f"  b_gecen std (OLS)   = {b_gecen_ols.std():.6f}")

# --- TEST tarafinda ayni kurallarla dort varyant (giris = v50, uretimdeki taban)
r50_te = np.log1p(v50) - log_guc
tan_te = te["tanim"].to_numpy()

varyant = {}
mS = sicak
varyant["URETIM (tum sicak, dengesiz)"] = (tan_te[mS], gun[mS], r50_te[mS], gun_etkisi)
mT = sicak & te["tanim"].isin(tam_sicak).to_numpy()
varyant["TAM PANEL (122 gun)"] = (tan_te[mT], gun[mT], r50_te[mT], gun_etkisi)
m9 = sicak & te["tanim"].isin(d90_sicak).to_numpy()
varyant[">=%90 (capa kuralinin AYNISI)"] = (tan_te[m9], gun[m9], r50_te[m9], gun_etkisi)
varyant["TUM SICAK, OLS iki yonlu"] = (tan_te[mS], gun[mS], r50_te[mS], gun_etkisi_ols)

print()
sonuc = {}
for ad, (t_, g_, r_, fn) in varyant.items():
    b = fn(t_, g_, r_)
    oran = float(b_gecen.std() / b.std())
    ia = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)
    ib = pd.Series(b.values, index=pd.to_datetime(b.index).dayofyear)
    ortak = ia.index.intersection(ib.index)
    kor = float(np.corrcoef(ia[ortak], ib[ortak])[0, 1])
    c = kor * oran
    sonuc[ad] = (float(b.std()), oran, kor, c, int(len(t_)))
    print(f"{ad:32s} n={len(t_):7d} std={b.std():.6f} oran={oran:6.3f} kor={kor:.4f}  c={c:6.4f}")

print("\nURETIMDE KULLANILAN c = 1,335 (formul 1,492 x lb-kalibre 0,893)")
print("LB'nin cozdugu c*        = 1,332")
for ad, (s, o, k, c, nn) in sonuc.items():
    print(f"  {ad:32s} c={c:.4f}   c* farki = {c - 1.332:+.4f}")
