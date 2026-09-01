"""p27-04: GUNLUK SEVIYE CAPASI.

Kahin gun-duzeyi seviye duzeltmesi tavani +0.0317 LB. Sorun: ogrenilebilir mi?
Test doneminde GERCEK gunluk ulusal tuketim (ulusal_gunluk) VERILMIS (122 gun,
NaN yok). Yani gun-duzeyi seviye icin GERCEK bir capa var.

Sinav: gun-duzeyi yanlilik b_t, ulusal seriden BLOK-DISI ogrenilebiliyor mu?
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
R = {}
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b) for b in BLOKLAR}

KOL = ["tarih", "ulusal_gunluk", "ulusal_tepe", "ulusal_yil_once",
       "ulusal_yillik_buyume", "sicaklik_ort", "cdd22", "gun_uzunlugu_saat"]
E = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=KOL + ["_blok"])
GUN = E.groupby("tarih")[KOL[1:]].mean()



def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def bil_of(d, p):
    sg = d.soguk_mu.values == 1
    return bilesik(rmsle(d.y.values[~sg], p[~sg]), rmsle(d.y.values[sg], p[sg]))


# ---------------- gun duzeyi yanlilik tablosu ----------------
sat = []
for b in BLOKLAR:
    d = D[b]
    sg = d.soguk_mu.values == 1
    t = pd.to_datetime(d.tarih)
    for rej, m in (("sicak", ~sg), ("soguk", sg)):
        g = pd.DataFrame(dict(t=t.values[m], r=d.r.values[m])).groupby("t").r.agg(
            ["mean", "count"])
        for ts, row in g.iterrows():
            sat.append(dict(blok=b, rejim=rej, tarih=ts, b=row["mean"], n=row["count"]))
G = pd.DataFrame(sat)
G = G.merge(GUN.reset_index().rename(columns={"tarih": "tarih"}), on="tarih", how="left")
G["lu"] = np.log(G.ulusal_gunluk)
G["lyo"] = np.log(G.ulusal_yil_once)
G["byy"] = G.lu - G.lyo          # yillik buyume (log)
G["dow"] = G.tarih.dt.dayofweek

ozet = []
for (b, rej), gg in G.groupby(["blok", "rejim"]):
    ozet.append(dict(blok=b, rejim=rej, n_gun=int(len(gg)),
                     b_ort=round(float(gg.b.mean()), 4),
                     b_std=round(float(gg.b.std()), 4),
                     kor_lu=round(float(np.corrcoef(gg.b, gg.lu)[0, 1]), 3),
                     kor_byy=round(float(np.corrcoef(gg.b, gg.byy)[0, 1]), 3),
                     kor_sic=round(float(np.corrcoef(gg.b, gg.sicaklik_ort)[0, 1]), 3),
                     kor_gunuz=round(float(np.corrcoef(gg.b, gg.gun_uzunlugu_saat)[0, 1]), 3)))
R["01_gun_yanliligi"] = ozet
print("1) GUN-DUZEYI YANLILIK b_t ve ULUSAL SERIYLE KORELASYON")
print(f"{'blok':7}{'rejim':7}{'n':>5}{'b_ort':>9}{'b_std':>8}{'kor_lu':>8}"
      f"{'kor_byy':>9}{'kor_sic':>9}{'kor_gunuz':>10}")
for x in ozet:
    print(f"{x['blok']:7}{x['rejim']:7}{x['n_gun']:>5}{x['b_ort']:>+9.4f}{x['b_std']:>8.4f}"
          f"{x['kor_lu']:>+8.3f}{x['kor_byy']:>+9.3f}{x['kor_sic']:>+9.3f}"
          f"{x['kor_gunuz']:>+10.3f}")

# blok ici (blok ortalamasi cikarilmis) -- SAF gun-ici degiskenlik
print("\n   blok-ici (b - blok ort):")
for (b, rej), gg in G.groupby(["blok", "rejim"]):
    bb = gg.b - gg.b.mean()
    print(f"   {b:7}{rej:7} kor(lu)={np.corrcoef(bb, gg.lu)[0,1]:+.3f}"
          f"  kor(byy)={np.corrcoef(bb, gg.byy)[0,1]:+.3f}"
          f"  kor(sic)={np.corrcoef(bb, gg.sicaklik_ort)[0,1]:+.3f}")

# ---------------- BLOK-DISI SINAV ----------------
# Model: b_t = a + B'x_t  (gun agirlikli EKK), iki blokta fit, ucuncude uygula.
OZELLIK = {
    "S1_yalniz_lu": ["lu"],
    "S2_lu+byy": ["lu", "byy"],
    "S3_lu+byy+sic": ["lu", "byy", "sicaklik_ort"],
    "S4_lu+byy+sic+gunuz": ["lu", "byy", "sicaklik_ort", "gun_uzunlugu_saat"],
    "S5_yalniz_byy": ["byy"],
}
sinav = []
for hed in BLOKLAR:
    d = D[hed]
    taban = bil_of(d, d.p.values)
    sg = d.soguk_mu.values == 1
    th = pd.to_datetime(d.tarih)
    for ad, kols in OZELLIK.items():
        p2 = d.p.values.copy()
        for rej, m in (("sicak", ~sg), ("soguk", sg)):
            tr = G[(G.blok != hed) & (G.rejim == rej)]
            te = G[(G.blok == hed) & (G.rejim == rej)]
            X = np.column_stack([np.ones(len(tr))] + [tr[k].values for k in kols])
            w = tr.n.values.astype(float)
            beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], tr.b.values * np.sqrt(w),
                                   rcond=None)[0]
            Xe = np.column_stack([np.ones(len(te))] + [te[k].values for k in kols])
            tah = pd.Series(Xe @ beta, index=te.tarih.values)
            p2[m] = p2[m] + th.values[m].astype("datetime64[ns]").astype("O")
            p2[m] = d.p.values[m] + tah.reindex(th.values[m]).to_numpy()
        bl = bil_of(d, p2)
        sinav.append(dict(hedef=hed, model=ad, bilesik=round(bl, 5),
                          kazanc=round(taban - bl, 5)))
R["02_blokdisi_sinav"] = sinav
print("\n2) BLOK-DISI SINAV (2 blokta fit, 3.'de uygula):")
for x in sinav:
    print(f"   {x['hedef']:7} {x['model']:22} bilesik {x['bilesik']:.5f}"
          f"  kazanc {x['kazanc']:+.5f}")

# ---------------- TAVAN: gun-ici kahin, sicak/soguk ayri ----------------
tv = []
for hed in BLOKLAR:
    d = D[hed]
    taban = bil_of(d, d.p.values)
    sg = d.soguk_mu.values == 1
    th = pd.to_datetime(d.tarih).values
    p_gun = d.p.values.copy()
    for rej, m in (("sicak", ~sg), ("soguk", sg)):
        s = pd.Series(d.r.values[m]).groupby(th[m]).mean()
        p_gun[m] = p_gun[m] + s.reindex(th[m]).to_numpy()
    # yalniz sicak / yalniz soguk
    p_s = d.p.values.copy()
    s = pd.Series(d.r.values[~sg]).groupby(th[~sg]).mean()
    p_s[~sg] += s.reindex(th[~sg]).to_numpy()
    p_c = d.p.values.copy()
    s = pd.Series(d.r.values[sg]).groupby(th[sg]).mean()
    p_c[sg] += s.reindex(th[sg]).to_numpy()
    tv.append(dict(hedef=hed, taban=round(taban, 5),
                   kahin_gun_ikisi=round(taban - bil_of(d, p_gun), 5),
                   kahin_gun_yalniz_sicak=round(taban - bil_of(d, p_s), 5),
                   kahin_gun_yalniz_soguk=round(taban - bil_of(d, p_c), 5)))
R["03_gun_kahin_ayrisim"] = tv
print("\n3) GUN KAHIN TAVANI AYRISIMI:")
for x in tv:
    print("   ", x)

# ---------------- TEST DONEMI CAPA VERISI ----------------
T = pd.read_parquet(os.path.join(DN, "test.parquet"),
                    columns=["tarih", "ulusal_gunluk", "ulusal_yil_once"])
tg = T.groupby("tarih")[["ulusal_gunluk", "ulusal_yil_once"]].mean()
tg["byy"] = np.log(tg.ulusal_gunluk) - np.log(tg.ulusal_yil_once)
R["04_test_capa"] = dict(
    n_gun=int(len(tg)), nan=int(T.ulusal_gunluk.isna().sum()),
    ulusal_ort=round(float(tg.ulusal_gunluk.mean()), 1),
    byy_ort=round(float(tg.byy.mean()), 4), byy_std=round(float(tg.byy.std()), 4),
    egitim_byy_ort=round(float(G.byy.mean()), 4),
    NOT="test doneminin GERCEK gunluk ulusal tuketimi ozellik olarak VERILMIS",
)
print("\n4) TEST CAPA:", R["04_test_capa"])

with open(os.path.join(CIK, "p27_04.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p27_04.json")
