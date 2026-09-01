"""p27-02: hata anatomisi -- cepler, ufuk, artik yapisi."""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import blok, agirlik, rmsle, HEDEF_SOGUK

CIK = os.path.dirname(os.path.abspath(__file__))
R = {}
D = {b: blok(b) for b in ("yaz25", "guz25", "kis26")}


def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def cep(d, w, mask, ad):
    """maskeli cebin agirlikli MSE payi ve o cep MUKEMMEL olsaydi bilesik."""
    r2 = d.r.values**2
    top = float(np.sum(w * r2))
    pay = float(np.sum(w[mask] * r2[mask]) / top)
    sg = d.soguk_mu.values == 1
    p2 = d.p.values.copy()
    p2[mask] = d.y.values[mask]
    yeni = bilesik(rmsle(d.y[~sg], p2[~sg]), rmsle(d.y[sg], p2[sg]))
    return dict(ad=ad, n=int(mask.sum()), satir_pay=round(float(mask.mean()), 4),
                mse_pay=round(pay, 4), mukemmel_bilesik=round(yeni, 5))


# ---------- A. UFUK ----------
ufuk = []
for b, d in D.items():
    w = agirlik(d)
    u = d.ufuk_gun.values
    kn = [0, 15, 30, 45, 60, 75, 90, 105, 122, 10**9]
    for i in range(len(kn) - 1):
        m = (u > kn[i]) & (u <= kn[i + 1])
        if m.sum() < 500:
            continue
        sg = d.soguk_mu.values == 1
        ufuk.append(dict(blok=b, ufuk=f"{kn[i]+1}-{kn[i+1] if kn[i+1]<10**9 else 'ust'}",
                         n=int(m.sum()),
                         rmsle=round(rmsle(d.y[m], d.p.values[m]), 5),
                         rmsle_sicak=round(rmsle(d.y[m & ~sg], d.p.values[m & ~sg]), 5) if (m & ~sg).sum() > 100 else None,
                         rmsle_soguk=round(rmsle(d.y[m & sg], d.p.values[m & sg]), 5) if (m & sg).sum() > 100 else None,
                         ort_artik=round(float(d.r.values[m].mean()), 4)))
R["A_ufuk"] = ufuk
print("UFUK:")
for x in ufuk:
    print("  ", x)
print("test ufuk araligi: 1-122 gun; blok ufuk max:",
      {b: int(d.ufuk_gun.max()) for b, d in D.items()})

# ---------- B. CEPLER ----------
cepler = []
for b, d in D.items():
    w = agirlik(d)
    y = d.tuketim.values
    sg = d.soguk_mu.values == 1
    z = y <= 0
    grup = [
        ("sifir_TUM", z),
        ("sifir_soguk", z & sg),
        ("sifir_sicak", z & ~sg),
        ("soguk_TUM", sg),
        ("soguk_pozitif", sg & ~z),
        ("sicak_pozitif", (~sg) & (~z)),
        ("kucuk_pozitif_y<10", (~z) & (y > 0) & (y < 10)),
        ("orta_10-100", (y >= 10) & (y < 100)),
        ("buyuk_y>=100", y >= 100),
    ]
    for ad, m in grup:
        if m.sum() < 50:
            continue
        c = cep(d, w, m, ad)
        c["blok"] = b
        cepler.append(c)
R["B_cepler"] = cepler
print("\nCEPLER (mukemmel_bilesik = o cep sifir hatali olsa CV bilesigi):")
for x in cepler:
    print("  ", x)

# ---------- C. TRAFO YOGUNLASMASI ----------
yog = []
for b, d in D.items():
    w = agirlik(d)
    r2 = d.r.values**2
    g = pd.DataFrame(dict(t=d.tanim.values, wr2=w * r2))
    s = g.groupby("t").wr2.sum().sort_values(ascending=False)
    top = s.sum()
    n = len(s)
    yog.append(dict(blok=b, n_trafo=int(n),
                    **{f"ust_%{p}": round(float(s.iloc[: max(1, int(n * p / 100))].sum() / top), 4)
                       for p in (1, 5, 10, 25, 50)}))
R["C_trafo_yogunlasmasi"] = yog
print("\nTRAFO YOGUNLASMASI:", yog)

# ---------- D. ARTIK YAPISI: haftanin gunu / tatil / ay / sicaklik ----------
yapi = {}
for eks, kol in (("haftanin_gunu", "hg"), ("ay", "ay"), ("tatil_mi", "tatil_mi")):
    sat = []
    for b, d in D.items():
        for k, gg in d.groupby(kol):
            sat.append(dict(blok=b, eksen=str(k), n=int(len(gg)),
                            ort_artik=round(float(gg.r.mean()), 4),
                            rmse=round(float(np.sqrt((gg.r**2).mean())), 4)))
    yapi[eks] = sat
# sicaklik tepkisi: artik vs sicaklik_ort desili
sat = []
for b, d in D.items():
    q = pd.qcut(d.sicaklik_ort, 10, duplicates="drop", labels=False)
    for k, gg in d.groupby(q):
        sat.append(dict(blok=b, desil=int(k), n=int(len(gg)),
                        sic_ort=round(float(gg.sicaklik_ort.mean()), 1),
                        ort_artik=round(float(gg.r.mean()), 4),
                        rmse=round(float(np.sqrt((gg.r**2).mean())), 4)))
yapi["sicaklik_desili"] = sat
R["D_artik_yapisi"] = yapi
print("\nARTIK YAPISI (ort_artik buyukse sistematik kayma var):")
for eks in yapi:
    vals = [x["ort_artik"] for x in yapi[eks]]
    print(f"  {eks}: ort_artik aralik [{min(vals):.4f},{max(vals):.4f}] yayilim={max(vals)-min(vals):.4f}")

# ---------- E. GUC / ILCE HUCRELERI ----------
hucre = []
for b, d in D.items():
    w = agirlik(d)
    r2 = d.r.values**2
    top = float(np.sum(w * r2))
    gk = pd.qcut(d.guc, 8, duplicates="drop", labels=False)
    for k in sorted(pd.unique(gk.dropna())):
        m = (gk == k).values
        hucre.append(dict(blok=b, eksen="guc_oktil", k=int(k), n=int(m.sum()),
                          guc_ort=round(float(d.guc.values[m].mean()), 1),
                          mse_pay=round(float(np.sum(w[m] * r2[m]) / top), 4),
                          rmse=round(float(np.sqrt(np.average(r2[m], weights=w[m]))), 4),
                          ort_artik=round(float(np.average(d.r.values[m], weights=w[m])), 4)))
R["E_guc_hucreleri"] = hucre
print("\nGUC OKTILLERI:")
for x in hucre:
    print("  ", x)

with open(os.path.join(CIK, "p27_02.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nyazildi p27_02.json")
