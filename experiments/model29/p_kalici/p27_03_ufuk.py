"""p27-03: UFUK EKSENI / SEVIYE KAYMASI.

Sorular:
  1. Hata ufukla nasil buyuyor? (sicak/soguk ayri)
  2. Yanlilik mi varyans mi?
  3. Kahin seviye duzeltmesi tavani (ufuk kovasi / takvim ayi / gun)
  4. Ogrenilebilir mi? (blok-disi transfer)
  5. TESHIS: ufuk mu, yoksa "egitimde GORULMEYEN takvim ayi" mi?
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, agirlik, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
R = {}
BLOKLAR = ("yaz25", "guz25", "kis26")
D = {b: blok(b) for b in BLOKLAR}

# ---- her blogun EGITIMINDE etiket olarak gorunen takvim aylari -------------
# tuketim_model.py: kokenleri_ayikla, hedef blokla KESISEN her kokeni atar.
PENCERE = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
    "sub25": ("2025-02-01", "2025-03-31"),
    "bah25": ("2025-05-01", "2025-08-31"),
    "yaz25b": ("2025-07-01", "2025-10-31"),
    "guz25b": ("2025-09-01", "2025-12-31"),
    "kis26b": ("2025-11-01", "2026-02-28"),
    "bah26": ("2026-01-01", "2026-03-31"),
}


def gorulen_aylar(hedef):
    hb, hs = pd.Timestamp(PENCERE[hedef][0]), pd.Timestamp(PENCERE[hedef][1])
    aylar = set()
    for ad, (b, s) in PENCERE.items():
        b, s = pd.Timestamp(b), pd.Timestamp(s)
        if s < hb or b > hs:  # kesismiyor -> egitimde KALIR
            aylar |= set(pd.date_range(b, s, freq="D").month)
    return sorted(aylar)


KAPSAM = {b: gorulen_aylar(b) for b in BLOKLAR}
# TEST: 2026-04-01..07-31; hicbir koken atilmaz -> tum kokenler egitimde
TEST_KAPSAM = sorted(
    set().union(*[set(pd.date_range(b, s, freq="D").month) for b, s in PENCERE.values()])
)
R["00_kapsam"] = dict(
    aciklama="her blogun EGITIMINDE etiket olarak gorunen takvim aylari",
    blok_kapsami={b: KAPSAM[b] for b in BLOKLAR},
    blok_hedef_aylari={
        "yaz25": [4, 5, 6, 7], "guz25": [8, 9, 10, 11], "kis26": [12, 1, 2, 3]},
    test_kapsami=TEST_KAPSAM,
    test_hedef_aylari=[4, 5, 6, 7],
    NOT="TEST'in dort hedef ayi da egitimde etiket olarak VAR (yaz25+bah25 kokenleri).",
)
for b in BLOKLAR:
    hed = R["00_kapsam"]["blok_hedef_aylari"][b]
    R["00_kapsam"].setdefault("hedef_ayi_gorulmus_mu", {})[b] = {
        str(a): (a in KAPSAM[b]) for a in hed}
print("KAPSAM:", json.dumps(R["00_kapsam"], ensure_ascii=False, indent=1))


def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))


def bil_of(d, p):
    sg = d.soguk_mu.values == 1
    return bilesik(rmsle(d.y.values[~sg], p[~sg]), rmsle(d.y.values[sg], p[sg]))


KENAR = [0, 15, 30, 45, 60, 75, 90, 105, 122]


def kova(u):
    return np.clip(np.digitize(u, KENAR[1:-1], right=True), 0, len(KENAR) - 2)


# =========================================================================
# 1-2. UFUK EGRISI: yanlilik / sacilim ayrisimi (sicak & soguk ayri)
# =========================================================================
egri = []
for b in BLOKLAR:
    d = D[b]
    u = d.ufuk_gun.values
    k = kova(u)
    sg = d.soguk_mu.values == 1
    z = d.tuketim.values <= 0
    for rejim, m0 in (("sicak", ~sg), ("soguk", sg)):
        for i in range(len(KENAR) - 1):
            m = m0 & (k == i)
            if m.sum() < 300:
                continue
            r = d.r.values[m]
            rp = d.r.values[m & ~z]
            egri.append(dict(
                blok=b, rejim=rejim, kova=f"{KENAR[i]+1}-{KENAR[i+1]}", n=int(m.sum()),
                rmse=round(float(np.sqrt((r**2).mean())), 5),
                yanlilik=round(float(r.mean()), 4),
                sacilim=round(float(r.std()), 4),
                yanlilik_payi=round(float(r.mean() ** 2 / (r**2).mean()), 4),
                yanlilik_poz=round(float(rp.mean()), 4) if m.sum() > (m & z).sum() else None,
                sacilim_poz=round(float(rp.std()), 4) if m.sum() > (m & z).sum() else None,
                sifir_orani=round(float(z[m].mean()), 4),
                ay_dagilimi=sorted(set(d.ay.values[m].tolist())),
            ))
R["01_ufuk_egrisi"] = egri
print("\n1-2) UFUK EGRISI (yanlilik/sacilim):")
print(f"{'blok':7}{'rejim':7}{'kova':10}{'n':>9}{'rmse':>9}{'yanlilik':>10}"
      f"{'sacilim':>9}{'yan_payi':>10}{'yan_poz':>9}{'sac_poz':>9}{'sifir%':>8}")
for x in egri:
    print(f"{x['blok']:7}{x['rejim']:7}{x['kova']:10}{x['n']:>9,}{x['rmse']:>9.4f}"
          f"{x['yanlilik']:>+10.4f}{x['sacilim']:>9.4f}{x['yanlilik_payi']:>10.4f}"
          f"{(x['yanlilik_poz'] if x['yanlilik_poz'] is not None else 0):>+9.4f}"
          f"{(x['sacilim_poz'] if x['sacilim_poz'] is not None else 0):>9.4f}"
          f"{100*x['sifir_orani']:>8.2f}")

# =========================================================================
# 3. KAHIN SEVIYE DUZELTMESI TAVANLARI
# =========================================================================
def kahin(d, gruplar, rejim_ayri=True):
    """gruplar: satir basina grup kodu. Her grupta (rejim ayri) artik ort'yi sifirla."""
    p = d.p.values.copy()
    sg = d.soguk_mu.values == 1
    r = d.r.values
    for rej_m in ((~sg, sg) if rejim_ayri else (np.ones(len(d), bool),)):
        gv = np.asarray(gruplar)[rej_m]
        g = pd.Series(gv)
        ort = g.map(pd.Series(r[rej_m]).groupby(gv).mean())
        p[rej_m] = p[rej_m] + ort.to_numpy()
    return p


tavan = []
for b in BLOKLAR:
    d = D[b]
    sg = d.soguk_mu.values == 1
    taban = bil_of(d, d.p.values)
    tas = pd.to_datetime(d.tarih).values
    secenek = {
        "yok(taban)": None,
        "A_global_sabit": np.zeros(len(d), int),
        "B_ufuk_8kova": kova(d.ufuk_gun.values),
        "C_takvim_ayi": d.ay.values,
        "D_gun(122)": pd.factorize(tas)[0],
        "E_gun x hafta_gunu": pd.factorize(
            pd.Series(tas).astype(str).to_numpy()
            + "_" + d.hg.values.astype(str))[0],
    }
    for ad, g in secenek.items():
        if g is None:
            rec = dict(blok=b, duzeltme=ad, bilesik=round(taban, 5),
                       sicak=round(rmsle(d.y[~sg], d.p.values[~sg]), 5),
                       soguk=round(rmsle(d.y[sg], d.p.values[sg]), 5), kazanc=0.0)
        else:
            p2 = kahin(d, g)
            bl = bil_of(d, p2)
            rec = dict(blok=b, duzeltme=ad, bilesik=round(bl, 5),
                       sicak=round(rmsle(d.y[~sg], p2[~sg]), 5),
                       soguk=round(rmsle(d.y[sg], p2[sg]), 5),
                       kazanc=round(taban - bl, 5), n_grup=int(len(np.unique(g))))
        tavan.append(rec)
R["02_kahin_seviye_tavani"] = tavan
print("\n3) KAHIN SEVIYE DUZELTMESI TAVANI (rejim ayri, kohort agirlikli bilesik):")
print(f"{'blok':7}{'duzeltme':22}{'sicak':>9}{'soguk':>9}{'bilesik':>10}{'kazanc':>9}")
for x in tavan:
    print(f"{x['blok']:7}{x['duzeltme']:22}{x['sicak']:>9.4f}{x['soguk']:>9.4f}"
          f"{x['bilesik']:>10.5f}{x['kazanc']:>+9.5f}")

# blok ortalamasi + LB olcegi
OLCEK = 0.93907  # p27_01: LB 1.00115 / CV bilesik 1.0661
ozet = {}
for ad in ("yok(taban)", "A_global_sabit", "B_ufuk_8kova", "C_takvim_ayi",
           "D_gun(122)", "E_gun x hafta_gunu"):
    v = [x for x in tavan if x["duzeltme"] == ad]
    ort = float(np.mean([x["bilesik"] for x in v]))
    ozet[ad] = dict(cv_bilesik_ort=round(ort, 5), lb_olcekli=round(ort * OLCEK, 5),
                    lb_kazanc=round((1.0661 - ort) * OLCEK, 5))
R["03_tavan_ozeti"] = dict(olcek=OLCEK, tablo=ozet)
print("\n   BLOK ORTALAMASI -> LB olcegi:")
for a, v in ozet.items():
    print(f"   {a:22} CV {v['cv_bilesik_ort']:.5f}  LB~{v['lb_olcekli']:.5f}"
          f"  kazanc {v['lb_kazanc']:+.5f}")

# =========================================================================
# 4. OGRENILEBILIR MI? -- blok-disi transfer
# =========================================================================
transfer = []
for hed in BLOKLAR:
    d = D[hed]
    sg = d.soguk_mu.values == 1
    taban = bil_of(d, d.p.values)
    kh = kova(d.ufuk_gun.values)
    for kay_ad, kaynaklar in ([(k, [k]) for k in BLOKLAR if k != hed]
                              + [("DIGER_IKISI_ORT", [k for k in BLOKLAR if k != hed])]):
        p2 = d.p.values.copy()
        for rej_m, rej in ((~sg, "sicak"), (sg, "soguk")):
            duz = np.zeros(len(KENAR) - 1)
            for ks in kaynaklar:
                dk = D[ks]
                sgk = dk.soguk_mu.values == 1
                mk = (~sgk) if rej == "sicak" else sgk
                kk = kova(dk.ufuk_gun.values)[mk]
                rr = dk.r.values[mk]
                s = pd.Series(rr).groupby(kk).mean()
                duz += np.array([s.get(i, 0.0) for i in range(len(KENAR) - 1)])
            duz /= len(kaynaklar)
            p2[rej_m] = p2[rej_m] + duz[kh[rej_m]]
        bl = bil_of(d, p2)
        transfer.append(dict(hedef=hed, kaynak=kay_ad, bilesik=round(bl, 5),
                             kazanc=round(taban - bl, 5)))
R["04_ufuk_transfer"] = transfer
print("\n4) BLOK-DISI UFUK YANLILIK TRANSFERI (p11'in yaptigi sey):")
for x in transfer:
    print(f"   {x['hedef']:7} <- {x['kaynak']:18} bilesik {x['bilesik']:.5f}"
          f"  kazanc {x['kazanc']:+.5f}")

# global sabit transferi
gtr = []
for hed in BLOKLAR:
    d = D[hed]
    sg = d.soguk_mu.values == 1
    taban = bil_of(d, d.p.values)
    p2 = d.p.values.copy()
    for rej_m, rej in ((~sg, "sicak"), (sg, "soguk")):
        vals = []
        for ks in BLOKLAR:
            if ks == hed:
                continue
            dk = D[ks]
            sgk = dk.soguk_mu.values == 1
            vals.append(float(dk.r.values[(~sgk) if rej == "sicak" else sgk].mean()))
        p2[rej_m] += float(np.mean(vals))
    gtr.append(dict(hedef=hed, kazanc=round(taban - bil_of(d, p2), 5)))
R["05_global_sabit_transfer"] = gtr
print("\n5) BLOK-DISI GLOBAL SABIT KAYMA:", gtr)

# =========================================================================
# 5b. TESHIS: yanlilik ~ ufuk mu, "gorulmemis ay" mi?
# =========================================================================
ay_tab = []
for b in BLOKLAR:
    d = D[b]
    sg = d.soguk_mu.values == 1
    for a, gg in d.groupby("ay"):
        m = (d.ay.values == a)
        ay_tab.append(dict(
            blok=b, ay=int(a), n=int(len(gg)),
            gorulmus_mu=bool(a in KAPSAM[b]),
            ufuk_ort=round(float(d.ufuk_gun.values[m].mean()), 1),
            yanlilik=round(float(gg.r.mean()), 4),
            yanlilik_sicak=round(float(d.r.values[m & ~sg].mean()), 4),
            yanlilik_soguk=round(float(d.r.values[m & sg].mean()), 4)
            if (m & sg).sum() > 100 else None,
            rmse=round(float(np.sqrt((gg.r**2).mean())), 4),
        ))
R["06_takvim_ayi"] = ay_tab
print("\n5b) TESHIS -- TAKVIM AYI x GORULMUSLUK:")
print(f"{'blok':7}{'ay':>4}{'gorulmus':>10}{'ufuk_ort':>10}{'n':>9}"
      f"{'yanlilik':>10}{'yan_sicak':>11}{'yan_soguk':>11}{'rmse':>9}")
for x in ay_tab:
    ys = x["yanlilik_soguk"]
    print(f"{x['blok']:7}{x['ay']:>4}{str(x['gorulmus_mu']):>10}{x['ufuk_ort']:>10.1f}"
          f"{x['n']:>9,}{x['yanlilik']:>+10.4f}{x['yanlilik_sicak']:>+11.4f}"
          f"{(ys if ys is not None else 0):>+11.4f}{x['rmse']:>9.4f}")

# kis26 dogal deney: 12,1 GORULMEYEN vs 2,3 GORULEN (ikisi de ayni blok)
d = D["kis26"]
sg = d.soguk_mu.values == 1
kd = {}
for ad, aylar in (("12,01_GORULMEYEN", [12, 1]), ("02,03_GORULEN", [2, 3])):
    m = np.isin(d.ay.values, aylar)
    kd[ad] = dict(n=int(m.sum()), ufuk_ort=round(float(d.ufuk_gun.values[m].mean()), 1),
                  yanlilik=round(float(d.r.values[m].mean()), 4),
                  yanlilik_sicak=round(float(d.r.values[m & ~sg].mean()), 4),
                  rmse_sicak=round(float(np.sqrt((d.r.values[m & ~sg]**2).mean())), 4))
R["07_kis26_dogal_deney"] = kd
print("\n   KIS26 DOGAL DENEYI (ayni blok, ayni model, ufuk 2,3'te DAHA UZUN):")
for a, v in kd.items():
    print(f"   {a:20} {v}")

with open(os.path.join(CIK, "p27_03.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nyazildi p27_03.json")
