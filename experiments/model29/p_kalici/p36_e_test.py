"""p36-e: p35_d ANOMALISI -- TEST'te bayrakli satirlarda ort_p=1.98 ama yaz25'te 0.36.
Gercek mi, yoksa bayrak/populasyon tanimi mi farkli?  KIYAS ADIL OLMALI:
ayni bayrak, ayni agirliksiz ortalama, sicak/soguk ayri.
"""
import os
import numpy as np, pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
import sys; sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import blok  # noqa

te = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
raw = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
sub = pd.read_csv(os.path.join(KOK, "submissions/tuketim_YP_seviye.csv"))
assert np.array_equal(sub.id.values, raw.id.values)
assert np.array_equal(te.id.values, raw.id.values)
p_te = np.log1p(sub.tuketim.values.astype(np.float64))
ks_te = np.nan_to_num(te.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
sg_te = te.soguk_mu.values == 1
print("TEST n=%d soguk=%.4f" % (len(te), sg_te.mean()))

rows = []
def ekle(ad, p, ks, sg, ozet):
    f = ks >= 1
    fs = f & ~sg
    rows.append(dict(kume=ad, n=len(p), soguk_pay=round(float(sg.mean()), 4),
        bayrak_pay=round(float(f.mean()), 4),
        bayrak_pay_sicakta=round(float(fs.sum() / max((~sg).sum(), 1)), 4),
        ort_p_bayrak=round(float(p[f].mean()), 4),
        rms_p_bayrak=round(float(np.sqrt((p[f] ** 2).mean())), 4),
        medyan_p_bayrak=round(float(np.median(p[f])), 4),
        p_bayrak_ust10=round(float(np.quantile(p[f], 0.9)), 4),
        ort_p_bayraksiz=round(float(p[~f].mean()), 4),
        ozet_pencere=ozet))

ekle("TEST(gonderilen dosya)", p_te, ks_te, sg_te,
     float(np.nanmedian(te.ozet_pencere_gun.values)) if "ozet_pencere_gun" in te else np.nan)
for b in ("yaz25", "guz25", "kis26"):
    d = blok(b, soguk_harman="cat", son_islem=True)
    ks = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    ekle(b, d.p.values, ks, d.soguk_mu.values == 1,
         float(np.nanmedian(d.ozet_pencere_gun.values)))
    if b == "yaz25":
        Y = d
print(pd.DataFrame(rows).T.to_string())

# --- neden farkli? bayrakli satirlarda t_kuyruk_sifir dagilimi ve gecmis ---
print("\n--- bayrakli satirlarda ozetler (medyan) ---")
kar = []
for ad, ks, df in (("TEST", ks_te, te), ("yaz25", np.nan_to_num(
        Y.t_kuyruk_sifir.values.astype(float), nan=-1.0), Y)):
    f = ks >= 1
    row = dict(kume=ad, n_bayrak=int(f.sum()))
    for c in ("t_kuyruk_sifir", "t_sifir_orani", "t_log_ort", "t_log_son30",
              "t_son30_gun", "t_son_kayit_yasi", "t_gun_sayisi", "guc",
              "ozet_pencere_gun", "t_olu_mu", "ufuk_gun"):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").values[f]
            row[c] = round(float(np.nanmedian(v)), 3)
    kar.append(row)
print(pd.DataFrame(kar).T.to_string())

# --- t_olu_mu=1 olanlar ayri ---
print("\n--- t_olu_mu==1 alt kumesi ---")
for ad, df, p in (("TEST", te, p_te), ("yaz25", Y, Y.p.values)):
    o = np.nan_to_num(pd.to_numeric(df.t_olu_mu, errors="coerce").values, nan=0.0) > 0.5
    print(ad, "pay=%.4f ort_p=%.3f medyan_p=%.3f rms_p=%.3f" %
          (o.mean(), p[o].mean(), np.median(p[o]), np.sqrt((p[o]**2).mean())))

# --- gonderilen dosya URETIM harmani mi? baska dosyalarla kiyas ---
print("\n--- diger gonderimlerde ayni bayrakta ort_p ---")
f = ks_te >= 1
for fn in ("tuketim_YP_seviye.csv", "tuketim_TS_taban.csv", "tam_pipeline.csv",
           "tuketim_H2_harman311.csv", "tuketim_K_NIHAI_BUZMELI.csv"):
    yol = os.path.join(KOK, "submissions", fn)
    if not os.path.exists(yol): continue
    s = pd.read_csv(yol)
    if not np.array_equal(s.id.values, raw.id.values): print(fn, "id farkli"); continue
    pp = np.log1p(s.tuketim.values.astype(np.float64))
    print("%-32s ort_p_bayrak=%.4f medyan=%.4f ort_p_hepsi=%.4f" %
          (fn, pp[f].mean(), np.median(pp[f]), pp.mean()))
print("\nyaz25 ort_p_hepsi=%.4f" % Y.p.values.mean())
