"""p04b: p04'un DUZELTMESI + hata kaynagi ayrismasi.

p04'ta tum dogrusal adaylar buyuk NEGATIF cikti. Sebep olculdu: blok
ortalamalari cok farkli (yaz25 +0.100, guz25 -0.230, kis26 +0.205) ve
ham regresyon bu SEVIYEYI de tasiyor. Seviye tasinmiyor (p01). Bu yuzden
burada YALNIZCA EGIM tasinir: hem r hem x, blok x ay hucresi icinde
merkezlenir. Boylece "sicaklik artinca artik nasil degisir" sorusu,
"bu blok genelde ne kadar sapiyor" sorusundan ayrilir.

Uygulama tarafinda da duzeltme yaz25'in kendi ortalamasini DEGISTIRMEZ
(merkezlenmis x), yani seviye kazanci sayilmaz -- sadece egim kazanci.

SIZINTI: katsayilar yalniz guz25+kis26 artiklarindan; yaz25 hedefi sadece
degerlendirmede.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.222

KOL = ["tanim", "tarih", "tuketim", "ilce_key", "soguk_mu", "_blok", "ufuk_gun",
       "guc", "tarim_orani", "yerlesim_orani", "cdd18", "cdd22", "cdd24",
       "sicaklik_ort", "sicaklik_max", "et0_toplam", "toprak_nem_ort", "vpd_ort",
       "gunes_radyasyon", "t_sifir_orani", "t_log_ort", "t_egim_cdd22"]
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=KOL)


def blok_artik(ad):
    blk = e[e._blok == ad]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [np.load(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy")).astype(np.float64)
         for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")
         if os.path.exists(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy"))]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{ad}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    bf["r"] = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    s = bf.soguk_mu.values.astype(np.float64)
    w = np.where(s == 1, HEDEF_SOGUK / s.mean(), (1 - HEDEF_SOGUK) / (1 - s.mean()))
    bf["w"] = w / w.mean()
    bf["tarih"] = pd.to_datetime(bf.tarih)
    bf["_hucre"] = bf._blok.astype(str) + "_" + bf.tarih.dt.month.astype(str)
    return bf.reset_index(drop=True)


B = {a: blok_artik(a) for a in ("yaz25", "guz25", "kis26")}
Y = B["yaz25"]
DIS = pd.concat([B["guz25"], B["kis26"]], ignore_index=True)
m0 = float((Y.w * Y.r**2).mean())
R = {"taban_rmsle": float(np.sqrt(m0))}
print(f"YAZ25 taban RMSLE={np.sqrt(m0):.6f}\n", flush=True)

# ---------- 0. HATA KAYNAGI AYRISMASI (yaz25) ----------
tot = float((Y.w * Y.r**2).sum())
sifir = (Y.tuketim.values == 0)
buyuk = (Y.r.values.__abs__() > 1.0)
print("=== yaz25 hata ayrismasi ===", flush=True)
for ad, m in (("hedef SIFIR", sifir), ("soguk trafo", Y.soguk_mu.values == 1),
              ("|artik|>1", buyuk), ("sifir & |artik|>1", sifir & buyuk)):
    print(f"  {ad:20s} satir%={m.mean():.4f}  SSE%={(Y.w[m]*Y.r[m]**2).sum()/tot:.4f}",
          flush=True)
    R[f"ayrisma_{ad}"] = {"satir_payi": float(m.mean()),
                          "sse_payi": float((Y.w[m] * Y.r[m] ** 2).sum() / tot)}

# ---------- 1. EGIM TASIMA (blok x ay icinde merkezlenmis) ----------
def hucre_merkez(df, x):
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nanmean(x))
    return x - pd.Series(x).groupby(df._hucre.values).transform("mean").values


def egim_kazanc(ad, xd, xy, kirp=0.6):
    xdc = hucre_merkez(DIS, xd)
    rdc = DIS.r.values - pd.Series(DIS.r.values).groupby(DIS._hucre.values).transform("mean").values
    wd = DIS.w.values
    den = float((wd * xdc * xdc).sum())
    if den <= 0:
        return None
    b = float((wd * xdc * rdc).sum() / den)
    se = float(np.sqrt(((wd * (rdc - b * xdc) ** 2).sum() / (len(wd) - 1)) / den))
    d = np.clip(b * hucre_merkez(Y, xy), -kirp, kirp)
    m1 = float((Y.w * (Y.r - d) ** 2).mean())
    g = float(np.sqrt(m0) - np.sqrt(m1))
    print(f"  {ad:26s} beta={b:+.5f} t={b/max(se,1e-12):+7.1f}  kazanc={g:+.6f}", flush=True)
    return {"beta": b, "t": float(b / max(se, 1e-12)), "kazanc": g}


KIYI = {"cesme", "karaburun", "urla", "seferihisar", "foca", "dikili", "selcuk",
        "guzelbahce", "menderes", "aliaga"}
SULAMA = {"saruhanli", "salihli", "alasehir", "turgutlu", "akhisar", "kinik",
          "bergama", "menemen", "torbali", "odemis", "tire", "bayindir", "sarigol",
          "kirkagac", "golmarmara", "kula", "gordes", "selendi", "demirci", "soma",
          "beydag", "kiraz", "kemalpasa"}
ki_d, ki_y = DIS.ilce_key.isin(KIYI).astype(float).values, Y.ilce_key.isin(KIYI).astype(float).values
su_d, su_y = DIS.ilce_key.isin(SULAMA).astype(float).values, Y.ilce_key.isin(SULAMA).astype(float).values
ta_d, ta_y = DIS.tarim_orani.fillna(0).values, Y.tarim_orani.fillna(0).values

ad2 = {}
for nm in ("cdd18", "cdd22", "cdd24", "sicaklik_ort", "sicaklik_max", "et0_toplam",
           "toprak_nem_ort", "vpd_ort", "gunes_radyasyon"):
    ad2[f"H_{nm}"] = (DIS[nm].values, Y[nm].values)
for nm in ("et0_toplam", "cdd24", "toprak_nem_ort"):
    ad2[f"S_tarim_x_{nm}"] = (ta_d * DIS[nm].values, ta_y * Y[nm].values)
    ad2[f"S_sulamailce_x_{nm}"] = (su_d * DIS[nm].values, su_y * Y[nm].values)
    ad2[f"T_kiyi_x_{nm}"] = (ki_d * DIS[nm].values, ki_y * Y[nm].values)

print("\n=== EGIM TASIMA (blok x ay merkezli; SEVIYE tasinmaz) ===", flush=True)
R["egim"] = {k: v for k, v in ((a, egim_kazanc(a, *xy)) for a, xy in ad2.items()) if v}

# ---------- 2. EGIMIN KENDISI TASINIYOR MU? blok basina ayri ----------
print("\n=== EGIM KARARLILIGI: her blokta ayri olculur ===", flush=True)
R["egim_bloklar"] = {}
for nm in ("cdd24", "et0_toplam", "sicaklik_ort", "toprak_nem_ort"):
    sat = []
    for bd, df in B.items():
        xc = hucre_merkez(df, df[nm].values)
        rc = df.r.values - pd.Series(df.r.values).groupby(df._hucre.values).transform("mean").values
        w = df.w.values
        sat.append((bd, float((w * xc * rc).sum() / max((w * xc * xc).sum(), 1e-12))))
    print(f"  {nm:18s} " + "  ".join(f"{b}={v:+.5f}" for b, v in sat), flush=True)
    R["egim_bloklar"][nm] = {b: v for b, v in sat}

with open(os.path.join(BURA, "p04b_egim.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("\nyazildi p04b_egim.json", flush=True)
