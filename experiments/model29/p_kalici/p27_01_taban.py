"""p27-01: taban RMSLE'ler, izo-egri cebiri, sifir anatomisi."""
import json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import blok, agirlik, rmsle, HEDEF_SOGUK

CIK = os.path.dirname(os.path.abspath(__file__))
R = {}

# ---------- 1. blok tabanlari ----------
D = {}
tablo = []
for b in ("yaz25", "guz25", "kis26"):
    d = blok(b)
    D[b] = d
    sg = d.soguk_mu.values == 1
    sic_r = rmsle(d.y[~sg], d.p[~sg])
    sog_r = rmsle(d.y[sg], d.p[sg])
    bil = float(np.sqrt(HEDEF_SOGUK * sog_r**2 + (1 - HEDEF_SOGUK) * sic_r**2))
    tablo.append(dict(blok=b, n=int(len(d)), n_soguk=int(sg.sum()),
                      soguk_pay=round(float(sg.mean()), 4),
                      sicak_rmsle=round(sic_r, 5), soguk_rmsle=round(sog_r, 5),
                      test_bilesimi=round(bil, 5),
                      tarih_min=str(d.tarih.min())[:10], tarih_max=str(d.tarih.max())[:10]))
    print(tablo[-1])
R["01_blok_tabanlari"] = tablo

# blok ortalamasi (esit agirlik)
sic_ort = float(np.mean([t["sicak_rmsle"] for t in tablo]))
sog_ort = float(np.mean([t["soguk_rmsle"] for t in tablo]))
bil_ort = float(np.sqrt(HEDEF_SOGUK * sog_ort**2 + (1 - HEDEF_SOGUK) * sic_ort**2))
print(f"\nBLOK ORT: sicak={sic_ort:.5f} soguk={sog_ort:.5f} bilesik={bil_ort:.5f}")
R["02_blok_ortalamasi"] = dict(sicak=round(sic_ort, 5), soguk=round(sog_ort, 5),
                               bilesik=round(bil_ort, 5))

# ---------- 2. IZO-EGRI ----------
# LB olcegine tasi: LB=1.00115 gozlendi. CV bilesigini olcek katsayisiyla LB'ye esle.
LB = 1.00115
HEDEF = 0.98038          # 1. sira
HEDEF3 = 0.99556         # 3. sira
olcek = LB / bil_ort
sic_lb = sic_ort * olcek
sog_lb = sog_ort * olcek
print(f"olcek={olcek:.5f}  LB-olcekli sicak={sic_lb:.5f} soguk={sog_lb:.5f}")

def bilesik(sic, sog):
    return float(np.sqrt(HEDEF_SOGUK * sog**2 + (1 - HEDEF_SOGUK) * sic**2))

izo = {}
for ad, hedef in (("1_sira_0.98038", HEDEF), ("3_sira_0.99556", HEDEF3)):
    ihtiyac = hedef**2
    # (a) yalniz sicak
    q = (ihtiyac - HEDEF_SOGUK * sog_lb**2) / (1 - HEDEF_SOGUK)
    sic_gerek = float(np.sqrt(q)) if q > 0 else None
    # (b) yalniz soguk
    q2 = (ihtiyac - (1 - HEDEF_SOGUK) * sic_lb**2) / HEDEF_SOGUK
    sog_gerek = float(np.sqrt(q2)) if q2 > 0 else None
    # (c) orantili
    k = hedef / LB
    izo[ad] = dict(
        hedef=hedef,
        yalniz_sicak=dict(gerekli=None if sic_gerek is None else round(sic_gerek, 5),
                          mevcut=round(sic_lb, 5),
                          iyilesme_yuzde=None if sic_gerek is None else round(100 * (1 - sic_gerek / sic_lb), 2),
                          mumkun_mu="IMKANSIZ (negatif kare)" if sic_gerek is None else "matematiksel olarak mumkun"),
        yalniz_soguk=dict(gerekli=None if sog_gerek is None else round(sog_gerek, 5),
                          mevcut=round(sog_lb, 5),
                          iyilesme_yuzde=None if sog_gerek is None else round(100 * (1 - sog_gerek / sog_lb), 2),
                          mumkun_mu="IMKANSIZ (negatif kare)" if sog_gerek is None else "matematiksel olarak mumkun"),
        orantili=dict(carpan=round(k, 5), iyilesme_yuzde=round(100 * (1 - k), 2),
                      yeni_sicak=round(sic_lb * k, 5), yeni_soguk=round(sog_lb * k, 5)),
    )
    print(ad, json.dumps(izo[ad], ensure_ascii=False, indent=1))
R["03_izo_egri"] = dict(LB=LB, olcek=round(olcek, 5), sicak_lb=round(sic_lb, 5),
                        sog_lb=round(sog_lb, 5), hedefler=izo)

# izo tablo: soguk RMSLE grid -> gereken sicak RMSLE
grid = []
for sog_v in np.arange(0.85, 1.75, 0.05):
    q = (HEDEF**2 - HEDEF_SOGUK * sog_v**2) / (1 - HEDEF_SOGUK)
    grid.append(dict(soguk=round(float(sog_v), 3),
                     gereken_sicak_1sira=round(float(np.sqrt(q)), 5) if q > 0 else None,
                     bilesik_mevcut_sicakla=round(bilesik(sic_lb, float(sog_v)), 5)))
R["04_izo_tablo"] = grid

# ---------- 3. SIFIR ANATOMISI ----------
sifir = []
for b in ("yaz25", "guz25", "kis26"):
    d = D[b]
    w = agirlik(d)
    z = d.tuketim.values <= 0
    sg = d.soguk_mu.values == 1
    r2 = (d.r.values) ** 2
    top = float(np.sum(w * r2))
    rec = dict(blok=b, sifir_pay=round(float(z.mean()), 4),
               sifir_pay_soguk=round(float(z[sg].mean()), 4),
               sifir_pay_sicak=round(float(z[~sg].mean()), 4),
               mse_payi_sifir=round(float(np.sum(w[z] * r2[z]) / top), 4),
               mse_payi_sifir_soguk=round(float(np.sum(w[z & sg] * r2[z & sg]) / top), 4),
               mse_payi_sifir_sicak=round(float(np.sum(w[z & ~sg] * r2[z & ~sg]) / top), 4))
    # kahin sifir dedektoru: sifir satirlarda tahmin=0 (log 0)
    p2 = d.p.values.copy(); p2[z] = 0.0
    rec["kahin_soguk"] = round(rmsle(d.y[sg], p2[sg]), 5)
    rec["kahin_sicak"] = round(rmsle(d.y[~sg], p2[~sg]), 5)
    rec["kahin_bilesik"] = round(bilesik(rec["kahin_sicak"], rec["kahin_soguk"]), 5)
    # yalniz soguk kahin
    p3 = d.p.values.copy(); p3[z & sg] = 0.0
    rec["kahin_yalniz_soguk_bilesik"] = round(
        bilesik(rmsle(d.y[~sg], d.p.values[~sg]), rmsle(d.y[sg], p3[sg])), 5)
    sifir.append(rec)
    print(rec)
R["05_sifir_anatomisi"] = sifir

kah_sic = float(np.mean([s["kahin_sicak"] for s in sifir]))
kah_sog = float(np.mean([s["kahin_soguk"] for s in sifir]))
kah_bil = bilesik(kah_sic, kah_sog)
R["06_kahin_tavani"] = dict(
    sicak=round(kah_sic, 5), soguk=round(kah_sog, 5), bilesik_cv=round(kah_bil, 5),
    lb_olcekli=round(kah_bil * olcek, 5),
    lb_kazanc=round(LB - kah_bil * olcek, 5),
    yalniz_soguk_kahin_lb=round(olcek * float(np.mean([s["kahin_yalniz_soguk_bilesik"] for s in sifir])), 5),
)
print("\nKAHIN SIFIR TAVANI:", R["06_kahin_tavani"])

with open(os.path.join(CIK, "p27_01.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print("\nyazildi p27_01.json")
