"""p03: rakip farki nereden geliyor -- yapisal kesif (veri tarafi)."""

import json

import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
tr = pd.read_csv(f"{K}/data/raw/train.csv", parse_dates=["tarih"])
te = pd.read_csv(f"{K}/data/raw/test.csv", parse_dates=["tarih"])
R = {}

# --- 1. panel yapisi ---
gun_tr = tr.tarih.nunique()
gun_te = te.tarih.nunique()
c = tr.groupby("tanim").size()
ct = te.groupby("tanim").size()
R["panel"] = {
    "egitim_gun": int(gun_tr),
    "test_gun": int(gun_te),
    "egitim_trafo": int(tr.tanim.nunique()),
    "test_trafo": int(te.tanim.nunique()),
    "egitim_tam_panel_orani": float((c == gun_tr).mean()),
    "test_tam_panel_orani": float((ct == gun_te).mean()),
    "egitim_gun_dagilim": c.describe().to_dict(),
    "test_gun_dagilim": ct.describe().to_dict(),
    "beklenen_tam": int(tr.tanim.nunique() * gun_tr),
    "gercek": int(len(tr)),
}

# --- 2. soguk / sicak satir payi ---
sicak = set(tr.tanim) & set(te.tanim)
te["soguk"] = ~te.tanim.isin(sicak)
R["soguk"] = {
    "soguk_trafo": int(te.loc[te.soguk, "tanim"].nunique()),
    "soguk_satir": int(te.soguk.sum()),
    "soguk_satir_orani": float(te.soguk.mean()),
    "sadece_egitimde_trafo": int(len(set(tr.tanim) - set(te.tanim))),
}

# --- 3. hiyerarsi ---
sp = tr.lokasyon.str.split(">", expand=True)
tr["il"], tr["bolge"], tr["ilce"] = sp[0], sp[1], sp[2]
spt = te.lokasyon.str.split(">", expand=True)
te["il"], te["bolge"], te["ilce"] = spt[0], spt[1], spt[2]
R["hiyerarsi"] = {
    "il": sorted(tr.il.unique().tolist()),
    "bolge_sayisi": int(tr.bolge.nunique()),
    "bolge": sorted(tr.bolge.unique().tolist()),
    "ilce_sayisi": int(tr.ilce.nunique()),
    "lokasyon_sayisi": int(tr.lokasyon.nunique()),
    "test_yeni_lokasyon": sorted(set(te.lokasyon) - set(tr.lokasyon)),
    "trafo_lokasyon_degisen": int((tr.groupby("tanim").lokasyon.nunique() > 1).sum()),
    "trafo_guc_degisen": int((tr.groupby("tanim").guc.nunique() > 1).sum()),
    "test_yeni_guc": sorted(int(x) for x in set(te.guc) - set(tr.guc)),
}

# --- 4. tanim yapisi (kimlik numarasi bilgi tasiyor mu) ---
trf = tr.drop_duplicates("tanim")[["tanim", "lokasyon", "guc"]]
tn = pd.to_numeric(tr.tanim, errors="coerce")
te_tn = pd.to_numeric(te.tanim, errors="coerce")
R["tanim"] = {
    "uzunluk_dagilim": tr.tanim.str.len().value_counts().to_dict(),
    "sayisal_olmayan_trafo": int(pd.to_numeric(trf.tanim, errors="coerce").isna().sum()),
    "sayisal_olmayan_ornek": trf.tanim[pd.to_numeric(trf.tanim, errors="coerce").isna()]
    .head(10)
    .tolist(),
    "min": int(tn.min()),
    "max": int(tn.max()),
    "test_min": int(te_tn.min()),
    "test_max": int(te_tn.max()),
    "on_ek2_sayisi": int(tr.tanim.str[:2].nunique()),
    "on_ek3_sayisi": int(tr.tanim.str[:3].nunique()),
    "on_ek4_sayisi": int(tr.tanim.str[:4].nunique()),
}
# on ek lokasyon ile ortusuyor mu?
for p in (2, 3, 4, 5):
    g = trf.assign(pre=trf.tanim.str[:p]).groupby("pre").lokasyon.nunique()
    R["tanim"][f"on_ek{p}_basina_lokasyon_ort"] = float(g.mean())

# --- 5. hedef dagilimi ve sifirlar ---
y = tr.tuketim.to_numpy()
ly = np.log1p(y)
kesikler = [0, 1e-9, 1, 10, 50, 100, 500, 1000, 5000, 1e5, 1e9]
etiket = [
    "=0",
    "(0,1]",
    "(1,10]",
    "(10,50]",
    "(50,100]",
    "(100,500]",
    "(500,1e3]",
    "(1e3,5e3]",
    "(5e3,1e5]",
    ">1e5",
]
kova = pd.cut(y, bins=kesikler, labels=etiket, include_lowest=True, right=True)
R["hedef"] = {
    "sifir_orani": float((y == 0).mean()),
    "kucuk_10_orani": float((y <= 10).mean()),
    "log1p_ort": float(ly.mean()),
    "log1p_std": float(ly.std()),
    "kova_pay": pd.Series(kova).value_counts(normalize=True).sort_index().to_dict(),
}

# --- 6. sifirlarin yapisi: trafo bazinda mi, tarih bazinda mi ---
tr["sifir"] = tr.tuketim == 0
ps = tr.groupby("tanim").sifir.mean()
R["sifir_yapisi"] = {
    "hic_sifiri_olmayan_trafo": int((ps == 0).sum()),
    "tamamen_sifir_trafo": int((ps == 1).sum()),
    "kismi_sifir_trafo": int(((ps > 0) & (ps < 1)).sum()),
    "sifir_orani_gun_bazinda_std": float(tr.groupby("tarih").sifir.mean().std()),
    "aylik_sifir_orani": tr.groupby(tr.tarih.dt.to_period("M"))
    .sifir.mean()
    .rename(lambda p: str(p))
    .to_dict(),
}

# --- 7. yaz25 blok tanimi ---
yaz = tr[(tr.tarih >= "2025-04-01") & (tr.tarih <= "2025-07-31")]
R["yaz25"] = {
    "satir": int(len(yaz)),
    "trafo": int(yaz.tanim.nunique()),
    "sifir_orani": float((yaz.tuketim == 0).mean()),
    "kucuk_10_orani": float((yaz.tuketim <= 10).mean()),
    "medyan": float(yaz.tuketim.median()),
}
# yaz25 ufkunda soguk olacak trafolar (2025-03-31 kesimine gore)
gec = tr[tr.tarih <= "2025-03-31"]
yaz_soguk = ~yaz.tanim.isin(set(gec.tanim))
R["yaz25"]["soguk_satir_orani"] = float(yaz_soguk.mean())
R["yaz25"]["soguk_trafo"] = int(yaz.loc[yaz_soguk.to_numpy(), "tanim"].nunique())

print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
with open(f"{K}/experiments/model29/p03_kesif.json", "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False, default=str)
