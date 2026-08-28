"""J3 - tahmin dagiliminin uclari; ust/alt kirpma potansiyeli; asiri trafolar."""

import json

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"
CIK = KOK + r"\experiments\joker"


def p(*a):
    print(*a, flush=True)


tr = pd.read_csv(KOK + r"\data\raw\train.csv", parse_dates=["tarih"])
te = pd.read_csv(KOK + r"\data\raw\test.csv", parse_dates=["tarih"])
v83 = pd.read_csv(KOK + r"\submissions\tuketim_v83_sicak_optimum.csv")
v93 = pd.read_csv(KOK + r"\submissions\tuketim_v93_gram_optimum.csv")
assert (v83["id"].values == te["id"].values).all()
assert (v93["id"].values == te["id"].values).all()
te["p83"] = v83["tuketim"].values
te["p93"] = v93["tuketim"].values
out = {}

for ad in ["p83", "p93"]:
    x = te[ad].values
    p("=== %s ===" % ad)
    p("min %.6f max %.3f ort %.2f" % (x.min(), x.max(), x.mean()))
    for q in [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999, 1.0]:
        p("   q%-8s %14.4f" % (q, np.quantile(x, q)))
    p("   <1 kWh: %d   <0.1: %d   ==0: %d" % ((x < 1).sum(), (x < 0.1).sum(), (x == 0).sum()))
    p("   >50k: %d  >100k: %d  >200k: %d" % ((x > 5e4).sum(), (x > 1e5).sum(), (x > 2e5).sum()))
    out[ad] = {"min": float(x.min()), "max": float(x.max()), "q9999": float(np.quantile(x, 0.9999))}

p()
p("=== EN BUYUK 15 TAHMIN (v83) ===")
b = te.nlargest(15, "p83")[["tanim", "guc", "tarih", "p83", "p93"]]
tr_ozet = tr.groupby("tanim")["tuketim"].agg(["max", "mean", "count"])
b = b.join(tr_ozet, on="tanim", rsuffix="_tr")
p(b.to_string())

p()
p("=== EN KUCUK 15 TAHMIN (v83) ===")
s = te.nsmallest(15, "p83")[["tanim", "guc", "tarih", "p83", "p93"]]
s = s.join(tr_ozet, on="tanim")
p(s.to_string())

p()
p("=== FIZIKSEL TAVAN: tahmin / (guc*24) ===")
oran = te["p83"] / (te["guc"] * 24.0)
p("max oran %.3f ; >1 olan satir %d ; >2 %d" % (oran.max(), (oran > 1).sum(), (oran > 2).sum()))
p("egitimde ayni oranin q0.9999 = %.3f" % np.quantile(tr["tuketim"] / (tr["guc"] * 24.0), 0.9999))
out["fiziksel"] = {"tahmin_max_oran": float(oran.max()), "tahmin_oran_1ustu": int((oran > 1).sum())}

p()
p("=== ASIRI TRAFOLAR (train'de tuketim > 2*guc*24) ===")
r = tr["tuketim"] / (tr["guc"] * 24.0)
asiri = sorted(tr.loc[r > 2.0, "tanim"].unique())
p("n=%d" % len(asiri))
alt_tr = tr[tr["tanim"].isin(asiri)]
alt_te = te[te["tanim"].isin(asiri)]
p("train satir %d, test satir %d" % (len(alt_tr), len(alt_te)))
tab = []
for t in asiri:
    a = alt_tr[alt_tr["tanim"] == t]
    b2 = alt_te[alt_te["tanim"] == t]
    if len(b2) == 0:
        continue
    # train'in son 122 gunu
    sonu = a[a["tarih"] >= a["tarih"].max() - pd.Timedelta(days=121)]
    tab.append(
        {
            "tanim": t,
            "guc": a["guc"].iloc[0],
            "n_tr": len(a),
            "n_te": len(b2),
            "tr_logort": float(np.log1p(a["tuketim"]).mean()),
            "tr_son122_logort": float(np.log1p(sonu["tuketim"]).mean()),
            "tr_son_tarih": str(a["tarih"].max().date()),
            "p83_logort": float(np.log1p(b2["p83"]).mean()),
            "fark": float(np.log1p(b2["p83"]).mean() - np.log1p(sonu["tuketim"]).mean()),
        }
    )
tab = pd.DataFrame(tab)
p(tab.to_string())
p("toplam test satiri:", int(tab["n_te"].sum()))
p("agirlikli ort fark (log): %.4f" % np.average(tab["fark"], weights=tab["n_te"]))
p(
    "bu satirlarin MSE'ye potansiyel katkisi (fark^2 * n / 714688): %.6f"
    % (float((tab["fark"] ** 2 * tab["n_te"]).sum()) / 714688)
)
tab.to_csv(CIK + r"\j03_asiri_trafolar.csv", index=False)

p()
p("=== ALT UC: egitimde sifir/kucuk etiketler ===")
y = tr["tuketim"].values
p(
    "sifir %d (%.3f%%)  (0,1) %d  [1,10) %d"
    % (
        (y == 0).sum(),
        100 * (y == 0).mean(),
        ((y > 0) & (y < 1)).sum(),
        ((y >= 1) & (y < 10)).sum(),
    )
)
# sifirlarin trafo dagilimi
sf = tr[tr["tuketim"] == 0].groupby("tanim").size()
p("sifir iceren trafo sayisi: %d" % len(sf))
p(
    "sifir orani yuksek trafolar (>%%80): %d"
    % int((sf / tr.groupby("tanim").size().reindex(sf.index) > 0.8).sum())
)

with open(CIK + r"\j03_ozet.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
p("YAZILDI")
