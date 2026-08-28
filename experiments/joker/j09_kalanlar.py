"""J9 - kalan serbest gozlem: v83 taban yapisi, lokasyon/guc uzayi, asiri trafo riski."""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"


def p(*a):
    print(*a, flush=True)


tr = pd.read_csv(KOK + r"\data\raw\train.csv", dtype={"tanim": str}, parse_dates=["tarih"])
te = pd.read_csv(KOK + r"\data\raw\test.csv", dtype={"tanim": str}, parse_dates=["tarih"])
v83 = pd.read_csv(KOK + r"\submissions\tuketim_v83_sicak_optimum.csv")
v93 = pd.read_csv(KOK + r"\submissions\tuketim_v93_gram_optimum.csv")

p("=== 1. v83 TABAN YAPISI ===")
x = v83["tuketim"].to_numpy()
vc = pd.Series(x).value_counts().head(8)
p("en sik gecen tahmin degerleri:")
p(vc.to_string())
p("min degerinde kac satir: %d" % int((x == x.min()).sum()))
p(
    "v93 icin: min=%.6f, min'de satir=%d, sifirda satir=%d"
    % (
        v93["tuketim"].min(),
        int((v93["tuketim"] == v93["tuketim"].min()).sum()),
        int((v93["tuketim"] == 0).sum()),
    )
)
# olu maske: v83 ile v89 farki
try:
    v89 = pd.read_csv(KOK + r"\submissions\tuketim_v89_genis_taban.csv")
    d = np.abs(v89["tuketim"].to_numpy() - x) / np.maximum(np.abs(x), 1e-9)
    p("v89 vs v83 bagil-tolerans maskesi: %d satir" % int((d >= 1e-9).sum()))
except Exception as e:
    p("v89 okunamadi:", e)

p()
p("=== 2. LOKASYON UZAYI ===")
for ad, d in [("train", tr), ("test", te)]:
    seg = d["lokasyon"].str.count(">") + 1
    p("%s: parca sayisi dagilimi %s" % (ad, seg.value_counts().to_dict()))
tr_l = set(tr["lokasyon"].unique())
te_l = set(te["lokasyon"].unique())
p(
    "train essiz lokasyon: %d, test: %d, test'te olup train'de olmayan: %d"
    % (len(tr_l), len(te_l), len(te_l - tr_l))
)
p("ornek yeni lokasyon:", list(te_l - tr_l)[:5])


def ilce(s):
    return s.str.split(">").str[-1].str.strip()


tr_i, te_i = set(ilce(tr["lokasyon"]).unique()), set(ilce(te["lokasyon"]).unique())
p(
    "ilce: train %d, test %d, test-only %d -> %s"
    % (len(tr_i), len(te_i), len(te_i - tr_i), sorted(te_i - tr_i)[:10])
)
yeni_ilce = te_i - tr_i
if yeni_ilce:
    m = ilce(te["lokasyon"]).isin(yeni_ilce)
    p("  yeni ilcedeki test satiri: %d" % int(m.sum()))

p()
p("=== 3. GUC UZAYI ===")
tg, teg = set(tr["guc"].unique()), set(te["guc"].unique())
p("train guc essiz %d, test %d, test-only %s" % (len(tg), len(teg), sorted(teg - tg)))
if teg - tg:
    p("  test-only guc satirlari: %d" % int(te["guc"].isin(teg - tg).sum()))
p("train-only guc:", sorted(tg - teg))

p()
p("=== 4. ASIRI ETIKET RISKI (guc>=10900 trafolar) ===")
dev = tr[tr["guc"] >= 10900]
p("train: %d satir, %d trafo" % (len(dev), dev["tanim"].nunique()))
devt = te[te["guc"] >= 10900]
p("test : %d satir, %d trafo" % (len(devt), devt["tanim"].nunique()))
oran = (dev["tuketim"] > 1e7).mean()
p("train'de bu trafolarda >1e7 satir orani: %.5f (%d satir)" % (oran, (dev["tuketim"] > 1e7).sum()))
p("test'te beklenen patlama satiri: %.1f" % (oran * len(devt)))
lp = np.log1p(v83["tuketim"].to_numpy())
mte = te["guc"] >= 10900
patlama_log = np.log1p(dev.loc[dev["tuketim"] > 1e7, "tuketim"]).mean()
p(
    "patlama ort log1p=%.3f ; v83 bu trafolarda ort log1p=%.3f ; fark=%.3f"
    % (patlama_log, lp[mte.to_numpy()].mean(), patlama_log - lp[mte.to_numpy()].mean())
)
n_bek = oran * len(devt)
p(
    "KORUNMASIZ MSE maliyeti ~ %.6f"
    % (n_bek * (patlama_log - lp[mte.to_numpy()].mean()) ** 2 / len(te))
)
# optimum hedge: her satira delta ekle
g = patlama_log - lp[mte.to_numpy()].mean()
q = oran
dopt = q * g
p(
    "optimum hedge delta=%.5f ; kazanc/satir=%.6f ; toplam dMSE=%.7f"
    % (
        dopt,
        q * g**2 - (q * (g - dopt) ** 2 + (1 - q) * dopt**2),
        -len(devt) * (q * g**2 - (q * (g - dopt) ** 2 + (1 - q) * dopt**2)) / len(te),
    )
)

p()
p("=== 5. TEST GUN BASINA TRAFO SAYISI: HAFTALIK DESEN VAR MI ===")
gs = te.groupby("tarih")["tanim"].nunique()
df = pd.DataFrame({"n": gs})
df["hg"] = df.index.dayofweek
p(df.groupby("hg")["n"].agg(["mean", "size"]).to_string())
p("en dusuk 8 gun:")
p(gs.nsmallest(8).to_string())
p("en yuksek 5 gun:")
p(gs.nlargest(5).to_string())
