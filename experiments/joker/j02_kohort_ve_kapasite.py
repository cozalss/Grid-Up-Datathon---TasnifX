"""J2 - kohort yapisi (yeni trafolar / 82 gun) + fiziksel kapasite tavani."""

import json

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"
CIK = KOK + r"\experiments\joker"


def p(*a):
    print(*a, flush=True)


tr = pd.read_csv(KOK + r"\data\raw\train.csv", parse_dates=["tarih"])
te = pd.read_csv(KOK + r"\data\raw\test.csv", parse_dates=["tarih"])
out = {}

p("=== A. FIZIKSEL KAPASITE TAVANI ===")
tr["tavan"] = tr["guc"] * 24.0
r = tr["tuketim"] / tr["tavan"]
p("guc essiz degerler:", sorted(tr["guc"].unique())[:30], "...n=", tr["guc"].nunique())
for q in [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0]:
    p("  oran q%-8s %.4f" % (q, np.quantile(r, q)))
for k in [1.0, 1.2, 2.0, 5.0, 10.0]:
    n = int((r > k).sum())
    p(
        "  tuketim > %.1f x guc x 24 : %d satir (%.4f%%)  %d trafo"
        % (k, n, 100 * n / len(tr), tr.loc[r > k, "tanim"].nunique())
    )
asiri = tr[r > 2.0]
p("asiri satirlarin tanim'lari (ilk 20):", asiri["tanim"].unique()[:20])
p("asiri trafo sayisi:", asiri["tanim"].nunique())
# bu trafolar test'te var mi
p(
    "asiri trafolardan test'te olan:",
    te["tanim"].isin(asiri["tanim"].unique()).sum(),
    "satir /",
    asiri["tanim"].isin(te["tanim"].unique()).sum(),
    "?",
)
ast = set(asiri["tanim"].unique())
p("asiri trafolarin test satiri:", int(te["tanim"].isin(ast).sum()))
out["kapasite"] = {
    "oran_q9999": float(np.quantile(r, 0.9999)),
    "oran_max": float(r.max()),
    "asiri_2x_satir": int((r > 2).sum()),
    "asiri_2x_trafo": int(asiri["tanim"].nunique()),
    "asiri_test_satir": int(te["tanim"].isin(ast).sum()),
}

p()
p("=== B. TEST KOHORT YAPISI ===")
gun_say = te.groupby("tanim")["tarih"].nunique()
ilk = te.groupby("tanim")["tarih"].min()
son = te.groupby("tanim")["tarih"].max()
p("ilk gorunum tarihi dagilimi (en sik 10):")
p(ilk.value_counts().head(10))
p("son gorunum tarihi dagilimi (en sik 10):")
p(son.value_counts().head(10))

g82 = gun_say[gun_say == 82].index
p("82 gunluk trafolar: n=%d" % len(g82))
p(
    "  ilk gorunum essiz:",
    te[te["tanim"].isin(g82)].groupby("tanim")["tarih"].min().value_counts().head(),
)
p(
    "  son gorunum essiz:",
    te[te["tanim"].isin(g82)].groupby("tanim")["tarih"].max().value_counts().head(),
)

tr_trafo = set(tr["tanim"].unique())
yeni = set(te["tanim"].unique()) - tr_trafo
p("yeni (train'de yok) trafo:", len(yeni))
p("  bunlarin kac tanesi 82 gunluk:", len(set(g82) & yeni))
p("  82 gunluklerin kac tanesi yeni:", len(set(g82) & yeni), "/", len(g82))
p("  yeni trafolarin gun sayisi dagilimi:")
p(gun_say.loc[list(yeni)].value_counts().head(10))
p("  eski trafolarin gun sayisi dagilimi:")
eski = list(set(te["tanim"].unique()) & tr_trafo)
p(gun_say.loc[eski].value_counts().head(10))
out["kohort"] = {
    "yeni_trafo": len(yeni),
    "g82": int(len(g82)),
    "yeni_ve_82": len(set(g82) & yeni),
}

p()
p("=== C. TRAIN'DE DOLULUK DESENI ===")
tr_gun = tr.groupby("tanim")["tarih"].nunique()
tr_ilk = tr.groupby("tanim")["tarih"].min()
tr_son = tr.groupby("tanim")["tarih"].max()
p("train trafo basina gun: ", tr_gun.describe())
p("train ilk gorunum en sik 10:")
p(tr_ilk.value_counts().head(10))
p("train son gorunum en sik 10:")
p(tr_son.value_counts().head(10))

p()
p("=== D. GUN SAYISI <-> TUKETIM SEVIYESI (train, etiketli) ===")
ort = tr.groupby("tanim")["tuketim"].apply(lambda s: np.log1p(s).mean())
df = pd.DataFrame({"gun": tr_gun, "logort": ort, "guc": tr.groupby("tanim")["guc"].first()})
p("korelasyon(gun sayisi, log ort tuketim): %.4f" % df["gun"].corr(df["logort"]))
p("spearman: %.4f" % df["gun"].corr(df["logort"], method="spearman"))
kova = pd.cut(df["gun"], [0, 60, 120, 200, 300, 400, 456])
p(
    df.groupby(kova, observed=True).agg(
        n=("logort", "size"), logort=("logort", "mean"), guc=("guc", "median")
    )
)

p()
p("=== E. SON GORUNUMU ERKEN OLAN TRAFOLAR (olu?) ===")
df["son"] = tr_son
df["ilk"] = tr_ilk
son_kova = pd.cut(
    df["son"],
    pd.to_datetime(
        ["2024-12-31", "2025-06-30", "2025-12-31", "2026-02-28", "2026-03-30", "2026-03-31"]
    ),
)
p(df.groupby(son_kova, observed=True).agg(n=("logort", "size"), logort=("logort", "mean")))
p("2026-03-31'e kadar yasayan trafo:", int((df["son"] == pd.Timestamp("2026-03-31")).sum()))
canli = df[df["son"] == pd.Timestamp("2026-03-31")].index
p(
    "test'teki eski trafolardan train sonuna kadar yasayan:",
    len(set(eski) & set(canli)),
    "/",
    len(eski),
)

with open(CIK + r"\j02_ozet.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
p("YAZILDI j02_ozet.json")
