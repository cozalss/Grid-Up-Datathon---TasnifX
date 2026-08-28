"""J1 - yapisal denetim: metrik, gonderim dosyasi, uclar, test penceresi, zaman, mukerrer trafo."""

import json

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"
CIK = KOK + r"\experiments\joker"

out = {}


def p(*a):
    print(*a, flush=True)


tr = pd.read_csv(KOK + r"\data\raw\train.csv", parse_dates=["tarih"])
te = pd.read_csv(KOK + r"\data\raw\test.csv", parse_dates=["tarih"])
ss = pd.read_csv(KOK + r"\data\raw\sample_submission.csv")

p("=== 0. BOYUT ===")
p("train", tr.shape, "test", te.shape, "sample_sub", ss.shape)
p("ss kolonlari", list(ss.columns))
p("ss tuketim dolu mu:", ss["tuketim"].notna().sum())
out["boyut"] = {
    "train": list(tr.shape),
    "test": list(te.shape),
    "ss": list(ss.shape),
    "ss_kolon": list(ss.columns),
}

p()
p("=== 2. GONDERIM YAPISI ===")
p("ss id mukerrer:", int(ss["id"].duplicated().sum()))
p("test id mukerrer:", int(te["id"].duplicated().sum()))
p("id sirasi ss==test:", bool((ss["id"].values == te["id"].values).all()))
# id formati tanim_tarih mi?
yeniden = te["tanim"].astype(str) + "_" + te["tarih"].dt.strftime("%Y-%m-%d")
p("id == tanim_tarih:", bool((yeniden.values == te["id"].values).all()))
# (tanim,tarih) cifti mukerrer mi
p("test (tanim,tarih) mukerrer:", int(te.duplicated(["tanim", "tarih"]).sum()))
p("train (tanim,tarih) mukerrer:", int(tr.duplicated(["tanim", "tarih"]).sum()))
out["gonderim"] = {
    "ss_id_mukerrer": int(ss["id"].duplicated().sum()),
    "te_id_mukerrer": int(te["id"].duplicated().sum()),
    "sira_ayni": bool((ss["id"].values == te["id"].values).all()),
    "id_format_dogru": bool((yeniden.values == te["id"].values).all()),
    "te_cift_mukerrer": int(te.duplicated(["tanim", "tarih"]).sum()),
    "tr_cift_mukerrer": int(tr.duplicated(["tanim", "tarih"]).sum()),
}

p()
p("=== 6. TRAFO METADATA TUTARLILIGI ===")
g_tr = tr.groupby("tanim")["guc"].nunique()
g_te = te.groupby("tanim")["guc"].nunique()
p("train'de birden fazla guc'lu trafo:", int((g_tr > 1).sum()))
p("test'te birden fazla guc'lu trafo:", int((g_te > 1).sum()))
l_tr = tr.groupby("tanim")["lokasyon"].nunique()
l_te = te.groupby("tanim")["lokasyon"].nunique()
p("train'de birden fazla lokasyonlu trafo:", int((l_tr > 1).sum()))
p("test'te birden fazla lokasyonlu trafo:", int((l_te > 1).sum()))

tr_g = tr.groupby("tanim")["guc"].agg(lambda s: s.mode().iloc[0])
te_g = te.groupby("tanim")["guc"].agg(lambda s: s.mode().iloc[0])
ortak = te_g.index.intersection(tr_g.index)
farkli_guc = (te_g.loc[ortak] != tr_g.loc[ortak]).sum()
p(
    "ortak trafo:",
    len(ortak),
    "/ test trafo:",
    te["tanim"].nunique(),
    "/ train trafo:",
    tr["tanim"].nunique(),
)
p("ortak trafolardan guc'u DEGISEN:", int(farkli_guc))
tr_l = tr.groupby("tanim")["lokasyon"].agg(lambda s: s.mode().iloc[0])
te_l = te.groupby("tanim")["lokasyon"].agg(lambda s: s.mode().iloc[0])
farkli_lok = (te_l.loc[ortak] != tr_l.loc[ortak]).sum()
p("ortak trafolardan lokasyonu DEGISEN:", int(farkli_lok))
yeni_trafo = te_g.index.difference(tr_g.index)
p("test'te olup train'de OLMAYAN trafo:", len(yeni_trafo))
p("bu trafolarin test satir sayisi:", int(te["tanim"].isin(yeni_trafo).sum()))
out["metadata"] = {
    "tr_cok_guc": int((g_tr > 1).sum()),
    "te_cok_guc": int((g_te > 1).sum()),
    "tr_cok_lok": int((l_tr > 1).sum()),
    "te_cok_lok": int((l_te > 1).sum()),
    "guc_degisen": int(farkli_guc),
    "lok_degisen": int(farkli_lok),
    "yeni_trafo": int(len(yeni_trafo)),
    "yeni_trafo_satir": int(te["tanim"].isin(yeni_trafo).sum()),
    "te_trafo": int(te["tanim"].nunique()),
    "tr_trafo": int(tr["tanim"].nunique()),
}

p()
p("=== 3/EGITIM: HEDEF DAGILIMI ===")
y = tr["tuketim"].values
p("train tuketim: min %.4f  max %.4f  ort %.3f" % (y.min(), y.max(), y.mean()))
for q in [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.0]:
    p("  q%-9s %12.3f" % (q, np.quantile(y, q)))
p("negatif:", int((y < 0).sum()), " sifir:", int((y == 0).sum()), " <1:", int((y < 1).sum()))
p("en buyuk 20:", np.sort(y)[-20:])
out["train_hedef"] = {
    "min": float(y.min()),
    "max": float(y.max()),
    "q999": float(np.quantile(y, 0.999)),
    "q9999": float(np.quantile(y, 0.9999)),
    "sifir": int((y == 0).sum()),
    "negatif": int((y < 0).sum()),
    "alt1": int((y < 1).sum()),
}
# son 12 ay
son = tr[tr["tarih"] >= "2025-04-01"]
p(
    "2025-04-01 sonrasi max:",
    son["tuketim"].max(),
    " q0.9999:",
    np.quantile(son["tuketim"].values, 0.9999),
)
# nisan-temmuz penceresi (test ile ayni mevsim)
mev = tr[(tr["tarih"].dt.month >= 4) & (tr["tarih"].dt.month <= 7)]
p(
    "Nis-Tem penceresi n=%d max=%.1f q9999=%.1f"
    % (len(mev), mev["tuketim"].max(), np.quantile(mev["tuketim"].values, 0.9999))
)
out["train_hedef"]["nis_tem_max"] = float(mev["tuketim"].max())

p()
p("=== 4. TEST PENCERESI DOLULUK ===")
p(
    "test tarih:",
    te["tarih"].min().date(),
    "->",
    te["tarih"].max().date(),
    " gun:",
    te["tarih"].nunique(),
)
p(
    "train tarih:",
    tr["tarih"].min().date(),
    "->",
    tr["tarih"].max().date(),
    " gun:",
    tr["tarih"].nunique(),
)
n_tr_trafo = te["tanim"].nunique()
p(
    "doluluk: %d / (%d x %d) = %.4f"
    % (len(te), n_tr_trafo, te["tarih"].nunique(), len(te) / (n_tr_trafo * te["tarih"].nunique()))
)
gun_say = te.groupby("tanim")["tarih"].nunique()
p("trafo basina test gun sayisi dagilimi:")
p(gun_say.describe())
vc = gun_say.value_counts().sort_index()
p("en sik 15 deger:")
p(vc.sort_values(ascending=False).head(15))
p("tam 122 gun goren trafo:", int((gun_say == 122).sum()))
# gun basina trafo sayisi
trafo_say = te.groupby("tarih")["tanim"].nunique()
p(
    "gun basina trafo sayisi: min %d max %d ort %.1f"
    % (trafo_say.min(), trafo_say.max(), trafo_say.mean())
)
out["test_pencere"] = {
    "gun": int(te["tarih"].nunique()),
    "trafo": int(n_tr_trafo),
    "doluluk": float(len(te) / (n_tr_trafo * te["tarih"].nunique())),
    "tam122": int((gun_say == 122).sum()),
    "gun_say_med": float(gun_say.median()),
    "gunluk_trafo_min": int(trafo_say.min()),
    "gunluk_trafo_max": int(trafo_say.max()),
}
trafo_say.to_csv(CIK + r"\j01_gunluk_trafo_sayisi.csv")
gun_say.to_csv(CIK + r"\j01_trafo_gun_sayisi.csv")

with open(CIK + r"\j01_ozet.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
p()
p("YAZILDI", CIK + r"\j01_ozet.json")
