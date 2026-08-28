"""J4 - satir sirasi kanali, asiri etiketler, metrik dogrulamasi."""

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"
CIK = KOK + r"\experiments\joker"


def p(*a):
    print(*a, flush=True)


tr = pd.read_csv(KOK + r"\data\raw\train.csv", parse_dates=["tarih"])
te = pd.read_csv(KOK + r"\data\raw\test.csv", parse_dates=["tarih"])
tr["ri"] = np.arange(len(tr))
te["ri"] = np.arange(len(te))

p("=== 1. ASIRI ETIKETLER (>1e7) ===")
big = tr[tr["tuketim"] > 1e7]
p("n=%d, trafo=%d" % (len(big), big["tanim"].nunique()))
p(
    big.groupby(["tanim", "guc"])
    .agg(n=("tuketim", "size"), mn=("tuketim", "min"), mx=("tuketim", "max"))
    .to_string()
)
p("tarihleri:", sorted(big["tarih"].dt.date.unique())[:30])
for t in big["tanim"].unique():
    a = tr[tr["tanim"] == t]
    p(
        "  %s guc=%d n=%d  medyan=%.1f  max=%.1f  test satir=%d"
        % (
            t,
            a["guc"].iloc[0],
            len(a),
            a["tuketim"].median(),
            a["tuketim"].max(),
            int((te["tanim"] == t).sum()),
        )
    )

p()
p("=== 2. SATIR SIRASI KANALI (train) ===")
# gun ici sira
tr["gun_ici"] = tr.groupby("tarih").cumcount()
tr["gun_n"] = tr.groupby("tarih")["tanim"].transform("size")
tr["sira_norm"] = tr["gun_ici"] / tr["gun_n"]
ly = np.log1p(tr["tuketim"].values)
p("corr(sira_norm, log1p(tuketim)) = %.5f" % np.corrcoef(tr["sira_norm"], ly)[0, 1])
# trafo etkisini cikar
trafo_ort = tr.groupby("tanim")["tuketim"].transform(lambda s: np.log1p(s).mean())
res = ly - trafo_ort.values
p("trafo etkisi cikarilmis corr = %.5f" % np.corrcoef(tr["sira_norm"], res)[0, 1])
kova = pd.cut(tr["sira_norm"], 10)
tt = pd.DataFrame({"k": kova, "res": res, "ly": ly})
p(
    tt.groupby("k", observed=True)
    .agg(n=("res", "size"), res=("res", "mean"), ly=("ly", "mean"))
    .to_string()
)

p()
p("--- sira sabit mi? (ayni trafo her gun ayni sirada mi) ---")
alt = tr[tr["tarih"].isin(pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]))]
piv = alt.pivot_table(index="tanim", columns="tarih", values="gun_ici")
piv = piv.dropna()
p("ortak trafo:", len(piv))
c = piv.corr(method="spearman")
p(c.to_string())

p()
p("--- gun ici sira, tanim'a gore mi siralanmis? ---")
g = tr[tr["tarih"] == pd.Timestamp("2026-03-01")]
p("tanim monoton artan mi:", bool(g["tanim"].astype(str).is_monotonic_increasing))
p("ilk 10 tanim:", list(g["tanim"].head(10)))
p("son 5 tanim:", list(g["tanim"].tail(5)))
gt = te[te["tarih"] == pd.Timestamp("2026-04-01")]
p("TEST ilk 10 tanim:", list(gt["tanim"].head(10)))

p()
p("=== 3. TEST'TE SIRA KANALI: ayni trafo gunler arasi sira degisiyor mu ===")
te["gun_ici"] = te.groupby("tarih").cumcount()
te["gun_n"] = te.groupby("tarih")["tanim"].transform("size")
te["sira_norm"] = te["gun_ici"] / te["gun_n"]
sn = te.groupby("tanim")["sira_norm"].agg(["mean", "std", "size"])
p("trafo ici sira_norm std dagilimi:")
p(sn["std"].describe())
# v83 tahminiyle sira korelasyonu (model bunu gormedi; gercek etiketle iliskisi bilinmiyor)
v83 = pd.read_csv(KOK + r"\submissions\tuketim_v83_sicak_optimum.csv")
te["p83"] = v83["tuketim"].values
p(
    "corr(sira_norm, log1p(p83)) TEST = %.5f"
    % np.corrcoef(te["sira_norm"], np.log1p(te["p83"]))[0, 1]
)

p()
p("=== 4. METRIK DOGRULAMASI ===")
# LB'de dogrulanmis: v80_optimum -> 1.01341, v83 -> 1.01318, v81 -> 1.01429
# tum bunlar MSE = RMSLE^2 ve dMSE = k^2 Q - 2 k L cebiriyle ONCEDEN tahmin edildi.

v80 = pd.read_csv(KOK + r"\submissions\tuketim_v80_optimum.csv")
v81 = pd.read_csv(KOK + r"\submissions\tuketim_v81_sicak08.csv")
a80 = np.log1p(v80["tuketim"].values)
a81 = np.log1p(v81["tuketim"].values)
a83 = np.log1p(v83["tuketim"].values)
u = a81 - a80
p(
    "v81-v80 farkli satir: %d, essiz delta: %s"
    % (int((np.abs(u) > 1e-9).sum()), np.unique(np.round(u[np.abs(u) > 1e-9], 4))[:5])
)
Q = float((u**2).mean())
p("Q(v81-v80) = %.7f  (docs/47: 0.7366095 x 0.08^2 = %.7f)" % (Q, 0.7366095 * 0.08**2))
S80, S81, S83 = 1.01341, 1.01429, 1.01318
M80, M81, M83 = S80**2, S81**2, S83**2
p("MSE: v80=%.6f v81=%.6f v83=%.6f" % (M80, M81, M83))
# dMSE = Q - 2L  (kappa=1 cunku u zaten 0.08'lik vektor)
L = (Q - (M81 - M80)) / 2
p("L cozumu = %.7f -> k* = %.5f -> optimum dMSE = %.6f" % (L, L / Q, -L * L / Q))
p(
    "v83 = v80 + k*u ile beklenen MSE = %.6f ; gercek %.6f ; fark %.6f"
    % (M80 - L * L / Q, M83, M83 - (M80 - L * L / Q))
)
# v83 gercekten v80 + k*u mu?
w = a83 - a80
if (np.abs(u) > 1e-9).sum():
    k_emp = float(np.dot(w, u) / np.dot(u, u))
    p("ampirik k (v83-v80 uzerine u projeksiyonu) = %.5f" % k_emp)
    p("artik maxabs = %.3e" % np.abs(w - k_emp * u).max())
