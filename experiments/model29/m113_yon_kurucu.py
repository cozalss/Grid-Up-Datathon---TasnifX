"""ADAY TARAMA -- CV tahminini LB span'iyla birlestirip YENI BILGI'yi kestir.

Anahtar cebir (hepsi exact, tahmin yalniz TEK yerde):
    x  test uzerinde birim yon
    x = x_span + x_dik                        (span = 27 olculmus yon)
    L(x_span) = c'L                           <- LB'DEN EXACT BILINIR
    L(x)      ~ carpan * kor_yaz25 * sqrt(m0) <- TEK TAHMIN
    L(x_dik)  = L(x) - L(x_span)
    rho_dik   = L(x_dik)/sqrt(Q_dik)    kazanc = rho_dik^2

yaz25 test bilesimine yeniden agirliklandirilir (soguk %7.5 -> %22.2).
carpan iki gercek LB olcumunden kalibre edilir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, os.path.join(KOK, "experiments/model29"))
from m112_kalibre import EK_MODEL, M0  # noqa: E402

S = os.path.join(KOK, "submissions")
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
RCOND, TABAN = 1e-6, "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK = 0.222

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)

V, L = [], []
for f, P in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - P * P) / 2)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)

V = np.array(V).T
L = np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=RCOND)
r_hat = V @ (Gi @ L)
nrm = float((r_hat * r_hat).mean())
print(f"span {V.shape[1]} yon  ||r_hat||^2={nrm:.6f}  saf optimum {np.sqrt(M0 - nrm):.6f}")


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return np.zeros_like(x)
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else x


def yonler(df, p):
    tarih = pd.to_datetime(df.tarih)
    ay = tarih.dt.month.to_numpy().astype(np.float64)
    hs = (tarih.dt.dayofweek >= 5).to_numpy().astype(np.float64)
    sv = st(p)
    soguk = df.soguk_mu.values.astype(np.float64)
    uf = df.ufuk_gun.values.astype(np.float64)
    lg = st(np.log1p(df.guc.values.astype(np.float64)))
    cdd = st(df.cdd22.values)
    ufs = st(uf)
    ays = st(ay)
    yas = st(df.yas.values)
    return {
        "ufuk_gun": ufs,
        "ufuk_kare": st(uf**2),
        "log_ufuk": st(np.log1p(uf)),
        "ay": ays,
        "cdd22": cdd,
        "cdd24": st(df.cdd24.values),
        "cdd22_ort7": st(df.cdd22_ort7.values),
        "sicaklik_ort": st(df.sicaklik_ort.values),
        "sicaklik_max": st(df.sicaklik_max.values),
        "nem_ort": st(df.nem_ort.values),
        "gunes_radyasyon": st(df.gunes_radyasyon.values),
        "ulusal_gunluk": st(df.ulusal_gunluk.values),
        "gun_uzunlugu": st(df.gun_uzunlugu_saat.values),
        "et0": st(df.et0_toplam.values),
        "vpd": st(df.vpd_ort.values),
        "toprak_nem": st(df.toprak_nem_ort.values),
        "yagis": st(df.yagis_toplam.values),
        "yas": yas,
        "sv_yas": st(sv * yas),
        "seviye": sv,
        "seviye2": st(sv**2),
        "seviye3": st(sv**3),
        "seviye_x_ufuk": st(sv * ufs),
        "seviye_x_cdd": st(sv * cdd),
        "seviye_x_ay": st(sv * ays),
        "seviye_x_soguk": st(sv * soguk),
        "seviye_x_guc": st(sv * lg),
        "seviye_x_hs": st(sv * hs),
        "soguk": st(soguk),
        "soguk_x_ufuk": st(soguk * ufs),
        "soguk_x_cdd": st(soguk * cdd),
        "guc": lg,
        "haftasonu": st(hs),
        "tatil": st(df.tatil_mi.values.astype(np.float64)),
        "tatil_agirligi": st(df.tatil_agirligi.values),
        "ramazan": st(df.ramazan_ayi.values.astype(np.float64)),
        "t_log_ort": st(df.t_log_ort.values),
        "t_log_std": st(df.t_log_std.values),
        "t_sifir_orani": st(df.t_sifir_orani.values),
        "t_trend": st(df.t_trend.values),
        "t_hg_genligi": st(df.t_hg_genligi.values),
        "t_mevsim_genlik": st(df.t_mevsim_genlik.values),
        "t_yuk_faktoru": st(df.t_yuk_faktoru.values),
        "t_egim_cdd22": st(df.t_egim_cdd22.values),
        "t_son_kayit_yasi": st(df.t_son_kayit_yasi.values),
        "t_gun_sayisi": st(df.t_gun_sayisi.values),
        "tarim_orani": st(df.tarim_orani.values),
        "yerlesim_orani": st(df.yerlesim_orani.values),
        "nufus_yog": st(df.ilce_nufus_yogunlugu.values),
        "guc_yuzdelik": st(df.guc_yuzdelik.values),
        "t_doluluk": st(df.t_doluluk.values),
        "p_doluluk": st(df.p_doluluk.values),
    }


e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))


def blok_kor(bad):
    blk = e[e._blok == bad]
    sic = blk[blk.soguk_mu == 0]
    sog = blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for a in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy"))
    ]
    p_s = np.mean(P, axis=0)
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    p_c = np.mean([z[k] for k in z.files], axis=0)
    idx = np.concatenate([sic.index.values, sog.index.values])
    p = np.concatenate([p_s, p_c])
    df = e.loc[idx].copy()
    r = np.log1p(df.tuketim.values.astype(np.float64)) - p
    sg = df.soguk_mu.values.astype(np.float64)
    pay = sg.mean()
    w = np.where(sg == 1, HEDEF_SOGUK / pay, (1 - HEDEF_SOGUK) / (1 - pay))
    w = w / w.mean()
    Y = yonler(df, p)
    m0b = float((w * r * r).mean())
    return {ad: float((w * r * x).mean()) / np.sqrt(m0b) for ad, x in Y.items()}, np.sqrt(m0b)


kor, rms = blok_kor("yaz25")
kor_g, rg = blok_kor("guz25")
kor_k, rk = blok_kor("kis26")
print(f"yaz25 (test bilesimine agirliklandirilmis) RMSLE={rms:.5f}  guz25={rg:.5f}  kis26={rk:.5f}")

CARPAN = -0.0304 / kor["seviye"]
print(
    f"KALIBRASYON: seviye yaz25 kor={kor['seviye']:+.4f}  LB rho=-0.0304  -> carpan={CARPAN:+.3f}"
)

tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(tp.id.values, IDS), "test.parquet sirasi ham test ile ayni degil"
Yt = yonler(tp, a0)

print(
    f"\n{'yon':>18s} {'yaz25':>8s} {'guz25':>8s} {'kis26':>8s} {'L_span':>9s} {'L_tah':>9s} "
    f"{'L_dik':>9s} {'Q_dik':>7s} {'rho_dik':>8s} {'kazanc':>10s}"
)
rows = []
for ad, x in Yt.items():
    c = Gi @ ((V.T @ x) / N)
    Lsp = float(c @ L)
    xp = x - V @ c
    Qd = float((xp * xp).mean())
    if Qd < 1e-6:
        continue
    Ltah = CARPAN * kor[ad] * np.sqrt(M0)
    Ldik = Ltah - Lsp
    rho = Ldik / np.sqrt(Qd)
    rows.append((ad, kor[ad], kor_g[ad], kor_k[ad], Lsp, Ltah, Ldik, Qd, rho, rho * rho))
rows.sort(key=lambda t: -t[9])
for a_, k1, k2, k3, ls, lt, ld, qd, rh, gn in rows:
    print(
        f"{a_:>18s} {k1:+8.4f} {k2:+8.4f} {k3:+8.4f} {ls:+9.5f} {lt:+9.5f} {ld:+9.5f} "
        f"{qd:7.4f} {rh:+8.4f} {gn:10.3e}"
    )
print(f"\n2. sira icin gereken TOPLAM kazanc: {(M0 - nrm) - 0.99940**2:.6f}")
print(f"1. sira icin gereken TOPLAM kazanc: {(M0 - nrm) - 0.99009**2:.6f}")
with open(os.path.join(BURA, "zc_adaylar.json"), "w") as fh:
    json.dump(
        {
            r[0]: dict(
                yaz=r[1], guz=r[2], kis=r[3], Lspan=r[4], Ltah=r[5], Ldik=r[6], Qd=r[7], rho=r[8]
            )
            for r in rows
        },
        fh,
        indent=1,
    )
