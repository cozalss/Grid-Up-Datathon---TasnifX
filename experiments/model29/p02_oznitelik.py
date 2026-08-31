"""p02: SIFIRDAN temiz taban -- oznitelik kurucu.

Tasarim:
  Her BLOK'un bir KESIM tarihi var. Bir satirin trafo-gecmisi oznitelikleri
  YALNIZCA kesim tarihinden ONCEKI ham tuketimlerden hesaplanir.
    yaz25 : kesim 2025-04-01, gecmis 2025-01-01..03-31   (DEGERLENDIRME)
    guz25 : kesim 2025-08-01, gecmis 2025-01-01..07-31   (egitim)
    kis26 : kesim 2025-12-01, gecmis 2025-01-01..11-30   (egitim)
    test  : kesim 2026-04-01, gecmis 2025-01-01..2026-03-31

  Trafo-gecmisi oznitelikleri HAM CSV'den KENDIM hesapliyorum.
  Disgüdumlu (hava/takvim/cografya/nufus/ulusal) sutunlari mevcut
  parquet'ten SADECE BEYAZ LISTE ile aliyorum -- bunlar hedefe bagli degil.
"""
import numpy as np
import pandas as pd

K = "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = f"{K}/data/interim/deney"

# hedefe BAGLI OLMAYAN sutunlar (t_*, g_*, gp_*, p_*, ozet_*, soguk_mu HARIC)
DIS = [
    "sicaklik_ort", "sicaklik_max", "sicaklik_min", "hissedilen_max",
    "yagis_toplam", "ruzgar_max", "nem_ort", "vpd_ort", "toprak_nem_ort",
    "et0_toplam", "gunes_ghi_gunluk", "gun_uzunlugu_saat",
    "cdd18", "cdd22", "cdd24", "cdd22_ort7", "cdd24_ort7",
    "sicaklik_ort_ort7", "sicaklik_ort_ort14",
    "tatil_mi", "tatil_agirligi", "tatil_mesafe", "tatil_veya_haftasonu",
    "ramazan_ayi", "ramazan_ilerleme", "ramazan_bayrama_kalan",
    "agac_orani", "tarim_orani", "yerlesim_orani", "bitki_ortusu_orani",
    "osm_direk_yogunlugu", "osm_hat_yogunlugu",
    "ilce_trafo_sayisi", "ilce_toplam_guc", "ilce_guc_medyan",
    "nufus", "alan_km2", "ilce_nufus_yogunlugu", "trafo_basina_nufus",
    "kva_basina_nufus", "guc_yuzdelik", "guc_payi", "guc_medyan_orani",
    "ulusal_gunluk", "ulusal_tepe_orani", "ulusal_yillik_buyume",
]

KESIM = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
    "test": ("2026-04-01", "2026-07-31"),
}


def ham():
    tr = pd.read_csv(f"{K}/data/raw/train.csv", parse_dates=["tarih"])
    te = pd.read_csv(f"{K}/data/raw/test.csv", parse_dates=["tarih"])
    tr["tanim"] = tr.tanim.astype(str)
    te["tanim"] = te.tanim.astype(str)
    tr["y"] = np.log1p(tr.tuketim.clip(lower=0).astype(np.float64))
    return tr, te


def gecmis_ozet(h, kesim):
    """h: kesimden ONCEKI ham satirlar (tanim, tarih, y). Trafo bazli ozet."""
    kesim = pd.Timestamp(kesim)
    h = h.copy()
    h["ofs"] = (kesim - h.tarih).dt.days  # 1 = kesimden onceki gun
    g = h.groupby("tanim", observed=True)
    o = g.agg(
        h_n=("y", "size"),
        h_ort=("y", "mean"),
        h_std=("y", "std"),
        h_med=("y", "median"),
        h_min=("y", "min"),
        h_max=("y", "max"),
        h_ilk_ofs=("ofs", "max"),
        h_son_ofs=("ofs", "min"),
    )
    o["h_p10"] = g.y.quantile(0.10)
    o["h_p90"] = g.y.quantile(0.90)
    o["h_sifir"] = g.y.apply(lambda s: float((s <= 1e-9).mean()))
    o["h_kapsam"] = o.h_n / (o.h_ilk_ofs - o.h_son_ofs + 1).clip(lower=1)
    # son N gun ortalamalari
    for n in (7, 14, 30, 60, 90, 180):
        s = h.loc[h.ofs <= n].groupby("tanim", observed=True).y.mean()
        o[f"h_son{n}"] = s
        o[f"h_son{n}_fark"] = o[f"h_son{n}"] - o.h_ort
    # egim (log tuketim ~ -ofs), son 120 gun
    hh = h.loc[h.ofs <= 120]
    gg = hh.groupby("tanim", observed=True)
    x = -hh.ofs.astype(np.float64)
    hh = hh.assign(x=x)
    m = hh.groupby("tanim", observed=True)[["x", "y"]].mean()
    hh = hh.join(m, on="tanim", rsuffix="_m")
    hh["dx"] = hh.x - hh.x_m
    hh["dy"] = hh.y - hh.y_m
    hh["xy"] = hh.dx * hh.dy
    hh["xx"] = hh.dx * hh.dx
    sm = hh.groupby("tanim", observed=True)[["xy", "xx"]].sum()
    o["h_egim"] = (sm.xy / sm.xx.replace(0, np.nan)) * 30.0  # 30 gunde degisim
    # haftanin gunu profili: hg ortalamasi - genel ortalama
    h["hg"] = h.tarih.dt.dayofweek
    hp = h.groupby(["tanim", "hg"], observed=True).y.mean().unstack()
    hp = hp.sub(o.h_ort, axis=0)
    hp.columns = [f"h_hg{c}" for c in hp.columns]
    o = o.join(hp)
    # olu kuyruk: son 30 gunde hic kayit yok / hep sifir
    o["h_olu"] = ((o.h_son_ofs > 30) | (o.get("h_son30", np.nan).fillna(1) <= 1e-9)).astype(int)
    return o


def grup_onceligi(h, meta):
    """Sadece GECMIS'ten grup ortalamalari (soguk trafolar icin)."""
    m = h.groupby("tanim", observed=True).y.mean().rename("ty").reset_index()
    m = m.merge(meta, on="tanim", how="left")
    m["gk"] = np.log2(m.guc.clip(lower=1)).round().astype(int)
    r = {}
    r["gr_guc"] = m.groupby("guc").ty.mean()
    r["gr_gk"] = m.groupby("gk").ty.mean()
    r["gr_ilce"] = m.groupby("ilce").ty.mean()
    r["gr_ilce_gk"] = m.groupby(["ilce", "gk"]).ty.mean()
    r["gr_ilce_gk_n"] = m.groupby(["ilce", "gk"]).ty.size()
    return r


def blok_kur(satir, h, meta, kesim, grp):
    """satir: (tanim, guc, tarih, lokasyon[, y]) cerceve."""
    o = gecmis_ozet(h[["tanim", "tarih", "y"]], kesim)
    d = satir.merge(o, left_on="tanim", right_index=True, how="left")
    d["soguk"] = d.h_n.isna().astype(int)
    d["h_n"] = d.h_n.fillna(0)
    # takvim
    t = d.tarih
    d["tk_hg"] = t.dt.dayofweek
    d["tk_ay"] = t.dt.month
    d["tk_gun"] = t.dt.day
    d["tk_doy"] = t.dt.dayofyear
    d["tk_hs"] = (d.tk_hg >= 5).astype(int)
    for k in (1, 2, 3):
        d[f"sin{k}"] = np.sin(2 * np.pi * k * d.tk_doy / 365.25)
        d[f"cos{k}"] = np.cos(2 * np.pi * k * d.tk_doy / 365.25)
    d["ufuk"] = (t - pd.Timestamp(kesim)).dt.days + 1
    # guc
    d["log_guc"] = np.log(d.guc.clip(lower=1))
    # lokasyon
    sp = d.lokasyon.fillna(">").str.split(">", expand=True)
    d["il"] = sp[0]
    d["bolge"] = sp[1] if sp.shape[1] > 1 else ""
    d["ilce"] = (sp[2] if sp.shape[1] > 2 else pd.Series("", index=d.index)).fillna("YOK")
    d["gk"] = np.log2(d.guc.clip(lower=1)).round().astype(int)
    # grup oncelikleri
    d["gr_guc"] = d.guc.map(grp["gr_guc"])
    d["gr_gk"] = d.gk.map(grp["gr_gk"])
    d["gr_ilce"] = d.ilce.map(grp["gr_ilce"])
    ik = pd.MultiIndex.from_arrays([d.ilce, d.gk])
    d["gr_ilce_gk"] = grp["gr_ilce_gk"].reindex(ik).to_numpy()
    d["gr_ilce_gk_n"] = grp["gr_ilce_gk_n"].reindex(ik).to_numpy()
    # gecmis merkezli grup farki
    d["h_ort_gr"] = d.h_ort - d.gr_ilce_gk
    return d
