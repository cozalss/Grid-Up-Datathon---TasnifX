"""Dis veri ozellik aileleri -- m30/m33/m34 tezgahina SARMALAYICI.

m30_ozellik.kur()'un urettigi matrise `data/external/` kolonlarini EKLER.
Hedef degiskeninden turetilen tek sey trafo-duzeyi sicaklik EGIMI (G ailesi) ve
o da yalnizca KESIM ONCESI gecmisten hesaplanir (h_egim ile ayni sizinti profili).

SONUC (huber a=2.0 l2=20, 3 tohum ortalamasi, taban - aday):
    A sicaklik      +0.0152 (2025-11-30)  +0.0373 (2025-09-30)   <- ANA KAZANC
    C nem/yagis     +0.0109              +0.0261
    G trafo x sic.  +0.0066              +0.0210
    B gunes         +0.0001              +0.0156
    E turizm/su     +0.0018              +0.0098
    D statik ilce   -0.0007              -0.0034                 <- IKI KESIMDE DE KAYBETTI
    A+C+G+E         +0.0153              +0.0386  (tohum std ~0.002)
Aileler buyuk olcude ayni sinyali tasiyor: A tek basina toplamin neredeyse tamami.

Kod cagirimi (m34_supurme.kos ile ayni protokol):
    import m61_hava as h
    h.kos("2025-11-30", aileler=("A", "C", "G", "E"),
          objective="huber", alpha=2.0, lambda_l2=20.0)
Uretim matrisi icin:
    X = h.zenginlestir(m30_ozellik.kur(gec, hed, kesim, sicak), meta, eg, ("A","C","G","E"))
    # meta: hed'in ilce_key / il_key / tanim / tarih / ay kolonlari
    # eg  : h.trafo_egim(gec, h.ortam()["gun"], kesim)   (yalnizca G ailesi icin)

Komutlar:
    python m61_hava.py kapsam    # ilce_key esleme raporu
    python m61_hava.py aile      # aile aile olcum
    python m61_hava.py kombo     # kazanan ailelerin birlesimleri
    python m61_hava.py tohum     # 3 tohumlu kararlilik testi
    python m61_hava.py uretim    # gercek test matrisinde kolon/NaN denetimi
Hepsi m61_hava.json icine yazar.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, BURA)
DIS = os.path.join(KOK, "data", "external")
REF = os.path.join(KOK, "data", "reference")

from m30_ozellik import kur, yukle_ham  # noqa: E402
from m33_durust import VARSAYILAN, hizala  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

# ------------------------------------------------------------------ anahtarlar


def anahtarla(d):
    """lokasyon 'IL>BOLGE>ILCE' -> il_key / ilce_key (aksansiz, kucuk harf)."""
    p = d.lokasyon.str.split(">")
    d["il_key"] = p.str[0].str.strip().map(join_key)
    d["ilce_key"] = p.str[-1].str.strip().map(join_key)
    return d


# ------------------------------------------------------------------ gunluk panel

ISIL_PENCERE = (3, 7, 14)


def _oku(dosya, kolonlar):
    d = pd.read_parquet(os.path.join(DIS, dosya), columns=["ilce_key", "tarih", *kolonlar])
    d["ilce_key"] = d["ilce_key"].astype(object)
    d["tarih"] = pd.to_datetime(d["tarih"]).dt.normalize()
    return d.drop_duplicates(["ilce_key", "tarih"]).set_index(["ilce_key", "tarih"])


def gunluk_panel():
    """Tum gunluk dis kaynaklari (ilce_key, tarih) ekseninde birlestir."""
    parcalar = [
        _oku(
            "hava_gunluk.parquet",
            [
                "sicaklik_max",
                "sicaklik_min",
                "sicaklik_ort",
                "hissedilen_max",
                "yagis_toplam",
                "yagmur_toplam",
                "kar_toplam",
                "yagis_saati",
                "ruzgar_max",
                "firtina_max",
                "gunes_radyasyon",
                "asiri_sicak",
                "asiri_soguk",
            ],
        ),
        _oku(
            "nem_toprak_gunluk.parquet",
            [
                "nem_ort",
                "nem_min",
                "nem_max",
                "sis_saat",
                "ciy_ort",
                "ciy_max",
                "toprak_nem_ort",
                "vpd_ort",
                "vpd_max",
                "bulut_dusuk_ort",
                "et0_toplam",
            ],
        ),
        _oku("hava_saatlik_turev.parquet", ["basinc_ort", "basinc_dusus_3s", "ruzgar_8ms_saat"]),
        _oku("hava_kalitesi_gunluk.parquet", ["pm10_ort", "toz_ort", "toz_tasinim_saat"]),
    ]
    g = pd.read_parquet(
        os.path.join(DIS, "gunes_gunluk.parquet"),
        columns=[
            "anahtar",
            "tarih",
            "gunes_ghi_gunluk",
            "gunes_dni_gunluk",
            "gunes_dhi_gunluk",
            "gunes_ghi_tepe",
            "gun_uzunlugu_saat",
            "gunes_ogle_yuksekligi",
        ],
    )
    g["ilce_key"] = g["anahtar"].str.split("|").str[-1].astype(object)
    g["tarih"] = pd.to_datetime(g["tarih"]).dt.normalize()
    parcalar.append(
        g.drop(columns=["anahtar"])
        .drop_duplicates(["ilce_key", "tarih"])
        .set_index(["ilce_key", "tarih"])
    )
    h = pd.concat(parcalar, axis=1).reset_index()
    h = h[(h.tarih >= "2024-06-01") & (h.tarih <= "2026-08-31")].copy()

    # derece-gun
    for t in (18, 22, 24):
        h[f"cdd{t}"] = (h["sicaklik_ort"] - t).clip(lower=0.0)
    h["hdd18"] = (18.0 - h["sicaklik_ort"]).clip(lower=0.0)
    h["hdd15"] = (15.0 - h["sicaklik_ort"]).clip(lower=0.0)
    h["cdd22_his"] = (h["hissedilen_max"] - 22.0).clip(lower=0.0)
    h["gun_farki"] = h["sicaklik_max"] - h["sicaklik_min"]

    h = h.sort_values(["ilce_key", "tarih"])
    gr = h.groupby("ilce_key", observed=True)
    ISIL = ["sicaklik_ort", "sicaklik_max", "cdd18", "cdd22", "cdd24", "hdd18"]
    NEMLI = ["yagis_toplam", "nem_ort", "toprak_nem_ort", "vpd_ort"]
    GUNESLI = ["gunes_ghi_gunluk", "gunes_radyasyon"]
    for kol in ISIL + NEMLI + GUNESLI:
        for p in ISIL_PENCERE:
            h[f"{kol}_ort{p}"] = gr[kol].transform(
                lambda s, _p=p: s.rolling(_p, min_periods=1).mean()
            )
    # sicaklik ivmesi (bugun - son 7 gun ortalamasi): klima acilis sinyali
    h["sic_sapma7"] = h["sicaklik_ort"] - h["sicaklik_ort_ort7"]
    h["cdd22_sapma7"] = h["cdd22"] - h["cdd22_ort7"]
    return h.set_index(["ilce_key", "tarih"])


# ------------------------------------------------------------------ aileler

A_KOL = (
    ["sicaklik_ort", "sicaklik_max", "sicaklik_min", "hissedilen_max", "gun_farki"]
    + ["cdd18", "cdd22", "cdd24", "cdd22_his", "hdd18", "hdd15", "asiri_sicak", "asiri_soguk"]
    + [
        f"{k}_ort{p}"
        for k in ("sicaklik_ort", "sicaklik_max", "cdd18", "cdd22", "cdd24", "hdd18")
        for p in ISIL_PENCERE
    ]
    + ["sic_sapma7", "cdd22_sapma7"]
)
B_KOL = [
    "gunes_ghi_gunluk",
    "gunes_dni_gunluk",
    "gunes_dhi_gunluk",
    "gunes_ghi_tepe",
    "gun_uzunlugu_saat",
    "gunes_ogle_yuksekligi",
    "gunes_radyasyon",
] + [f"{k}_ort{p}" for k in ("gunes_ghi_gunluk", "gunes_radyasyon") for p in ISIL_PENCERE]
C_KOL = [
    "yagis_toplam",
    "yagmur_toplam",
    "kar_toplam",
    "yagis_saati",
    "nem_ort",
    "nem_min",
    "nem_max",
    "sis_saat",
    "ciy_ort",
    "ciy_max",
    "toprak_nem_ort",
    "vpd_ort",
    "vpd_max",
    "bulut_dusuk_ort",
    "et0_toplam",
    "ruzgar_max",
    "firtina_max",
    "ruzgar_8ms_saat",
    "basinc_ort",
    "basinc_dusus_3s",
    "pm10_ort",
    "toz_ort",
    "toz_tasinim_saat",
] + [
    f"{k}_ort{p}"
    for k in ("yagis_toplam", "nem_ort", "toprak_nem_ort", "vpd_ort")
    for p in ISIL_PENCERE
]

GUNLUK_AILE = {"A": A_KOL, "B": B_KOL, "C": C_KOL}


def statik_tablo(tr, te):
    """D ailesi: ilce duzeyinde zamanla degismeyen yapi."""
    a = pd.read_parquet(os.path.join(DIS, "arazi_ortusu_ilce.parquet")).drop(
        columns=["il_key", "ortu_piksel"]
    )
    o = pd.read_parquet(os.path.join(DIS, "osm_altyapi_ilce.parquet")).drop(columns=["il_key"])
    r = pd.read_parquet(
        os.path.join(REF, "ilceler_gdz_adm.parquet"), columns=["ilce_key", "nufus", "alan_km2"]
    ).drop_duplicates("ilce_key")
    for _d in (a, o, r):
        _d["ilce_key"] = _d["ilce_key"].astype(object)
    s = a.merge(o, on="ilce_key", how="outer").merge(r, on="ilce_key", how="outer")
    # trafo filosu yapisi (hedeften bagimsiz, testte de bilinir)
    filo = pd.concat(
        [tr[["tanim", "guc", "ilce_key"]], te[["tanim", "guc", "ilce_key"]]]
    ).drop_duplicates("tanim")
    g = filo.groupby("ilce_key", observed=True)["guc"]
    y = pd.DataFrame(
        {
            "ilce_trafo_sayisi": g.size().astype("float64"),
            "ilce_toplam_guc": g.sum(),
            "ilce_guc_medyan": g.median(),
        }
    ).reset_index()
    s = s.merge(y, on="ilce_key", how="left")
    s["ilce_key"] = s["ilce_key"].astype(object)
    s["ilce_nufus_yogunlugu"] = s.nufus / s.alan_km2
    s["trafo_basina_nufus"] = s.nufus / s.ilce_trafo_sayisi
    s["kva_basina_nufus"] = s.nufus / s.ilce_toplam_guc
    s["trafo_basina_hat"] = s.osm_toplam_hat_km / s.ilce_trafo_sayisi
    s["trafo_yogunlugu"] = s.ilce_trafo_sayisi / s.alan_km2
    s = s.set_index("ilce_key")
    s.columns = ["d_" + c for c in s.columns]
    return s.astype("float64")


def turizm_tablo():
    """E ailesi: (il_key, ay) mevsimsel turizm endeksi + (ilce_key, ay) su endeksi.

    2026 Nis-Tem turizm verisi YOK, o yuzden zaman serisi degil MEVSIMSEL PROFIL
    kullanilir (2023-2025 ortalamasi), boylece test doneminde de tanimlidir.
    """
    t = pd.read_parquet(os.path.join(DIS, "turizm_aylik_il.parquet"))
    t = t[(t.kapsam == "isletme_basit") & (t.yil.between(2023, 2025))].copy()
    t["il_key"] = t["il_key"].astype(object)
    t["ay"] = t["ay"].astype("int64")
    ay = t.groupby(["il_key", "ay"])[["gelis", "geceleme", "doluluk"]].mean()
    yil = t.groupby("il_key")[["gelis", "geceleme"]].mean()
    lv = np.asarray(ay.index.get_level_values("il_key"), dtype=object)
    ay["e_gelis_endeks"] = ay.gelis.to_numpy(float) / yil.gelis.reindex(lv).to_numpy(float)
    ay["e_geceleme_endeks"] = ay.geceleme.to_numpy(float) / yil.geceleme.reindex(lv).to_numpy(float)
    ay = ay.rename(
        columns={"gelis": "e_gelis", "geceleme": "e_geceleme", "doluluk": "e_doluluk"}
    ).astype("float64")

    gc = pd.read_parquet(os.path.join(DIS, "turizm_geceleme.parquet")).copy()
    gc["ilce_key"] = gc["ilce_key"].astype(object)
    gc = gc[gc.yil == gc.yil.max()].set_index("ilce_key")[["geceleme", "tesise_gelis"]]
    gc.columns = ["e_ilce_geceleme", "e_ilce_gelis"]

    su = pd.read_parquet(os.path.join(DIS, "izsu_su_profili.parquet")).copy()
    su["ilce_key"] = su["ilce_key"].astype(object)
    su["ay"] = su["ay"].astype("int64")
    su = su.set_index(["ilce_key", "ay"])
    su.columns = ["e_su_ay", "e_su_yaz_kis"]
    return ay, gc.astype("float64"), su.astype("float64")


# ------------------------------------------------------------------ G: trafo x sicaklik


def trafo_egim(gec, gun, kesim, pencere=365):
    """Trafonun KENDI gecmisinde log-tuketimin cdd22 / hdd18 duyarliligi.

    Sadece kesim oncesi satirlardan; h_egim ile ayni sizinti profili.
    """
    k = pd.Timestamp(kesim)
    s = gec[gec.tarih > k - pd.Timedelta(days=pencere)][["tanim", "ilce_key", "tarih", "ly"]]
    idx = pd.MultiIndex.from_arrays([s.ilce_key.to_numpy(), s.tarih.to_numpy()])
    d = gun[["cdd22", "hdd18"]].reindex(idx)
    s = s.assign(cdd=d.cdd22.to_numpy(), hdd=d.hdd18.to_numpy())
    out = {}
    for ad, kol in (("t_egim_cdd22", "cdd"), ("t_egim_hdd18", "hdd")):
        gg = s.groupby("tanim", observed=True)
        n = gg.size()
        xm = gg[kol].mean()
        ym = gg["ly"].mean()
        xy = gg.apply(lambda d, _k=kol: float((d[_k] * d["ly"]).mean()), include_groups=False)
        xx = gg[kol].apply(lambda v: float((v * v).mean()))
        var = xx - xm * xm
        egim = (xy - xm * ym) / var.replace(0.0, np.nan)
        egim[(n < 30) | (var < 1e-6)] = np.nan
        out[ad] = egim
    return pd.DataFrame(out)


# ------------------------------------------------------------------ tezgah

AY = [
    "2025-03-31",
    "2025-04-30",
    "2025-05-31",
    "2025-06-30",
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
    "2025-12-31",
]

_ORTAM = {}


def ortam():
    if not _ORTAM:
        tr, te = yukle_ham()
        anahtarla(tr)
        anahtarla(te)
        gun = gunluk_panel()
        _ORTAM.update(tr=tr, te=te, gun=gun, stat=statik_tablo(tr, te), tur=turizm_tablo(), baz={})
    return _ORTAM


def baz(kesim, tavan, ay=4):
    """Sizintisiz parca -- m33.parca ile AYNI mantik, ek olarak meta saklar."""
    o = ortam()
    a = (kesim, tavan)
    if a in o["baz"]:
        return o["baz"][a]
    tr = o["tr"]
    k = pd.Timestamp(kesim)
    son = k + pd.DateOffset(months=ay)
    if tavan is not None:
        son = min(son, pd.Timestamp(tavan))
    gec = tr[tr.tarih <= k]
    hed = tr[(tr.tarih > k) & (tr.tarih <= son)]
    if len(hed) == 0:
        o["baz"][a] = None
        return None
    X = kur(gec, hed, kesim, set(gec.tanim))
    meta = pd.DataFrame(
        {
            "ilce_key": hed.ilce_key.to_numpy(),
            "il_key": hed.il_key.to_numpy(),
            "tanim": hed.tanim.to_numpy(),
            "tarih": hed.tarih.to_numpy(),
        }
    )
    meta["ilce_key"] = meta.ilce_key.astype(object)
    meta["il_key"] = meta.il_key.astype(object)
    meta["ay"] = meta.tarih.dt.month.astype("int64")
    y = np.log1p(hed.tuketim.to_numpy())
    eg = trafo_egim(gec, o["gun"], kesim)
    o["baz"][a] = (X, y, meta, eg)
    return o["baz"][a]


def _gunluk_ekle(X, meta, kolonlar, gun):
    idx = pd.MultiIndex.from_arrays([meta.ilce_key.to_numpy(), meta.tarih.to_numpy()])
    d = gun[kolonlar].reindex(idx)
    for c in kolonlar:
        X[c] = d[c].to_numpy(dtype=np.float32)
    return X


def zenginlestir(X, meta, eg, aileler):
    """Aile kolonlarini TEK seferde ekle (parcalanma yok)."""
    o = ortam()
    yeni = {}
    kol = [c for a in aileler if a in GUNLUK_AILE for c in GUNLUK_AILE[a]]
    if "G" in aileler:
        kol = kol + [c for c in ("cdd22_ort7", "hdd18_ort7") if c not in kol]
    if kol:
        idx = pd.MultiIndex.from_arrays([meta.ilce_key.to_numpy(), meta.tarih.to_numpy()])
        d = o["gun"][kol].reindex(idx)
        for c in kol:
            yeni[c] = d[c].to_numpy(dtype=np.float32)
    if "D" in aileler:
        st = o["stat"].reindex(meta.ilce_key.to_numpy())
        for c in st.columns:
            yeni[c] = st[c].to_numpy(dtype=np.float32)
        gucv = X.guc.to_numpy(dtype=np.float32)
        yeni["d_guc_payi"] = gucv / yeni["d_ilce_toplam_guc"]
        yeni["d_guc_medyan_orani"] = gucv / yeni["d_ilce_guc_medyan"]
    if "E" in aileler:
        ta, gc, su = o["tur"]
        i2 = pd.MultiIndex.from_arrays([meta.il_key.to_numpy(), meta.ay.to_numpy()])
        d = ta.reindex(i2)
        for c in ta.columns:
            yeni[c] = d[c].to_numpy(dtype=np.float32)
        g = gc.reindex(meta.ilce_key.to_numpy())
        for c in gc.columns:
            yeni[c] = g[c].to_numpy(dtype=np.float32)
        i3 = pd.MultiIndex.from_arrays([meta.ilce_key.to_numpy(), meta.ay.to_numpy()])
        sw = su.reindex(i3)
        for c in su.columns:
            yeni[c] = sw[c].to_numpy(dtype=np.float32)
    if "G" in aileler:
        for c in eg.columns:
            yeni[c] = meta.tanim.map(eg[c]).to_numpy(dtype=np.float32)
        yeni["t_cdd_etki"] = yeni["t_egim_cdd22"] * yeni["cdd22_ort7"]
        yeni["t_hdd_etki"] = yeni["t_egim_hdd18"] * yeni["hdd18_ort7"]
    if not yeni:
        return X.copy()
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(yeni)], axis=1)


def kos(dog, aileler=(), drop=(), tur=None, **pk):
    """m34_supurme.kos ile AYNI protokol; ek olarak `aileler` kolonlarini ekler."""
    Xva, yva, mva, egv = baz(dog, None)
    Xva = zenginlestir(Xva, mva, egv, aileler)
    sog = Xva.soguk.values.astype(bool)
    Xs, ys = [], []
    for k in [x for x in AY if x < dog]:
        r = baz(k, dog)
        if r is None:
            continue
        Xs.append(zenginlestir(r[0], r[2], r[3], aileler))
        ys.append(r[1])
    Xtr = pd.concat(Xs, ignore_index=True)
    del Xs
    ytr = np.concatenate(ys)
    Xtr = Xtr.drop(columns=list(drop))
    Xva2 = Xva.drop(columns=list(drop))
    Xtr, Xva2 = hizala(Xtr, Xva2)
    p = dict(VARSAYILAN)
    p.update(pk)
    ds = lgb.Dataset(Xtr, ytr)
    if tur is None:
        m = lgb.train(
            p,
            ds,
            4000,
            valid_sets=[lgb.Dataset(Xva2, yva, reference=ds)],
            callbacks=[lgb.early_stopping(150, verbose=False)],
        )
        n = m.best_iteration
    else:
        m = lgb.train(p, ds, tur)
        n = tur
    pv = m.predict(Xva2, num_iteration=n)
    L = (pv - yva) ** 2
    return dict(
        n_ozellik=int(Xtr.shape[1]),
        tur=int(n),
        rmsle=float(np.sqrt(L.mean())),
        soguk=float(np.sqrt(L[sog].mean())),
        sicak=float(np.sqrt(L[~sog].mean())),
        karisik=float(np.sqrt(0.7784 * L[~sog].mean() + 0.2216 * L[sog].mean())),
    )


# ------------------------------------------------------------------ kapsam raporu


def kapsam():
    o = ortam()
    tr, te, gun = o["tr"], o["te"], o["gun"]
    rap = {}
    ilce_dis = set(gun.index.get_level_values(0))
    for ad, d in (("train", tr), ("test", te)):
        eksik_ilce = sorted(set(d.ilce_key) - ilce_dis)
        idx = pd.MultiIndex.from_arrays([d.ilce_key.to_numpy(), d.tarih.to_numpy()])
        s = gun["sicaklik_ort"].reindex(idx)
        eslesmez = int(s.isna().sum())
        rap[ad] = dict(
            satir=int(len(d)),
            ilce=int(d.ilce_key.nunique()),
            eksik_ilce=eksik_ilce,
            eslesmeyen_satir=eslesmez,
            esleme_orani=round(1 - eslesmez / len(d), 6),
            tarih=[str(d.tarih.min().date()), str(d.tarih.max().date())],
        )
    # test doneminde her ilce icin gun kapsami
    t0, t1 = te.tarih.min(), te.tarih.max()
    gerek = int((t1 - t0).days) + 1
    alt = gun.loc[(slice(None), slice(t0, t1)), :]
    say = alt.groupby(level=0).sicaklik_ort.apply(lambda v: int(v.notna().sum()))
    say = say.reindex(sorted(set(te.ilce_key)))
    rap["test_donemi"] = dict(
        gereken_gun=gerek,
        ilce_sayisi=int(len(say)),
        tam_kapsanan_ilce=int((say == gerek).sum()),
        eksik_gunlu_ilceler={k: int(v) for k, v in say[say != gerek].items()},
    )
    # aile bazinda test doneminde NaN orani
    aile_nan = {}
    idx = pd.MultiIndex.from_arrays([te.ilce_key.to_numpy(), te.tarih.to_numpy()])
    for a, kols in GUNLUK_AILE.items():
        d = gun[kols].reindex(idx)
        aile_nan[a] = {"kolon": len(kols), "nan_orani": round(float(d.isna().mean().mean()), 6)}
    s = o["stat"].reindex(sorted(set(te.ilce_key)))
    aile_nan["D"] = {"kolon": int(s.shape[1]), "nan_orani": round(float(s.isna().mean().mean()), 6)}
    ta, gc, su = o["tur"]
    i2 = pd.MultiIndex.from_arrays([te.il_key.to_numpy(), te.tarih.dt.month.to_numpy()])
    i3 = pd.MultiIndex.from_arrays([te.ilce_key.to_numpy(), te.tarih.dt.month.to_numpy()])
    aile_nan["E"] = {
        "turizm_il_nan": round(float(ta.reindex(i2).isna().mean().mean()), 6),
        "turizm_ilce_nan": round(float(gc.reindex(te.ilce_key.to_numpy()).isna().mean().mean()), 6),
        "su_nan": round(float(su.reindex(i3).isna().mean().mean()), 6),
    }
    rap["aile_test_nan"] = aile_nan
    return rap


# ------------------------------------------------------------------ uretim matrisi


def uretim_matris(aileler=("A",)):
    """GERCEK test matrisi: gec = tum train, hed = test.csv. Kolonlar olusuyor mu?"""
    o = ortam()
    tr, te = o["tr"], o["te"]
    kesim = tr.tarih.max()
    hed = te.copy()
    X = kur(tr, hed, kesim, set(tr.tanim))
    meta = pd.DataFrame(
        {
            "ilce_key": hed.ilce_key.to_numpy().astype(object),
            "il_key": hed.il_key.to_numpy().astype(object),
            "tanim": hed.tanim.to_numpy(),
            "tarih": hed.tarih.to_numpy(),
        }
    )
    meta["ay"] = meta.tarih.dt.month.astype("int64")
    eg = trafo_egim(tr, o["gun"], kesim)
    Xz = zenginlestir(X, meta, eg, aileler)
    yeni = [c for c in Xz.columns if c not in X.columns]
    nan = Xz[yeni].isna().mean()
    return dict(
        kesim=str(pd.Timestamp(kesim).date()),
        satir=int(len(Xz)),
        taban_kolon=int(X.shape[1]),
        eklenen_kolon=int(len(yeni)),
        nan_olan_kolonlar={k: round(float(v), 6) for k, v in nan[nan > 0].items()},
        tam_dolu_kolon=int((nan == 0).sum()),
    )


def tohum_testi(deney, tohumlar=(7, 17, 27)):
    sonuc = {}
    for dog in ["2025-11-30", "2025-09-30"]:
        print(f"\n########## DOGRULAMA {dog} ##########", flush=True)
        for ad, ail in deney:
            k = []
            for t in tohumlar:
                r = kos(
                    dog, aileler=ail, seed=t, bagging_seed=t, feature_fraction_seed=t, **TABAN_PK
                )
                k.append(r["karisik"])
            sonuc.setdefault(ad, {})[dog] = dict(
                karisik=[round(x, 4) for x in k],
                ort=round(float(np.mean(k)), 4),
                std=round(float(np.std(k)), 4),
            )
            print(
                f"  {ad:26s} karisik ort {np.mean(k):.4f} std {np.std(k):.4f}  {[round(x, 4) for x in k]}",
                flush=True,
            )
    yaz({"tohum": sonuc})
    return sonuc


# ------------------------------------------------------------------ ana

TABAN_PK = dict(objective="huber", alpha=2.0, lambda_l2=20.0)

DENEY = [
    ("TABAN (dis veri yok)", ()),
    ("A sicaklik", ("A",)),
    ("B gunes/gun uzunlugu", ("B",)),
    ("C nem/toprak/yagis", ("C",)),
    ("D statik ilce", ("D",)),
    ("E turizm/su", ("E",)),
    ("G trafo x sicaklik", ("G",)),
    ("F HEPSI", ("A", "B", "C", "D", "E", "G")),
]

YOL = os.path.join(BURA, "m61_hava.json")


def yaz(d):
    eski = {}
    if os.path.exists(YOL):
        try:
            eski = json.load(open(YOL, encoding="utf-8"))
        except Exception:
            eski = {}
    eski.update(d)
    json.dump(eski, open(YOL, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def olc(deney, etiket):
    sonuc = {}
    for dog in ["2025-11-30", "2025-09-30"]:
        print(f"\n########## DOGRULAMA {dog} ##########", flush=True)
        t0 = time.time()
        for ad, ail in deney:
            try:
                r = kos(dog, aileler=ail, **TABAN_PK)
                print(
                    f"  {ad:26s} oz {r['n_ozellik']:3d} tur {r['tur']:4d} RMSLE {r['rmsle']:.4f} "
                    f"soguk {r['soguk']:.4f} sicak {r['sicak']:.4f} | karisik {r['karisik']:.4f}"
                    f"  ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                sonuc.setdefault(ad, {})[dog] = r
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"  {ad:26s} HATA {e}", flush=True)
    yaz({etiket: sonuc})
    return sonuc


if __name__ == "__main__":
    komut = sys.argv[1] if len(sys.argv) > 1 else "aile"
    if komut == "kapsam":
        r = kapsam()
        print(json.dumps(r, indent=1, ensure_ascii=False))
        yaz({"kapsam": r})
    elif komut == "aile":
        s = olc(DENEY, "aile")
        print("\n===== OZET (taban - aile, + = kazanc) =====")
        tb = s.get("TABAN (dis veri yok)", {})
        for ad, v in s.items():
            d = {k: round(tb.get(k, {}).get("karisik", np.nan) - v[k]["karisik"], 4) for k in v}
            print(f"  {ad:26s} {d}")
    elif komut == "uretim":
        r = {a: uretim_matris(tuple(a.split("+"))) for a in ["A", "A+C+G+E"]}
        print(json.dumps(r, indent=1, ensure_ascii=False))
        yaz({"uretim_matris": r})
    elif komut == "tohum":
        s = tohum_testi(
            [
                ("TABAN (dis veri yok)", ()),
                ("A", ("A",)),
                ("A+C+G", ("A", "C", "G")),
                ("A+C+G+E", ("A", "C", "G", "E")),
            ]
        )
        print("\n===== TOHUM ORTALAMASI: taban - aday (+ = kazanc) =====")
        tb = s["TABAN (dis veri yok)"]
        for ad, v in s.items():
            print(f"  {ad:26s} " + str({k: round(tb[k]["ort"] - v[k]["ort"], 4) for k in v}))
    elif komut == "kombo":
        KOMBO = [
            ("TABAN (dis veri yok)", ()),
            ("A", ("A",)),
            ("A+C", ("A", "C")),
            ("A+G", ("A", "G")),
            ("A+C+G", ("A", "C", "G")),
            ("A+B+C+G", ("A", "B", "C", "G")),
            ("A+C+G+E", ("A", "C", "G", "E")),
            ("A+B+C+G+E", ("A", "B", "C", "G", "E")),
        ]
        s = olc(KOMBO, "kombo")
        print("\n===== OZET (taban - kombo, + = kazanc) =====")
        tb = s.get("TABAN (dis veri yok)", {})
        for ad, v in s.items():
            print(
                f"  {ad:26s} "
                + str(
                    {k: round(tb.get(k, {}).get("karisik", np.nan) - v[k]["karisik"], 4) for k in v}
                )
            )
    elif komut == "birlesik":
        ail = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("A",)
        olc([("TABAN (dis veri yok)", ()), ("+".join(ail), ail)], "birlesik_" + "".join(ail))
