"""K5 -- EPIAS PLANSIZ KESINTI paneli: kapsam denetimi + ozellik insasi.

Kaynak: data/external/epias/panel_ilce_gun.parquet  (SADE surum -- yalnizca
EPIAS'in gercekten yayimladigi gunler; kardesi panel_ilce_gun_tam.parquet
kapsanmayan gunleri SAHTE SIFIR ile doldurur ve sources.yml onu modele girdi
olmaktan men eder).

Eksen: (ilce_key, tarih). Trafo duzeyinde kesinti verisi YOK; ilce uzerinden
baglanir. Bu, hedeften turetilmemis, sebeke isletim kaynakli bir bilgi turudur.

Komut:
    python k5_kesinti_veri.py           # kapsam raporu -> k5_kesinti_veri.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, BURA)

from gridup.turkish import join_key  # noqa: E402

PANEL = os.path.join(KOK, "data", "external", "epias", "panel_ilce_gun.parquet")
HAM = ["kesinti_adet", "kesinti_dk", "etkilenen_abone"]
PENCERE = (3, 7)
# OLCULDU (bkz. kayit_kaymasi raporu): gunluk "kesintisi olan ilce orani"
# CIFT TEPELI -- tam kayit rejiminde 0,45-0,48, kayit deliklerinde 0,02-0,30.
# Kayma MEVSIMSEL DEGIL, KAYIT kaynakli: 2025 Mayis-Temmuz (testin mevsimsel
# ikizi) %65,6 sifir / 3,81 adet iken 2025 Agustos %5,2 / 12,33 -- yani ayni
# yaz icinde iki ayri rejim. Bu esigin altindaki gunlerde SEVIYE kolonlari
# NaN'a cekilir (sahte sifir GIRMEZ), bayrak kolonlari kalir.
DUSUK_ESIK = 0.30


def anahtarla(d):
    p = d.lokasyon.str.split(">")
    d["il_key"] = p.str[0].str.strip().map(join_key)
    d["ilce_key"] = p.str[-1].str.strip().map(join_key)
    return d


def ham_panel():
    d = pd.read_parquet(PANEL)
    d = d[d.kapsanan_gun == 1].copy()
    d["ilce_key"] = d["ilce_key"].astype(object)
    d["tarih"] = pd.to_datetime(d["gun"]).dt.normalize()
    return d.drop(columns=["gun", "kapsanan_gun"])


def kesinti_panel(filo_sayisi=None):
    """(ilce_key, tarih) ekseninde kesinti ozellikleri.

    Turevler yalnizca dis veriden; hedef degiskene HIC dokunmaz.
    Hareketli toplamlar TAKVIM izgarasi uzerinde alinir (kapsanmayan gun = NaN,
    sahte sifir DEGIL), sonra kapsanan gunlere geri yazilir.
    """
    d = ham_panel().sort_values(["ilce_key", "tarih"])
    # --- GUN duzeyi kayit butunlugu (96 ilcenin tamami uzerinden)
    gunluk = d.groupby("tarih").agg(
        ks_gun_toplam_dk=("kesinti_dk", "sum"),
        ks_gun_toplam_adet=("kesinti_adet", "sum"),
        ks_gun_toplam_abone=("etkilenen_abone", "sum"),
        ks_gun_aktif_oran=("kesinti_adet", lambda v: float((v > 0).mean())),
    )
    dusuk = gunluk.ks_gun_aktif_oran < DUSUK_ESIK

    # takvim izgarasi: kapsanmayan gunler NaN kalir
    izgara = pd.MultiIndex.from_product(
        [sorted(set(d.ilce_key)), pd.date_range(d.tarih.min(), d.tarih.max())],
        names=["ilce_key", "tarih"],
    )
    g = d.set_index(["ilce_key", "tarih"]).reindex(izgara)
    gun_ekseni = g.index.get_level_values(1)
    guvenilir = ~pd.Series(gun_ekseni).map(dusuk).fillna(True).astype(bool).to_numpy()

    f = pd.DataFrame(index=g.index)
    for c in HAM:
        v = g[c].to_numpy(dtype="float64")
        # DUSUK KAYIT gunlerinde seviye bilgisi YOK sayilir -- sahte sifir girmez
        f["ks_" + c] = np.where(guvenilir, v, np.nan)
    f["ks_log_dk"] = np.log1p(f.ks_kesinti_dk)
    f["ks_log_adet"] = np.log1p(f.ks_kesinti_adet)
    f["ks_log_abone"] = np.log1p(f.ks_etkilenen_abone)
    f["ks_var"] = (f.ks_kesinti_adet > 0).astype("float64").where(f.ks_kesinti_adet.notna())
    f["ks_ort_sure"] = f.ks_kesinti_dk / f.ks_kesinti_adet.replace(0.0, np.nan)
    f["ks_abone_basi_dk"] = f.ks_kesinti_dk / f.ks_etkilenen_abone.replace(0.0, np.nan)

    gr = f.groupby(level=0, observed=True)
    for kol in ("ks_log_dk", "ks_log_adet", "ks_log_abone", "ks_var"):
        for p in PENCERE:
            f[f"{kol}_ort{p}"] = gr[kol].transform(
                lambda s, _p=p: s.rolling(_p, min_periods=1).mean()
            )
    # ilcenin KENDI normaline gore z-skoru (tum tarihce uzerinden)
    for kol in ("ks_log_dk", "ks_log_abone"):
        m = gr[kol].transform("mean")
        s = gr[kol].transform("std")
        f[kol + "_z"] = (f[kol] - m) / s.replace(0.0, np.nan)
    # dun / yarin (mekanik etki ayni gun, ama kaydirma taraf bilgisi verir)
    for kol in ("ks_log_dk", "ks_var"):
        f[kol + "_dun"] = gr[kol].shift(1)
        f[kol + "_yarin"] = gr[kol].shift(-1)
    # --- KAYIT REJIMINDEN BAGIMSIZ turevler: ayni GUN icinde oransal olculur,
    # aylik kayit yogunlugu degisimi bunlarda sadelesir.
    for c in ("ks_gun_toplam_dk", "ks_gun_toplam_adet", "ks_gun_toplam_abone"):
        f[c] = np.log1p(pd.Series(gun_ekseni).map(gunluk[c]).to_numpy(dtype="float64"))
    f["ks_gun_aktif_oran"] = pd.Series(gun_ekseni).map(gunluk.ks_gun_aktif_oran).to_numpy()
    f["ks_dusuk_kayit"] = (~guvenilir).astype("float64")
    f["ks_pay_dk"] = f.ks_kesinti_dk / np.expm1(f.ks_gun_toplam_dk).replace(0.0, np.nan)
    f["ks_pay_adet"] = f.ks_kesinti_adet / np.expm1(f.ks_gun_toplam_adet).replace(0.0, np.nan)
    f["ks_pay_abone"] = f.ks_etkilenen_abone / np.expm1(f.ks_gun_toplam_abone).replace(0.0, np.nan)
    # gun ICINDE ilceler arasi z-skoru (rejim seviyesinden tamamen bagimsiz)
    gg = f.groupby(level=1, observed=True)["ks_log_dk"]
    f["ks_log_dk_gunz"] = (f.ks_log_dk - gg.transform("mean")) / gg.transform("std").replace(
        0.0, np.nan
    )
    if filo_sayisi is not None:
        n = pd.Series(f.index.get_level_values(0)).map(filo_sayisi).to_numpy(dtype="float64")
        f["ks_abone_trafo"] = f.ks_etkilenen_abone.to_numpy() / n
        f["ks_adet_trafo"] = f.ks_kesinti_adet.to_numpy() / n
    # kapsanmayan gunler: TUM kolonlar NaN (sahte sifir yok)
    kapsam = g[HAM[0]].notna().to_numpy()
    f.loc[~kapsam, :] = np.nan
    f["ks_kapsandi"] = kapsam.astype("float64")
    return f.astype("float32")


KOLONLAR = None


def kolonlar(f):
    return [c for c in f.columns]


def filo(tr, te):
    a = pd.concat([tr[["tanim", "ilce_key"]], te[["tanim", "ilce_key"]]]).drop_duplicates("tanim")
    return a.groupby("ilce_key").size().astype("float64")


def kayit_kaymasi(d, test_ilce):
    """MEVSIMSEL MI KAYIT MI? Testin mevsimsel ikizi 2025 Nis-Tem ile karsilastir.

    Yalnizca 47 test ilcesi uzerinden olculur (96'lik izgara kucuk ilceleri de
    icerdigi icin sifir oranini yukari cekiyor).
    """
    w = d[d.ilce_key.isin(test_ilce)].copy()
    w["ay"] = w.tarih.dt.to_period("M").astype(str)

    def blok(a, b):
        z = w[(w.tarih >= a) & (w.tarih <= b)]
        if not len(z):
            return dict(gun=0)
        return dict(
            gun=int(z.tarih.nunique()),
            sifir_orani=round(float((z.kesinti_adet == 0).mean()), 4),
            ort_adet=round(float(z.kesinti_adet.mean()), 2),
            ort_dk=round(float(z.kesinti_dk.mean()), 1),
        )

    ay = w[w.tarih >= "2025-05-01"].groupby("ay")
    gunluk = d.groupby("tarih").kesinti_adet.apply(lambda v: float((v > 0).mean()))
    gunluk = gunluk[gunluk.index >= "2025-05-01"]
    dusuk = gunluk < DUSUK_ESIK
    return {
        "SORU": "egitim %38,5 sifir vs test %9,2 -- mevsimsel mi kayit kaymasi mi?",
        "HUKUM": "KAYIT KAYMASI (b). Testin mevsimsel ikizi 2025 Nis-Tem ayni "
        "seviyede DEGIL, ama 2025 Agustos/Aralik TAM test seviyesinde. Yani "
        "fark mevsimle degil, EPIAS kayit yogunlugunun aylik rejimiyle geliyor.",
        "2025_nisan": "PANELDE HIC YOK (kapsam 2025-05-08'de basliyor)",
        "test_mevsimsel_ikizi_2025_May08_Tem31": blok("2025-05-08", "2025-07-31"),
        "2025_temmuz": blok("2025-07-01", "2025-07-31"),
        "2025_agustos": blok("2025-08-01", "2025-08-31"),
        "2025_aralik": blok("2025-12-01", "2025-12-31"),
        "2026_subat": blok("2026-02-01", "2026-02-28"),
        "TEST_2026_Nis_Tem": blok("2026-04-01", "2026-07-31"),
        "aylik_sifir_orani_47ilce": {
            k: round(float(v), 4)
            for k, v in ay.kesinti_adet.apply(lambda v: (v == 0).mean()).items()
        },
        "aylik_ort_adet_47ilce": {k: round(float(v), 2) for k, v in ay.kesinti_adet.mean().items()},
        "gunluk_aktif_ilce_orani": dict(
            aciklama="96 ilcenin kacinda o gun kesinti kaydi var -- CIFT TEPELI",
            tam_rejim_tipik=round(float(gunluk[gunluk >= DUSUK_ESIK].mean()), 3),
            delik_rejim_tipik=round(float(gunluk[gunluk < DUSUK_ESIK].mean()), 3),
            esik=DUSUK_ESIK,
            dusuk_gun_toplam=int(dusuk.sum()),
            dusuk_gun_egitim=int((dusuk & (dusuk.index <= "2026-03-31")).sum()),
            dusuk_gun_test=int(
                (dusuk & (dusuk.index >= "2026-04-01") & (dusuk.index <= "2026-07-31")).sum()
            ),
        ),
        "ALINAN_ONLEM": [
            "kapsanmayan gun -> TUM kesinti kolonlari NaN (sahte sifir GIRMEZ), "
            "ayrica ks_kapsandi bayragi",
            f"gunluk aktif-ilce orani < {DUSUK_ESIK} olan gunler DUSUK KAYIT sayilir; "
            "SEVIYE kolonlari NaN'a cekilir, ks_dusuk_kayit bayragi kalir",
            "egitim kesimleri 2025-07-31..2025-12-31 secildi -- hedef pencereleri "
            "kapsamin basladigi doneme dusuyor (2025-01..04 hic yok)",
            "rejimden BAGIMSIZ turevler eklendi: ilce ici z-skoru (ks_log_dk_z), "
            "GUN ICI ilceler arasi z-skoru (ks_log_dk_gunz), gunun toplamina oran "
            "(ks_pay_dk/adet/abone) -- aylik kayit yogunlugu bunlarda sadelesir",
        ],
    }


def kapsam_raporu():
    tr = pd.read_csv(
        os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    te = pd.read_csv(
        os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    anahtarla(tr)
    anahtarla(te)
    d = ham_panel()
    pk = set(d.ilce_key)
    rap = {
        "kaynak": "data/external/epias/panel_ilce_gun.parquet (SADE)",
        "satir": int(len(d)),
        "ilce": int(d.ilce_key.nunique()),
        "il": int(d.il_key.nunique()),
        "tarih": [str(d.tarih.min().date()), str(d.tarih.max().date())],
        "farkli_gun": int(d.tarih.nunique()),
    }
    for ad, x in (("train", tr), ("test", te)):
        idx = pd.MultiIndex.from_arrays([x.ilce_key.to_numpy(), x.tarih.to_numpy()])
        s = d.set_index(["ilce_key", "tarih"]).kesinti_adet.reindex(idx)
        rap[ad] = dict(
            satir=int(len(x)),
            ilce=int(x.ilce_key.nunique()),
            panelde_olmayan_ilce=sorted(set(x.ilce_key) - pk),
            tarih=[str(x.tarih.min().date()), str(x.tarih.max().date())],
            eslesen_satir=int(s.notna().sum()),
            esleme_orani=round(float(s.notna().mean()), 6),
        )
        t0, t1 = x.tarih.min(), x.tarih.max()
        gerek = int((t1 - t0).days) + 1
        w = d[(d.tarih >= t0) & (d.tarih <= t1)]
        say = w.groupby("ilce_key").size()
        eksik = sorted(set(pd.date_range(t0, t1)) - set(w.tarih))
        rap[ad]["pencere_kapsam"] = dict(
            gereken_gun=gerek,
            kapsanan_farkli_gun=int(w.tarih.nunique()),
            hic_kayit_olmayan_gun=len(eksik),
            ilk_10_eksik_gun=[str(z.date()) for z in eksik[:10]],
            ilce_basina_gun_min=int(say.min()),
            ilce_basina_gun_med=float(say.median()),
            tam_kapsanan_ilce=int((say == gerek).sum()),
            ilce_sayisi=int(len(say)),
        )
        w2 = w[w.ilce_key.isin(set(x.ilce_key))]
        rap[ad]["dagilim"] = {
            c: dict(
                sifir_orani=round(float((w2[c] == 0).mean()), 4),
                ort=round(float(w2[c].mean()), 2),
                medyan=float(w2[c].median()),
                p95=round(float(w2[c].quantile(0.95)), 1),
                maks=float(w2[c].max()),
            )
            for c in HAM
        }
    rap["kayit_kaymasi"] = kayit_kaymasi(d, set(te.ilce_key))
    f = kesinti_panel(filo(tr, te))
    idx = pd.MultiIndex.from_arrays([te.ilce_key.to_numpy(), te.tarih.to_numpy()])
    n = f.reindex(idx).isna().mean()
    rap["ozellik"] = dict(
        kolon=int(f.shape[1]),
        adlar=list(f.columns),
        test_nan_orani={k: round(float(v), 6) for k, v in n.items() if v > 0},
        test_tam_dolu_kolon=int((n == 0).sum()),
    )
    idx2 = pd.MultiIndex.from_arrays([tr.ilce_key.to_numpy(), tr.tarih.to_numpy()])
    n2 = f.reindex(idx2).isna().mean()
    rap["ozellik"]["train_nan_ortalama"] = round(float(n2.mean()), 6)
    return rap


if __name__ == "__main__":
    r = kapsam_raporu()
    print(json.dumps(r, indent=1, ensure_ascii=False))
    json.dump(
        r,
        open(os.path.join(BURA, "k5_kesinti_veri.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
