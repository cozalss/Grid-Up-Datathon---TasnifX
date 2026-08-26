# ruff: noqa
"""H5 -- KONUM TOPLAMI / YUK DEVRI. Adim 1-3 (olay calismasi + regresyon).

SORU: Bir lokasyonda YENI trafo dogdugunda, ayni lokasyondaki MEVCUT
trafolarin yuku DUSUYOR mu? Ve yeni trafonun seviyesi "devredilen yuk" ile
iliskili mi?

KURAL 8 (AS-OF): on-pencere dogumdan ONCE biter [d0-28, d0-1].
KURAL 7: yaz25 ZORUNLU; guz25 kontrol blogu.

Cikti: reports/eksen_h5/adim13.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIK = KOK / "reports" / "eksen_h5"
CIK.mkdir(parents=True, exist_ok=True)
ARA = KOK / "data" / "interim" / "eksen_h5"
ARA.mkdir(parents=True, exist_ok=True)

PENCERE = 28
PARTI_ESIGI = 20  # ayni GUN >= 20 dogum -> TOPLU (idari yukleme)
MIN_GUN = 20  # her pencerede bir trafodan istenen asgari kayit gunu


def yukle() -> pd.DataFrame:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    return tr


def panel_kur(tr: pd.DataFrame):
    """Yogun [trafo x gun] tuketim matrisi (eksik = NaN)."""
    trafolar = np.sort(tr["tanim"].unique())
    gunler = pd.date_range(tr["tarih"].min(), tr["tarih"].max(), freq="D")
    ti = pd.Series(np.arange(len(trafolar)), index=trafolar)
    gi = pd.Series(np.arange(len(gunler)), index=gunler)
    M = np.full((len(trafolar), len(gunler)), np.nan, dtype="float64")
    r = ti.reindex(tr["tanim"]).to_numpy()
    c = gi.reindex(tr["tarih"]).to_numpy()
    M[r, c] = tr["tuketim"].to_numpy(dtype="float64")
    return M, trafolar, gunler, ti, gi


def main() -> int:
    cikti = []

    def yaz(s=""):
        print(s)
        cikti.append(s)

    tr = yukle()
    M, trafolar, gunler, ti, gi = panel_kur(tr)
    yaz(f"panel {M.shape}  dolu %{np.isfinite(M).mean() * 100:.2f}")

    # --- trafo meta ---
    g = tr.groupby("tanim", observed=True)
    meta = pd.DataFrame(
        {
            "ilk": g["tarih"].min(),
            "son": g["tarih"].max(),
            "guc": g["guc"].first(),
            "lok": g["lokasyon"].first(),
            "n": g.size(),
        }
    )
    lok_tekil = g["lokasyon"].nunique()
    yaz(f"lokasyonu DEGISEN trafo: {int((lok_tekil > 1).sum())}")
    meta = meta.loc[trafolar]
    meta["ilce"] = meta["lok"]
    meta["bolge"] = meta["lok"].str.rsplit(">", n=1).str[0]
    yaz(f"trafo {len(meta):,} | ilce {meta['ilce'].nunique()} | bolge {meta['bolge'].nunique()}")

    ilk_idx = gi.reindex(meta["ilk"]).to_numpy()
    meta["ilk_idx"] = ilk_idx

    # --- dogum olaylari ---
    TR_BAS = gunler[0]
    dogan = meta[meta["ilk"] > TR_BAS].copy()
    parti = dogan["ilk"].value_counts()
    toplu_tarih = set(parti[parti >= PARTI_ESIGI].index)
    dogan["toplu"] = dogan["ilk"].isin(toplu_tarih)
    yaz(
        f"\ntrain icinde dogan trafo {len(dogan):,}  "
        f"TOPLU {int(dogan['toplu'].sum()):,} / TEKIL {int((~dogan['toplu']).sum()):,}"
    )

    for seviye in ("ilce", "bolge"):
        yaz(
            f"\n{'=' * 78}\nLOKASYON SEVIYESI: {seviye}  "
            f"({meta[seviye].nunique()} birim, ort {len(meta) / meta[seviye].nunique():.1f} trafo)"
        )
        olcum(M, meta, dogan, gunler, gi, seviye, yaz)

    (CIK / "adim13.txt").write_text("\n".join(cikti), encoding="utf-8")
    return 0


def pencere_ort(M, satirlar, a, b):
    """[a,b) gun araliginda trafo basi gunluk ortalama + gecerli gun sayisi."""
    blok = M[np.ix_(satirlar, np.arange(a, b))]
    n = np.isfinite(blok).sum(axis=1)
    with np.errstate(invalid="ignore"):
        ort = np.nansum(blok, axis=1) / np.maximum(n, 1)
    return ort, n


def olcum(M, meta, dogan, gunler, gi, seviye, yaz):
    G = len(gunler)
    lok_kod = meta[seviye]
    # lokasyon -> trafo satir indeksleri
    idx = pd.Series(np.arange(len(meta)), index=meta.index)
    lok_uyeler = {k: idx.loc[v].to_numpy() for k, v in meta.groupby(seviye).groups.items()}

    ilk_idx = meta["ilk_idx"].to_numpy()
    guc = meta["guc"].to_numpy(dtype="float64")

    # --- KONTROL SERISI: hic dogmamis (ilk = train basi) trafolarin gunluk toplami
    temiz = np.where(ilk_idx == 0)[0]
    kontrol_gun = np.nansum(M[temiz], axis=0)
    kontrol_n = np.isfinite(M[temiz]).sum(axis=0)
    yaz(f"kontrol paneli: {len(temiz):,} 'her zaman vardi' trafo")

    # bolge bazli kontrol
    bolge_of = meta["bolge"].to_numpy()
    bolgeler = np.unique(bolge_of)
    bolge_kontrol = {}
    for b in bolgeler:
        s = np.where((bolge_of == b) & (ilk_idx == 0))[0]
        if len(s) >= 15:
            bolge_kontrol[b] = np.nansum(M[s], axis=0)

    kayitlar = []
    for (lok, d0), grup in dogan.groupby([seviye, "ilk"], observed=True):
        d = int(gi[d0])
        if d - PENCERE < 0 or d + 1 + PENCERE > G:
            continue
        uyeler = lok_uyeler.get(lok)
        if uyeler is None:
            continue
        # YERLESIK: on-pencere basindan once dogmus VE olay penceresinde
        # (d-28, d+28) dogum yasamamis
        yerlesik = uyeler[
            (ilk_idx[uyeler] <= d - PENCERE)
            & ~((ilk_idx[uyeler] > d - PENCERE - 1) & (ilk_idx[uyeler] < d + PENCERE))
        ]
        if len(yerlesik) == 0:
            continue
        on_o, on_n = pencere_ort(M, yerlesik, d - PENCERE, d)
        so_o, so_n = pencere_ort(M, yerlesik, d + 1, d + 1 + PENCERE)
        ok = (on_n >= MIN_GUN) & (so_n >= MIN_GUN)
        if ok.sum() == 0:
            continue
        on_top = float(on_o[ok].sum())
        so_top = float(so_o[ok].sum())
        if on_top <= 0:
            continue

        k_on = float(np.nanmean(kontrol_gun[d - PENCERE : d]))
        k_so = float(np.nanmean(kontrol_gun[d + 1 : d + 1 + PENCERE]))
        c_glob = np.log(k_so / k_on) if k_on > 0 and k_so > 0 else np.nan

        b = meta["bolge"].iloc[uyeler[0]]
        bk = bolge_kontrol.get(b)
        if bk is not None:
            b_on = float(np.nanmean(bk[d - PENCERE : d]))
            b_so = float(np.nanmean(bk[d + 1 : d + 1 + PENCERE]))
            c_bolge = np.log(b_so / b_on) if b_on > 0 and b_so > 0 else np.nan
        else:
            c_bolge = np.nan

        yeni = idx.loc[grup.index].to_numpy()
        y_o, y_n = pencere_ort(M, yeni, d, d + PENCERE)
        yeni_yuk = float(y_o[y_n >= MIN_GUN].sum())

        kayitlar.append(
            {
                "lok": lok,
                "d0": d0,
                "d": d,
                "n_yeni": len(yeni),
                "n_yerlesik": int(ok.sum()),
                "toplu": bool(grup["toplu"].iloc[0]),
                "guc_yeni": float(grup["guc"].sum()),
                "guc_yerlesik": float(guc[yerlesik][ok].sum()),
                "on_top": on_top,
                "so_top": so_top,
                "y": np.log(max(so_top, 1e-6) / on_top),
                "c_glob": c_glob,
                "c_bolge": c_bolge,
                "yeni_yuk": yeni_yuk,
                "n_yeni_gecerli": int((y_n >= MIN_GUN).sum()),
            }
        )

    E = pd.DataFrame(kayitlar)
    if len(E) == 0:
        yaz("OLAY YOK")
        return
    E["etki_g"] = E["y"] - E["c_glob"]
    E["etki_b"] = E["y"] - E["c_bolge"]
    E["bekl_so"] = E["on_top"] * np.exp(E["c_glob"].fillna(0.0))
    E["devir"] = E["bekl_so"] - E["so_top"]  # + = yerlesiklerden gitmis kWh/gun
    E["mevsim"] = np.where(
        (E["d0"] >= "2025-04-01") & (E["d0"] <= "2025-07-31"),
        "yaz25",
        np.where((E["d0"] >= "2025-08-01") & (E["d0"] <= "2025-11-30"), "guz25", "diger"),
    )
    E.to_parquet(ARA / f"olaylar_{seviye}.parquet")
    yaz(f"olay sayisi {len(E):,}  (lok x dogum-gunu)")

    def ozet(x, ad):
        x = np.asarray(x, dtype="float64")
        x = x[np.isfinite(x)]
        if len(x) < 3:
            return f"  {ad:<28} n={len(x):>5}  --"
        sh = x.std(ddof=1) / np.sqrt(len(x))
        return (
            f"  {ad:<28} n={len(x):>5}  ort {x.mean():+.4f}  sh {sh:.4f}"
            f"  t {x.mean() / sh:+6.2f}  medyan {np.median(x):+.4f}"
        )

    yaz("\n--- (2) YERLESIK YUK DEGISIMI (DiD, log oran) ---")
    for ad, alt in (
        ("TUMU", E),
        ("TEKIL", E[~E["toplu"]]),
        ("TOPLU", E[E["toplu"]]),
        ("yaz25", E[E["mevsim"] == "yaz25"]),
        ("yaz25 TEKIL", E[(E["mevsim"] == "yaz25") & ~E["toplu"]]),
        ("guz25", E[E["mevsim"] == "guz25"]),
        ("guz25 TEKIL", E[(E["mevsim"] == "guz25") & ~E["toplu"]]),
    ):
        yaz(ozet(alt["etki_g"], f"{ad} (global kontrol)"))
        yaz(ozet(alt["etki_b"], f"{ad} (bolge kontrol)"))

    yaz("\n--- PLASEBO: dogum olmayan (lok, gun) ciftleri ---")
    plasebo(M, meta, gunler, gi, seviye, E, kontrol_gun, yaz, idx, lok_uyeler, ilk_idx)

    yaz("\n--- (3) REGRESYON: yeni_yuk ~ devir ---")
    for ad, alt in (
        ("TUMU", E),
        ("TEKIL", E[~E["toplu"]]),
        ("yaz25", E[E["mevsim"] == "yaz25"]),
        ("yaz25 TEKIL", E[(E["mevsim"] == "yaz25") & ~E["toplu"]]),
        ("guz25", E[E["mevsim"] == "guz25"]),
    ):
        a = alt[
            np.isfinite(alt["devir"]) & np.isfinite(alt["yeni_yuk"]) & (alt["n_yeni_gecerli"] > 0)
        ]
        if len(a) < 10:
            yaz(f"  {ad:<14} n={len(a)} -- yetersiz")
            continue
        x = a["devir"].to_numpy()
        yv = a["yeni_yuk"].to_numpy()
        r = np.corrcoef(x, yv)[0, 1]
        egim = np.polyfit(x, yv, 1)
        # log-log
        m2 = (x > 0) & (yv > 0)
        r_log = np.corrcoef(np.log(x[m2]), np.log(yv[m2]))[0, 1] if m2.sum() > 10 else np.nan
        yaz(
            f"  {ad:<14} n={len(a):>5}  R2={r * r:.4f}  r={r:+.3f}  egim={egim[0]:+.4f}"
            f"  | log-log n={int(m2.sum())} R2={r_log * r_log if np.isfinite(r_log) else float('nan'):.4f}"
        )
        # kapasite payi ile karsilastirma (referans model)
        pay = a["guc_yeni"] / (a["guc_yeni"] + a["guc_yerlesik"])
        tahmin_pay = pay * (a["on_top"] + a["yeni_yuk"])
        rp = np.corrcoef(tahmin_pay, yv)[0, 1]
        yaz(f"  {'':<14}   referans 'kapasite payi x lok toplam': R2={rp * rp:.4f}")


def plasebo(M, meta, gunler, gi, seviye, E, kontrol_gun, yaz, idx, lok_uyeler, ilk_idx):
    """Ayni lokasyonlarda, dogum OLMAYAN tarihlerde ayni istatistik."""
    rng = np.random.default_rng(1000)
    G = len(gunler)
    dogum_gunleri = {}
    for lok, d in zip(E["lok"], E["d"]):
        dogum_gunleri.setdefault(lok, []).append(d)
    vals = []
    for lok, gunlist in dogum_gunleri.items():
        uyeler = lok_uyeler[lok]
        yasak = set()
        for d in gunlist:
            yasak.update(range(d - 2 * PENCERE, d + 2 * PENCERE))
        aday = [d for d in range(PENCERE, G - PENCERE - 1) if d not in yasak]
        if len(aday) < 3:
            continue
        for d in rng.choice(aday, size=min(3, len(aday)), replace=False):
            d = int(d)
            yerlesik = uyeler[ilk_idx[uyeler] <= d - PENCERE]
            if len(yerlesik) == 0:
                continue
            on_o, on_n = pencere_ort(M, yerlesik, d - PENCERE, d)
            so_o, so_n = pencere_ort(M, yerlesik, d + 1, d + 1 + PENCERE)
            ok = (on_n >= MIN_GUN) & (so_n >= MIN_GUN)
            if ok.sum() == 0:
                continue
            on_top = float(on_o[ok].sum())
            so_top = float(so_o[ok].sum())
            if on_top <= 0:
                continue
            k_on = float(np.nanmean(kontrol_gun[d - PENCERE : d]))
            k_so = float(np.nanmean(kontrol_gun[d + 1 : d + 1 + PENCERE]))
            if k_on <= 0 or k_so <= 0:
                continue
            vals.append(np.log(so_top / on_top) - np.log(k_so / k_on))
    v = np.asarray(vals)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        yaz("  plasebo yetersiz")
        return
    sh = v.std(ddof=1) / np.sqrt(len(v))
    yaz(f"  PLASEBO n={len(v):,}  ort {v.mean():+.4f}  sh {sh:.4f}  t {v.mean() / sh:+6.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
