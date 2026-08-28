"""ADAY AILESI K -- AGAC OLMAYAN model siniflari.

Uretim hattinin tamami GBM (CatBoost/LGBM/XGBoost) + kucuk bir sinir agi.
GBM'in yapisal olarak ifade EDEMEDIGI uc sey var:

    1. 364 gunluk gecikme (gecen yilin AYNI GUNU) -- uretimde yalnizca
       ``t_gy_log_ort`` yani gecen yilin SEVIYESI var, GUN SEKLI yok.
    2. Trafo bazli TREND EKSTRAPOLASYONU -- agac bir yaprakta sabit tahmin
       eder, 122 gun ileri egim tasiyamaz.
    3. Cok seviyeli KISMI HAVUZLAMA -- agac bolme yapar, buzme yapmaz;
       seyrek hucrede varyansi kendi kendine kismaz.

Dort aday:
    K1  mevsimsel naif (lag-364 gun sekli + surukleme duzeltmesi)
    K2  ETS/Theta (SES seviye + sonumlu trend + haftalik profil + kuresel
        sicaklik tepkisi)
    K3  hiyerarsik kismi havuzlama  il > ilce > kVA kovasi > ilce x kVA > trafo
    K4  komsuluk (kNN): ayni ilce + yakin kVA + profil benzerligi, komsularin
        GECEN YILKI ayni takvim gunundeki sapmasi

TEMEL KURAL: her aday YALNIZ kesim tarihinden ONCEKI veriyi gorur. Bu, uretimin
"blok disi ozet" duzeninden DAHA SIKI bir kuraldir (uretim yaz25/guz25 icin
gelecegi de goruyor); bu yuzden yaz25 blogunda bu adaylar yapisal olarak
dezavantajlidir -- kapi buna gore okunmalidir.

Kaggle'a hicbir sey gondermez.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import ortak

KESIM = {
    "yaz25": "2025-04-01",
    "guz25": "2025-08-01",
    "kis26": "2025-12-01",
    "TEST": "2026-04-01",
}


# ----------------------------------------------------------------------------- panel
class Panel:
    """Yogun (trafo x gun) log1p matrisi ve yardimcilari."""

    def __init__(self) -> None:
        ham = ortak.ham_train()
        self.tanim = np.sort(ham["tanim"].unique())
        self.gun = pd.date_range("2025-01-01", "2026-03-31", freq="D")
        ti = pd.Index(self.tanim)
        gi = pd.Index(self.gun)
        M = np.full((len(ti), len(gi)), np.nan, dtype="float32")
        M[ti.get_indexer(ham["tanim"]), gi.get_indexer(ham["tarih"])] = ham["lg"].to_numpy(
            "float32"
        )
        self.M = M
        self.ti, self.gi = ti, gi
        self.guc = ham.groupby("tanim")["guc"].max().reindex(self.tanim).to_numpy("float64")
        self.lgc = np.log1p(self.guc)
        # trafo -> ilce/il/kova  (egitim + test cerceveleerinden)
        e = ortak.egitim(["tanim", "ilce_key", "il_key", "g_guc_kova"]).drop_duplicates("tanim")
        t = ortak.test(["tanim", "ilce_key", "il_key", "g_guc_kova"]).drop_duplicates("tanim")
        harita = pd.concat([e, t], ignore_index=True)
        harita["tanim"] = harita["tanim"].astype(str)
        harita = harita.drop_duplicates("tanim").set_index("tanim")
        self.harita = harita
        self.kuresel_gun = self._kuresel_gun_serisi()

    def _kuresel_gun_serisi(self) -> pd.DataFrame:
        """Gunluk kuresel hava ozeti (butun ilcelerin ortalamasi)."""
        h = pd.read_parquet(
            ortak.KOK / "data/external/hava_gunluk.parquet",
            columns=["ilce_key", "tarih", "sicaklik_ort", "sicaklik_max"],
        )
        ilce = set(self.harita["ilce_key"].dropna().unique())
        h = h[h["ilce_key"].isin(ilce)]
        g = h.groupby("tarih")[["sicaklik_ort", "sicaklik_max"]].mean()
        g["cdd22"] = np.maximum(g["sicaklik_ort"] - 22.0, 0.0)
        g["hdd18"] = np.maximum(18.0 - g["sicaklik_ort"], 0.0)
        return g

    # --- gun etkisi modeli
    def gun_etkisi(self, kesim: pd.Timestamp, hedef_gun: pd.DatetimeIndex) -> np.ndarray:
        """Kuresel gun etkisini kesim ONCESI veriden ogrenip hedefe tasir.

        Model: gunluk ortalama (lg - trafo seviyesi) ~ hafta gunu + cdd22 + hdd18
        + sicaklik + sicaklik^2. Ridge. Yaz tahmini kis verisinden EKSTRAPOLASYON
        oldugu icin risklidir; kapi bunu olcer.
        """
        gec = self.gi < kesim
        Mh = self.M[:, gec]
        seviye = np.nanmean(Mh, axis=1)
        sap = Mh - seviye[:, None]
        with np.errstate(invalid="ignore"):
            gunluk = np.nanmean(sap, axis=0)
        gh = self.gi[gec]
        ok = np.isfinite(gunluk)

        def tasarim(idx: pd.DatetimeIndex) -> np.ndarray:
            w = self.kuresel_gun.reindex(idx)
            s = w["sicaklik_ort"].to_numpy("float64")
            X = [np.ones(len(idx)), s, s * s, w["cdd22"].to_numpy(), w["hdd18"].to_numpy()]
            for k in range(6):
                X.append((idx.dayofweek.to_numpy() == k).astype("float64"))
            return np.column_stack(X)

        A = tasarim(gh)[ok]
        yv = gunluk[ok]
        lam = 1e-3 * len(yv)
        R = np.eye(A.shape[1]) * lam
        R[0, 0] = 0.0
        w = np.linalg.solve(A.T @ A + R, A.T @ yv)
        return tasarim(hedef_gun) @ w


_PANEL: Panel | None = None


def panel() -> Panel:
    global _PANEL
    if _PANEL is None:
        _PANEL = Panel()
    return _PANEL


# ----------------------------------------------------------------------------- hedef cerceveler
def hedefler() -> dict[str, pd.DataFrame]:
    e = ortak.egitim(["tanim", "guc", "tarih", "_blok"])
    t = ortak.test(["tanim", "guc", "tarih"])
    e = e.assign(tanim=e["tanim"].astype(str))
    t = t.assign(tanim=t["tanim"].astype(str))
    h = {b: e[e["_blok"].to_numpy() == b][["tanim", "guc", "tarih"]] for b in ortak.BLOKLAR}
    h["TEST"] = t
    return h


# ----------------------------------------------------------------------------- ortak parcalar
def _buz(toplam: np.ndarray, sayi: np.ndarray, k: float) -> np.ndarray:
    """James-Stein tarzi buzme: n/(n+k) x grup ortalamasi."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ort = np.where(sayi > 0, toplam / np.maximum(sayi, 1e-9), 0.0)
    return ort * (sayi / (sayi + k))


def _seviye(P: Panel, kesim: pd.Timestamp, pencere: int = 60) -> np.ndarray:
    """Trafo bazli guncel seviye (son ``pencere`` gunun ortalamasi, yoksa tumu)."""
    gec = P.gi < kesim
    Mh = P.M[:, gec]
    son = Mh[:, -pencere:] if Mh.shape[1] > pencere else Mh
    with np.errstate(invalid="ignore"):
        s = np.nanmean(son, axis=1)
        tum = np.nanmean(Mh, axis=1)
    return np.where(np.isfinite(s), s, tum)


def _hiyerarsi(P: Panel, kesim: pd.Timestamp, hedef_tanim: np.ndarray) -> np.ndarray:
    """K3'un cekirdegi: il > ilce > kova > ilce x kova > trafo kismi havuzlamasi.

    Doner: hedef trafolarin OFSET (log1p(tuketim) - log1p(guc)) seviyesi.
    """
    gec = P.gi < kesim
    Mh = P.M[:, gec]
    with np.errstate(invalid="ignore"):
        tr_ort = np.nanmean(Mh, axis=1)
    tr_n = np.isfinite(Mh).sum(axis=1).astype("float64")
    var = np.isfinite(tr_ort)
    r = tr_ort - P.lgc  # trafo ofseti

    hm = P.harita.reindex(P.tanim)
    il = hm["il_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    ilce = hm["ilce_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    kova = hm["g_guc_kova"].fillna(-1).astype(int).astype(str).to_numpy(dtype=object)
    ilce_kova = (pd.Series(ilce) + "|" + pd.Series(kova)).to_numpy()

    mu = float(np.average(r[var], weights=tr_n[var]))
    kalan = r - mu

    def katman(anahtar: np.ndarray, k: float) -> tuple[dict, np.ndarray]:
        s = pd.Series(np.where(var, kalan * tr_n, 0.0)).groupby(anahtar).sum()
        n = pd.Series(np.where(var, tr_n, 0.0)).groupby(anahtar).sum()
        etki = pd.Series(_buz(s.to_numpy(), n.to_numpy(), k), index=s.index)
        return etki.to_dict(), pd.Series(anahtar).map(etki).to_numpy("float64")

    e_il, v_il = katman(il, 500.0)
    kalan = kalan - v_il
    e_ilce, v_ilce = katman(ilce, 500.0)
    kalan = kalan - v_ilce
    e_kova, v_kova = katman(kova, 500.0)
    kalan = kalan - v_kova
    e_ik, v_ik = katman(ilce_kova, 300.0)
    kalan = kalan - v_ik
    # trafo katmani: kendi verisi olan trafolar icin
    tr_etki = np.where(var, kalan * (tr_n / (tr_n + 25.0)), 0.0)

    # hedef trafolara tasi
    hh = P.harita.reindex(hedef_tanim)
    h_il = hh["il_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    h_ilce = hh["ilce_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    h_kova = hh["g_guc_kova"].fillna(-1).astype(int).astype(str).to_numpy(dtype=object)
    h_ik = (pd.Series(h_ilce) + "|" + pd.Series(h_kova)).to_numpy()
    idx = P.ti.get_indexer(hedef_tanim)

    def al(sz: dict, anahtar: np.ndarray) -> np.ndarray:
        return pd.Series(anahtar).map(sz).fillna(0.0).to_numpy("float64")

    ofs = (
        mu
        + al(e_il, h_il)
        + al(e_ilce, h_ilce)
        + al(e_kova, h_kova)
        + al(e_ik, h_ik)
        + np.where(idx >= 0, tr_etki[np.maximum(idx, 0)], 0.0)
    )
    return ofs


# ----------------------------------------------------------------------------- adaylar
def k1_mevsimsel_naif(ad: str) -> None:
    """Gecen yilin AYNI GUNU (lag-364, hafta gunu hizali) + surukleme duzeltmesi."""
    P = panel()
    cv, testp = {}, None
    for etiket, hf in hedefler().items():
        kesim = pd.Timestamp(KESIM[etiket])
        gun_e = P.gun_etkisi(kesim, pd.DatetimeIndex(hf["tarih"]))
        sev = _seviye(P, kesim)
        ofs = _hiyerarsi(P, kesim, hf["tanim"].to_numpy())
        lgc_h = np.log1p(hf["guc"].to_numpy("float64"))
        idx_t = P.ti.get_indexer(hf["tanim"].to_numpy())
        sev_h = np.where(idx_t >= 0, sev[np.maximum(idx_t, 0)], np.nan)
        taban = np.where(np.isfinite(sev_h), sev_h, lgc_h + ofs)
        hedef_gun = pd.DatetimeIndex(hf["tarih"])
        gec_gun = hedef_gun - pd.Timedelta(days=364)
        idx_g = P.gi.get_indexer(gec_gun)
        # +-3 gunluk pencere medyani
        yig = []
        for k in (-3, -2, -1, 0, 1, 2, 3):
            j = idx_g + k
            v = np.full(len(hf), np.nan, dtype="float32")
            ok = (j >= 0) & (j < len(P.gi)) & (idx_t >= 0)
            v[ok] = P.M[idx_t[ok], j[ok]]
            yig.append(v)
        with np.errstate(invalid="ignore"):
            gy = np.nanmedian(np.vstack(yig), axis=0)
        kapsam = float(np.isfinite(gy).mean())
        # surukleme: son 45 gun seviyesi - gecen yil ayni pencerenin seviyesi
        sur = _surukleme(P, kesim)
        d = np.where(idx_t >= 0, sur[np.maximum(idx_t, 0)], 0.0)
        log_t = np.where(np.isfinite(gy), gy + 0.5 * d, taban + gun_e)
        p = np.clip(np.expm1(np.maximum(log_t, 0.0)), 0.0, None)
        print(f"  K1/{etiket}: lag-364 kapsami {kapsam:.1%}")
        if etiket == "TEST":
            testp = p
        else:
            cv[etiket] = p
    ortak.kaydet(ad, cv, testp)


def _surukleme(P: Panel, kesim: pd.Timestamp, pencere: int = 45) -> np.ndarray:
    """Son ``pencere`` gun seviyesi eksi gecen yil ayni pencere seviyesi."""
    gec = P.gi < kesim
    son_gun = P.gi[gec][-pencere:] if gec.sum() > pencere else P.gi[gec]
    gy_gun = son_gun - pd.Timedelta(days=364)
    i1 = P.gi.get_indexer(son_gun)
    i2 = P.gi.get_indexer(gy_gun)
    i2ok = i2[i2 >= 0]
    with np.errstate(invalid="ignore"):
        a = np.nanmean(P.M[:, i1], axis=1)
        bq = np.nanmean(P.M[:, i2ok], axis=1) if len(i2ok) else np.full(len(P.ti), np.nan)
    d = a - bq
    return np.where(np.isfinite(d), np.clip(d, -1.5, 1.5), 0.0)


def k2_ets_theta(ad: str) -> None:
    """SES seviye + sonumlu trend + trafo haftalik profili + kuresel gun etkisi."""
    P = panel()
    cv, testp = {}, None
    for etiket, hf in hedefler().items():
        kesim = pd.Timestamp(KESIM[etiket])
        gec = P.gi < kesim
        Mh = P.M[:, gec]
        gh = P.gi[gec]
        nh = Mh.shape[1]
        # SES: ustel agirlikli seviye (yarilanma 45 gun)
        w = 0.5 ** ((nh - 1 - np.arange(nh)) / 45.0)
        gec_var = np.isfinite(Mh)
        Mz = np.where(gec_var, Mh, 0.0)
        pay = Mz @ w
        payda = gec_var @ w
        seviye = np.where(payda > 1e-6, pay / np.maximum(payda, 1e-9), np.nan)
        # trend: son 180 gunun haftalik ortalamalari uzerinde OLS egim (gun basina)
        p180 = min(nh, 180)
        S = Mh[:, -p180:]
        x = np.arange(p180, dtype="float64")
        Sv = np.isfinite(S)
        n = Sv.sum(axis=1).astype("float64")
        Sz = np.where(Sv, S, 0.0)
        sx = Sv @ x
        sy = Sz.sum(axis=1)
        sxx = Sv @ (x * x)
        sxy = (Sz * x).sum(axis=1)
        det = n * sxx - sx * sx
        with np.errstate(invalid="ignore", divide="ignore"):
            egim = np.where(det > 1e-6, (n * sxy - sx * sy) / np.maximum(det, 1e-9), 0.0)
        egim = np.where(n >= 30, np.clip(egim, -0.006, 0.006), 0.0)
        # haftalik profil (trafo bazli, buzulmus)
        dow = gh.dayofweek.to_numpy()
        sap = Mh - np.where(np.isfinite(seviye), seviye, 0.0)[:, None]
        hp = np.zeros((len(P.ti), 7), dtype="float64")
        for k in range(7):
            m = dow == k
            with np.errstate(invalid="ignore"):
                v = np.nanmean(np.where(gec_var[:, m], sap[:, m], np.nan), axis=1)
            nk = gec_var[:, m].sum(axis=1).astype("float64")
            hp[:, k] = np.nan_to_num(v) * (nk / (nk + 10.0))
        hp -= hp.mean(axis=1, keepdims=True)

        hedef_gun = pd.DatetimeIndex(hf["tarih"])
        gun_e = P.gun_etkisi(kesim, hedef_gun)
        idx_t = P.ti.get_indexer(hf["tanim"].to_numpy())
        ufuk = (hedef_gun - kesim).days.to_numpy().astype("float64") + 1.0
        phi = 0.985
        birikim = phi * (1.0 - phi**ufuk) / (1.0 - phi)
        ofs = _hiyerarsi(P, kesim, hf["tanim"].to_numpy())
        lgc_h = np.log1p(hf["guc"].to_numpy("float64"))
        sev_h = np.where(idx_t >= 0, seviye[np.maximum(idx_t, 0)], np.nan)
        egim_h = np.where(idx_t >= 0, egim[np.maximum(idx_t, 0)], 0.0)
        hp_h = np.where(idx_t[:, None] >= 0, hp[np.maximum(idx_t, 0)], 0.0)[
            np.arange(len(hf)), hedef_gun.dayofweek.to_numpy()
        ]
        taban = np.where(np.isfinite(sev_h), sev_h, lgc_h + ofs)
        log_t = taban + egim_h * birikim + hp_h + gun_e
        p = np.clip(np.expm1(np.maximum(log_t, 0.0)), 0.0, None)
        if etiket == "TEST":
            testp = p
        else:
            cv[etiket] = p
    ortak.kaydet(ad, cv, testp)


def k3_hiyerarsi(ad: str) -> None:
    """Saf kismi havuzlama: hiyerarsik seviye + kuresel gun etkisi."""
    P = panel()
    cv, testp = {}, None
    for etiket, hf in hedefler().items():
        kesim = pd.Timestamp(KESIM[etiket])
        hedef_gun = pd.DatetimeIndex(hf["tarih"])
        ofs = _hiyerarsi(P, kesim, hf["tanim"].to_numpy())
        gun_e = P.gun_etkisi(kesim, hedef_gun)
        lgc_h = np.log1p(hf["guc"].to_numpy("float64"))
        log_t = lgc_h + ofs + gun_e
        p = np.clip(np.expm1(np.maximum(log_t, 0.0)), 0.0, None)
        if etiket == "TEST":
            testp = p
        else:
            cv[etiket] = p
    ortak.kaydet(ad, cv, testp)


def k4_knn(ad: str, K: int = 20) -> None:
    """Komsuluk: ayni ilce + yakin kVA + profil benzerligi.

    Seviye komsulardan agirlikli ortalama; GUN SEKLI komsularin gecen yilki
    ayni takvim gunundeki sapmasindan. Uretimde boyle bir kanal YOK.
    """
    P = panel()
    hm = P.harita.reindex(P.tanim)
    ilce_all = hm["ilce_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    il_all = hm["il_key"].fillna("NA").astype(str).to_numpy(dtype=object)
    cv, testp = {}, None
    for etiket, hf in hedefler().items():
        t0 = time.time()
        kesim = pd.Timestamp(KESIM[etiket])
        gec = P.gi < kesim
        Mh = P.M[:, gec]
        with np.errstate(invalid="ignore"):
            sev_all = np.nanmean(Mh, axis=1)
        var = np.isfinite(sev_all) & (np.isfinite(Mh).sum(axis=1) >= 30)
        ofs_all = sev_all - P.lgc

        hedef_tanim = pd.unique(hf["tanim"].to_numpy())
        hh = P.harita.reindex(hedef_tanim)
        h_ilce = hh["ilce_key"].fillna("NA").astype(str).to_numpy(dtype=object)
        h_il = hh["il_key"].fillna("NA").astype(str).to_numpy(dtype=object)
        h_guc = np.log1p(
            hf.drop_duplicates("tanim")
            .set_index("tanim")["guc"]
            .reindex(hedef_tanim)
            .to_numpy("float64")
        )
        idx_self = P.ti.get_indexer(hedef_tanim)

        # komsu havuzlari
        havuz_ilce: dict[str, np.ndarray] = {}
        for a in np.unique(ilce_all[var]):
            havuz_ilce[a] = np.where(var & (ilce_all == a))[0]
        havuz_il: dict[str, np.ndarray] = {}
        for a in np.unique(il_all[var]):
            havuz_il[a] = np.where(var & (il_all == a))[0]
        tum = np.where(var)[0]

        gy_gun = pd.DatetimeIndex(hf["tarih"]) - pd.Timedelta(days=364)
        gy_idx = P.gi.get_indexer(gy_gun)
        satir_trafo = pd.Index(hedef_tanim).get_indexer(hf["tanim"].to_numpy())

        komsu = np.zeros((len(hedef_tanim), K), dtype="int64")
        agir = np.zeros((len(hedef_tanim), K), dtype="float64")
        for i in range(len(hedef_tanim)):
            hav = havuz_ilce.get(h_ilce[i])
            if hav is None or len(hav) < K:
                hav = havuz_il.get(h_il[i], tum)
            if len(hav) < K:
                hav = tum
            hav = hav[hav != idx_self[i]] if idx_self[i] >= 0 else hav
            d = np.abs(P.lgc[hav] - h_guc[i])
            sec = hav[np.argsort(d)[:K]]
            w = 1.0 / (0.25 + np.abs(P.lgc[sec] - h_guc[i]))
            komsu[i, : len(sec)] = sec
            agir[i, : len(sec)] = w
            if len(sec) < K:
                komsu[i, len(sec) :] = sec[0] if len(sec) else 0
        agir = agir / np.maximum(agir.sum(axis=1, keepdims=True), 1e-9)

        komsu_ofs = (ofs_all[komsu] * agir).sum(axis=1)
        # kendi verisi varsa buzulmus karisim
        kendi_n = np.where(idx_self >= 0, np.isfinite(Mh[np.maximum(idx_self, 0)]).sum(axis=1), 0)
        kendi_ofs = np.where(idx_self >= 0, ofs_all[np.maximum(idx_self, 0)], np.nan)
        lam = kendi_n / (kendi_n + 40.0)
        ofs_t = (
            np.where(np.isfinite(kendi_ofs), lam * np.nan_to_num(kendi_ofs), 0.0)
            + (1.0 - np.where(np.isfinite(kendi_ofs), lam, 0.0)) * komsu_ofs
        )

        # gun sekli: komsularin gecen yilki sapmasi
        ok = (gy_idx >= 0) & (gy_idx < len(P.gi))
        sekil = np.zeros(len(hf), dtype="float64")
        if ok.any():
            kk = komsu[satir_trafo[ok]]
            ww = agir[satir_trafo[ok]]
            gg = gy_idx[ok]
            vals = P.M[kk, gg[:, None]].astype("float64") - sev_all[kk]
            gvar = np.isfinite(vals)
            wv = np.where(gvar, ww, 0.0)
            with np.errstate(invalid="ignore"):
                s = (np.where(gvar, vals, 0.0) * wv).sum(axis=1) / np.maximum(wv.sum(axis=1), 1e-9)
            sekil[ok] = np.nan_to_num(np.clip(s, -2.0, 2.0))
        kapsam = float(ok.mean())

        hedef_gun = pd.DatetimeIndex(hf["tarih"])
        gun_e = P.gun_etkisi(kesim, hedef_gun)
        lgc_h = np.log1p(hf["guc"].to_numpy("float64"))
        log_t = lgc_h + ofs_t[satir_trafo] + np.where(ok, sekil, gun_e)
        p = np.clip(np.expm1(np.maximum(log_t, 0.0)), 0.0, None)
        print(f"  K4/{etiket}: gy kapsami {kapsam:.1%}  {time.time() - t0:.0f}s")
        if etiket == "TEST":
            testp = p
        else:
            cv[etiket] = p
    ortak.kaydet(ad, cv, testp)


def main() -> None:
    isler = [
        ("K1_mevsimsel_naif", k1_mevsimsel_naif),
        ("K3_hiyerarsi", k3_hiyerarsi),
        ("K2_ets_theta", k2_ets_theta),
        ("K4_knn_komsu", k4_knn),
    ]
    for ad, fn in isler:
        if ortak.var_mi(ad):
            print(f"{ad}: onbellekte, atlaniyor")
            continue
        t0 = time.time()
        print(f"--- {ad}")
        fn(ad)
        print(f"    bitti {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
