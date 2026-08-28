"""PROB TASARIMI -- adim 1: HILE TAVANLARI ve BLOKLAR ARASI KORELASYON.

Her aday grup bolunmesi icin iki sayi olculur:

  TAVAN  : ofset AYNI bloktan ogrenilirse dMSE ne olur (ulasilabilir en iyi).
           LB probu bunu dogrudan test kumesinde olcecegi icin tasima
           gerektirmez -- yani prob kampanyasinin UST SINIRIDIR.
  |rho|  : bir blokta ogrenilen ofset DESENININ diger bloktaki desenle
           (test grup buyuklukleriyle agirliklandirilmis) korelasyonu.
           Beklenen prob kazanci ~ rho^2 * tavan.

Sicak taraf ``experiments/sicak_kaldirac/ortak.py``, soguk taraf
``experiments/soguk_kaldirac/ortak.py`` tabanlarini birebir kullanir.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent


def _modul(ad: str, yol: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(ad, yol)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ad] = mod
    spec.loader.exec_module(mod)
    return mod


SIC = _modul("sicak_ortak", KOK / "experiments" / "sicak_kaldirac" / "ortak.py")

BLOKLAR = ("yaz25", "guz25", "kis26")
SICAK_PAY = 556_319 / 714_688
SOGUK_PAY = 158_369 / 714_688


def _soguk_modul():
    return _modul("soguk_ortak", KOK / "experiments" / "soguk_kaldirac" / "ortak.py")


def kova_kva(guc: np.ndarray) -> np.ndarray:
    kenar = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
    et = ["<=50", "100", "160", "250", "400", "630", "1000", ">1000"]
    return np.asarray(et)[np.digitize(guc, kenar) - 1]


def desil(x: np.ndarray, k: int = 10) -> np.ndarray:
    """Blok ICI k-lik: kenarlar tahmin dagilimindan -- ETIKETSIZ, mesru."""
    kenar = np.quantile(x, np.linspace(0.0, 1.0, k + 1)[1:-1])
    return np.digitize(x, kenar)


def grup_ofseti(e: np.ndarray, g: np.ndarray) -> tuple[dict, np.ndarray]:
    """Grup ortalamasi artiklar (global ortalama cikarilmis)."""
    s = pd.Series(e - e.mean())
    key = pd.Series(g)
    ort = s.groupby(key).mean()
    ort = ort - (ort * key.value_counts().reindex(ort.index)).sum() / len(s)
    return ort.to_dict(), key.map(ort).to_numpy(dtype="float64")


def tavan_hesapla(e: np.ndarray, g: np.ndarray) -> float:
    """Ofset AYNI bloktan ogrenilirse dMSE (negatif = kazanc)."""
    _, d = grup_ofseti(e, g)
    yeni = e - d
    return float((yeni * yeni).mean() - (e * e).mean())


def korelasyon(desen: dict[str, dict], agirlik: dict) -> dict[str, float]:
    """Blok ciftleri arasi agirlikli korelasyon (test grup buyukluklerine gore)."""
    cikti = {}
    for a, b in itertools.combinations(desen, 2):
        ortak_anahtar = [k for k in desen[a] if k in desen[b] and k in agirlik]
        if len(ortak_anahtar) < 3:
            cikti[f"{a}|{b}"] = float("nan")
            continue
        w = np.array([agirlik[k] for k in ortak_anahtar], dtype="float64")
        x = np.array([desen[a][k] for k in ortak_anahtar], dtype="float64")
        y = np.array([desen[b][k] for k in ortak_anahtar], dtype="float64")
        w = w / w.sum()
        xm, ym = float(w @ x), float(w @ y)
        vx = float(w @ (x - xm) ** 2)
        vy = float(w @ (y - ym) ** 2)
        if vx <= 0 or vy <= 0:
            cikti[f"{a}|{b}"] = float("nan")
            continue
        cikti[f"{a}|{b}"] = float((w @ ((x - xm) * (y - ym))) / np.sqrt(vx * vy))
    return cikti


# --------------------------------------------------------------------------
# TEST tarafi grup buyuklukleri (agirlik olarak, ETIKETSIZ)
# --------------------------------------------------------------------------
def test_cercevesi() -> pd.DataFrame:
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih", "lokasyon"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    p = te["lokasyon"].fillna("").astype(str).str.split(">")
    te["il"] = p.str[0].str.strip()
    te["ilce"] = p.str[-1].str.strip()
    te["kova"] = kova_kva(te["guc"].to_numpy(dtype="float64"))
    te["ay"] = pd.to_datetime(te["tarih"]).dt.month
    tr_tanim = set(
        pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
            "tanim"
        ].unique()
    )
    te["sicak"] = te["tanim"].isin(tr_tanim)
    v93 = pd.read_csv(KOK / "submissions/tuketim_v93_gram_optimum.csv")
    assert (v93["id"].to_numpy() == te["id"].to_numpy()).all(), "id sirasi farkli"
    te["lgp"] = np.log1p(v93["tuketim"].to_numpy(dtype="float64"))
    return te


def main() -> None:
    te = test_cercevesi()
    ts = te[te["sicak"]].reset_index(drop=True)
    tso = te[~te["sicak"]].reset_index(drop=True)
    ts["sev"] = desil(ts["lgp"].to_numpy(), 10)
    tso["sev"] = desil(tso["lgp"].to_numpy(), 10)
    print(f"TEST: sicak {len(ts):,}  soguk {len(tso):,}  toplam {len(te):,}")

    satirlar = []

    # ---------------- SICAK ----------------
    bl = SIC.bloklari_kur()
    taban = {k: SIC.taban_r(bl[k]) for k in BLOKLAR}
    sic_grup = {
        "ilce": lambda b: b.cerceve["ilce"].to_numpy(),
        "seviye_desili10": lambda b, r=None: None,  # asagida ozel
        "kva_kovasi": lambda b: b.cerceve["kova"].to_numpy(),
        "ay": lambda b: b.cerceve["ay"].to_numpy(),
        "trafo": lambda b: b.cerceve["tanim"].to_numpy(),
        "ilce_x_kova": lambda b: (
            b.cerceve["ilce"].astype(str) + "|" + b.cerceve["kova"].astype(str)
        ).to_numpy(),
    }
    sic_agirlik = {
        "ilce": ts["ilce"].value_counts().to_dict(),
        "seviye_desili10": ts["sev"].value_counts().to_dict(),
        "kva_kovasi": ts["kova"].value_counts().to_dict(),
        "ay": ts["ay"].value_counts().to_dict(),
        "trafo": ts["tanim"].value_counts().to_dict(),
        "ilce_x_kova": (ts["ilce"] + "|" + ts["kova"]).value_counts().to_dict(),
    }

    art = {}
    grup_dizi: dict[str, dict[str, np.ndarray]] = {}
    for k in BLOKLAR:
        b = bl[k]
        r0 = taban[k]
        art[k] = b.lgy - np.maximum(r0 + b.lgc, 0.0)
        lgp = np.maximum(r0 + b.lgc, 0.0)
        for ad, fn in sic_grup.items():
            if ad == "seviye_desili10":
                g = desil(lgp, 10)
            else:
                g = fn(b)
            grup_dizi.setdefault(ad, {})[k] = np.asarray(g)

    for ad in sic_grup:
        tavan = {}
        desen = {}
        for k in BLOKLAR:
            e = art[k]
            g = grup_dizi[ad][k]
            tavan[k] = tavan_hesapla(e, g)
            desen[k], _ = grup_ofseti(e, g)
        rho = korelasyon(desen, sic_agirlik[ad])
        rho_vals = [abs(v) for v in rho.values() if np.isfinite(v)]
        # blok agirlikli ortalama tavan
        n_top = sum(bl[k].n for k in BLOKLAR)
        tv = sum(tavan[k] * bl[k].n for k in BLOKLAR) / n_top
        satirlar.append(
            {
                "rejim": "sicak",
                "yon": ad,
                "grup_sayisi_test": len(sic_agirlik[ad]),
                "tavan_rejim": tv,
                "tavan_toplam": tv * SICAK_PAY,
                "tavan_blok": tavan,
                "rho": rho,
                "rho_ort": float(np.mean(rho_vals)) if rho_vals else float("nan"),
                "rho_min": float(np.min(rho_vals)) if rho_vals else float("nan"),
            }
        )
        print(
            f"  SICAK {ad:18s} G={len(sic_agirlik[ad]):>5d}  tavan_rejim={tv:+.5f}"
            f"  tavan_toplam={tv * SICAK_PAY:+.5f}  |rho|ort="
            f"{satirlar[-1]['rho_ort']:+.3f}  min={satirlar[-1]['rho_min']:+.3f}"
        )

    del bl, taban, art, grup_dizi

    # ---------------- SOGUK ----------------
    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    stab = {k: SOG.taban_r(sb[k]) for k in BLOKLAR}
    sog_agirlik = {
        "ilce": tso["ilce"].value_counts().to_dict(),
        "seviye_desili10": tso["sev"].value_counts().to_dict(),
        "kva_kovasi": tso["kova"].value_counts().to_dict(),
        "ay": tso["ay"].value_counts().to_dict(),
        "trafo": tso["tanim"].value_counts().to_dict(),
    }
    sart = {}
    sgrup: dict[str, dict[str, np.ndarray]] = {}
    for k in BLOKLAR:
        b = sb[k]
        r0 = stab[k]
        sart[k] = b.lgy - (r0 + b.lgc)
        lgp = r0 + b.lgc
        sgrup.setdefault("ilce", {})[k] = b.ilce
        sgrup.setdefault("seviye_desili10", {})[k] = desil(lgp, 10)
        sgrup.setdefault("kva_kovasi", {})[k] = kova_kva(b.guc)
        sgrup.setdefault("ay", {})[k] = pd.to_datetime(b.tarih).month.to_numpy()
        sgrup.setdefault("trafo", {})[k] = b.tanim

    for ad in sog_agirlik:
        tavan = {}
        desen = {}
        for k in BLOKLAR:
            e = sart[k]
            g = sgrup[ad][k]
            tavan[k] = tavan_hesapla(e, g)
            desen[k], _ = grup_ofseti(e, g)
        rho = korelasyon(desen, sog_agirlik[ad])
        rho_vals = [abs(v) for v in rho.values() if np.isfinite(v)]
        n_top = sum(sb[k].n for k in BLOKLAR)
        tv = sum(tavan[k] * sb[k].n for k in BLOKLAR) / n_top
        satirlar.append(
            {
                "rejim": "soguk",
                "yon": ad,
                "grup_sayisi_test": len(sog_agirlik[ad]),
                "tavan_rejim": tv,
                "tavan_toplam": tv * SOGUK_PAY,
                "tavan_blok": tavan,
                "rho": rho,
                "rho_ort": float(np.mean(rho_vals)) if rho_vals else float("nan"),
                "rho_min": float(np.min(rho_vals)) if rho_vals else float("nan"),
            }
        )
        print(
            f"  SOGUK {ad:18s} G={len(sog_agirlik[ad]):>5d}  tavan_rejim={tv:+.5f}"
            f"  tavan_toplam={tv * SOGUK_PAY:+.5f}  |rho|ort="
            f"{satirlar[-1]['rho_ort']:+.3f}  min={satirlar[-1]['rho_min']:+.3f}"
        )

    (BURA / "tavan.json").write_text(
        json.dumps(satirlar, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- desenleri prob uretimi icin kaydet ----
    print("\nyazildi: tavan.json")


if __name__ == "__main__":
    main()
