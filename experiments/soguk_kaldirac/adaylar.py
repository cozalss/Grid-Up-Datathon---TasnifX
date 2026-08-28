"""SOGUK KOHORT ADAYLARI -- uc blokta sizintisiz olcum.

Her aday, uretim tabani (cat + beta 0,60 + delta 0,1046) UZERINE uygulanan
bir SON ISLEMDIR; yani v83'e dogrudan tasinabilir.

Parametreli adaylar (grup ofsetleri) BLOK-DISI uydurulur:
hedef blok icin katsayilar DIGER IKI bloktan ogrenilir. Uretimde uc blogun
tamami kullanilacagi icin bu, kazanci HAFIF EKSIK tahmin eden durust bir
kestirimdir.

KAPI: uc blokta da ayni isaret + toplam (test) dMSE >= 0,002.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, SOGUK_PAY, Blok, mse, taban_r, tum_bloklar  # noqa: E402

CIKTI = Path(__file__).resolve().parent / "adaylar.jsonl"


# ------------------------------------------------------------ yardimcilar


def kova_et(guc: np.ndarray) -> np.ndarray:
    kenar = [0, 60, 110, 175, 260, 410, 520, 720, 900, 1130, 1400, 1e9]
    et = ["50-", "100", "160", "250", "400", "500", "630", "800", "1000", "1250", "1600+"]
    return np.asarray(et)[np.digitize(guc, kenar) - 1]


def kaba_kova(guc: np.ndarray) -> np.ndarray:
    return np.where(guc <= 250, "kucuk", np.where(guc <= 630, "orta", "buyuk"))


def yas_kova(yas: np.ndarray) -> np.ndarray:
    kenar = [-1, 0, 2, 6, 13, 29, 59, 10**9]
    et = ["0", "1-2", "3-6", "7-13", "14-29", "30-59", "60+"]
    return np.asarray(et)[np.digitize(yas, kenar) - 1]


def artik(b: Blok, r: np.ndarray) -> np.ndarray:
    return b.lgy - (r + b.lgc)


def grup_etkisi(e: np.ndarray, g: np.ndarray, *, k: float = 200.0) -> dict[str, float]:
    """MERKEZLI grup ofsetleri, ampirik-Bayes buzmesiyle.

    ``k`` = buzme yarilanmasi: n satirlik bir grubun etkisi n/(n+k) ile
    carpilir. Genel ortalama CIKARILIR -- seviye ayri bir adaydir (C1).
    """
    d = pd.DataFrame({"g": g, "e": e})
    ort = float(d["e"].mean())
    t = d.groupby("g")["e"].agg(["mean", "size"])
    etki = (t["mean"] - ort) * (t["size"] / (t["size"] + k))
    return {str(i): float(v) for i, v in etki.items()}


def uygula(etki: dict[str, float], g: np.ndarray) -> np.ndarray:
    return np.array([etki.get(str(x), 0.0) for x in g], dtype="float64")


# ------------------------------------------------------------ adaylar


def c1_seviye(bloklar: dict[str, Blok]) -> list[dict]:
    """C1 -- global seviye ofseti (log uzayinda sabit)."""
    out = []
    for delta in (-0.20, -0.15, -0.10, -0.06, -0.03, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30):
        satir = {"aday": f"C1 seviye d={delta:+.2f}"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            satir[ad] = mse(b, r0) - mse(b, r0 + delta)
        out.append(satir)
    # blok-disi optimum delta
    satir = {"aday": "C1 seviye (blok-disi optimum d)"}
    for ad in BLOKLAR:
        b = bloklar[ad]
        d_opt = float(
            np.mean([artik(bloklar[o], taban_r(bloklar[o])).mean() for o in BLOKLAR if o != ad])
        )
        r0 = taban_r(b)
        satir[ad] = mse(b, r0) - mse(b, r0 + d_opt)
        satir[ad + "_d"] = round(d_opt, 4)
    out.append(satir)
    return out


def c2_genlik(bloklar: dict[str, Blok]) -> list[dict]:
    """C2 -- toplam buzme katsayisi (uretim 0,60)."""
    out = []
    for beta in (0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50):
        satir = {"aday": f"C2 genlik beta={beta:.2f}"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            r1 = taban_r(b, beta=beta)
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
    return out


def _lobo_grup(bloklar: dict[str, Blok], anahtar, ad_aday: str, *, k: float = 200.0) -> dict:
    """Blok-disi uydurulan merkezli grup ofseti."""
    satir = {"aday": ad_aday}
    for ad in BLOKLAR:
        birik: dict[str, list[float]] = {}
        for o in BLOKLAR:
            if o == ad:
                continue
            bo = bloklar[o]
            e = artik(bo, taban_r(bo))
            for gk, v in grup_etkisi(e, anahtar(bo), k=k).items():
                birik.setdefault(gk, []).append(v)
        etki = {gk: float(np.mean(v)) for gk, v in birik.items()}
        b = bloklar[ad]
        r0 = taban_r(b)
        satir[ad] = mse(b, r0) - mse(b, r0 + uygula(etki, anahtar(b)))
    return satir


def c3_kva(bloklar: dict[str, Blok]) -> list[dict]:
    return [
        _lobo_grup(bloklar, lambda b: kova_et(b.guc), "C3a kVA ince (11 kova)"),
        _lobo_grup(bloklar, lambda b: kaba_kova(b.guc), "C3b kVA kaba (3 kova)"),
        _lobo_grup(bloklar, lambda b: b.guc.astype("int64").astype(str), "C3c kVA ham deger"),
    ]


def c4_takvim(bloklar: dict[str, Blok]) -> list[dict]:
    hg = lambda b: pd.to_datetime(b.tarih).dayofweek.astype(str).to_numpy()  # noqa: E731
    hs = lambda b: np.where(pd.to_datetime(b.tarih).dayofweek >= 5, "hafta_sonu", "ic_gun")  # noqa: E731
    ay = lambda b: pd.to_datetime(b.tarih).month.astype(str).to_numpy()  # noqa: E731
    return [
        _lobo_grup(bloklar, hg, "C4a haftagunu (7)"),
        _lobo_grup(bloklar, hs, "C4b hafta sonu / ic gun"),
        _lobo_grup(bloklar, ay, "C4c ay (blok-disi -- mevsim kaymasi RISKLI)"),
    ]


def c5_konum(bloklar: dict[str, Blok]) -> list[dict]:
    return [
        _lobo_grup(bloklar, lambda b: b.il, "C5a il (2)"),
        _lobo_grup(bloklar, lambda b: b.ilce, "C5b ilce"),
        _lobo_grup(bloklar, lambda b: b.lokasyon.astype(str), "C5c tam lokasyon"),
        _lobo_grup(
            bloklar,
            lambda b: np.char.add(np.char.add(b.ilce.astype(str), "|"), kaba_kova(b.guc)),
            "C5d ilce x kVA kaba",
        ),
    ]


def c6_yas(bloklar: dict[str, Blok]) -> list[dict]:
    return [
        _lobo_grup(bloklar, lambda b: yas_kova(b.yas), "C6a panele giris yasi (7 kova)", k=50.0),
        _lobo_grup(
            bloklar,
            lambda b: np.where(b.yas == 0, "ilk_gun", "sonra"),
            "C6b yalnizca ILK GUN",
            k=50.0,
        ),
        _lobo_grup(
            bloklar, lambda b: np.where(b.yas <= 6, "ilk_hafta", "sonra"), "C6c ilk hafta", k=50.0
        ),
    ]


def c7_kova_harmani(bloklar: dict[str, Blok]) -> list[dict]:
    """C7 -- model tahminini blok-disi kVA-kovasi AMPIRIK seviyesine harmanla."""
    out = []
    for w in (0.15, 0.30, 0.50, 0.70, 1.00):
        satir = {"aday": f"C7 kVA kovasi ampirik harman w={w:.2f}"}
        for ad in BLOKLAR:
            birik: dict[str, list[float]] = {}
            for o in BLOKLAR:
                if o == ad:
                    continue
                bo = bloklar[o]
                ry = bo.lgy - bo.lgc
                d = pd.DataFrame({"g": kova_et(bo.guc), "r": ry})
                for gk, v in d.groupby("g")["r"].mean().items():
                    birik.setdefault(str(gk), []).append(float(v))
            hedef = {gk: float(np.mean(v)) for gk, v in birik.items()}
            b = bloklar[ad]
            r0 = taban_r(b)
            rk = uygula(hedef, kova_et(b.guc))
            # kovasi bulunmayan satirlar r0'da kalir
            var = np.array([str(x) in hedef for x in kova_et(b.guc)])
            r1 = r0.copy()
            r1[var] = (1 - w) * r0[var] + w * rk[var]
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
    return out


def c8_ici_arasi(bloklar: dict[str, Blok]) -> list[dict]:
    """C8 -- trafo-ARASI ve trafo-ICI genligi AYRI olcekle."""
    out = []
    for ba in (0.60, 1.00, 1.40, 2.00):
        for bi in (0.00, 0.50, 1.00):
            satir = {"aday": f"C8 arasi={ba:.2f} ici={bi:.2f}"}
            for ad in BLOKLAR:
                b = bloklar[ad]
                r0 = taban_r(b)
                s = pd.Series(r0).groupby(pd.Series(b.tanim)).transform("mean").to_numpy()
                r1 = r0.mean() + ba * (s - r0.mean()) + bi * (r0 - s)
                satir[ad] = mse(b, r0) - mse(b, r1)
            out.append(satir)
    return out


def c9_aile(bloklar: dict[str, Blok]) -> list[dict]:
    """C9 -- SON ISLEM DEGIL: aile harmani (yeniden kosum gerektirir)."""
    out = []
    karisim = {
        "yalniz cat (URETIM)": {"cat": 1.0},
        "yalniz xgb": {"xgb": 1.0},
        "yalniz lgbm": {"lgbm": 1.0},
        "1/1/1": {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0},
        "3/1/1": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0},
        "1/2/2": {"cat": 1.0, "xgb": 2.0, "lgbm": 2.0},
    }
    for ad_k, w in karisim.items():
        satir = {"aday": f"C9 aile {ad_k}"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            tw = sum(w.values())
            p = sum(wv * b.ham[k] for k, wv in w.items()) / tw
            r = p - b.lgc
            r1 = r.mean() + 0.60 * (r - r.mean()) + 0.1046
            satir[ad] = mse(b, r0) - mse(b, r1)
        out.append(satir)
    return out


def yazdir(baslik: str, satirlar: list[dict]) -> list[dict]:
    print("\n" + "=" * 92)
    print(baslik)
    print("=" * 92)
    print(f"{'aday':46} {'yaz25':>10} {'guz25':>10} {'kis26':>10} {'GENEL':>10}  KAPI")
    for s in satirlar:
        v = [s[b] for b in BLOKLAR]
        hepsi_arti = all(x > 0 for x in v)
        hepsi_eksi = all(x < 0 for x in v)
        # genel beklenen: uc blogun ortalamasi x soguk payi
        genel = float(np.mean(v)) * SOGUK_PAY
        if hepsi_arti and genel >= 0.002:
            kapi = "KABUL"
        elif hepsi_arti:
            kapi = "RED (uc blok + ama esik alti)"
        elif hepsi_eksi:
            kapi = "RED (uc blokta da ZARARLI)"
        else:
            kapi = "RED (isaret ters)"
        s["genel_dmse"] = genel
        s["kapi"] = kapi
        print(
            f"{s['aday']:46} {v[0]:>+10.5f} {v[1]:>+10.5f} {v[2]:>+10.5f} {genel:>+10.5f}  {kapi}"
        )
    return satirlar


def main() -> int:
    bloklar = tum_bloklar()
    hepsi: list[dict] = []
    hepsi += yazdir("C1 -- GLOBAL SEVIYE OFSETI (dMSE > 0 = IYILESME)", c1_seviye(bloklar))
    hepsi += yazdir("C2 -- GENLIK / BUZME KATSAYISI (uretim 0,60)", c2_genlik(bloklar))
    hepsi += yazdir("C3 -- kVA BAZLI SEVIYE (blok-disi uydurma)", c3_kva(bloklar))
    hepsi += yazdir("C4 -- TAKVIM (blok-disi uydurma)", c4_takvim(bloklar))
    hepsi += yazdir("C5 -- KONUM (blok-disi uydurma)", c5_konum(bloklar))
    hepsi += yazdir("C6 -- PANELE GIRIS YASI (blok-disi uydurma)", c6_yas(bloklar))
    hepsi += yazdir("C7 -- kVA KOVASI AMPIRIK HARMAN (blok-disi)", c7_kova_harmani(bloklar))
    hepsi += yazdir("C8 -- TRAFO-ARASI / TRAFO-ICI AYRI GENLIK", c8_ici_arasi(bloklar))
    hepsi += yazdir("C9 -- AILE HARMANI (SON ISLEM DEGIL, yeniden kosum gerekir)", c9_aile(bloklar))

    with CIKTI.open("w", encoding="utf-8") as f:
        for s in hepsi:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nyazildi: {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
