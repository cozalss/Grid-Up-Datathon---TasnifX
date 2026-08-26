"""27 Ağustos skorlarını çöz ve nihai üç-rejim submission'ını üret.

Bu araç Kaggle'a gönderim yapmaz. v80 banka skoru ile v81 sıcak-çekirdek ve
v82 kuyruk problarının skorlarını alır; iki probun kuadratik özdeşliğinden
optimum sıcak çekirdek/kuyruk deltalarını, düzeltilmiş ortak denklemden de
soğuk deltayı çözer. Son dosya v80_a (yalnız ölçülmüş soğuk gün ekseni)
üstüne üç ayrık rejim deltası tek adımda uygulanarak üretilir.

Kullanım::

    uv run python scripts/yarin_coz.py --sabitler
    uv run python scripts/yarin_coz.py --banka-score 1.01341 \
        --sicak-prob-score 1.01108 --kuyruk-prob-score 1.01388
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
SUB = KOK / "submissions"
TABAN_DOSYA = SUB / "tuketim_v80_a.csv"
BANKA_DOSYA = SUB / "tuketim_v80_optimum.csv"
SICAK_PROB_DOSYA = SUB / "tuketim_v81_sicak08.csv"
KUYRUK_PROB_DOSYA = SUB / "tuketim_v82_ayirici.csv"

BEKLENEN_SATIR = 714_688
BEKLENEN_SOGUK = 158_369
BEKLENEN_KUYRUK = 29_873
BEKLENEN_SICAK_CEKIRDEK = 526_446

MEVCUT_SOGUK = 0.1046
MEVCUT_KUYRUK = 0.1664
ORTAK_L = 0.00753522
ORTAK_SOGUK_ADIM = 0.22
ORTAK_KUYRUK_ADIM = 0.35
LIDER_RMSLE = 1.00635


@dataclass(frozen=True)
class ProbSonucu:
    katsayi: float
    optimum_ek_delta: float
    kazanc_mse: float
    gerceklesen_dmse: float


@dataclass(frozen=True)
class NihaiCozum:
    b_sicak_cekirdek: float
    b_soguk: float
    b_kuyruk: float
    tahmini_mse: float


def prob_coz(taban_score: float, prob_score: float, *, q: float, adim: float) -> ProbSonucu:
    """Tek kuadratik probdan tabana göre optimum ek deltayı çöz."""
    if taban_score <= 0 or prob_score <= 0 or q <= 0 or adim == 0:
        raise ValueError("score, q ve adım pozitif olmalı")
    dmse = prob_score**2 - taban_score**2
    ic_carpim = (q - dmse) / 2.0
    katsayi = ic_carpim / q
    return ProbSonucu(
        katsayi=float(katsayi),
        optimum_ek_delta=float(katsayi * adim),
        kazanc_mse=float(-(ic_carpim**2) / q),
        gerceklesen_dmse=float(dmse),
    )


def ortak_denklemden_soguk_coz(
    b_kuyruk: float,
    *,
    l_ortak: float = ORTAK_L,
    p_soguk: float = BEKLENEN_SOGUK / BEKLENEN_SATIR,
    p_kuyruk: float = BEKLENEN_KUYRUK / BEKLENEN_SATIR,
) -> float:
    """v79 ile ölçülen ortak iç-çarpımdan soğuk grup optimumunu ayır."""
    payda = p_soguk * ORTAK_SOGUK_ADIM
    if payda <= 0:
        raise ValueError("soğuk payı pozitif olmalı")
    return float((l_ortak - p_kuyruk * ORTAK_KUYRUK_ADIM * b_kuyruk) / payda)


def nihai_cozum(
    banka_score: float,
    sicak_sonuc: ProbSonucu,
    *,
    kuyruk_sonuc: ProbSonucu | None,
    p_soguk: float = BEKLENEN_SOGUK / BEKLENEN_SATIR,
    p_kuyruk: float = BEKLENEN_KUYRUK / BEKLENEN_SATIR,
) -> NihaiCozum:
    """İki skorla sıcak-only, üç skorla tam optimum çözümü kur."""
    b_sicak = sicak_sonuc.optimum_ek_delta
    tahmini_mse = banka_score**2 + sicak_sonuc.kazanc_mse
    if kuyruk_sonuc is None:
        return NihaiCozum(
            b_sicak_cekirdek=b_sicak,
            b_soguk=MEVCUT_SOGUK,
            b_kuyruk=MEVCUT_KUYRUK,
            tahmini_mse=float(tahmini_mse),
        )
    b_kuyruk = MEVCUT_KUYRUK + kuyruk_sonuc.optimum_ek_delta
    b_soguk = ortak_denklemden_soguk_coz(
        b_kuyruk,
        p_soguk=p_soguk,
        p_kuyruk=p_kuyruk,
    )
    tahmini_mse += kuyruk_sonuc.kazanc_mse - p_soguk * (b_soguk - MEVCUT_SOGUK) ** 2
    return NihaiCozum(
        b_sicak_cekirdek=b_sicak,
        b_soguk=b_soguk,
        b_kuyruk=b_kuyruk,
        tahmini_mse=float(tahmini_mse),
    )


def uc_rejim_deltasi_uygula(
    taban: np.ndarray,
    *,
    soguk: np.ndarray,
    kuyruk: np.ndarray,
    b_soguk: float,
    b_kuyruk: float,
    b_sicak_cekirdek: float,
) -> np.ndarray:
    """Ayrık soğuk/kuyruk/sıcak-çekirdek gruplarına sabit log deltası uygula."""
    y = np.asarray(taban, dtype="float64")
    soguk = np.asarray(soguk, dtype=bool)
    kuyruk = np.asarray(kuyruk, dtype=bool)
    if y.ndim != 1 or soguk.shape != y.shape or kuyruk.shape != y.shape:
        raise ValueError("taban ve maskeler aynı tek-boyutlu şekle sahip olmalı")
    if (soguk & kuyruk).any():
        raise ValueError("soğuk ve kuyruk grupları ayrık olmalı")
    sicak_cekirdek = ~(soguk | kuyruk)
    delta = np.select(
        [soguk, kuyruk, sicak_cekirdek],
        [b_soguk, b_kuyruk, b_sicak_cekirdek],
    )
    ham = np.expm1(np.log1p(np.clip(y, 0.0, None)) + delta)
    if (ham < -1e-12).any():
        raise ValueError("negatif delta kırpma gerektiriyor; kuadratik çözüm artık tam değil")
    return np.maximum(ham, 0.0)


def _oku(yol: Path) -> pd.DataFrame:
    d = pd.read_csv(yol, encoding="utf-8")
    if list(d.columns) != ["id", "tuketim"]:
        raise RuntimeError(f"{yol.name}: kolonlar yanlış {list(d.columns)}")
    if len(d) != BEKLENEN_SATIR or d["id"].duplicated().any():
        raise RuntimeError(f"{yol.name}: satır/id sözleşmesi bozuk")
    if d["tuketim"].isna().any() or (d["tuketim"] < 0).any():
        raise RuntimeError(f"{yol.name}: NaN/negatif tahmin")
    return d


def _prob_vektoru(banka: pd.DataFrame, prob: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    if not prob["id"].equals(banka["id"]):
        raise RuntimeError("prob id sırası bankayla aynı değil")
    fark = np.log1p(prob["tuketim"].to_numpy()) - np.log1p(banka["tuketim"].to_numpy())
    maske = np.abs(fark) > 1e-7
    if not maske.any() or (~maske & (np.abs(fark) > 1e-12)).any():
        raise RuntimeError("prob vektörü iki-değerli değil")
    adim = float(fark[maske].mean())
    if float(np.abs(fark[maske] - adim).max()) > 1e-5:
        raise RuntimeError("prob adımı grup içinde sabit değil")
    q = float((fark**2).mean())
    return maske, adim, q


def _rejim_maskeleri(idler: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim"],
        dtype={"tanim": str},
        encoding="utf-8",
    )
    if not te["id"].equals(idler):
        raise RuntimeError("test.csv id sırası submission ile aynı değil")
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    sicak_set = set(tr["tanim"].unique())
    soguk = ~te["tanim"].isin(sicak_set).to_numpy()
    ilk = tr.groupby("tanim")["tarih"].min()
    kuyruk_set = set(ilk[ilk >= pd.Timestamp("2026-03-26")].index)
    kuyruk = te["tanim"].isin(kuyruk_set).to_numpy()
    if (soguk & kuyruk).any():
        raise RuntimeError("soğuk/kuyruk kesişiyor")
    sicak_cekirdek = ~(soguk | kuyruk)
    sayilar = (int(soguk.sum()), int(kuyruk.sum()), int(sicak_cekirdek.sum()))
    beklenen = (BEKLENEN_SOGUK, BEKLENEN_KUYRUK, BEKLENEN_SICAK_CEKIRDEK)
    if sayilar != beklenen:
        raise RuntimeError(f"rejim sayıları {sayilar}, beklenen {beklenen}")
    return soguk, kuyruk, sicak_cekirdek


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sabitler", action="store_true")
    ap.add_argument("--banka-score", type=float)
    ap.add_argument("--sicak-prob-score", type=float)
    ap.add_argument("--kuyruk-prob-score", type=float)
    ap.add_argument("--cikis", type=Path)
    ap.add_argument("--rapor", type=Path, default=KOK / "reports" / "yarin_cozum.json")
    ar = ap.parse_args()

    taban = _oku(TABAN_DOSYA)
    banka = _oku(BANKA_DOSYA)
    sicak_prob = _oku(SICAK_PROB_DOSYA)
    kuyruk_prob = _oku(KUYRUK_PROB_DOSYA)
    if not taban["id"].equals(banka["id"]):
        raise RuntimeError("v80_a id sırası bankayla aynı değil")

    m_sicak, adim_sicak, q_sicak = _prob_vektoru(banka, sicak_prob)
    m_kuyruk, adim_kuyruk, q_kuyruk = _prob_vektoru(banka, kuyruk_prob)
    soguk, kuyruk, sicak_cekirdek = _rejim_maskeleri(banka["id"])
    if not np.array_equal(m_sicak, sicak_cekirdek):
        raise RuntimeError("v81 yalnız sıcak çekirdeği değiştirmiyor")
    if not np.array_equal(m_kuyruk, kuyruk):
        raise RuntimeError("v82 yalnız kuyruğu değiştirmiyor")

    print("DOSYA SABİTLERİ — etiketsiz")
    print(f"  sıcak çekirdek n={int(m_sicak.sum()):,} adım={adim_sicak:.8f} Q={q_sicak:.9f}")
    print(f"  kuyruk        n={int(m_kuyruk.sum()):,} adım={adim_kuyruk:.8f} Q={q_kuyruk:.9f}")
    print(f"  soğuk         n={int(soguk.sum()):,}")
    if ar.sabitler and ar.banka_score is None:
        return 0

    if ar.banka_score is None or ar.sicak_prob_score is None:
        raise SystemExit("çözüm için banka ve sıcak-prob score zorunlu")
    banka_score = float(ar.banka_score)
    sicak_score = float(ar.sicak_prob_score)

    sicak_sonuc = prob_coz(banka_score, sicak_score, q=q_sicak, adim=adim_sicak)
    kuyruk_sonuc = (
        prob_coz(
            banka_score,
            float(ar.kuyruk_prob_score),
            q=q_kuyruk,
            adim=adim_kuyruk,
        )
        if ar.kuyruk_prob_score is not None
        else None
    )
    cozum = nihai_cozum(
        banka_score,
        sicak_sonuc,
        kuyruk_sonuc=kuyruk_sonuc,
        p_soguk=float(soguk.mean()),
        p_kuyruk=float(kuyruk.mean()),
    )
    b_sicak = cozum.b_sicak_cekirdek
    b_soguk = cozum.b_soguk
    b_kuyruk = cozum.b_kuyruk
    if not (-0.10 <= b_sicak <= 0.40 and -0.10 <= b_soguk <= 0.50 and -0.10 <= b_kuyruk <= 0.70):
        raise RuntimeError(
            f"çözülen delta güvenlik aralığı dışında: sıcak={b_sicak:.4f} "
            f"soğuk={b_soguk:.4f} kuyruk={b_kuyruk:.4f}"
        )

    tahmini_mse = cozum.tahmini_mse
    tahmini_score = float(np.sqrt(max(tahmini_mse, 0.0)))
    mod = "TAM" if kuyruk_sonuc is not None else "SICAK-ONLY (3. hak adayı)"
    print(f"\nÇÖZÜM — {mod}")
    print(f"  b_sıcak-çekirdek = {b_sicak:+.6f}")
    print(f"  b_soğuk          = {b_soguk:+.6f}")
    print(f"  b_kuyruk         = {b_kuyruk:+.6f}")
    print(f"  tahmini RMSLE    = {tahmini_score:.6f}")
    lider_hukmu = "GEÇİYOR" if tahmini_score < LIDER_RMSLE else "YETMİYOR"
    print(f"  lider            = {LIDER_RMSLE:.5f} -> {lider_hukmu}")

    if ar.cikis is None:
        ad = "tuketim_v84_tam_optimum.csv" if kuyruk_sonuc else "tuketim_v83_sicak_optimum.csv"
        ar.cikis = SUB / ad

    yeni = uc_rejim_deltasi_uygula(
        taban["tuketim"].to_numpy(),
        soguk=soguk,
        kuyruk=kuyruk,
        b_soguk=b_soguk,
        b_kuyruk=b_kuyruk,
        b_sicak_cekirdek=b_sicak,
    )
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("nihai tahminde NaN/negatif")
    ar.cikis.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": taban["id"], "tuketim": yeni}).to_csv(ar.cikis, index=False)

    rapor = {
        "skorlar": {
            "banka": banka_score,
            "sicak_prob": sicak_score,
            "kuyruk_prob": ar.kuyruk_prob_score,
        },
        "sicak_prob": asdict(sicak_sonuc),
        "kuyruk_prob": asdict(kuyruk_sonuc) if kuyruk_sonuc else None,
        "deltalar": {"sicak_cekirdek": b_sicak, "soguk": b_soguk, "kuyruk": b_kuyruk},
        "tahmini_mse": tahmini_mse,
        "tahmini_rmsle": tahmini_score,
        "lider_rmsle": LIDER_RMSLE,
        "cikis": str(ar.cikis),
    }
    ar.rapor.parent.mkdir(parents=True, exist_ok=True)
    ar.rapor.write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  yazıldı          = {ar.cikis}")
    print(f"  rapor            = {ar.rapor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
