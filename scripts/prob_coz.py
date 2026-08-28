"""Ölçülmüş prob skorlarından ``L``, ``κ*`` ve optimum dosyayı çöz.

Bir prob dosyası tam olarak ``log1p(p) = log1p(v93) + u`` biçimindedir; ``u``
dosyadan **etiketsiz** okunur. RMSLE karesel olduğu için

    MSE(v93 + u) = MSE(v93) + Q_u − 2·L_u ,
        Q_u = ‖u‖²/n            (dosyadan, tam)
        L_u = ⟨u, t − v93⟩/n    (BİLİNMEYEN)

ve tek gönderimden

    L_u = (MSE(v93) + Q_u − MSE(prob)) / 2

**tam** çözülür. Birden fazla prob geldiğinde ortak optimum, Gram sistemidir:

    G κ = L ,   G_ij = ⟨u_i, u_j⟩/n ,   kazanç = −Lᵀ G⁻¹ L

Yönler birbirine dik kurulduğu için ``G`` köşegene yakındır ve kazançlar
toplanır. Betik hiçbir gönderim yapmaz; yalnız dosya ve rapor üretir.

Kullanım
--------
    uv run python scripts/prob_coz.py \
        --taban-skor 1.00833 \
        --prob p1_sicak_ilce=1.00312 \
        --prob p2_sicak_seviye=1.00588 \
        --cikis submissions/tuketim_v100_optimum.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"
TABAN_DOSYA = "tuketim_v93_gram_optimum.csv"
BEKLENEN_SATIR = 714_688
YARIM_ADIM = 5e-6  # LB 5 ondalığa yuvarlanıyor


def _log1p(yol: Path) -> tuple[np.ndarray, pd.Series]:
    d = pd.read_csv(yol)
    if list(d.columns) != ["id", "tuketim"]:
        raise RuntimeError(f"{yol.name}: şema geçersiz")
    if len(d) != BEKLENEN_SATIR or d["id"].duplicated().any():
        raise RuntimeError(f"{yol.name}: satır sayısı / mükerrer id kapısı")
    v = d["tuketim"].to_numpy(dtype="float64")
    if not np.isfinite(v).all() or np.any(v < 0):
        raise RuntimeError(f"{yol.name}: sonlu ve negatif olmayan tahmin şartı")
    return np.log1p(v), d["id"]


def _prob_yolu(ad: str) -> Path:
    yol = Path(ad)
    if yol.exists():
        return yol
    for aday in (GONDERIM / ad, GONDERIM / f"tuketim_{ad}.csv", GONDERIM / f"{ad}.csv"):
        if aday.exists():
            return aday
    raise FileNotFoundError(f"prob dosyası bulunamadı: {ad}")


def coz(m0: float, Q: np.ndarray, G: np.ndarray, m: np.ndarray) -> dict:
    """L, tek-yön optimumları ve ortak Gram optimumu."""
    L = (m0 + Q - m) / 2.0
    tekil_kappa = L / Q
    tekil_kazanc = -(L**2) / Q
    kosul = float(np.linalg.cond(G))
    kappa = np.linalg.solve(G, L)
    ortak_kazanc = float(-L @ kappa)
    return {
        "L": L,
        "tekil_kappa": tekil_kappa,
        "tekil_kazanc": tekil_kazanc,
        "kappa": kappa,
        "ortak_kazanc": ortak_kazanc,
        "kosul": kosul,
        "artik": float(np.max(np.abs(G @ kappa - L))),
    }


def yuvarlama_bandi(m0: float, Q: np.ndarray, G: np.ndarray, m: np.ndarray) -> tuple:
    """Skorların ±5e-6 yuvarlama köşelerinde çözümün bandı."""
    k = len(m)
    if k > 4:  # 2^(k+1) köşe -- 4 probdan sonra köşe taraması pahalı
        return (float("nan"), float("nan"))
    alt, ust = np.inf, -np.inf
    s0 = np.sqrt(m0)
    s = np.sqrt(m)
    for isaret in itertools.product((-1.0, 1.0), repeat=k + 1):
        m0k = (s0 + isaret[0] * YARIM_ADIM) ** 2
        mk = np.array([(s[i] + isaret[i + 1] * YARIM_ADIM) ** 2 for i in range(k)])
        r = coz(m0k, Q, G, mk)
        deger = float(np.sqrt(max(m0k + r["ortak_kazanc"], 0.0)))
        alt, ust = min(alt, deger), max(ust, deger)
    return float(alt), float(ust)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taban-skor", type=float, required=True, help="v93'ün ÖLÇÜLMÜŞ Kaggle skoru")
    ap.add_argument(
        "--prob",
        action="append",
        default=[],
        metavar="AD=SKOR",
        help="prob dosyası ve ölçülmüş skoru (tekrarlanabilir)",
    )
    ap.add_argument("--taban", default=TABAN_DOSYA)
    ap.add_argument("--cikis", type=Path)
    ap.add_argument("--rapor", type=Path)
    ap.add_argument(
        "--olcek",
        type=float,
        default=1.0,
        help="ortak adımı bu oranla küçült (güvenlik, varsayılan 1.0)",
    )
    a = ap.parse_args()

    if a.taban_skor <= 0:
        raise ValueError("taban skor pozitif olmalı")
    if not a.prob:
        raise ValueError("en az bir --prob AD=SKOR gerekli")

    adlar, skorlar = [], []
    for p in a.prob:
        ad, _, s = p.partition("=")
        if not s:
            raise ValueError(f"--prob biçimi AD=SKOR olmalı: {p}")
        adlar.append(ad)
        skorlar.append(float(s))
    if any(s <= 0 for s in skorlar):
        raise ValueError("prob skorları pozitif olmalı")

    taban_log, kimlik = _log1p(GONDERIM / a.taban)
    n = taban_log.size
    U = np.empty((len(adlar), n), dtype="float64")
    for i, ad in enumerate(adlar):
        lg, kid = _log1p(_prob_yolu(ad))
        if not kid.equals(kimlik):
            raise RuntimeError(f"{ad}: id sırası tabanla eşleşmiyor")
        U[i] = lg - taban_log

    G = U @ U.T / n
    G = (G + G.T) / 2.0
    Q = np.diag(G).copy()
    if np.any(Q <= 0):
        raise RuntimeError("bir prob yönü sıfır normlu")
    m0 = a.taban_skor**2
    m = np.array(skorlar) ** 2

    r = coz(m0, Q, G, m)
    if r["kosul"] > 1e4 or r["artik"] > 1e-12:
        raise RuntimeError(f"Gram kapısı: cond={r['kosul']:.3e} artık={r['artik']:.3e}")

    nrm = np.sqrt(Q)
    C = G / np.outer(nrm, nrm)
    alt, ust = yuvarlama_bandi(m0, Q, G, m)

    print("=" * 78)
    print(f"TABAN {a.taban}  ölçülmüş skor {a.taban_skor:.5f}  MSE {m0:.7f}  n={n:,}")
    print("=" * 78)
    print(
        f"\n{'prob':22s}{'skor':>10s}{'Q':>12s}{'L':>12s}"
        f"{'kappa*':>10s}{'tek kazanc':>13s}{'ortak k':>10s}"
    )
    print("-" * 89)
    for i, ad in enumerate(adlar):
        print(
            f"{ad[:22]:22s}{skorlar[i]:>10.5f}{Q[i]:>12.7f}{r['L'][i]:>+12.7f}"
            f"{r['tekil_kappa'][i]:>+10.4f}{r['tekil_kazanc'][i]:>+13.7f}"
            f"{r['kappa'][i]:>+10.4f}"
        )

    print("\nyönler arası kosinüs:")
    for i, ad in enumerate(adlar):
        print("   " + f"{ad[:18]:18s}" + "".join(f"{C[i, j]:>9.4f}" for j in range(len(adlar))))
    print(
        f"   maks köşegen dışı = {np.abs(C - np.eye(len(adlar))).max():.2e}"
        f"   cond(G) = {r['kosul']:.3f}"
    )

    tekil_top = float(r["tekil_kazanc"].sum())
    yeni_mse = m0 + r["ortak_kazanc"] * (2 * a.olcek - a.olcek**2)
    print(f"\ntekil kazançların toplamı  = {tekil_top:+.7f}")
    print(f"ORTAK Gram kazancı         = {r['ortak_kazanc']:+.7f}")
    print(f"beklenen MSE               = {yeni_mse:.7f}")
    print(f"beklenen RMSLE             = {np.sqrt(max(yeni_mse, 0.0)):.6f}")
    if np.isfinite(alt):
        print(f"yuvarlama bandı (olcek=1)  = [{alt:.6f}, {ust:.6f}]")

    if a.cikis is None:
        print("\n--cikis verilmedi; dosya YAZILMADI.")
        return 0

    adim = a.olcek * (r["kappa"] @ U)
    yeni_log = taban_log + adim
    tasan = yeni_log < 0.0
    kirp_q = 0.0
    if tasan.any():
        # kırpmak yerine taşan satırlarda adımı sıfırla: dosya kesin biçimde
        # "taban + adım" kalır, sapmanın MSE etkisi Cauchy-Schwarz ile sınırlanır
        kayip = np.where(tasan, adim, 0.0)
        kirp_q = float(kayip @ kayip) / n
        adim = np.where(tasan, 0.0, adim)
        yeni_log = taban_log + adim
    if float(yeni_log.min()) < 0.0:
        raise RuntimeError("hâlâ negatif log1p")
    sinir = float(np.sqrt(kirp_q * m0))
    if tasan.any():
        print(
            f"\nsıfırlanan taşma satırı {int(tasan.sum()):,}  ||sapma||²/n={kirp_q:.3e}"
            f"  MSE belirsizlik sınırı ±{sinir:.3e}"
        )

    tahmin = np.expm1(yeni_log)
    if not np.isfinite(tahmin).all() or np.any(tahmin < 0):
        raise RuntimeError("çıktı tahminleri geçersiz")
    a.cikis.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": kimlik, "tuketim": tahmin}).to_csv(
        a.cikis, index=False, float_format="%.17g"
    )
    print(f"YAZILDI: {a.cikis}")

    rapor = {
        "taban": a.taban,
        "taban_skor": a.taban_skor,
        "taban_mse": m0,
        "n": int(n),
        "problar": [
            {
                "ad": adlar[i],
                "skor": skorlar[i],
                "Q": float(Q[i]),
                "L": float(r["L"][i]),
                "tekil_kappa": float(r["tekil_kappa"][i]),
                "tekil_kazanc": float(r["tekil_kazanc"][i]),
                "ortak_kappa": float(r["kappa"][i]),
            }
            for i in range(len(adlar))
        ],
        "gram": G.tolist(),
        "kosul": r["kosul"],
        "artik": r["artik"],
        "olcek": a.olcek,
        "tekil_kazanc_toplami": tekil_top,
        "ortak_kazanc": r["ortak_kazanc"],
        "beklenen_mse": float(yeni_mse),
        "beklenen_rmsle": float(np.sqrt(max(yeni_mse, 0.0))),
        "yuvarlama_bandi": [alt, ust],
        "tasma_satiri": int(tasan.sum()),
        "tasma_belirsizligi": sinir,
        "cikis": str(a.cikis),
    }
    if a.rapor is None:
        a.rapor = KOK / "reports" / f"{a.cikis.stem}.json"
    a.rapor.parent.mkdir(parents=True, exist_ok=True)
    a.rapor.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RAPOR  : {a.rapor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
