"""EKSEN 2b -- SABIT SEVIYE KAYMASI delta icin YANLILIK b'nin OLCUMU.

Bu betik SAYI URETIR, hukum vermez. Dort soru:
  (a1) kis26 TUM blok yanliligi (satir / trafo / test-agirlikli)
  (a2) yalnizca 2026 Sub-Mar; ay ay ayrisim
  (a3) ufuk profili -- b(ufuk) egrisi ve test ufkuna izdusum
  (b)  soguk satirlarda yanlilik (soguk_tahmin_kis26.npz)
  (c)  optimal delta ve dMSE(delta) egrisi

Kural 1 geregi her sayi TRAFO BAZINDA ayristirilir ve en buyuk K trafo
atilinca tablo verilir.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut  # noqa: E402
import tuketim_model as tm  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
SICAK_ONB = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
SOGUK_ONB = KOK / "data" / "interim" / "deney" / "soguk_tahmin_kis26.npz"
MSE0 = 1.03207
SICAK_PAY = 0.7784


def harman(z, blok: str, n: int) -> np.ndarray:
    pay = sum(AGIRLIK)
    loglar = [
        sum(AGIRLIK[i] * z[f"{blok}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
        for t in di.TOHUMLAR
    ]
    out = np.mean(loglar, axis=0)
    assert out.shape == (n,), (out.shape, n)
    return out


def harman_soguk(z, n: int) -> np.ndarray:
    pay = sum(AGIRLIK)
    loglar = [
        sum(AGIRLIK[i] * z[f"{t}_{a}"] for i, a in enumerate(AILELER)) / pay for t in di.TOHUMLAR
    ]
    out = np.mean(loglar, axis=0)
    assert out.shape == (n,), (out.shape, n)
    return out


def trafo_tablo(tanim: np.ndarray, b: np.ndarray, ad: str) -> None:
    """Trafo bazinda ortalama yanlilik; en buyuk K trafo atilinca ne kaliyor."""
    df = pd.DataFrame({"t": tanim, "b": b})
    g = df.groupby("t")["b"]
    ort = g.mean()
    n = g.size()
    # satir-agirlikli ortalama = sum(n_i * ort_i) / sum(n_i)
    print(f"  {ad}: {len(ort):,} trafo, {len(df):,} satir")
    print(
        f"    satir-ort {df['b'].mean():+.4f}   trafo-ort {ort.mean():+.4f}   "
        f"trafo-med {ort.median():+.4f}   trafo-std {ort.std(ddof=1):.4f}   "
        f"POZ %{100 * (ort > 0).mean():.1f}"
    )
    sira = ort.abs().sort_values(ascending=False).index
    satir = []
    for K in (0, 1, 5, 10, 25, 50):
        kalan = ort.drop(sira[:K]) if K else ort
        nk = n.reindex(kalan.index)
        satir.append(f"K={K}:{float((kalan * nk).sum() / nk.sum()):+.4f}")
    print("    en buyuk K trafo atilinca satir-agirlikli b -> " + "  ".join(satir))


def main() -> int:
    t0 = time.time()
    print("=" * 100)
    print("EKSEN 2b -- SEVIYE YANLILIGI b'nin BAGIMSIZ OLCUMLERI")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(SICAK_ONB)

    # ---------------- kis26 SICAK ----------------
    _, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dog[~soguk].reset_index(drop=True)
    y = gercek[~soguk]
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    r = harman(z, "kis26", int((~soguk).sum())) - lg
    g_ofs = np.log1p(y) - lg
    b = g_ofs - r
    tar = pd.to_datetime(dg["tarih"])
    ay = tar.dt.to_period("M").astype(str).to_numpy()
    ufuk = dg["ufuk_gun"].to_numpy(dtype="float64")
    tanim = dg["tanim"].to_numpy()

    print("\n### (a1) kis26 TUM BLOK, SICAK")
    trafo_tablo(tanim, b, "kis26 tum")

    print("\n### (a2) AY AY")
    print(f"  {'ay':10}{'n':>10}{'ufuk':>12}{'satir-b':>10}{'trafo-b':>10}{'m0':>9}{'trafo':>8}")
    for a in sorted(set(ay)):
        m = ay == a
        tb = pd.DataFrame({"t": tanim[m], "b": b[m]}).groupby("t")["b"].mean()
        print(
            f"  {a:10}{int(m.sum()):10,}{ufuk[m].min():6.0f}-{ufuk[m].max():<5.0f}"
            f"{b[m].mean():+10.4f}{tb.mean():+10.4f}{(g_ofs[m] - r[m]).var() + b[m].mean() ** 2:9.4f}"
            f"{tb.size:8,}"
        )

    sm = np.isin(ay, ["2026-02", "2026-03"])
    print("\n  --- yalniz 2026 Sub-Mar (mevsimsel ikizi EGITIM ETIKETLERINDE olan kisim) ---")
    trafo_tablo(tanim[sm], b[sm], "kis26 Sub-Mar")
    print("\n  --- yalniz 2025 Ara-Oca (ikizi YOK; mevsim ekstrapolasyonu) ---")
    am = np.isin(ay, ["2025-12", "2026-01"])
    trafo_tablo(tanim[am], b[am], "kis26 Ara-Oca")

    # AYNI TRAFO KUMESI uzerinde ay karsilastirmasi (karisim etkisini ayikla)
    ort_ay = pd.DataFrame({"t": tanim, "a": ay, "b": b}).pivot_table(
        index="t", columns="a", values="b", aggfunc="mean"
    )
    tam = ort_ay.dropna()
    print(f"\n  AYNI {len(tam):,} trafo, her ayda kaydi olanlar (karisim sabit):")
    print("    " + "  ".join(f"{c}={tam[c].mean():+.4f}" for c in tam.columns))

    # ---------------- (a3) UFUK PROFILI ----------------
    print("\n### (a3) UFUK PROFILI  b(ufuk)")
    # trafo etkisi cikarilmis: her trafonun kendi ortalamasindan sapma + genel ort
    df = pd.DataFrame({"t": tanim, "u": ufuk, "b": b, "ay": ay})
    genel = b.mean()
    icsel = b - df.groupby("t")["b"].transform("mean").to_numpy() + genel
    kenar = [0, 15, 31, 46, 62, 77, 92, 107, 122]
    kod = np.clip(np.searchsorted(kenar[1:-1], ufuk, side="right"), 0, len(kenar) - 2)
    print(f"  {'dilim':14}{'n':>10}{'ham b':>10}{'trafo-ici b':>13}")
    for j in range(len(kenar) - 1):
        m = kod == j
        if not m.any():
            continue
        print(
            f"  {kenar[j] + 1:>3}-{kenar[j + 1]:<10}{int(m.sum()):10,}"
            f"{b[m].mean():+10.4f}{icsel[m].mean():+13.4f}"
        )
    eg, ke = np.polyfit(ufuk, b, 1)
    print(
        f"  dogrusal uydurma  b = {ke:+.4f} {eg:+.6f} * ufuk"
        f"   -> ufuk 61'de {ke + eg * 61:+.4f}, ufuk 1'de {ke + eg:+.4f},"
        f" ufuk 122'de {ke + eg * 122:+.4f}"
    )
    # SADECE Sub-Mar icinde ufuk egimi (mevsim sabitken)
    for etiket, msk in (
        ("2026-02", ay == "2026-02"),
        ("2026-03", ay == "2026-03"),
        ("Sub-Mar", sm),
    ):
        e2, k2 = np.polyfit(ufuk[msk], b[msk], 1)
        print(
            f"  {etiket:8} ic-ufuk egimi {e2:+.6f}/gun  (ufuk {ufuk[msk].min():.0f}"
            f"-{ufuk[msk].max():.0f}, uc-uca fark {e2 * (ufuk[msk].max() - ufuk[msk].min()):+.4f})"
        )

    # ---------------- TEST-AGIRLIKLI ----------------
    print("\n### TEST-AGIRLIKLI (olcut.py: bayatlik x kVA x ufuk)")
    tr_t = pd.read_csv(
        KOK / "data/raw/train.csv", usecols=["tanim"], encoding="utf-8", dtype={"tanim": str}
    )
    te_h = pd.read_csv(KOK / "data/raw/test.csv", encoding="utf-8", dtype={"tanim": str})
    sicak_te = test[test["tanim"].isin(set(tr_t["tanim"]))]
    ke_g = olcut.guc_kenarlari(test)
    for etiket, msk in (("tum blok", np.ones(len(b), bool)), ("Sub-Mar", sm)):
        alt = dg[msk]
        w, tani = olcut.test_agirliklari(alt, sicak_te, ke_g)
        print(
            f"  {etiket:10} b_agirlikli {float(np.dot(w, b[msk]) / w.sum()):+.4f}"
            f"   (ham {b[msk].mean():+.4f})  ESS {tani['ess_orani']:.3f}"
            f" kapsanmayan {tani['kapsanmayan']:.3f} guvenilir {tani['guvenilir']}"
        )
        for eks in (("bayatlik",), ("guc",), ("ufuk",)):
            w2, t2 = olcut.test_agirliklari(alt, sicak_te, ke_g, eksenler=eks)
            print(
                f"      yalniz {eks[0]:9} -> {float(np.dot(w2, b[msk]) / w2.sum()):+.4f}"
                f"  ESS {t2['ess_orani']:.3f}"
            )

    # ---------------- (b) SOGUK ----------------
    print("\n### (b) SOGUK satirlar, kis26")
    zs = np.load(SOGUK_ONB)
    ds = dog[soguk].reset_index(drop=True)
    ys = gercek[soguk]
    lgs = np.log1p(ds["guc"].to_numpy(dtype="float64"))
    rs = harman_soguk(zs, int(soguk.sum())) - lgs
    bs = np.log1p(ys) - lgs - rs
    ays = pd.to_datetime(ds["tarih"]).dt.to_period("M").astype(str).to_numpy()
    trafo_tablo(ds["tanim"].to_numpy(), bs, "soguk tum")
    print(f"  {'ay':10}{'n':>10}{'satir-b':>10}{'trafo-b':>10}")
    for a in sorted(set(ays)):
        m = ays == a
        tb = pd.DataFrame({"t": ds["tanim"].to_numpy()[m], "b": bs[m]}).groupby("t")["b"].mean()
        print(f"  {a:10}{int(m.sum()):10,}{bs[m].mean():+10.4f}{tb.mean():+10.4f}")
    sms = np.isin(ays, ["2026-02", "2026-03"])
    trafo_tablo(ds["tanim"].to_numpy()[sms], bs[sms], "soguk Sub-Mar")
    print(
        f"  soguk tahmin dagilimi: trafo-ort std "
        f"{pd.DataFrame({'t': ds['tanim'].to_numpy(), 'r': rs}).groupby('t')['r'].mean().std():.4f}"
    )

    # ---------------- (c) dMSE EGRISI ----------------
    print("\n### (c) dMSE(delta) -- p = sicak pay 0,7784")
    print(
        f"  {'delta':>7}"
        + "".join(f"{f'b={bb:.2f}':>12}" for bb in (0.00, 0.05, 0.08, 0.11, 0.15, 0.19))
    )
    for de in (0.00, 0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
        s = f"  {de:7.2f}"
        for bb in (0.00, 0.05, 0.08, 0.11, 0.15, 0.19):
            dm = SICAK_PAY * (de**2 - 2 * de * bb)
            s += f"{np.sqrt(MSE0 + dm):12.5f}"
        print(s)
    print("  basabas delta (dMSE=0) = 2*b ; her delta icin gereken en kucuk b = delta/2")

    print(f"\nTAMAM {(time.time() - t0) / 60:.1f} dk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
