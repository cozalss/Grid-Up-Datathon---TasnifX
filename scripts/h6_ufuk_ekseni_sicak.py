# ruff: noqa
"""H6 -- UFUK EKSENI (SICAK): hata ~ kesmeden gecen gun sayisi.

r(ufuk) = ortalama[ log1p(gercek) - log1p(tahmin) ]   ufuk = 1..122

UFUK ile MEVSIM bu veride birebir karisik: bir blok icinde ufuk = takvim gunu.
Ayirmanin TEK yolu iki blok:
    yaz25  ufuk 1 = 1 Nisan     ufuk 122 = 31 Temmuz
    guz25  ufuk 1 = 1 Agustos   ufuk 122 = 30 Kasim
Ayni isaret + benzer buyukluk  -> etki UFUK.
Ters isaret                    -> etki MEVSIM, duzeltme yanlis -> CURUDU.

Onbelleklenmis sicak tahminler (deney/sicak_tahmin.npz, cat*3+xgb+lgbm,
3 tohum torbalanmis) kullanilir -- fit YOK.

    uv run python scripts/h6_ufuk_ekseni_sicak.py
"""

from __future__ import annotations

import json
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
import tuketim_model as tm  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
CIKTI = KOK / "reports" / "h6_ufuk"


def gun_kumeli_egim(u: np.ndarray, r: np.ndarray, gun: np.ndarray) -> tuple[float, float, float]:
    """r ~ a + b*u dogrusal uydurma; SE GUN kumeli (ayni gun artiklari bagimli).

    Doner: (b, se_b, a)
    """
    X = np.column_stack([np.ones_like(u), u])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ r)
    e = r - X @ beta
    # kume-saglam (CR0) sandvic
    meat = np.zeros((2, 2))
    for g in np.unique(gun):
        m = gun == g
        Xg = X[m]
        sg = Xg.T @ e[m]
        meat += np.outer(sg, sg)
    V = XtX_inv @ meat @ XtX_inv
    return float(beta[1]), float(np.sqrt(V[1, 1])), float(beta[0])


def main() -> int:
    t0 = time.time()
    CIKTI.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("H6 -- UFUK EKSENI, SICAK REJIM")
    print("=" * 96)

    if not ONBELLEK.exists():
        raise RuntimeError(f"onbellek yok: {ONBELLEK}")

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(ONBELLEK)

    veri: dict[str, dict[str, np.ndarray]] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = parcala(egitim, b.ad)
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_t = np.mean(loglar, axis=0)  # log1p(tahmin)
        dg = dogrulama[~soguk]
        y = gercek[~soguk]
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        veri[b.ad] = {
            "r": np.log1p(y) - log_t,  # ARTIK: gercek - tahmin (log uzayi)
            "log_t": log_t,
            "y": y,
            "lg": lg,
            "ufuk": dg["ufuk_gun"].to_numpy(dtype="float64"),
            "tanim": dg["tanim"].to_numpy(),
            "tarih": dg["tarih"].to_numpy(),
        }
        print(f"  {b.ad}: {len(y):,} sicak satir  {len(np.unique(veri[b.ad]['tanim'])):,} trafo")

    te_ufuk = test["ufuk_gun"].to_numpy(dtype="float64")
    print(f"  TEST ufku {te_ufuk.min():.0f}-{te_ufuk.max():.0f}  n={len(te_ufuk):,}")

    # ------------------------------------------------ 1) 10 gunluk kova egrileri
    print("\n" + "-" * 96)
    print("1) r(ufuk) EGRILERI -- 10 gunluk kovalar  [ham | trafo-etkisi-cikarilmis]")
    print("-" * 96)
    kenar = np.arange(0, 131, 10)
    satirlar = []
    print(f"  {'kova':>9}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad + ' ham':>13}{b.ad + ' dm':>13}", end="")
    print()
    tablo = {}
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        # trafo etkisi cikarma: her trafonun blok-ici ortalama artigini dus
        s = pd.Series(v["r"])
        tr_ort = s.groupby(pd.Series(v["tanim"])).transform("mean").to_numpy()
        v["r_dm"] = v["r"] - tr_ort
        k = np.clip(np.searchsorted(kenar, v["ufuk"], side="right") - 1, 0, len(kenar) - 2)
        v["kova"] = k
        tablo[b.ad] = {
            "ham": np.array(
                [v["r"][k == j].mean() if (k == j).any() else np.nan for j in range(len(kenar) - 1)]
            ),
            "dm": np.array(
                [
                    v["r_dm"][k == j].mean() if (k == j).any() else np.nan
                    for j in range(len(kenar) - 1)
                ]
            ),
            "n": np.array([int((k == j).sum()) for j in range(len(kenar) - 1)]),
        }
    for j in range(len(kenar) - 1):
        if all(tablo[b.ad]["n"][j] == 0 for b in tm.BLOKLAR):
            continue
        etiket = f"{kenar[j] + 1}-{kenar[j + 1]}"
        print(f"  {etiket:>9}", end="")
        sat = {"kova": etiket}
        for b in tm.BLOKLAR:
            h = tablo[b.ad]["ham"][j]
            dm = tablo[b.ad]["dm"][j]
            print(f"{h:+13.4f}{dm:+13.4f}", end="")
            sat[f"{b.ad}_ham"] = h
            sat[f"{b.ad}_dm"] = dm
            sat[f"{b.ad}_n"] = int(tablo[b.ad]["n"][j])
        print()
        satirlar.append(sat)
    pd.DataFrame(satirlar).to_csv(CIKTI / "sicak_ufuk_egrileri.csv", index=False)

    # ------------------------------------------------ 2) dogrusal egim + gun kumeli SE
    print("\n" + "-" * 96)
    print("2) DOGRUSAL EGIM  r = a + b*ufuk   (SE gun-kumeli)")
    print("-" * 96)
    print(
        f"  {'blok':8}{'n':>10}{'b_ham':>11}{'se':>9}{'t':>8}{'b_dm':>11}{'se':>9}{'t':>8}{'122g_etki':>11}"
    )
    egimler = {}
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        gun = v["ufuk"].astype("int64")
        bh, sh, ah = gun_kumeli_egim(v["ufuk"], v["r"], gun)
        bd, sd, ad = gun_kumeli_egim(v["ufuk"], v["r_dm"], gun)
        egimler[b.ad] = {"b_ham": bh, "se_ham": sh, "a_ham": ah, "b_dm": bd, "se_dm": sd}
        print(
            f"  {b.ad:8}{len(v['r']):10,}{bh:+11.6f}{sh:9.6f}{bh / sh:+8.2f}"
            f"{bd:+11.6f}{sd:9.6f}{bd / sd:+8.2f}{bh * 121:+11.4f}"
        )

    # ------------------------------------------------ 3) ISARET HUKMU
    print("\n" + "-" * 96)
    print("3) ISARET TUTARLILIGI  (yaz25 vs guz25 -- MEVSIM/UFUK AYRIMI)")
    print("-" * 96)
    by = egimler["yaz25"]["b_ham"]
    bg = egimler["guz25"]["b_ham"]
    bk = egimler["kis26"]["b_ham"]
    print(
        f"  yaz25 b = {by:+.6f}   guz25 b = {bg:+.6f}   kis26 b = {bk:+.6f}  (kis26 HUKUM VERMEZ)"
    )
    if np.sign(by) == np.sign(bg):
        oran = max(abs(by), abs(bg)) / max(min(abs(by), abs(bg)), 1e-12)
        print(f"  ISARETLER AYNI. buyukluk orani {oran:.2f}x")
        print("  -> etki UFUK olabilir; transfer testine gec.")
    else:
        print("  ISARETLER TERS -> etki MEVSIM. Ufuk duzeltmesi YANLIS.")

    # ------------------------------------------------ 4) CAPRAZ BLOK TRANSFER TESTI
    print("\n" + "-" * 96)
    print("4) CAPRAZ BLOK TRANSFER  -- egim kaynak blokta uydurulur, hedefe uygulanir")
    print("   duzeltme: log_t' = log_t + (a_k + b_k*ufuk)      [ARTIK ORTALAMASI EKLENIR]")
    print("-" * 96)
    kayitlar = []
    for hedef in tm.BLOKLAR:
        v = veri[hedef.ad]
        onceki_mse = float(np.mean(v["r"] ** 2))
        onceki = float(np.sqrt(onceki_mse))
        for kaynak in tm.BLOKLAR:
            if kaynak.ad == hedef.ad:
                continue
            vk = veri[kaynak.ad]
            a_k, b_k = np.polyfit(vk["ufuk"], vk["r"], 1)[::-1]
            duzeltme = a_k + b_k * v["ufuk"]
            yeni_r = v["r"] - duzeltme
            yeni_mse = float(np.mean(yeni_r**2))
            # sadece EGIM (kesme atilir -- kesme = sabit delta, YASAK BOLGE)
            duz_egim = b_k * (v["ufuk"] - vk["ufuk"].mean())
            egim_mse = float(np.mean((v["r"] - duz_egim) ** 2))
            print(
                f"  {kaynak.ad} -> {hedef.ad}:  MSE {onceki_mse:.6f}"
                f" | tam(a+b) {yeni_mse:.6f} ({yeni_mse - onceki_mse:+.6f})"
                f" | yalniz-egim {egim_mse:.6f} ({egim_mse - onceki_mse:+.6f})"
            )
            kayitlar.append(
                {
                    "kaynak": kaynak.ad,
                    "hedef": hedef.ad,
                    "mse_once": onceki_mse,
                    "mse_tam": yeni_mse,
                    "d_tam": yeni_mse - onceki_mse,
                    "mse_egim": egim_mse,
                    "d_egim": egim_mse - onceki_mse,
                }
            )
    pd.DataFrame(kayitlar).to_csv(CIKTI / "sicak_capraz_transfer.csv", index=False)

    # ------------------------------------------------ 5) ORAKUL TAVANI
    print("\n" + "-" * 96)
    print("5) ORAKUL TAVANI -- egim BLOGUN KENDISINDE uydurulursa (ust sinir, ulasilamaz)")
    print("-" * 96)
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        a, bb = np.polyfit(v["ufuk"], v["r"], 1)[::-1]
        onceki_mse = float(np.mean(v["r"] ** 2))
        tam = float(np.mean((v["r"] - (a + bb * v["ufuk"])) ** 2))
        egim = float(np.mean((v["r"] - bb * (v["ufuk"] - v["ufuk"].mean())) ** 2))
        print(
            f"  {b.ad}:  MSE {onceki_mse:.6f} -> orakul-tam {tam:.6f} ({tam - onceki_mse:+.6f})"
            f"  orakul-yalniz-egim {egim:.6f} ({egim - onceki_mse:+.6f})"
        )

    ozet = {
        "egimler": egimler,
        "capraz": kayitlar,
    }
    (CIKTI / "sicak_ozet.json").write_text(
        json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika  -> {CIKTI}")
    return 0


def parcala(egitim: pd.DataFrame, blok: str):
    dogrulama = egitim[egitim["_blok"] == blok]
    return (
        None,
        dogrulama,
        dogrulama[tm.HEDEF].to_numpy(),
        (dogrulama["soguk_mu"] == 1).to_numpy(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
