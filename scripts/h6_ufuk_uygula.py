# ruff: noqa
"""H6 (c) -- UFUK DUZELTMESINI UYGULA ve dMSE OLC (test karisimina agirlikli).

Iki aday duzeltme:
  (A) GUVENLI EGIM  b = +0,000330/gun  -- gun sabit etkili kestirim
      (h6_ufuk_gun_sabit_etki.py §2). MEVSIMDEN ARINDIRILMIS tek sayi.
  (B) KOVA PROFILI  c(kova) -- ayni betigin §4 gunFE profili.

Her ikisi de SICAK tarafa uygulanir (soguk trafonun bayatlayacak capasi yok).
Sonuc uc blokta AYRI verilir; havuzlanmis sayiya guvenilmez.

Ek: SOGUK tarafta kis26 ufuk egrisi (BILGI AMACLI -- kural 9, hukum vermez).

    uv run python scripts/h6_ufuk_uygula.py
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
from olcut import agirlikli_rmsle, guc_kenarlari, test_agirliklari  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
SICAK_ONB = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
SOGUK_ONB = KOK / "data" / "interim" / "deney" / "soguk_tahmin_kis26.npz"
CIKTI = KOK / "reports" / "h6_ufuk"

#: gun sabit etkili SAF ufuk egimi (h6_ufuk_gun_sabit_etki.py §2)
GUVENLI_EGIM = 0.000330
#: gunFE kova profili (§4), referans kova 1-10
KOVA_PROFIL = np.array(
    [
        0.0000,
        0.0044,
        0.0055,
        0.0172,
        0.0195,
        0.0201,
        0.0252,
        0.0281,
        0.0292,
        0.0306,
        0.0355,
        0.0354,
        0.0388,
    ]
)
P_SICAK = 0.77841
P_SOGUK = 0.22159


def main() -> int:
    t0 = time.time()
    CIKTI.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("H6 (c) -- UFUK DUZELTMESI UYGULAMASI")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(SICAK_ONB)
    gk = guc_kenarlari(test)
    te_ufuk = test["ufuk_gun"].to_numpy(dtype="float64")
    te_kova = np.clip(te_ufuk.astype("int64") // 10, 0, 12)
    print(
        f"  TEST ufuk ortalamasi {te_ufuk.mean():.2f}  kova dagilimi {np.bincount(te_kova) / len(te_kova)}"
    )

    sonuc = []
    print("\n" + "-" * 96)
    print("SICAK TARAF -- dMSE (blok bazli, hem duz hem test-agirlikli)")
    print("-" * 96)
    print(
        f"  {'blok':8}{'n':>9}{'MSE_once':>11}{'A:egim':>11}{'dMSE_A':>11}{'B:profil':>11}{'dMSE_B':>11}{'ESS':>7}"
    )
    for b in tm.BLOKLAR:
        dogrulama = egitim[egitim["_blok"] == b.ad]
        gercek = dogrulama[tm.HEDEF].to_numpy()
        soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_t = np.mean(loglar, axis=0)
        dg = dogrulama[~soguk]
        y = gercek[~soguk]
        u = dg["ufuk_gun"].to_numpy(dtype="float64")
        kova = np.clip(u.astype("int64") // 10, 0, 12)
        r = np.log1p(y) - log_t

        w, tani = test_agirliklari(dg, test, gk)

        # (A) guvenli egim, TEST ufuk ortalamasina gore merkezli
        cA = GUVENLI_EGIM * (u - u.mean())  # BLOK ORTALAMASINA merkezli -> SAF EGIM, sabit yok
        # (B) kova profili, TEST kova dagilimina gore merkezli
        cB = KOVA_PROFIL[kova] - float(KOVA_PROFIL[kova].mean())  # SAF SEKIL, sabit yok

        mse0 = float(np.mean(r**2))
        mseA = float(np.mean((r - cA) ** 2))
        mseB = float(np.mean((r - cB) ** 2))
        # test-agirlikli
        wm0 = float(np.dot(w, r**2) / w.sum())
        wmA = float(np.dot(w, (r - cA) ** 2) / w.sum())
        wmB = float(np.dot(w, (r - cB) ** 2) / w.sum())
        print(
            f"  {b.ad:8}{len(y):9,}{mse0:11.6f}{mseA:11.6f}{mseA - mse0:+11.6f}"
            f"{mseB:11.6f}{mseB - mse0:+11.6f}{tani['ess_orani']:7.2f}"
        )
        print(
            f"  {'':8}{'agirlikli':>9}{wm0:11.6f}{wmA:11.6f}{wmA - wm0:+11.6f}"
            f"{wmB:11.6f}{wmB - wm0:+11.6f}   guvenilir={tani['guvenilir']}"
        )
        sonuc.append(
            {
                "blok": b.ad,
                "n": int(len(y)),
                "mse0": mse0,
                "dA": mseA - mse0,
                "dB": mseB - mse0,
                "w_mse0": wm0,
                "w_dA": wmA - wm0,
                "w_dB": wmB - wm0,
                "ess": tani["ess_orani"],
            }
        )

    print("\n  BLOK KIRILIMI (agirlikli dMSE, sicak taraf):")
    for s in sonuc:
        print(f"    {s['blok']:8} A {s['w_dA']:+.6f}   B {s['w_dB']:+.6f}")
    ort_A = float(np.mean([s["w_dA"] for s in sonuc]))
    ort_B = float(np.mean([s["w_dB"] for s in sonuc]))
    kaz_A = sum(1 for s in sonuc if s["w_dA"] < 0)
    kaz_B = sum(1 for s in sonuc if s["w_dB"] < 0)
    print(f"    ORT   A {ort_A:+.6f} ({kaz_A}/3 blok)   B {ort_B:+.6f} ({kaz_B}/3 blok)")
    print(f"    yaz25 HARIC (mevsim tutulmasi artifakti):")
    ha = float(np.mean([s["w_dA"] for s in sonuc if s["blok"] != "yaz25"]))
    hb = float(np.mean([s["w_dB"] for s in sonuc if s["blok"] != "yaz25"]))
    print(f"      A {ha:+.6f}   B {hb:+.6f}")
    print(
        f"    genel dMSE tahmini (p_sicak={P_SICAK}): A {ort_A * P_SICAK:+.6f}  B {ort_B * P_SICAK:+.6f}"
    )

    # ---------------------------------------------------------------- SOGUK
    print("\n" + "-" * 96)
    print("SOGUK TARAF -- kis26 ufuk egrisi  (BILGI AMACLI, kural 9: HUKUM VERMEZ)")
    print("  yaz25/guz25 icin onbelleklenmis SOGUK tahmin YOK -> mevsimsel hukum IMKANSIZ")
    print("-" * 96)
    if SOGUK_ONB.exists():
        zs = np.load(SOGUK_ONB)
        dogrulama = egitim[egitim["_blok"] == "kis26"]
        soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
        ds = dogrulama[soguk]
        ys = dogrulama[tm.HEDEF].to_numpy()[soguk]
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * zs[f"{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_s = np.mean(loglar, axis=0)
        rs = np.log1p(ys) - log_s
        us = ds["ufuk_gun"].to_numpy(dtype="float64")
        ks = np.clip(us.astype("int64") // 10, 0, 12)
        print(f"  n {len(ys):,}  {ds['tanim'].nunique():,} soguk trafo")
        print(f"  {'kova':>9}{'r_ort':>11}{'n':>10}")
        for j in range(13):
            m = ks == j
            if not m.any():
                continue
            print(
                f"  {j * 10 + 1:>4}-{min((j + 1) * 10, 122):<4}{rs[m].mean():+11.4f}{int(m.sum()):10,}"
            )
        bs = np.polyfit(us, rs, 1)[0]
        print(f"  kis26 soguk egim b = {bs:+.6f}   122g {bs * 121:+.4f}")

    (CIKTI / "uygulama.json").write_text(
        json.dumps({"sicak": sonuc, "ort_A": ort_A, "ort_B": ort_B}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
