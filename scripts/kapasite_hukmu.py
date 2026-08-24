"""KAPASITE HUKMU -- eslenik standart hata ile, toplu tabloya guvenmeden.

NEDEN AYRI BIR BETIK
--------------------
``deney_kapasite.py`` toplu (tohum-torbalanmis) skor ve blok kirilimi basiyor
ama ESLENIK STANDART HATA basmiyor. Soguk kapasite deneyi bunun neden
yetmedigini gosterdi: toplu tablo "d6i500 en iyi, +0,00223" diyordu; eslenik
tohum bazinda bakinca SH 0,00699 ve t=+0,18 cikti -- saf gurultu. Ayni
tabloda "en iyi" gorunen uc kol da t<1 idi. Tek istatistiksel olarak anlamli
sonuc, bir kolun DAHA KOTU oldugu idi.

Bu betik hukmu dogru birim uzerinde verir: **(blok, tohum) ciftleri**. Uc
blok x uc tohum = 9 eslenik gozlem, tek blokta 3 gozlemden cok daha saglam.
Her kol uretim kolu ("250") ile ayni (blok, tohum) ciftinde karsilastirilir --
ayni veri, ayni maske, ayni tohum; tek fark kapasite.

Skor ``olcut.py`` ile TESTE AGIRLIKLANDIRILMIS ve URETIM HARMANI icinde
(cat3/xgb1/lgbm1/ag1,4), cunku degistirdigimiz sey harmanin bir uyesi.

    python scripts/kapasite_hukmu.py
    python scripts/kapasite_hukmu.py --onek pg     # p_/g_/gp_ deneyinin kollari
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907

#: Karsilastirma kumeleri: ad -> (uretim kolu dosya soneki, aday sonekleri)
KUMELER = {
    "kapasite": ("cat_kap250", ("cat_kap500", "cat_kap900", "cat_kap500d7")),
    "pg": ("cat_kap250", ("cat_pg14", "cat_pg10", "cat_maske0")),
}


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kume", default="kapasite", choices=sorted(KUMELER))
    ar = ap.parse_args()
    uretim_sonek, aday_sonekler = KUMELER[ar.kume]

    t0 = time.time()
    print("=" * 100)
    print(f"KAPASITE HUKMU -- eslenik SH, kume '{ar.kume}'")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in ("xgb", "lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
        veri[b.ad] = blok

    def skor(blok: str, tohum: int, sonek: str) -> float | None:
        yol = DIZIN / f"{blok}_{tohum}_{sonek}.npy"
        if not yol.exists():
            return None
        v = veri[blok]
        cat = np.load(yol).astype("float64")
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        pay = (
            GBDT_AGIRLIK[0] * cat
            + GBDT_AGIRLIK[1] * v[(tohum, "xgb")]
            + GBDT_AGIRLIK[2] * v[(tohum, "lgbm")]
            + AG_AGIRLIK * v[(tohum, "sinir_agi")]
        )
        tahmin = np.clip(np.expm1(pay / top), 0.0, None)
        return ol.agirlikli_rmsle(v["gercek"], tahmin, v["w"])

    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    u = {c: skor(*c, uretim_sonek) for c in ciftler}
    eksik = [c for c, s in u.items() if s is None]
    if eksik:
        print(f"  URETIM KOLU EKSIK: {eksik}")
        print(f"  once: python scripts/deney_kapasite.py  ({uretim_sonek})")
        return 1
    print(f"  uretim kolu {uretim_sonek}: {len(ciftler)} eslenik gozlem")
    print(f"  (blok, tohum) ortalamasi {np.mean(list(u.values())):.5f}")

    print("\n" + "-" * 100)
    print("ESLENIK FARK (uretim - aday; POZITIF = aday IYI)")
    print("-" * 100)
    print(f"  {'aday':>14}{'fark':>10}{'SH':>9}{'t':>7}{'kazanan':>9}{'genel etki':>12}  hukum")
    for sonek in aday_sonekler:
        s = {c: skor(*c, sonek) for c in ciftler}
        var = [c for c in ciftler if s[c] is not None]
        if len(var) < len(ciftler):
            print(f"  {sonek:>14}  {len(var)}/{len(ciftler)} gozlem hazir -- atlandi")
            continue
        f = np.array([u[c] - s[c] for c in ciftler])
        ort = float(f.mean())
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = ort / sh if sh > 0 else 0.0
        hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")
        print(
            f"  {sonek:>14}{ort:+10.5f}{sh:9.5f}{t_d:+7.2f}"
            f"{(f > 0).sum():5d}/{len(f):<3d}{-ort * SICAK_KATSAYI:+12.5f}  {hukum}"
        )
        for b in tm.BLOKLAR:
            bf = np.array([u[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR])
            print(f"       {b.ad:8} {bf.mean():+.5f}  ({(bf > 0).sum()}/{len(bf)} tohum)")

    print("\n  KARAR KURALI: t >= 2 VE uc blokta da pozitif ortalama.")
    print("  Ikisi birden saglanmadikca uretim yapilandirmasi DEGISMEZ --")
    print("  30 tohumdan dusmenin maliyeti zaten +0,0006 (docs/40 §4).")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
