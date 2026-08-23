"""SICAK HARMANA 5. UYE: duzenlilestirilmis DOGRUSAL model.

NEDEN
-----
Krogh & Vedelsby ayrismasi (NeurIPS 1994) log uzayinda aritmetik ortalama
icin bir OZDESLIKtir:

    E_harman = E_uye_ortalamasi - A_cesitlilik

Yani bir uye TEK BASINA daha kotu olsa bile, hatalari digerlerinden
yeterince farkliysa harmani duzeltir. Sinir agi tam bunu yapti: uc GBDT
ailesinin birbiriyle korelasyonu 0,914 iken ag ile 0,77, ve ag harmani
-0,0015 iyilestirdi.

Geriye denenmemis EN AYRIK tumevarim onyargisi kaldi: DOGRUSAL model.
Agaclar parcali sabit, ag duzgun ama esnek; ridge KATI dogrusal. Ofset
uzayinda calisildigi icin (hedef log1p(y) - log1p(guc)) dogrusal bir
model mantikli bir taban kurabilir.

Beklenti dusuk ama olcum ucuz: ridge deterministik (tohum yok), yalnizca
3 blok icin bir kez fit edilir; cat/xgb/lgbm tahminleri onbellekten gelir.

On isleme: yalnizca sayisal kolonlar, medyan doldurma + eksiklik
gostergesi, standartlastirma. Maskeleme AYNEN uygulanir -- yoksa dogrusal
uye soguk satirlarda gecmis gorur ve harman kirlenir.

    python scripts/deney_dogrusal_uye.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KAYIT = KOK / "experiments" / "dogrusal_uye.jsonl"
SICAK_MASKE = 0.15
#: Dogrusal uyenin harmandaki agirligi (cat 3 / xgb 1 / lgbm 1 uzerine).
AGIRLIKLAR = (0.0, 0.15, 0.3, 0.6, 1.0)
ALFALAR = (1.0, 10.0, 100.0)


def hazirla(egitim, hedef, kolonlar):  # noqa: ANN001, ANN201
    """Sayisal matris + medyan doldurma + eksiklik gostergesi + olcekleme."""
    sayisal = [k for k in kolonlar
               if egitim[k].dtype.kind in "ifb" and hedef[k].dtype.kind in "ifb"]
    xe = egitim[sayisal].to_numpy(dtype="float32")
    xh = hedef[sayisal].to_numpy(dtype="float32")
    eksik_e = np.isnan(xe)
    eksik_h = np.isnan(xh)
    medyan = np.nanmedian(xe, axis=0)
    medyan = np.where(np.isnan(medyan), 0.0, medyan)
    xe = np.where(eksik_e, medyan, xe)
    xh = np.where(eksik_h, medyan, xh)
    # yalnizca gercekten eksik olan kolonlar icin gosterge
    gosterge = eksik_e.any(axis=0)
    xe = np.hstack([xe, eksik_e[:, gosterge].astype("float32")])
    xh = np.hstack([xh, eksik_h[:, gosterge].astype("float32")])
    ort = xe.mean(axis=0)
    std = xe.std(axis=0)
    std[std < 1e-6] = 1.0
    return (xe - ort) / std, (xh - ort) / std, len(sayisal), int(gosterge.sum())


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SICAK HARMANA 5. UYE: duzenlilestirilmis dogrusal model")
    print("=" * 92)

    from sklearn.linear_model import Ridge

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    z = np.load(ONBELLEK)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    dogrusal: dict[tuple[str, float], np.ndarray] = {}
    veri: dict[str, dict] = {}
    for b in tm.BLOKLAR:
        parca, dogrulama, gercek, soguk = parcalar[b.ad]
        maskeli = d.soguk_maskele(parca, kol, SICAK_MASKE, di.TOHUMLAR[0])
        xe, xh, n_say, n_gos = hazirla(maskeli, dogrulama, kol)
        ye = (np.log1p(maskeli[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64"))
              - np.log1p(maskeli["guc"].to_numpy(dtype="float64")))
        lg = np.log1p(dogrulama["guc"].to_numpy(dtype="float64"))
        for alfa in ALFALAR:
            reg = Ridge(alpha=alfa, solver="lsqr")
            reg.fit(xe, ye)
            dogrusal[(b.ad, alfa)] = (reg.predict(xh) + lg)[~soguk]
        veri[b.ad] = {"y": gercek[~soguk], "lg": lg[~soguk]}
        print(f"  {b.ad}: {xe.shape[0]:,} x {xe.shape[1]} ({n_say} sayisal + {n_gos} gosterge)"
              f"  ({time.time() - t0:.0f} sn)")

    print(f"\n  {'alfa':>7}{'blok':>8}{'tek basina':>12}{'cat':>10}")
    for alfa in ALFALAR:
        for b in tm.BLOKLAR:
            v = veri[b.ad]
            tek = tm.rmsle(v["y"], np.clip(np.expm1(dogrusal[(b.ad, alfa)]), 0.0, None))
            c = np.mean([z[f"{b.ad}_{t}_cat"] for t in di.TOHUMLAR], axis=0)
            c_sk = tm.rmsle(v["y"], np.clip(np.expm1(c), 0.0, None))
            print(f"  {alfa:7.0f}{b.ad:>8}{tek:12.5f}{c_sk:10.5f}")

    print("\n  HARMAN (cat3/xgb1/lgbm1 + w*dogrusal), 3 tohum torbalanmis")
    print("  " + f"{'alfa':>7}" + "".join(f"{f'w={w:.2f}':>11}" for w in AGIRLIKLAR))
    kayitlar = []
    for alfa in ALFALAR:
        satir = []
        for w in AGIRLIKLAR:
            skorlar = []
            for b in tm.BLOKLAR:
                v = veri[b.ad]
                loglar = []
                for t in di.TOHUMLAR:
                    gbdt = sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER))
                    loglar.append((gbdt + w * dogrusal[(b.ad, alfa)]) / (sum(AGIRLIK) + w))
                harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
                skorlar.append(tm.rmsle(v["y"], harman))
            satir.append(float(np.mean(skorlar)))
            kayitlar.append({"alfa": alfa, "w": w, "rmsle": satir[-1]})
        print(f"  {alfa:7.0f}" + "".join(f"{v:11.5f}" for v in satir))

    taban = next(k["rmsle"] for k in kayitlar if k["w"] == 0.0)
    en_iyi = min(kayitlar, key=lambda k: k["rmsle"])
    kazanc = taban - en_iyi["rmsle"]
    print(f"\n  TABAN (w=0): {taban:.5f}")
    print(f"  EN IYI alfa={en_iyi['alfa']:.0f} w={en_iyi['w']:.2f} -> {en_iyi['rmsle']:.5f}"
          f"   kazanc {kazanc:+.5f}")
    print(f"  genel skora tahmini etki {-kazanc * 0.528:+.5f}")
    print(f"  HUKUM: {'AL' if kazanc > 0.002 else 'REDDET (esik alti)'}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
