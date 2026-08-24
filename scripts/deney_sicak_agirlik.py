"""SICAK HARMAN AGIRLIKLARI -- gonderilen tahminciyle AYNI kurguda.

NEDEN YENIDEN
-------------
``deney_agirlik.py`` izgarasi (3,3,1)'i uretim (3,1,1)'e gore +0,0032 iyi
bulmustu ama karar REDDEDILDI: izgara 3 tohum TORBALANMIS tahminler
uzerindeydi, uretimin dogrulama adimi ise TEK tohum (42). Iki farkli
tahminci karsilastiriliyordu.

Ama GONDERILEN dosya 3 tohum torbali (``tuketim_model.py`` son egitim
dongusu). Yani izgara gonderilen tahminciyi olcuyordu, tek-tohumlu
dogrulama olcmuyordu. Ret gerekcesi ters yone isaret ediyor.

Bu betik ayrimi kapatir: aile tahminlerini bir kez uretip onbellege alir,
sonra agirlik izgarasini HEM tek tohumda HEM 3 tohum torbalanmis olarak
hesaplar. Ikisi ayni yonu gostermiyorsa degisiklik YAPILMAZ.

sinir_agi izgaraya GIREMEZ (tek fit ~20 dakika, 27 fit imkansiz). Onun
agirligi ayri olculdu (docs/33) ve burada sabit tutulur; izgara yalnizca
cat/xgb/lgbm oranlarini arar.

    python scripts/deney_sicak_agirlik.py
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
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
SICAK_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
SICAK_MASKE = 0.15
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KAYIT = KOK / "experiments" / "sicak_agirlik.jsonl"

#: Test karisimina agirlik (uretimdeki dogrulama raporuyla ayni).
TEST_SOGUK_PAY = 0.2216


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SICAK HARMAN AGIRLIKLARI  --  tek tohum vs 3 tohum torbalanmis")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    if ONBELLEK.exists():
        z = np.load(ONBELLEK)
        ham = {
            (b.ad, t, a): z[f"{b.ad}_{t}_{a}"]
            for b in tm.BLOKLAR
            for t in di.TOHUMLAR
            for a in AILELER
        }
        print(f"  tahminler onbellekten: {ONBELLEK.name}")
    else:
        ham = {}
        for b in tm.BLOKLAR:
            parca, dogrulama, _, soguk = parcalar[b.ad]
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(parca, kol, SICAK_MASKE, tohum)
                for aile in AILELER:
                    ust = SICAK_USTYAZIM if aile == "cat" else {}
                    ham[(b.ad, tohum, aile)] = di.egit_tahmin(
                        aile, maskeli, dogrulama, kol, tohum, **ust
                    )[~soguk]
                print(f"  {b.ad} tohum {tohum} hazir ({time.time() - t0:.0f} sn)")
        ONBELLEK.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(ONBELLEK, **{f"{b}_{t}_{a}": v for (b, t, a), v in ham.items()})
        print(f"  onbellege yazildi: {ONBELLEK.name}")

    izgara = [w for w in product((1, 2, 3, 4, 5), (1, 2, 3), (1, 2, 3)) if w[0] >= 1]
    sonuc: dict[tuple[int, int, int], dict[str, float]] = {}
    for w in izgara:
        pay = float(sum(w))
        tekil, torba = [], []
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            y = gercek[~soguk]
            loglar = []
            for tohum in di.TOHUMLAR:
                lt = sum(w[i] * ham[(b.ad, tohum, a)] for i, a in enumerate(AILELER)) / pay
                loglar.append(lt)
                tekil.append(tm.rmsle(y, np.clip(np.expm1(lt), 0.0, None)))
            torba.append(tm.rmsle(y, np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)))
        sonuc[w] = {
            "tek": float(np.mean(tekil)),
            "torba": float(np.mean(torba)),
            "bloklar": [float(v) for v in torba],
        }

    taban = sonuc[(3, 1, 1)]
    print(f"\n  URETIM (3,1,1): tek {taban['tek']:.5f}   torbalanmis {taban['torba']:.5f}")
    print(f"\n  {'agirlik':12}{'tek tohum':>11}{'torbali':>10}{'torba fark':>12}   bloklar")
    sirali = sorted(sonuc.items(), key=lambda kv: kv[1]["torba"])
    for w, s in sirali[:10]:
        blk = " ".join(f"{v:.5f}" for v in s["bloklar"])
        fark = s["torba"] - taban["torba"]
        print(f"  {str(w):12}{s['tek']:11.5f}{s['torba']:10.5f}{fark:+12.5f}   {blk}")

    en_iyi = sirali[0]
    ayni_yon = (en_iyi[1]["tek"] < taban["tek"]) and (en_iyi[1]["torba"] < taban["torba"])
    kazanc = taban["torba"] - en_iyi[1]["torba"]
    print(f"\n  EN IYI {en_iyi[0]}  torbali kazanc {kazanc:+.5f}")
    print(f"  tek tohum da ayni yonu gosteriyor mu: {'EVET' if ayni_yon else 'HAYIR'}")
    print(f"  genel skora tahmini etki {-kazanc * 0.528:+.5f}")
    hukum = "AL" if (ayni_yon and kazanc > 0.002) else "REDDET"
    print(f"  HUKUM: {hukum}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for w, s in sirali[:10]:
            fh.write(json.dumps({"agirlik": list(w), **s}, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
