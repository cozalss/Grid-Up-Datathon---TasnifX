"""SOGUK UZMANA SINIR AGI: 4. uye burada da ise yariyor mu?

NEDEN
-----
Sinir agi yalnizca SICAK rejimde olculdu ve orada uretime alindi (harmanin
%21,9'u). SOGUK rejimde HIC denenmedi -- uretim orada ``{"cat": 1.0}``.

Ceilismeli iki gerekce var:

  LEHINE  Sicakta uc GBDT ailesinin hata korelasyonu 0,914, sinir agi ile
          0,77. Sogukta uzmanin elinde hicbir ``t_*`` yok (hepsi NaN); geriye
          kalan sinyal guc, konum ve takvim. Farkli tumevarim onyargisi olan
          bir uye burada daha da degerli olabilir.

  ALEYHINE Sogukta HARMAN ZATEN ZARARLI: cat tek basina 1,82250, 3/1/1
          harmani 1,83041 (beta=0,25, gun korumali son islem). Yani ikinci
          bir aile eklemek sogukta simdiye kadar hep kotulestirdi.

Ve son islem beta=0,25 ile modelin katkisini dortte bire indiriyor: soguk
modeldeki her kazanc genel skora ~0,25 x 0,377 = 0,094 katsayisiyla geciyor.
Yani bu kolun tavani dusuk -- ama olculmemis.

Karar mercii kis26 (ezber orani %0 olan tek durust kat, docs/35).

Maliyet: cat tahminleri ONBELLEKTEN gelir (``soguk_tahmin_kis26.npz``),
yalnizca sinir agi fit edilir -- 3 tohum, kis26 soguk.

    python scripts/deney_soguk_sinir.py
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
import deney_soguk_taban as st  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
CAT_ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
AG_ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_sinir_{BLOK}.npz"
KAYIT = KOK / "experiments" / "soguk_sinir.jsonl"
BETALAR = (1.00, 0.40, 0.25, 0.15)
#: Sinir agi agirliklari (cat agirligi 1,0 sabit).
AG_AGIRLIKLARI = (0.0, 0.3, 0.6, 1.0, 1.4)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print(f"SOGUK UZMANA SINIR AGI  --  {BLOK}, {len(TOHUMLAR)} tohum")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    y = gercek[soguk]
    log_guc = np.log1p(dogrulama["guc"].to_numpy(dtype="float64"))[soguk]
    gun = pd.to_datetime(dogrulama["tarih"]).to_numpy()[soguk]
    print(f"  {BLOK} soguk {len(y):,} satir")

    if not CAT_ONBELLEK.exists():
        raise RuntimeError(f"cat onbellegi yok: {CAT_ONBELLEK}")
    zc = np.load(CAT_ONBELLEK)
    cat = {t: zc[f"{t}_cat"] for t in TOHUMLAR}

    if AG_ONBELLEK.exists():
        za = np.load(AG_ONBELLEK)
        ag_tahmin = {t: za[str(t)] for t in TOHUMLAR}
        print(f"  sinir agi onbellekten: {AG_ONBELLEK.name}")
    else:
        from sinir_agi import SinirAgi

        ag_tahmin = {}
        for tohum in TOHUMLAR:
            maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
            hedef = np.log1p(maskeli[tm.HEDEF].clip(lower=0.0)) - np.log1p(maskeli["guc"])
            ag = SinirAgi(tohum=tohum, rejim="soguk", sessiz=True)
            ag.fit(maskeli[kol], hedef)
            tam = ag.predict(dogrulama[kol]) + np.log1p(dogrulama["guc"].to_numpy(dtype="float64"))
            ag_tahmin[tohum] = tam[soguk]
            print(f"  tohum {tohum} sinir agi hazir ({time.time() - t0:.0f} sn)")
        AG_ONBELLEK.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(AG_ONBELLEK, **{str(t): v for t, v in ag_tahmin.items()})
        print(f"  onbellege yazildi: {AG_ONBELLEK.name}")

    tabanlar = st.tabanlari_kur(parca, dogrulama, soguk)
    hucre = tabanlar["ilcexkova_M2000"]

    def gun_ort(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(gun).transform("mean").to_numpy()

    etki = hucre - gun_ort(hucre)

    def skorla(ofs: np.ndarray) -> float:
        return tm.rmsle(y, np.clip(np.expm1(ofs + log_guc), 0.0, None))

    # tek basina aileler
    print("\n  TEK BASINA (son islem yok)")
    for ad, tab in (("cat", cat), ("sinir_agi", ag_tahmin)):
        sk = [skorla(tab[t] - log_guc) for t in TOHUMLAR]
        print(f"    {ad:10} {np.mean(sk):.5f}   tekil " + " ".join(f"{v:.4f}" for v in sk))
    kor = float(np.mean([
        np.corrcoef(cat[t] - log_guc, ag_tahmin[t] - log_guc)[0, 1] for t in TOHUMLAR
    ]))
    print(f"    cat <-> sinir_agi tahmin korelasyonu {kor:.4f}")

    print("\n  HARMAN x SON ISLEM   (cat agirligi 1,0 sabit)")
    print("  " + f"{'ag agirligi':>12}" + "".join(f"{f'beta={b:.2f}':>12}" for b in BETALAR))
    kayitlar = []
    for w in AG_AGIRLIKLARI:
        satir = []
        for beta in BETALAR:
            sk = []
            for t in TOHUMLAR:
                m = (cat[t] + w * ag_tahmin[t]) / (1.0 + w) - log_guc
                tb = gun_ort(m) + etki
                sk.append(skorla(tb + beta * (m - tb)))
            satir.append(float(np.mean(sk)))
            kayitlar.append({"ag_agirligi": w, "beta": beta, "rmsle": satir[-1]})
        print(f"  {w:12.1f}" + "".join(f"{v:12.5f}" for v in satir))

    taban = next(k["rmsle"] for k in kayitlar if k["ag_agirligi"] == 0.0 and k["beta"] == 0.25)
    en_iyi = min(kayitlar, key=lambda k: k["rmsle"])
    kazanc = taban - en_iyi["rmsle"]
    print(f"\n  URETIM (cat tek, beta=0,25): {taban:.5f}")
    print(f"  EN IYI: ag={en_iyi['ag_agirligi']:.1f} beta={en_iyi['beta']:.2f} "
          f"-> {en_iyi['rmsle']:.5f}   kazanc {kazanc:+.5f}")
    print(f"  genel skora tahmini etki {-kazanc * 0.377:+.5f}   (d(genel)/d(soguk) = 0,377)")
    print(f"  HUKUM: {'AL' if kazanc > 0.004 else 'REDDET (esik alti)'}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
