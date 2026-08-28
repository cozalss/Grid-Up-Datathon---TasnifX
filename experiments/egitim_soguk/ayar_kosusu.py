"""SOGUK UZMAN AYAR KOSUSU -- uretim rig'ini (deney_uretim_ayarlari) saran kosucu.

Rig yeniden kurulmuyor: kolon secimi, egitim nufusu, maske sozlesmesi, olcut
(np.clip(np.expm1(.),0,None)) ve eslenik ozet fonksiyonlari OLDUGU GIBI
deney_uretim_ayarlari'ndan ithal ediliyor. Tek fark: uc blok x N tohum x
COKLU aday tek gecişte, ve hukum kis26 ile veriliyor (yaz25/guz25 soguk
tarafta %97 ezberlenebilir -- gecersiz olcum araci, yalniz isaret kontrolu).

    uv run python experiments/egitim_soguk/ayar_kosusu.py --tohum 1000 1001 1002
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import deney_uretim_ayarlari as rig  # noqa: E402  -- URETIM TEZGAHI
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

P_SOGUK = 0.22159

ADAYLAR: dict[str, dict[str, object]] = {
    "TABAN_d7": {"depth": 7},
    "random_4": {"depth": 7, "random_strength": 4.0},
    "lr003_random4": {"depth": 7, "learning_rate": 0.03, "iterations": 400, "random_strength": 4.0},
    "depth_5": {"depth": 5},
    "depth_6": {"depth": 6},
}


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tohum", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--blok", nargs="+", default=["kis26", "yaz25", "guz25"])
    ap.add_argument("--aday", nargs="+", default=list(ADAYLAR))
    ap.add_argument(
        "--cikti", type=Path, default=KOK / "experiments" / "egitim_soguk" / "ayar_kosusu.jsonl"
    )
    ar = ap.parse_args()

    adaylar = {a: ADAYLAR[a] for a in ar.aday}
    print("=" * 100)
    print("SOGUK UZMAN AYAR KOSUSU -- uretim rig'i")
    print(
        f"  aday={list(adaylar)}  blok={ar.blok}  tohum={ar.tohum}  "
        f"fit={len(adaylar) * len(ar.blok) * len(ar.tohum)}"
    )
    print("=" * 100)

    t0 = time.time()
    dar, test = d.cerceveleri_kur()
    assert rig.rejim_kaynagi("soguk", dar, dar) is dar
    kolonlar = rig.uretim_kolonlari(dar, test)
    tm.kategorik_kodla(dar, test)
    print(f"  egitim={len(dar):,} kolon={len(kolonlar)} (uretim yalin seti)")

    skor: dict[str, dict[str, list[float]]] = {a: {b: [] for b in ar.blok} for a in adaylar}
    for blok in ar.blok:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(dar, blok)
        if not soguk.any():
            raise RuntimeError(f"{blok} dogrulamasinda soguk satir yok")
        for tohum in ar.tohum:
            tf = time.time()
            maskeli = d.soguk_maskele(kalan, kolonlar, 1.0, tohum)
            for ad, par in adaylar.items():
                lg = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, tohum, **par)
                s = rig._skorla(gercek, soguk, lg)  # uretim olcutu, kirpmali
                skor[ad][blok].append(s)
                print(f"  {blok} t={tohum} {ad:15} RMSLE={s:.6f}", flush=True)
            del maskeli
            print(f"    cift suresi {time.time() - tf:.0f} sn", flush=True)

    print("\n" + "=" * 100)
    print("SONUC -- dMSE NEGATIF = KAZANC (taban_MSE'den dusus)")
    print("=" * 100)
    hdr = f"{'aday':16}"
    for b in ar.blok:
        hdr += f"{b + ' dMSE':>14}"
    print(hdr + f"{'test dMSE(kis26)':>18}{'ayni_yon':>10}{'t(kis26)':>10}")

    kayitlar = []
    for ad in adaylar:
        if ad == "TABAN_d7":
            continue
        dler, satir = [], f"{ad:16}"
        for b in ar.blok:
            tb = np.asarray(skor["TABAN_d7"][b]) ** 2
            ay = np.asarray(skor[ad][b]) ** 2
            dm = float(ay.mean() - tb.mean())
            dler.append(dm)
            satir += f"{dm:>+14.6f}"
        i = ar.blok.index("kis26")
        f = np.asarray(skor["TABAN_d7"]["kis26"]) ** 2 - np.asarray(skor[ad]["kis26"]) ** 2
        sh = float(f.std(ddof=1) / np.sqrt(f.size)) if f.size > 1 else float("nan")
        tval = float(f.mean() / sh) if sh and sh > 0 else float("nan")
        ayni = all(v < 0 for v in dler) or all(v > 0 for v in dler)
        satir += f"{P_SOGUK * dler[i]:>+18.6f}{str(ayni):>10}{tval:>+10.2f}"
        print(satir)
        kayitlar.append(
            {
                "aday": ad,
                "parametreler": adaylar[ad],
                "bloklar": ar.blok,
                "tohumlar": ar.tohum,
                "dmse": dict(zip(ar.blok, dler)),
                "test_dmse_kis26": P_SOGUK * dler[i],
                "ayni_yon": ayni,
                "t_kis26": tval,
                "skorlar": {b: skor[ad][b] for b in ar.blok},
                "taban_skorlar": {b: skor["TABAN_d7"][b] for b in ar.blok},
            }
        )

    ar.cikti.parent.mkdir(parents=True, exist_ok=True)
    with ar.cikti.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM {(time.time() - t0) / 60:.1f} dakika | {ar.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
