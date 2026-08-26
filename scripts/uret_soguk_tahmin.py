"""SOGUK UZMAN TAHMINLERINI HERHANGI BIR BLOK ICIN URET.

ALTYAPI ACIGI (docs/45 tik 7)
------------------------------
Onbellekte YALNIZCA ``data/interim/deney/soguk_tahmin_kis26.npz`` var. Bu
yuzden bu gecenin BUTUN soguk hukumleri kis26 uzerinde verildi: b_soguk
sinif ayrisimi (H17), hurdle yeniden sinama (H15), h16'nin uc ekseni,
nufus denetimi (H14).

Ama KALICI KURAL 10 tam bunu yasakliyor: kis26'da olculen seviye kazanci
kesme-etiket mevsim bitisikliginden besleniyor ve TEST'in geometrisine
tasinmiyor; boyle oneriler **yaz25**'te olculmelidir. Yani MSE'nin %63'unu
tasiyan taraf, kural geregi kullanilmamasi gereken blokta olculuyor -- ve
bunun tek sebebi EKSIK BIR DOSYA.

Bu betik o dosyayi uretir. Bir kez uretilince kalan gunler boyunca HER
soguk karari mevsimsel ikizde yeniden dogrulanabilir hale gelir.

FORMAT: kis26 npz'siyle BIREBIR ayni ({tohum}_{aile} anahtarlari) ki mevcut
betikler DEGISMEDEN calissin. Ek olarak meta parquet yazilir (tanim, tarih,
guc, y) -- kis26'nin ``kis26_soguk_meta.parquet``inin karsiligi.

AYAR: ``deney_soguk_taban.py`` ile birebir ayni -- saf soguk uzman
(maske orani 1,00: hicbir trafonun gecmisini gormez), cat icin depth 7.

    uv run python scripts/uret_soguk_tahmin.py --blok yaz25 --tohum 1000 1001 1002 1003 1004
"""

from __future__ import annotations

import argparse
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
USTYAZIM: dict[str, object] = {"depth": 7}


def main() -> int:
    a = argparse.ArgumentParser(description="blok icin soguk uzman tahminleri")
    a.add_argument("--blok", required=True, choices=["yaz25", "guz25", "kis26"])
    a.add_argument("--tohum", type=int, nargs="+", default=[1000, 1001, 1002])
    a.add_argument("--aile", nargs="+", default=list(AILELER))
    ar = a.parse_args()

    cikti = KOK / "data/interim/deney" / f"soguk_tahmin_{ar.blok}.npz"
    meta_yol = KOK / "data/interim" / f"{ar.blok}_soguk_meta.parquet"
    print(f"BLOK {ar.blok}   tohum {ar.tohum}   aile {ar.aile}")
    print(f"cikti {cikti}")

    t0 = time.time()
    egitim, test = d.cerceveleri_kur()
    print(f"  cerceveler hazir ({time.time() - t0:.0f} sn)  egitim {egitim.shape}")

    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, ar.blok)
    print(
        f"  parca {parca.shape}  dogrulama {dogrulama.shape}  "
        f"SOGUK satir {int(soguk.sum()):,}  trafo "
        f"{dogrulama.loc[soguk, tm.GRUP].nunique():,}"
    )

    if int(soguk.sum()) == 0:
        raise SystemExit(f"{ar.blok} blogunda soguk satir yok")

    # meta -- kis26_soguk_meta.parquet ile ayni semada
    dg = dogrulama.loc[soguk]
    meta = pd.DataFrame(
        {
            "tanim": dg[tm.GRUP].to_numpy(),
            "tarih": pd.to_datetime(dg["tarih"].to_numpy()),
            "guc": dg["guc"].to_numpy(dtype="float64"),
            "y": dg[tm.HEDEF].to_numpy(dtype="float64"),
        }
    )
    meta_yol.parent.mkdir(parents=True, exist_ok=True)
    meta.to_parquet(meta_yol, index=False)
    print(f"  meta yazildi: {meta_yol.name}  ({len(meta):,} satir)")

    # var olan onbellegi koru, uzerine ekle
    ham: dict[str, np.ndarray] = {}
    if cikti.exists():
        z = np.load(cikti)
        ham = {k: z[k] for k in z.files}
        print(f"  mevcut onbellek okundu: {len(ham)} anahtar")

    for tohum in ar.tohum:
        maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)  # SAF soguk uzman
        for aile in ar.aile:
            anahtar = f"{tohum}_{aile}"
            if anahtar in ham:
                print(f"  {anahtar} zaten var, atlandi")
                continue
            ust = USTYAZIM if aile == "cat" else {}
            ham[anahtar] = di.egit_tahmin(aile, maskeli, dogrulama, kol, tohum, **ust)[soguk]
            print(f"  {anahtar} hazir ({time.time() - t0:.0f} sn)", flush=True)
            cikti.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cikti, **ham)  # her adimda kaydet
    print(f"\nBITTI ({time.time() - t0:.0f} sn)  {len(ham)} anahtar -> {cikti.name}")

    # hizli tani
    lgy = np.log1p(np.clip(meta["y"].to_numpy(dtype="float64"), 0, None))
    ort = np.mean([ham[k] for k in ham], axis=0)
    print(f"  yanlilik b = ort(log1p(y) - tahmin) = {float((lgy - ort).mean()):+.4f}")
    print(f"  RMSE = {float(np.sqrt(((lgy - ort) ** 2).mean())):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
