"""SOGUK UZMANDAN OLU KOLONLARI AT -- 33 t_* kolonu maske 1,00'da TAMAMEN NaN.

BULGU
-----
Soguk uzman ``maske: 1.00`` ile calisir, yani butun ``t_*`` kolonlari NaN
yapilir. Olculdu: 105 kolonun **33'u** ``t_`` onekli ve egitim cercevesinde
%100 NaN. Model onlari kullanamaz; yalnizca oznitelik listesini sisirirler.

MEKANIZMA ZAYIF, YINE DE OLCULUR
--------------------------------
Ilk bakista "rsm=0,75 bolme basina 79 kolon orneklıyor, 25'i colup" gibi
gorunuyor. Ama rsm ORANTILI ornekledigi icin kullanisli aday sayisi iki
durumda da ayni: 0,75 x 105 - 0,75 x 33 = 54 = 0,75 x 72. Yani ortalama
degismez; degisen yalnizca VARYANS (105 kolonda kullanisli aday sayisi
binom dagilimli, 72 kolonda sabit 54) ve CatBoost'un tamamen-NaN kolonlari
kuantillerken/isleme alirken yaptigi is.

Beklenti bu yuzden KUCUK. Ama olcum 3 fit ve ~3 dakika; teori yerine sayi.

Ayrica bu, uretim-sadik bir soru: soguk uzman ``ek_koken: False`` ile
``dar_egitim`` uzerinde calisir ve rig zaten uretimle eslesir.

HUKUM SIKI (bu gecenin dersi)
-----------------------------
Soguk tarafta bir kazanc gorulunce TRAFO BAZINDA ayristirilir. Bu gece
``deney_soguk_grup_kolon.py`` t=+13,71 verdi ve kazancin %116,4'u TEK bir
olu trafodan cikti (tanim=78040011, 97 gunun 97'sinde sifir). 1.223 trafolu
bir katta bu mumkun; artik her soguk bulgusunda kontrol ediliyor.

    python scripts/deney_soguk_kolon_temizle.py
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "soguk_temiz"
KAYIT = KOK / "experiments" / "soguk_kolon_temizle.jsonl"
BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
SOGUK_MASKE = 1.00
SOGUK_CAT: dict[str, object] = {"depth": 7}
BETA = 0.60
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print(f"SOGUK UZMANDAN OLU KOLONLARI AT  ({BLOK}, son islem sonrasi, kVA duzeltilmis)")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol105 = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    tanim = dg["tanim"].astype(str).to_numpy()
    te_c = test[test["soguk_mu"] == 1]
    w, tani = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))

    ornek = d.soguk_maskele(parca, kol105, SOGUK_MASKE, TOHUMLAR[0])
    olu = [k for k in kol105 if ornek[k].isna().all()]
    kol72 = [k for k in kol105 if k not in olu]
    print(f"  {BLOK} soguk {len(y):,} satir   egitim {len(parca):,}")
    print(f"  kolon {len(kol105)} -> {len(kol72)}  ({len(olu)} tamamen NaN atildi)")
    print(f"  kVA agirliklandirmasi ESS %{100 * tani['ess_orani']:.1f}")

    def buz(log_t: np.ndarray) -> np.ndarray:
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    DIZIN.mkdir(parents=True, exist_ok=True)
    tahmin: dict[str, dict[int, np.ndarray]] = {}
    for ad, kk in (("taban", kol105), ("temiz", kol72)):
        tahmin[ad] = {}
        for t in TOHUMLAR:
            yol = DIZIN / f"{BLOK}_{t}_{ad}.npy"
            if yol.exists():
                tahmin[ad][t] = np.load(yol).astype("float64")
                continue
            t1 = time.time()
            maskeli = d.soguk_maskele(parca, kk, SOGUK_MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kk, t, **SOGUK_CAT)
            v = log_t[soguk] if log_t.shape[0] == soguk.size else log_t
            np.save(yol, v.astype("float32"))
            tahmin[ad][t] = v.astype("float64")
            print(f"    {ad:6} tohum {t}  ({len(kk)} kolon, {time.time() - t1:.0f} sn)")

    tekil = {
        ad: np.array([ol.agirlikli_rmsle(y, buz(tahmin[ad][t]), w) for t in TOHUMLAR])
        for ad in ("taban", "temiz")
    }
    torba = {
        ad: ol.agirlikli_rmsle(y, buz(np.mean([tahmin[ad][t] for t in TOHUMLAR], axis=0)), w)
        for ad in ("taban", "temiz")
    }
    f = tekil["taban"] - tekil["temiz"]
    sh = float(f.std(ddof=1) / np.sqrt(len(f)))
    t_d = f.mean() / sh if sh > 0 else 0.0
    hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")

    print("\n" + "-" * 100)
    print(f"  torbalanmis   taban {torba['taban']:.5f}   temiz {torba['temiz']:.5f}")
    print(
        f"  ESLENIK FARK  {f.mean():+.5f}  SH {sh:.5f}  t {t_d:+.2f}"
        f"  ({(f > 0).sum()}/{len(f)} tohum)  genel {-f.mean() * SOGUK_KATSAYI:+.5f}  {hukum}"
    )

    # ---------------------------------------------- TEK-TRAFO KONTROLU
    g = np.log1p(np.clip(y, 0.0, None))
    du = sum((g - tahmin["taban"][t]) ** 2 for t in TOHUMLAR) / len(TOHUMLAR)
    ds = sum((g - tahmin["temiz"][t]) ** 2 for t in TOHUMLAR) / len(TOHUMLAR)
    katki = (du - ds) * w
    top = float(katki.sum())
    seri = pd.Series(katki).groupby(tanim).sum().sort_values(ascending=False)
    if abs(top) > 1e-12:
        print(
            f"\n  TEK-TRAFO KONTROLU: {seri.size:,} trafo, toplam d(MSE) {top:+.4f}"
            f"  EN BUYUK payi %{100 * seri.iloc[0] / top:.1f}"
            f"  ilk5 %{100 * float(seri.iloc[:5].sum()) / top:.1f}"
        )
        print("  (soguk grup bulgusunda tek trafo %116,4 idi ve hukum cokmustu)")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "atilan": len(olu),
                    "fark": float(f.mean()),
                    "sh": sh,
                    "t": float(t_d),
                    "hukum": hukum,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
