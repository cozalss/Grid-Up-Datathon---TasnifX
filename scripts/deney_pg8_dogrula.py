"""SICAK ek_kolon YENIDEN: 8 kolon, ve TEK-TRAFO tuzagi kontrolu.

NEDEN YENIDEN
-------------
``pg10`` olcumu (+0,0102, t=+3,59; uretim-sadik kurguda +0,0097, t=+3,16)
ON kolonluydu. Dusmanca sinama (2026-08-24 gece) ikisinin OLCULEMEZ oldugunu
gosterdi:

    gp_ilce_ay, gp_kova_ay -- EGITIMDE yalnizca Ocak-Mart satirlarinda dolu
    (338.187 satirda "dolu <=> ay in {1,2,3}" ihlali SIFIR), TESTTE %100 dolu
    ve aylar Nisan-Temmuz. Deger destegi de kopuk: test satirlarinin %26'si
    egitim maksimumunun ustunde, Temmuz'un TAMAMI destek disi.

Ikisi uretim yapilandirmasindan CIKARILDI. Geriye 8 kolon kaldi ve bu betik
o 8 kolonu tek basina olcer -- yani OLCULEN SEY ARTIK GONDERILEN SEY.

TEK-TRAFO KONTROLU
------------------
Ayni sinamada SOGUK bulgusu (t=+13,71) coktu: kVA-agirlikli d(MSE)'nin
%116,4'u TEK bir trafodan geliyordu (olu trafo, 97 gunun 97'sinde sifir).
Buyuk bir t, kucuk bir kat ve agir kuyruklu bir hedefte tek bir satir
kumesinin eseri olabilir.

Bu yuzden burada kazanc TRAFO BAZINDA ayristirilir: en cok katkiyi veren
trafo cikarilinca hukum ayakta kaliyor mu? Sicak tarafta 254-382 bin satir
ve on binlerce trafo var, yani beklenti "evet" -- ama artik varsayilmaz,
OLCULUR.

    python scripts/deney_pg8_dogrula.py
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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "pg8_dogrula.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15
URETIM_CAT: dict[str, object] = {
    "random_strength": 4.0,
    "l2_leaf_reg": 1.0,
    "depth": 6,
    "iterations": 250,
}
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print("SICAK ek_kolon = 8 -- olculen sey artik gonderilen sey")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    ek8 = list(tm.REJIM_AYARLARI["sicak"]["ek_kolon"])  # type: ignore[index]
    kol = taban_kol + ek8
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  taban {len(taban_kol)} + ek {len(ek8)} = {len(kol)} kolon")
    print(f"  ek_kolon: {ek8}")

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {
            "gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"),
            "w": w,
            "tanim": dogrulama.loc[~soguk, "tanim"].astype(str).to_numpy(),
        }
        for t in TOHUMLAR:
            for a in ("xgb", "lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
            blok[(t, "taban")] = np.load(DIZIN / f"{b.ad}_{t}_cat_kap250.npy").astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            yol = DIZIN / f"{b.ad}_{t}_cat_pg8.npy"
            if yol.exists():
                blok[(t, "pg8")] = np.load(yol).astype("float64")
                continue
            t1 = time.time()
            maskeli = d.soguk_maskele(parca, kol, MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, **URETIM_CAT)
            v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
            np.save(yol, v.astype("float32"))
            blok[(t, "pg8")] = v.astype("float64")
            print(f"    tohum {t}  pg8 ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    def harman(bad: str, tohum: int, ad: str) -> np.ndarray:
        v = veri[bad]
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        return (
            GBDT_AGIRLIK[0] * v[(tohum, ad)]
            + GBDT_AGIRLIK[1] * v[(tohum, "xgb")]
            + GBDT_AGIRLIK[2] * v[(tohum, "lgbm")]
            + AG_AGIRLIK * v[(tohum, "sinir_agi")]
        ) / top

    def skor(bad: str, tohum: int, ad: str) -> float:
        v = veri[bad]
        t = np.clip(np.expm1(harman(bad, tohum, ad)), 0.0, None)
        return ol.agirlikli_rmsle(v["gercek"], t, v["w"])

    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    u = {c: skor(*c, "taban") for c in ciftler}
    s = {c: skor(*c, "pg8") for c in ciftler}
    f = np.array([u[c] - s[c] for c in ciftler])
    ort, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
    t_d = ort / sh if sh > 0 else 0.0
    blok_ort = {
        b.ad: float(np.mean([u[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR])) for b in tm.BLOKLAR
    }
    uc = all(v > 0 for v in blok_ort.values())

    print("\n" + "-" * 100)
    print("ESLENIK FARK (taban - pg8; POZITIF = pg8 IYI)")
    print("-" * 100)
    print(
        f"  fark {ort:+.5f}  SH {sh:.5f}  t {t_d:+.2f}  {(f > 0).sum()}/{len(f)}"
        f"  uc blok {'EVET' if uc else 'HAYIR'}  genel {-ort * SICAK_KATSAYI:+.5f}"
    )
    for b in tm.BLOKLAR:
        print(f"    {b.ad:8} {blok_ort[b.ad]:+.5f}")

    # ------------------------------------------------- TEK-TRAFO KONTROLU
    print("\n" + "-" * 100)
    print("TEK-TRAFO KONTROLU (soguk bulgusunu yikan tuzak)")
    print("-" * 100)
    for b in tm.BLOKLAR:
        v = veri[b.ad]
        g = np.log1p(np.clip(v["gercek"], 0.0, None))
        du, ds = 0.0, 0.0
        for t in TOHUMLAR:
            du += (g - harman(b.ad, t, "taban")) ** 2
            ds += (g - harman(b.ad, t, "pg8")) ** 2
        katki = (du - ds) / len(TOHUMLAR) * v["w"]
        top = float(katki.sum())
        seri = pd.Series(katki).groupby(v["tanim"]).sum().sort_values(ascending=False)
        en = seri.iloc[0]
        pay = 100.0 * en / top if top != 0 else float("nan")
        ilk5 = 100.0 * float(seri.iloc[:5].sum()) / top if top != 0 else float("nan")
        print(
            f"  {b.ad:8} trafo {seri.size:6,}  toplam d(MSE) {top:+.4f}"
            f"  EN BUYUK TRAFO payi %{pay:.1f}  ilk5 %{ilk5:.1f}"
        )

    print("\n  Soguk bulgusunda tek trafonun payi %116,4 idi ve hukum coktu.")
    print("  Burada pay kucukse kazanc genis tabana yayilmis demektir.")

    hukum = "AL" if (t_d >= 2 and uc) else ("REDDET" if t_d <= -2 else "esik alti")
    print(f"\n  HUKUM: {hukum}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"kol": "pg8", "fark": ort, "sh": sh, "t": t_d, "blok": blok_ort, "hukum": hukum},
                ensure_ascii=False,
            )
            + "\n"
        )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
