"""SICAK UZMAN: geri verilen kolonlar (p_/g_/gp_) ve maske orani -- URETIM ESLI.

IKI SORU, IKISI DE AYNI RIG KUSURUNDAN
--------------------------------------
Rig provenans denetimi (2026-08-24 aksami) iki kararin ek_kokensiz kolda
alindigini gosterdi:

1. YALIN SET 144 -> 105 kolon, ``p_``/``g_``/``gp_`` ailesini (14 kolon)
   SICAK uzmandan da atiyor. Olcumu ``deney_ileri.py:731`` ve o rig uretimden
   DORT eksende birden ayri: ek kokensiz (1,04M vs 2,86M), maske 0,2216 (uretim
   0,15), ``random_strength`` 1 (uretim 4), ``depth`` 5 / ``l2`` 3 (uretim 6/1).
   Olculen sicak kazanc +0,00559, uc blokta da ayni isaret. Genele cevrimi
   x0,589 ile ~+0,0033, ama maske ve rs yanliliklari bu sayiyi SISIRIYOR.

2. MASKE ORANI 0,15 (``deney_ileri.py:306``, ayni ek kokensiz rig). Maske
   TRAFO BAZINDA (``tuketim_model.py:1013``), yani ek kokenli kolda maskelenen
   bir trafonun ~2,75 kopyasi BIREBIR ayni hale geliyor: butun ``t_*`` NaN ve
   hedef ayni. Bu tam olarak ``tuketim_model.py:849-852``de belgelenen "veri
   artirma degil kopya cogaltma" mekanizmasi -- maske 1,00'da ek kokenin
   SOGUK uzmana zarar vermesinin (-0,0327, t=-2,59) sebebi. Yani maskenin
   marjinal maliyeti ek kokenli kolda DAHA YUKSEK ve optimum 0'a kayar.
   Ayrica ayni deneyin tek-tohum okumasinda argmin 0,15 degil 0,00.

UCUNCU KOL: HAM GUN TEHLIKESI
-----------------------------
``p_gun_sayisi``, ``p_ilk_ofset``, ``p_son_ofset``, ``p_yayilma`` panel
penceresine gore HAM GUN. Ana bloklar 121-122 gun, TEST 122 gun, ama
``EK_KOKENLER`` icinde sub25 = 59 ve bah26 = 90 gun var. Yani ek satirlarin
bir kisminda bu dort kolon testte HIC gorulmeyen sikistirilmis bir olcekte
geliyor. ``p_doluluk`` ve ``p_pencere_payi`` normalize, ``g_``/``gp_`` grup
istatistigi -- onlarda bu sorun yok. Bu yuzden ayri bir "+10" kolu var.

TABAN BEDAVA: ``deney_kapasite.py``nin "250" kolu tam olarak uretim
yapilandirmasidir (105 kolon, d6, l2=1, rs=4, 250 agac, ek kokenli, maske
0,15) ve onbellekte duruyor. Bu betik yalnizca YENI kollari egitir.

    python scripts/deney_pg_maske.py
    python scripts/deney_pg_maske.py --kollar pg14,maske0
"""

from __future__ import annotations

import argparse
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
KAYIT = KOK / "experiments" / "pg_maske.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
URETIM_MASKE = 0.15
URETIM_CAT: dict[str, object] = {
    "random_strength": 4.0,
    "l2_leaf_reg": 1.0,
    "depth": 6,
    "iterations": 250,
}

#: Ham GUN cinsinden panel istatistikleri -- ek kokenlerde olcek heterojen.
HAM_GUN = ("p_gun_sayisi", "p_ilk_ofset", "p_son_ofset", "p_yayilma")

#: Kol -> (geri verilecek kolon oneki filtresi, maske orani).
#: "taban" onbellekten okunur (deney_kapasite.py "250" kolu), egitilmez.
KOLLAR: dict[str, tuple[str, float]] = {
    "taban": ("yok", URETIM_MASKE),
    "pg14": ("hepsi", URETIM_MASKE),
    "pg10": ("hamgunsuz", URETIM_MASKE),
    "maske0": ("yok", 0.00),
}
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def _geri_kolonlar(cerceve: pd.DataFrame, filtre: str) -> list[str]:
    if filtre == "yok":
        return []
    pg = [k for k in cerceve.columns if k.startswith(("p_", "g_", "gp_"))]
    if filtre == "hamgunsuz":
        pg = [k for k in pg if k not in HAM_GUN]
    return sorted(pg)


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kollar", default="taban,pg14,pg10,maske0")
    ar = ap.parse_args()
    istenen = [k.strip() for k in ar.kollar.split(",") if k.strip() in KOLLAR]

    t0 = time.time()
    print("=" * 100)
    print("SICAK UZMAN: geri verilen kolonlar + maske orani (URETIM ESLI rig)")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  ana {len(egitim):,} -> EK KOKENLI {len(genis):,}   taban kolon {len(kol)}")
    for ad in istenen:
        filtre, maske = KOLLAR[ad]
        gk = _geri_kolonlar(genis, filtre)
        print(f"    {ad:8} kolon {len(kol) + len(gk):3d}  maske {maske:.2f}  (+{len(gk)} geri)")

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in ("xgb", "lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            for ad in istenen:
                filtre, maske = KOLLAR[ad]
                yol = (
                    DIZIN / f"{b.ad}_{t}_cat_kap250.npy"
                    if ad == "taban"
                    else DIZIN / f"{b.ad}_{t}_cat_{ad}.npy"
                )
                if yol.exists():
                    blok[(t, ad)] = np.load(yol).astype("float64")
                    continue
                if ad == "taban":
                    raise SystemExit(
                        f"taban onbellegi yok: {yol}\n  once: python scripts/deney_kapasite.py"
                    )
                t1 = time.time()
                gk = _geri_kolonlar(parca, filtre)
                maskeli = d.soguk_maskele(parca, kol + gk, maske, t)
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol + gk, t, **URETIM_CAT)
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(yol, v.astype("float32"))
                blok[(t, ad)] = v.astype("float64")
                print(f"    tohum {t}  {ad:8} ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    def skorla(ad: str, blok: str | None = None):  # noqa: ANN202
        bloklar = [b.ad for b in tm.BLOKLAR] if blok is None else [blok]
        k_ag, w_top = 0.0, 0.0
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        for bad in bloklar:
            v = veri[bad]
            yig = []
            for t in TOHUMLAR:
                pay = (
                    GBDT_AGIRLIK[0] * v[(t, ad)]
                    + GBDT_AGIRLIK[1] * v[(t, "xgb")]
                    + GBDT_AGIRLIK[2] * v[(t, "lgbm")]
                    + AG_AGIRLIK * v[(t, "sinir_agi")]
                )
                yig.append(pay / top)
            tahmin = np.clip(np.expm1(np.mean(yig, axis=0)), 0.0, None)
            y, w = v["gercek"], v["w"]
            k_ag += ol.agirlikli_rmsle(y, tahmin, w) ** 2 * w.sum()
            w_top += w.sum()
        return float(np.sqrt(k_ag / w_top))

    print("\n" + "-" * 100)
    print("HUKUM (teste agirliklandirilmis, uretim harmani icinde)")
    print("-" * 100)
    print(f"  {'kol':>8}{'agirlikli':>12}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'tabana gore':>13}{'blok':>7}")
    tb = skorla("taban")
    tb_blok = [skorla("taban", b.ad) for b in tm.BLOKLAR]
    kayitlar = []
    for ad in istenen:
        g = skorla(ad)
        bs = [skorla(ad, b.ad) for b in tm.BLOKLAR]
        kaz = sum(1 for i in range(len(tm.BLOKLAR)) if bs[i] < tb_blok[i])
        etik = "URETIM" if ad == "taban" else f"{kaz}/3"
        print(
            f"  {ad:>8}{g:12.5f}" + "".join(f"{x:11.5f}" for x in bs) + f"{tb - g:+13.5f}{etik:>7}"
        )
        kayitlar.append({"kol": ad, "agirlikli": g, "blok": bs, "kazanan": kaz})

    en = min(
        (k for k in kayitlar if k["kol"] != "taban"), key=lambda k: k["agirlikli"], default=None
    )
    if en is not None:
        fark = tb - en["agirlikli"]
        print(f"\n  en iyi {en['kol']}  {en['agirlikli']:.5f}   uretim {tb:.5f}   fark {fark:+.5f}")
        print(f"  genel skora tahmini etki {-fark * SICAK_KATSAYI:+.5f}")
        print(f"  BLOK TUTARLILIGI: {en['kazanan']}/3")
        if en["kazanan"] == 3 and fark > 0:
            print("  UC BLOKTA DA KAZANIYOR -- hipotez GECERLI, uretim kosusuna deger.")
        else:
            print("  UYARI: uc blokta birden kazanmiyor -- hipotez SAYILMAZ.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
