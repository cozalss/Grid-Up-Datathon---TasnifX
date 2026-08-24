"""SOGUK UZMANA GRUP ISTATISTIKLERI -- hic sorulmadi, ve tam ihtiyaci olan sey.

NEDEN
-----
Soguk uzman maske 1,00'da calisir: butun ``t_*`` kolonlari NaN. Elinde
trafoyu ayirt eden HICBIR gecmis yoktur; yalnizca lokasyon, ``guc``, takvim,
hava ve statik ilce kolonlari kalir.

``g_*`` / ``gp_*`` ise GRUP ISTATISTIKLERIDIR -- ozet panelinden hesaplanan
(ilce x kVA kovasi) ortalamalari ve profil kolonlari. Geçmişi olmayan bir
model icin bu tam olarak eksik olan sey: bir SEVIYE kestirimi.

Bugun (2026-08-24) bu ailedan 10 kolon SICAK uzmana geri verildi ve
t=+3,59 ile kazandi. SOGUK uzmana HIC sorulmadi.

IKI CIDDI UYARI
---------------
1. LB GECMISI. 2026-08-23'te soguk uzmana ``ek_kolon`` (hafta gunu) eklendi
   ve harman 3/1/1 -> 1/1/1 yapildi; ikisi BIRLIKTE LB'de olculdu ve zararli
   cikti (v18 1,03370 -> v23 1,04820). Teshis docs/35: ``yaz25`` ezber
   yuzunden ikisini de onaylamis, ``kis26`` ikisine de hayir demis, LB
   kis26'yi hakli cikarmis. Yani soguk tarafta "kolon ekle" fikri bir kez
   LB'de yandi -- ama o paket HAFTA GUNU + HARMAN idi, grup istatistigi degil.

2. SON ISLEM GECMISI. ``son_islem_gun.py`` ilce x kova hucre tablosunu
   SON ISLEMDE kullanmayi denedi ve LB'de curudu (+0,00414). Buradaki fark:
   orada sabit bir ``0,40*etki`` agirligi DAYATILIYORDU; burada kolon
   MODELE veriliyor ve model ona ne kadar guvenecegini kendi ogreniyor.
   Yine de ayni bilgi kanali oldugu icin sonuc dikkatle okunmali.

Bu ikisi yuzunden hukum SIKI: yalnizca ``kis26`` (ezber %0), son islem
SONRASI, testin kVA karisimina agirliklandirilmis, ve eslenik SH ile.
Gecerse bile IZOLE bir LB gonderimiyle sinanir.

    python scripts/deney_soguk_grup_kolon.py
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
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "soguk_grup"
KAYIT = KOK / "experiments" / "soguk_grup_kolon.jsonl"
BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
SOGUK_MASKE = 1.00
SOGUK_CAT: dict[str, object] = {"depth": 7}
BETA = 0.60

#: Kol -> geri verilecek kolon oneki/adlari. "taban" uretimdir.
KOLLAR: dict[str, tuple[str, ...]] = {
    "taban": (),
    "grup": ("g_", "gp_"),
    "grup_panel": ("g_", "gp_", "p_doluluk", "p_pencere_payi"),
}
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 100)
    print(f"SOGUK UZMANA GRUP ISTATISTIKLERI  ({BLOK}, son islem sonrasi, kVA duzeltilmis)")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    te_c = test[test["soguk_mu"] == 1]
    w, tani = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))
    print(f"  {BLOK} soguk {len(y):,} satir   egitim {len(parca):,} (ek koken YOK)")
    print(f"  kVA agirliklandirmasi ESS %{100 * tani['ess_orani']:.1f}")

    adaylar = {}
    for ad, onekler in KOLLAR.items():
        if not onekler:
            adaylar[ad] = []
            continue
        c = sorted(
            k
            for k in egitim.columns
            if k.startswith(onekler) and k in test.columns and k not in taban_kol
        )
        adaylar[ad] = c
        print(f"    {ad:11} +{len(c):2d} kolon  {c}")

    def buz(log_t: np.ndarray) -> np.ndarray:
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    DIZIN.mkdir(parents=True, exist_ok=True)
    tahmin = {}
    for ad in KOLLAR:
        tahmin[ad] = {}
        kk = taban_kol + adaylar[ad]
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
            print(f"    {ad:11} tohum {t}  ({len(kk)} kolon, {time.time() - t1:.0f} sn)")

    print("\n" + "-" * 100)
    print("HUKUM")
    print("-" * 100)
    print(f"  {'kol':>12}{'HAM kis26':>12}{'kVA duzeltilmis':>18}{'uretime gore':>14}")
    torba = {ad: buz(np.mean([tahmin[ad][t] for t in TOHUMLAR], axis=0)) for ad in KOLLAR}
    duz = {ad: ol.agirlikli_rmsle(y, torba[ad], w) for ad in KOLLAR}
    ham = {ad: ol.agirlikli_rmsle(y, torba[ad]) for ad in KOLLAR}
    for ad in KOLLAR:
        bayrak = "  <- URETIM" if ad == "taban" else ""
        print(f"  {ad:>12}{ham[ad]:12.5f}{duz[ad]:18.5f}{duz['taban'] - duz[ad]:+14.5f}{bayrak}")

    print("\n  ESLENIK FARK (taban - aday; tohum bazinda, POZITIF = aday IYI)")
    kayitlar = []
    tekil = {
        ad: np.array([ol.agirlikli_rmsle(y, buz(tahmin[ad][t]), w) for t in TOHUMLAR])
        for ad in KOLLAR
    }
    for ad in KOLLAR:
        if ad == "taban":
            continue
        f = tekil["taban"] - tekil[ad]
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = f.mean() / sh if sh > 0 else 0.0
        hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")
        print(
            f"    {ad:>12} fark {f.mean():+.5f}  SH {sh:.5f}  t {t_d:+.2f}"
            f"  ({(f > 0).sum()}/{len(f)} tohum)  genel {-f.mean() * SOGUK_KATSAYI:+.5f}  {hukum}"
        )
        kayitlar.append(
            {
                "kol": ad,
                "n_kolon": len(adaylar[ad]),
                "fark": float(f.mean()),
                "sh": sh,
                "t": float(t_d),
                "hukum": hukum,
            }
        )

    print("\n  UYARI: soguk tarafta 'kolon ekle' fikri 2026-08-23'te LB'de yandi")
    print("  (v18 1,03370 -> v23 1,04820) -- ama o paket HAFTA GUNU + HARMAN degisikligiydi.")
    print("  Gecse bile IZOLE bir LB gonderimiyle sinanir; dogrudan uretime alinmaz.")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
