r"""SICAK HATA: ne kadari SEVIYE, ne kadari SEKIL.

NEDEN
-----
Oracle olculdu (yaz25, sicak satirlar, 254.296 satir)::

    tek sabit (global)                2,2267
    URETIM MODELI                     0,80081
    trafo SEVIYESI bilinseydi         0,6101   <- oracle
    trafo x ay bilinseydi             0,4033

Yani modelin sicak tarafta 0,19'luk bir acigi var ve bunun tamami trafo
SEVIYESINI yanlis tahmin etmekten geliyor olabilir. Soguk taraftaki 1,0'lik
ucuruma gore bu kapatilabilir gorunuyor -- ve olcegi buyuk: sicak tarafta
0,06 kazanc yaz25'i 0,953'e, LB'yi ~0,995'e indirir.

Ama once ayrim yapilmali. Artik = ln(gercek) - ln(tahmin) olmak uzere::

    artik = (trafo bazinda ORTALAMA artik)  +  (trafo ici SAPMA)
             \___________________________/     \_________________/
                   SEVIYE HATASI                  SEKIL HATASI

Ikisi dik oldugu icin MSE tam ayrisir::

    MSE = Var_arasi(seviye hatasi) + ortalama^2 + Ort_ici(sekil varyansi)

SEVIYE HATASI buyukse: trafo duzeyinde kalibrasyon / daha iyi seviye
tahmini yatirimi mantikli. SEKIL HATASI baskinsa seviye pesinde kosmak
bosuna -- gunluk dalgalanma zaten indirgenemez.

Ayrica seviye hatasinin ONGORULEBILIR olup olmadigi sinaniyor: trafo
bazindaki ortalama artik, trafonun GECMIS ozellikleriyle (gecmis uzunlugu,
oynaklik, sifir orani, kapasite) korele mi? Koreleyse duzeltilebilir.

    python scripts/teshis_seviye_sekil.py
"""

from __future__ import annotations

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

BLOK = "yaz25"
TOHUM = 1000
SICAK_MASKE = 0.15
USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SICAK HATA AYRISIMI -- {BLOK}, uretim sicak uzmani")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, TOHUM)
    log_t = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, TOHUM, **USTYAZIM)

    sic = ~soguk
    dv = dogrulama.loc[sic].reset_index(drop=True)
    dv["artik"] = np.log1p(gercek[sic]) - np.log1p(np.clip(np.expm1(log_t), 0.0, None))[sic]

    toplam_mse = float((dv["artik"] ** 2).mean())
    trafo_ort = dv.groupby("tanim", observed=True)["artik"].transform("mean")
    seviye = float((trafo_ort**2).mean())
    sekil = float(((dv["artik"] - trafo_ort) ** 2).mean())

    print(f"\n  sicak satir {len(dv):,} | trafo {dv['tanim'].nunique():,}")
    print(f"  RMSLE {np.sqrt(toplam_mse):.5f}   (MSE {toplam_mse:.5f})")
    print(f"\n  {'bilesen':32}{'MSE':>10}{'pay':>8}{'RMSLE karsiligi':>18}")
    print(f"  {'SEVIYE (trafo bazinda ort artik)':32}{seviye:>10.5f}{seviye / toplam_mse:>8.1%}"
          f"{np.sqrt(seviye):>18.4f}")
    print(f"  {'SEKIL (trafo ici sapma)':32}{sekil:>10.5f}{sekil / toplam_mse:>8.1%}"
          f"{np.sqrt(sekil):>18.4f}")
    print(f"\n  SEVIYE tamamen cozulseydi RMSLE {np.sqrt(sekil):.5f} olurdu")
    print("  (oracle seviye olcumu 0,6101 demisti -- ikisi ortusmeli)")

    # Seviye hatasi ONGORULEBILIR mi: trafo bazinda ort artik neyle korele
    trafo = dv.groupby("tanim", observed=True).agg(
        sapma=("artik", "mean"),
        gun=("t_gun_sayisi", "first"),
        oynaklik=("t_log_std", "first"),
        sifir=("t_sifir_orani", "first"),
        guc=("guc", "first"),
        seviye_g=("t_log_ort", "first"),
        son30=("t_log_son30", "first"),
        doluluk=("t_doluluk", "first"),
    )
    print(f"\n  SEVIYE HATASI ONGORULEBILIR MI ({len(trafo):,} trafo)")
    print(f"  {'ozellik':16}{'Pearson r':>12}{'|r|>0,15 mi':>14}")
    for k in ("gun", "oynaklik", "sifir", "guc", "seviye_g", "son30", "doluluk"):
        alt = trafo[["sapma", k]].dropna()
        if len(alt) < 50:
            print(f"  {k:16}{'yetersiz':>12}")
            continue
        r = float(np.corrcoef(alt["sapma"], alt[k])[0, 1])
        print(f"  {k:16}{r:>+12.3f}{'EVET' if abs(r) > 0.15 else '':>14}")

    print(f"\n  trafo bazinda ort artik: ort {trafo['sapma'].mean():+.4f} "
          f"std {trafo['sapma'].std():.4f} "
          f"p05 {trafo['sapma'].quantile(.05):+.3f} p95 {trafo['sapma'].quantile(.95):+.3f}")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
