"""Rejim yonlendirmesi ARTI harman -- gonderimi belirleyen deney.

NEDEN
-----
``deney_ileri.py --deney soguk_oran`` (2026-08-21 gece, 63 CatBoost fit)
sunu olctu:

    maske    sicak     soguk
    0,00     0,8136    1,8215
    0,15     0,8128    1,7851
    0,2216   0,8219    1,7792     <- mevcut uretim
    0,50     0,8181    1,7733
    0,70     0,8830    1,7688
    1,00     1,6299    1,7595     <- SAF SOGUK UZMANI

Iki egri TERS YONDE ve TEKDUZE. Tek bir oran ikisini birden en iyi
yapamaz; bugune kadar tam olarak onu deniyorduk. Satiri rejimine gore
yonlendirmek mesru, cunku test aninda bir trafonun gecmisi olup
olmadigini BILIYORUZ (``soguk_mu``).

CatBoost'ta olculdu: 1,10805 -> 1,09608, ucu blokta da ayni yonde.

AMA o olcum TEK AILEYLE yapildi. Uretim 3/1/1 CatBoost/XGBoost/LightGBM
harmani. Bu betik sorunun tamamini kapatir: yonlendirme kazanci harmanda
da duruyor mu, ve iki rejimin harman agirligi ayni olmali mi.

Fit sayisi: 3 blok x 2 maske x 3 aile x 3 tohum = 54.
Olculen sureler (cat 35 sn, xgb 45 sn, lgbm 23 sn) ile ~31 dakika.

BUZULME -- duzeltilmis
----------------------
Ilk olcum basarisizdi (-0,033) ve nedeni bulundu: capraz-blok regresyonun
KESIM terimi de tasinmisti, oysa kesim bloktan bloga cok degisiyor
(soguk: +0,10 / +0,28 / +0,38). Egim ise tutarli sekilde 1'in ALTINDA
(0,65-0,76), yani fazla varyans gercek. Duzeltme: yalnizca egimi tasi,
merkezi olarak bloğun KENDI tahmin ortalamasini kullan --

    r_duz = ort_b(r_sapka) + beta * (r_sapka - ort_b(r_sapka))

``ort_b`` yalnizca TAHMINLERDEN hesaplanir, etiket kullanmaz; sizinti yok.

Calistirma::

    python scripts/deney_yonlendirme.py
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

#: Sicak uzmani ve soguk uzmani. Egrilerin tepe noktalari.
SICAK_MASKE, SOGUK_MASKE = 0.15, 1.0

AILELER = ("cat", "xgb", "lgbm")

#: Denenecek harman agirliklari. Rejim basina AYRI aranacak.
AGIRLIKLAR: dict[str, dict[str, float]] = {
    "cat tek": {"cat": 1.0},
    "1/1/1": {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0},
    "2/1/1": {"cat": 2.0, "xgb": 1.0, "lgbm": 1.0},
    "3/1/1": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0},
    "4/1/1": {"cat": 4.0, "xgb": 1.0, "lgbm": 1.0},
    "6/1/1": {"cat": 6.0, "xgb": 1.0, "lgbm": 1.0},
    "cat+xgb 3/1": {"cat": 3.0, "xgb": 1.0},
    "cat+lgbm 3/1": {"cat": 3.0, "lgbm": 1.0},
}


def karisim(tahmin: dict[str, np.ndarray], agirlik: dict[str, float]) -> np.ndarray:
    toplam = sum(agirlik.values())
    return sum(w * tahmin[a] for a, w in agirlik.items()) / toplam


def main() -> int:  # noqa: PLR0915 - tek akisli deney betigi
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("REJIM YONLENDIRMESI x HARMAN")
    print("=" * 96)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    print(f"  egitim {len(egitim):,} satir | {len(kolonlar)} oznitelik")
    print(f"  sicak uzmani maske={SICAK_MASKE}  soguk uzmani maske={SOGUK_MASKE}")

    # blok -> maske -> aile -> torbalanmis log tahmin
    tahmin: dict[str, dict[float, dict[str, np.ndarray]]] = {}
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    log_guc: dict[str, np.ndarray] = {}

    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        gercekler[b.ad] = (gercek, soguk)
        log_guc[b.ad] = np.log1p(dogrulama["guc"].to_numpy())
        tahmin[b.ad] = {}
        for maske in (SICAK_MASKE, SOGUK_MASKE):
            tahmin[b.ad][maske] = {a: [] for a in AILELER}
            for tohum in di.TOHUMLAR:
                t0 = time.time()
                maskeli = d.soguk_maskele(kalan, kolonlar, maske, tohum)
                for a in AILELER:
                    tahmin[b.ad][maske][a].append(
                        di.egit_tahmin(a, maskeli, dogrulama, kolonlar, tohum)
                    )
                del maskeli
                print(
                    f"  {b.ad:6} maske {maske:.2f} tohum {tohum}  3 aile"
                    f"  ({time.time() - t0:.0f} sn)"
                )
            # tohum torbalamasi: log uzayinda ortalama
            tahmin[b.ad][maske] = {a: np.mean(v, axis=0) for a, v in tahmin[b.ad][maske].items()}

    def skorla_blok(blok: str, log_t: np.ndarray) -> di.Skor:
        return di.skorla(*gercekler[blok], log_t)

    print("\n  --- YONLENDIRMESIZ: tek maske, harman ---")
    for maske in (SICAK_MASKE, SOGUK_MASKE):
        for ad, agirlik in AGIRLIKLAR.items():
            blok_skorlari = {
                b.ad: [skorla_blok(b.ad, karisim(tahmin[b.ad][maske], agirlik))] for b in tm.BLOKLAR
            }
            di.yazdir(f"maske {maske:.2f}  {ad}", blok_skorlari)
            di.kaydet(
                f"maske {maske:.2f} {ad}",
                blok_skorlari,
                {"deney": "yonlendirme_harman", "maske": maske, "agirlik": agirlik},
            )

    print("\n  --- YONLENDIRMELI: sicak/soguk AYRI agirlik ---")
    print(
        f"  (sicak satirlar maske {SICAK_MASKE:.2f}'ten,"
        f" soguk satirlar maske {SOGUK_MASKE:.2f}'ten)"
    )
    en_iyi: tuple[float, str] = (1e9, "")
    for s_ad, s_ag in AGIRLIKLAR.items():
        for c_ad, c_ag in AGIRLIKLAR.items():
            blok_skorlari = {}
            for b in tm.BLOKLAR:
                _, soguk = gercekler[b.ad]
                birlesik = np.where(
                    soguk,
                    karisim(tahmin[b.ad][SOGUK_MASKE], c_ag),
                    karisim(tahmin[b.ad][SICAK_MASKE], s_ag),
                )
                blok_skorlari[b.ad] = [skorla_blok(b.ad, birlesik)]
            ad = f"sicak[{s_ad}] soguk[{c_ad}]"
            genel = di.yazdir(ad, blok_skorlari)
            di.kaydet(
                ad,
                blok_skorlari,
                {"deney": "yonlendirme_harman_rejim", "sicak": s_ag, "soguk": c_ag},
            )
            if genel < en_iyi[0]:
                en_iyi = (genel, ad)

    print(f"\n  EN IYI YONLENDIRMELI HARMAN: {en_iyi[1]}   {en_iyi[0]:.5f}")

    print("\n  --- BUZULME (DUZELTILMIS: yalnizca egim, blok-ici merkez) ---")
    print("  ilk deneme kesimi de tasidigi icin -0,033 vermisti; kesim bloktan")
    print("  bloga +0,10 ile +0,38 arasi degisiyor. Simdi yalnizca egim tasiniyor.")
    taban_ag = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
    yol: dict[str, np.ndarray] = {}
    for b in tm.BLOKLAR:
        _, soguk = gercekler[b.ad]
        yol[b.ad] = np.where(
            soguk,
            karisim(tahmin[b.ad][SOGUK_MASKE], taban_ag),
            karisim(tahmin[b.ad][SICAK_MASKE], taban_ag),
        )
    for beta_soguk in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7):
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            gercek, soguk = gercekler[b.ad]
            r_sapka = yol[b.ad] - log_guc[b.ad]
            duz = r_sapka.copy()
            if soguk.any():
                merkez = float(r_sapka[soguk].mean())
                duz[soguk] = merkez + beta_soguk * (r_sapka[soguk] - merkez)
            blok_skorlari[b.ad] = [di.skorla(gercek, soguk, duz + log_guc[b.ad])]
        di.yazdir(f"soguk buzulme beta={beta_soguk:.2f}", blok_skorlari)
        di.kaydet(
            f"buzulme beta={beta_soguk:.2f}",
            blok_skorlari,
            {"deney": "buzulme2", "beta": beta_soguk},
        )

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika   -> {di.SONUC_DOSYASI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
