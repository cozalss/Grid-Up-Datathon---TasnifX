"""UC AILENIN HATALARI NE KADAR ORTUSUYOR? -- 4. uye (sinir agi) yarar mi?

Krogh & Vedelsby ayrisimi (log uzayinda ortalama alan topluluklar icin GECERLI):

    E_topluluk = E_ortalama_uye - A_cesitlilik

Uyeler ayni hatayi yapiyorsa A ~ 0 ve topluluk hicbir sey kazandirmaz.
Uretimde uc aile de AGAC tabanli (CatBoost/XGBoost/LightGBM) -- ayni
tumevarim yanliligini paylasiyorlar. Farkli bir aile (sinir agi) eklemenin
degeri, tam olarak mevcut cesitliligin ne kadar DUSUK oldugudur.

Bu betik olcer:
  1. Aileler arasi HATA korelasyonu (yuksek = cesitlilik dusuk)
  2. Krogh-Vedelsby ayrisimi: ortalama uye hatasi, cesitlilik, topluluk
  3. 4. uye ne kadar kazandirir -- korelasyona gore analitik tahmin

    python scripts/teshis_cesitlilik.py
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

HAFTA = ("tk_haftanin_gunu", "tk_hafta_sonu")
KURULUM = (
    ("SICAK", 0.15, {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}, ()),
    ("SOGUK", 1.00, {"depth": 7}, HAFTA),
)
AILELER = ("cat", "xgb", "lgbm")
TOHUM = 1000
BLOK = "yaz25"


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print(f"AILE CESITLILIGI -- {BLOK} blogu, tohum {TOHUM}")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    log_gercek = np.log1p(gercek)

    for rejim, maske, ustyazim, ek in KURULUM:
        kol = uretim + [k for k in ek if k not in uretim]
        secim = soguk if rejim == "SOGUK" else ~soguk
        maskeli = d.soguk_maskele(parca, kol, maske, TOHUM)
        print(f"\n--- {rejim} rejimi ({int(secim.sum()):,} satir, maske {maske:.2f}) ---")

        hata = {}
        for aile in AILELER:
            ek_ust = ustyazim if aile == "cat" else {}
            log_t = di.egit_tahmin(aile, maskeli, dogrulama, kol, TOHUM, **ek_ust)
            hata[aile] = (log_t - log_gercek)[secim]
            print(f"  {aile:5} tek basina RMSLE {np.sqrt((hata[aile] ** 2).mean()):.5f}")

        H = np.column_stack([hata[a] for a in AILELER])
        print("\n  HATA KORELASYON MATRISI (1,0 = ayni hatayi yapiyorlar):")
        K = np.corrcoef(H.T)
        print("        " + "  ".join(f"{a:>7}" for a in AILELER))
        for i, a in enumerate(AILELER):
            print(f"  {a:5} " + "  ".join(f"{K[i, j]:7.4f}" for j in range(len(AILELER))))
        ort_dis = float((K.sum() - np.trace(K)) / (len(AILELER) * (len(AILELER) - 1)))
        print(f"  ortalama ikili korelasyon: {ort_dis:.4f}")

        topluluk = H.mean(axis=1)
        E_uye = float((H**2).mean())
        E_top = float((topluluk**2).mean())
        A = E_uye - E_top
        print("\n  KROGH-VEDELSBY:")
        print(f"    ortalama uye hatasi (MSE) {E_uye:.5f}  -> RMSLE {np.sqrt(E_uye):.5f}")
        print(f"    cesitlilik A             {A:.5f}  (%{100 * A / E_uye:.2f})")
        print(f"    TOPLULUK              {E_top:.5f}  -> RMSLE {np.sqrt(E_top):.5f}")

        # 4. uye: mevcut uyelerle korelasyonu r, tekil hatasi ayni olsun
        s2 = float((H**2).mean())
        for r4 in (0.99, 0.95, 0.90, 0.80, 0.70):
            # 4 uyeli esit agirlikli toplulugun beklenen MSE'si
            # kovaryans: kosegen s2, ilk uc arasi ort_dis*s2, 4. ile r4*s2
            C = np.full((4, 4), ort_dis * s2)
            C[3, :3] = C[:3, 3] = r4 * s2
            np.fill_diagonal(C, s2)
            yeni = float(np.ones(4) @ C @ np.ones(4) / 16)
            print(
                f"    4. uye korelasyon {r4:.2f} -> topluluk RMSLE {np.sqrt(yeni):.5f}"
                f"  ({np.sqrt(yeni) - np.sqrt(E_top):+.5f})"
            )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
