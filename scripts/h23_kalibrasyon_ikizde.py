"""H23 -- KALIBRASYON/BETA, artik DOGRU NUFUSTA (tek kalan BELIRSIZ).

DURUM
-----
docs/41 §4 "kalibrasyon doymus" hukmunu **kis26 soguk** uzerinde verdi
(taban RMSLE 1,98505, orakul tavani -0,002). Tik 5'in denetimi bunu
BELIRSIZ birakmisti: kis26 soguk nufusu (%59 tekil / %1 toplu) TEST
sogugunun (%12 tekil / %81 toplu) ikizi degil, ve o zaman yaz25 icin
soguk tahmin YOKTU.

Tik 7'de uretildi. Simdi olculebilir.

URETIM NE YAPIYOR
-----------------
``son_islem.py``: ofset uzayinda  r' = ort(r) + beta*(r - ort(r)),  beta=0,60.
Yani soguk tahminlerin YAYILIMINI genel ortalamaya dogru buzuyor. beta
hicbir veriden uydurulmadi -- sabit.

OLCULEN
-------
1. ORAKUL beta (her blokta kendi etiketiyle) -> ULASILABILIR UST SINIR
2. beta=0,60'in o tavana uzakligi
3. beta taramasi: yaz25 (IKIZ, kural 10) ve guz25 ve kis26
4. Kirpma tablosu (kural 1) secilen beta'da

Not: buzme SEVIYEYI korur (ortalama etrafinda), yani b_soguk'tan BAGIMSIZ
bir eksendir; ikisi ayni anda uygulanabilir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
BETA_URETIM = 0.60
P_SOGUK = 0.22159


def yukle(blok: str):
    npz = KOK / f"data/interim/deney/soguk_tahmin_{blok}.npz"
    meta = KOK / f"data/interim/{blok}_soguk_meta.parquet"
    if not npz.exists() or not meta.exists():
        return None
    z = np.load(npz)
    m = pd.read_parquet(meta).reset_index(drop=True)
    tohum = sorted({k.split("_")[0] for k in z.files})
    pay = sum(HARMAN.values())
    tah = {
        t: sum(HARMAN[a] * z[f"{t}_{a}"].astype("float64") for a in HARMAN) / pay
        for t in tohum
        if all(f"{t}_{a}" in z.files for a in HARMAN)
    }
    return m, tah


def buz(r: np.ndarray, beta: float) -> np.ndarray:
    mu = float(r.mean())
    return mu + beta * (r - mu)


def main() -> int:
    izgara = np.round(np.arange(0.30, 1.51, 0.10), 2)
    print("=" * 96)
    print("KALIBRASYON/BETA -- ofset uzayinda buzme, uretim beta=0,60")
    print("=" * 96)

    ozet = {}
    for blok in ("yaz25", "guz25", "kis26"):
        r0 = yukle(blok)
        if r0 is None:
            print(f"\n{blok}: onbellek yok")
            continue
        m, tah = r0
        lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
        lg_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
        # ofset uzayi: r = tahmin - log1p(guc);  hedef = lgy - log1p(guc)
        hedef = lgy - lg_guc
        etiket = "IKIZ" if blok == "yaz25" else ""
        print(
            f"\n--- {blok} {etiket}   {len(m):,} satir, {m.tanim.nunique():,} trafo, "
            f"{len(tah)} tohum"
        )

        # orakul beta (kapali form): min_b E[(hedef - (mu + b(r-mu)))^2]
        ork, mse_uretim, mse_ork, mse_b1 = [], [], [], []
        for t, v in tah.items():
            r = v - lg_guc
            mu = float(r.mean())
            x = r - mu
            y = hedef - mu
            b_ok = float((x @ y) / (x @ x))
            ork.append(b_ok)
            mse_b1.append(float(((hedef - r) ** 2).mean()))
            mse_uretim.append(float(((hedef - buz(r, BETA_URETIM)) ** 2).mean()))
            mse_ork.append(float(((hedef - buz(r, b_ok)) ** 2).mean()))
        ork = np.array(ork)
        print(
            f"    ORAKUL beta = {ork.mean():.4f} (std {ork.std(ddof=1):.4f})   uretim {BETA_URETIM}"
        )
        d_ur = np.array(mse_uretim) - np.array(mse_b1)
        d_ok = np.array(mse_ork) - np.array(mse_b1)
        print("    beta=1,00 (buzme yok) tabanina gore dMSE:")
        print(f"      uretim beta=0,60  {d_ur.mean():+.5f}")
        print(f"      ORAKUL beta       {d_ok.mean():+.5f}   (ULASILAMAZ ust sinir)")
        print(
            f"      uretimin oraküle uzakligi {d_ur.mean() - d_ok.mean():+.5f}"
            f"   -> test etkisi {P_SOGUK * (d_ur.mean() - d_ok.mean()):+.6f}"
        )

        # izgara
        sat = []
        for b in izgara:
            per = [
                float(((hedef - buz(v - lg_guc, b)) ** 2).mean())
                - float(((hedef - (v - lg_guc)) ** 2).mean())
                for v in tah.values()
            ]
            sat.append((b, float(np.mean(per)), float(np.std(per, ddof=1) / np.sqrt(len(per)))))
        en = min(sat, key=lambda s: s[1])
        print(f"    izgara optimumu beta={en[0]:.2f}  dMSE {en[1]:+.5f}")
        print(f"      {'beta':>6} {'dMSE':>10} {'SH':>9}")
        for b, v, s in sat:
            im = (
                "  <- uretim"
                if abs(b - BETA_URETIM) < 1e-9
                else ("  <- optimum" if b == en[0] else "")
            )
            print(f"      {b:>6.2f} {v:>+10.5f} {s:>9.5f}{im}")
        ozet[blok] = {
            "orakul": ork.mean(),
            "izgara": en[0],
            "kazanc_uretimden": en[1] - d_ur.mean(),
        }

    print("\n" + "=" * 96)
    print("HUKUM")
    print("=" * 96)
    for b, o in ozet.items():
        print(
            f"  {b:<7} orakul beta {o['orakul']:.3f}  izgara optimumu "
            f"{o['izgara']:.2f}  uretimden kazanc {o['kazanc_uretimden']:+.5f} "
            f"-> test {P_SOGUK * o['kazanc_uretimden']:+.6f}"
        )
    print("\n  Kural 10: hukum yaz25 (IKIZ) esasli verilir.")
    print("  Test etkisi -0,002'yi gecmiyorsa eksen TEMIZ KAPANIR (BELIRSIZ biter).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
