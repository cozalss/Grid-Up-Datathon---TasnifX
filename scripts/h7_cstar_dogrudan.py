"""H7-b -- c*'i VARSAYIMSIZ coz: parabolu GERCEK log-uzayi deltalarindan kur.

MSE(m) - MSE(0) = 2m*S + m^2*Q      d = log1p(v55) - log1p(v50)  (ALL rows)
    Q = mean(d^2)  ANALITIK olculur (kirpma dahil, varsayim yok)
    S = mean(d * e) LB'nin tek noktasindan cozulur
    m* = -S/Q ,  v66'nin konumu m66 = <d66,d55>/<d55,d55>  (izdusum)

Ayrica: kirpma sayimi, seviye sizintisi kaynagi, SOGUK gun ekseninin
kompozisyon-saglam olcumu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIK = KOK / "reports" / "h7_cstar"
CIK.mkdir(parents=True, exist_ok=True)

LB = {  # LB'de DONMUS skorlar (RMSLE)
    "tuketim_v50_nihai30.csv": 1.01686,
    "tuketim_v55_gunolcek.csv": 1.01591,
}


def gun_etkisi(tanim, gun, r):
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def main() -> int:
    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = (
        pd.read_csv(
            KOK / "data/raw/test.csv",
            usecols=["id", "tanim", "guc", "tarih"],
            encoding="utf-8",
            dtype={"tanim": str},
        )
        .set_index("id")
        .loc[ornek["id"]]
        .reset_index()
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    sicak = te["tanim"].isin(set(tr["tanim"])).to_numpy()
    soguk = ~sicak
    N = len(te)

    def yukle(ad):
        s = pd.read_csv(KOK / "submissions" / ad, encoding="utf-8")
        assert s["id"].equals(ornek["id"]), ad
        return s["tuketim"].to_numpy(dtype="float64")

    v50 = yukle("tuketim_v50_nihai30.csv")
    v55 = yukle("tuketim_v55_gunolcek.csv")
    v66 = yukle("tuketim_v66_c1335.csv")
    v67 = yukle("tuketim_v67_c1335_olay.csv")

    l50, l55, l66, l67 = (np.log1p(v) for v in (v50, v55, v66, v67))
    d55 = l55 - l50
    d66 = l66 - l50
    d67 = l67 - l50

    print("=== KIRPMA: c uygulamasi kac satiri tabana yasladi? ===")
    for ad, v, l in [("v55", v55, l55), ("v66", v66, l66)]:
        sifir_yeni = int(((v <= 0) & (v50 > 0)).sum())
        print(
            f"  {ad}: tahmin==0 satir {int((v <= 0).sum()):,} "
            f"(v50'de {int((v50 <= 0).sum()):,}, YENI sifirlanan {sifir_yeni:,})"
        )
    print(f"  v50 log1p<0.05 olan satir: {int((l50 < 0.05).sum()):,} / {N:,}")

    print("\n=== PARABOL: varsayimsiz Q, LB'den S ===")
    Q55 = float((d55**2).mean())
    Q66 = float((d66**2).mean())
    Q67 = float((d67**2).mean())
    mse50, mse55 = LB["tuketim_v50_nihai30.csv"] ** 2, LB["tuketim_v55_gunolcek.csv"] ** 2
    dm = mse55 - mse50
    S55 = (dm - Q55) / 2.0
    m_yildiz = -S55 / Q55
    m66 = float((d66 * d55).sum() / (d55 * d55).sum())
    m67 = float((d67 * d55).sum() / (d55 * d55).sum())
    A = mse50 - Q55 * m_yildiz**2  # MSE(m*) ... = mse50 + 2*0*S + 0 - ...
    en_iyi = mse50 - Q55 * m_yildiz**2
    print(f"  Q55 = mean(d55^2) = {Q55:.8f}   (=B, olcek m55=1 birimi cinsinden)")
    print(f"  Q66 = {Q66:.8f}   Q67 = {Q67:.8f}")
    print(f"  LB dMSE(v55-v50) = {dm:+.6f}  ->  S55 = {S55:+.8f}")
    print(f"  >>> OPTIMUM m* = {m_yildiz:.4f}  (v55 birimi; m=1 <-> uygulanan c~1,476)")
    print(f"      v66'nin konumu m66 = {m66:.4f}   v67'nin konumu m67 = {m67:.4f}")
    print(f"      >>> v67 uzerine GEREKEN EK OLCEK k = m*/m67 = {m_yildiz / m67:.4f}")
    kayip67 = Q55 * (m67 - m_yildiz) ** 2
    print(f"      v67'nin optimumdan KAYBI = {kayip67:+.8f} MSE")
    print(f"      optimumda ulasilabilir MSE = {en_iyi:.6f}  (RMSLE {np.sqrt(en_iyi):.5f})")
    print(f"      v50 tabanina gore toplam gun-ekseni kazanci = {en_iyi - mse50:+.6f}")

    # belirsizlik: LB skoru 5 haneli -> +-5e-6
    for eps in (+5e-6, -5e-6):
        dm2 = (LB["tuketim_v55_gunolcek.csv"] + eps) ** 2 - mse50
        S2 = (dm2 - Q55) / 2.0
        print(f"      skor {eps:+.0e} -> m* = {-S2 / Q55:.4f}  (k = {(-S2 / Q55) / m67:.4f})")

    print("\n=== EK OLCEK k icin dMSE tablosu (v67 tabani) ===")
    for k in [0.85, 0.90, 0.95, 1.00, 1.02, 1.05, 1.10, 1.12, 1.15, 1.20]:
        d = Q55 * ((k * m67 - m_yildiz) ** 2 - (m67 - m_yildiz) ** 2)
        print(f"   k={k:5.2f}   dMSE {d:+.8f}")

    # ---- (B) senaryosu: public LB alt kume olsaydi ----
    print("\n=== (B) senaryosu -- public LB alt kume, c*_formul=1,492 dogru ===")
    print("  v59_sicak20 GONDERILMEDI: (A)/(B) ayrimi LB'de COZULMEDI.")
    m_B = 0.492 / 0.476  # formul c=1.492'nin v55 birimindeki karsiligi
    print(f"  (B) dogruysa m* = {m_B:.4f}, gereken k = {m_B / m67:.4f}, ")
    print(
        f"     o k'da (A) dogruysa maliyet {Q55 * ((m_B - m_yildiz) ** 2 - (m67 - m_yildiz) ** 2):+.8f}"
    )
    print(f"     (B) dogruysa v67'nin kaybi {Q55 * (m67 - m_B) ** 2:+.8f} (Q55 (B)'de %63 kucuk)")

    # ---- SEVIYE SIZINTISI kaynagi ----
    print("\n=== SEVIYE: d'nin ortalamasi 0 olmali (kirpma bozuyor mu?) ===")
    for ad, d in [("d55", d55), ("d66", d66)]:
        print(
            f"  {ad}: ort(hepsi) {d.mean():+.3e}  ort(sicak) {d[sicak].mean():+.3e}  "
            f"ort(soguk) {d[soguk].mean():+.3e}   maks {d.max():+.4f} min {d.min():+.4f}"
        )
    # kirpilan satirlari cikar
    kirp = (v66 <= 0) & (v50 > 0) | (v55 <= 0) & (v50 > 0)
    print(
        f"  kirpilan {int(kirp.sum()):,} satir cikarilinca: "
        f"d55 ort {d55[~kirp & sicak].mean():+.3e}  d66 ort {d66[~kirp & sicak].mean():+.3e}"
    )
    print(
        f"  seviye kaymasinin MSE maliyeti ~ p*ort^2 = "
        f"{0.77841 * d66[sicak].mean() ** 2:.3e}  (ihmal edilebilir mi?)"
    )

    # ---- SOGUK gun ekseni: kompozisyon-saglam ----
    print("\n=== SOGUK GUN EKSENI: kompozisyon tuzagina karsi denetim ===")
    g = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31") & (tr["tuketim"] > 0)]
    rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        g["guc"].to_numpy(dtype="float64")
    )
    xg = pd.DataFrame({"t": g["tanim"].to_numpy(), "g": g["tarih"].to_numpy()})
    ng = xg["g"].nunique()
    tam = xg.groupby("t")["g"].nunique()
    tam = set(tam[tam >= 0.9 * ng].index)
    sel = np.isin(xg["t"].to_numpy(), list(tam))
    b_gecen = gun_etkisi(xg["t"].to_numpy()[sel], xg["g"].to_numpy()[sel], rg[sel])
    ia = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)

    log_guc = np.log1p(te["guc"].to_numpy(dtype="float64"))
    r67 = l67 - log_guc
    tan = te["tanim"].to_numpy()
    tar = te["tarih"].to_numpy()
    ngun_te = pd.Series(tar).nunique()
    say = pd.DataFrame({"t": tan[soguk], "g": tar[soguk]}).groupby("t")["g"].nunique()
    for esik, ad in [(0.0, "hepsi (kompozisyon KIRLI)"), (0.9, "%90 panel"), (0.99, "tam panel")]:
        tut = set(say[say >= esik * ngun_te].index)
        m = soguk & np.isin(tan, list(tut))
        if m.sum() < 1000:
            print(f"  {ad:28s}: {int(m.sum()):,} satir -- yetersiz")
            continue
        b = gun_etkisi(tan[m], tar[m], r67[m])
        ib = pd.Series(b.values, index=pd.to_datetime(b.index).dayofyear)
        ok = ia.index.intersection(ib.index)
        kor = float(np.corrcoef(ia[ok], ib[ok])[0, 1])
        print(
            f"  {ad:28s}: {len(tut):,} trafo, {int(m.sum()):,} satir  "
            f"sigma {b.std():.4f}  kor {kor:+.4f}  c_formul {kor * float(b_gecen.std()) / b.std():.3f}"
        )

    with (CIK / "dogrudan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "Q55": Q55,
                "Q66": Q66,
                "Q67": Q67,
                "S55": S55,
                "m_yildiz": m_yildiz,
                "m66": m66,
                "m67": m67,
                "gereken_k": m_yildiz / m67,
                "v67_kaybi": kayip67,
                "en_iyi_mse": en_iyi,
                "en_iyi_rmsle": float(np.sqrt(en_iyi)),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nyazildi: {CIK / 'dogrudan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
