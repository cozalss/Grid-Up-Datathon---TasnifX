"""H8f -- SOGUK GUN EKSENINDE IKI BLOKTA DA AYNI ISARETI VEREN BILESEN VAR MI?

DURUM (h8e)
-----------
Soguk gun ekseni TEK genlikle olceklenince bloklar ZIT yon soyluyor:
    yaz25 (T3, 94 temiz trafo)   c* = 2,78   dMSE -0,0675
    guz25 (T3, 272 temiz trafo)  c* = 0,67   dMSE -0,0041
Kural 9: iki ORTUSMEYEN zaman kesmesi ister. Tek genlik BU KAPIDAN GECMIYOR.

AMA frekans ayrimi iki blokta da AYNI YONU soyledi:
    yaz25  c_dusuk 3,33   c_yuksek 0,71
    guz25  c_dusuk 0,70   c_yuksek 0,39
YUKSEK FREKANS bileseni HER IKI BLOKTA DA 1'in ALTINDA -- yani modelin
soguk tahminlerindeki gunluk salinim GERCEKTEN FAZLA yayilmis.

MEKANIZMA (fiziksel beklenti, blok-bagimsiz):
Model bu trafolarin gecmisini GORMEDI. Gunluk salinimi komsu/benzer
trafolardan tahmin ediyor; bu tahmin GURULTU. Kuadratik kayipta gurultulu
bir bileseni ortalamaya dogru BUZMEK her zaman kazandirir. Bu bir mevsim
iddiasi degil, BUZULME iddiasidir -- ve bu yuzden bloklar arasi tasinir.

DUSUK FREKANS (mevsimsel rampa) ise mevsime bagli: yazin genis, guzun dar.
Onun genligi bloklar arasi TASINMAZ ve etiketsiz capa ister.

BU BETIK
--------
(c_dusuk, c_yuksek) izgarasini IKI BLOKTA da tarar, ikisinin de dMSE<0
verdigi ORTAK BOLGEYI bulur. Sonra o bolgede muhafazakar bir nokta secip
kirpma tablosunu koser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"
PENCERE = 31  # dusuk frekans hareketli ortalama penceresi
P_SOGUK = 0.22159


def iki_yonlu(v, bi, gi, nb, ng, tur=400):
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return a, b, mu


def blok_hazirla(ad: str) -> dict:
    m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
    ilk = m.groupby("tanim")["tarih"].transform("min")
    yas = (m["tarih"] - ilk).dt.days.to_numpy()
    say = m.groupby("tanim")["tanim"].transform("size").to_numpy()
    mask = (yas >= 7) & (say >= 60)  # T3 temiz panel
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(a["tanim"])
    gi, gun = pd.factorize(a["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    sira = np.argsort(gun.values)

    parcalar = []
    for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
        pr = np.load(p).astype("float64")[mask.to_numpy() if hasattr(mask, "to_numpy") else mask]
        am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
        bs = bm[sira]
        lo_s = pd.Series(bs).rolling(PENCERE, center=True, min_periods=1).mean().to_numpy()
        lo = np.empty(ng)
        lo[sira] = lo_s
        hi = bm - lo
        parcalar.append({"r": lgy - pr, "lo": lo, "hi": hi, "bm": bm})
    return {
        "ad": ad,
        "gi": gi,
        "bi": bi,
        "ng": ng,
        "nb": nb,
        "n": len(a),
        "trafo": a.tanim.nunique(),
        "parcalar": parcalar,
        "lgy": lgy,
    }


def dmse(bl: dict, c_lo: float, c_hi: float) -> np.ndarray:
    gi = bl["gi"]
    out = []
    for p in bl["parcalar"]:
        r = p["r"]
        duz = (c_lo - 1) * p["lo"][gi] + (c_hi - 1) * p["hi"][gi]
        out.append(float(((r - duz) ** 2).mean()) - float((r**2).mean()))
    return np.array(out)


def main() -> int:
    bloklar = {ad: blok_hazirla(ad) for ad in ("yaz25", "guz25")}
    for ad, bl in bloklar.items():
        print(
            f"{ad}: T3 temiz panel {bl['n']:,} satir, {bl['trafo']} trafo, "
            f"{len(bl['parcalar'])} tohum, {bl['ng']} gun"
        )

    print("\n" + "=" * 92)
    print(f"1. YALNIZ YUKSEK FREKANS BUZULMESI (c_dusuk = 1 SABIT, pencere={PENCERE})")
    print("=" * 92)
    print(
        f"  {'c_yuksek':>9} | {'yaz25 dMSE':>11} {'SH':>8} {'t':>7} {'poz':>5} "
        f"| {'guz25 dMSE':>11} {'SH':>8} {'t':>7} {'poz':>5} | {'ikisi de<0':>10}"
    )
    ortak = []
    for chi in np.round(np.arange(0.0, 1.05, 0.1), 2):
        sat = f"  {chi:9.2f} |"
        hepsi_neg = True
        for ad in ("yaz25", "guz25"):
            v = dmse(bloklar[ad], 1.0, float(chi))
            sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            t = v.mean() / sh if sh > 0 else float("nan")
            sat += f" {v.mean():+11.5f} {sh:8.5f} {t:+7.2f} {int((v < 0).sum()):>2}/{len(v)} |"
            if not (v.mean() < 0 and (v < 0).all()):
                hepsi_neg = False
        sat += f" {'EVET' if hepsi_neg else '':>10}"
        if hepsi_neg:
            ortak.append(float(chi))
        print(sat)

    print("\n" + "=" * 92)
    print("2. (c_dusuk, c_yuksek) IZGARASI -- ikisi de dMSE<0 olan ORTAK BOLGE")
    print("=" * 92)
    lo_izgara = np.round(np.arange(0.6, 2.05, 0.2), 2)
    hi_izgara = np.round(np.arange(0.0, 1.25, 0.2), 2)
    baslik = "c_lo/c_hi"
    print(f"  {baslik:>10}" + "".join(f"{h:>16.2f}" for h in hi_izgara))
    for lo in lo_izgara:
        sat = f"  {lo:>10.2f}"
        for hi in hi_izgara:
            vy = dmse(bloklar["yaz25"], float(lo), float(hi))
            vg = dmse(bloklar["guz25"], float(lo), float(hi))
            ok = vy.mean() < 0 and vg.mean() < 0 and (vy < 0).all() and (vg < 0).all()
            im = "*" if ok else " "
            sat += f"  {vy.mean():+7.4f}/{vg.mean():+6.4f}{im}"
        print(sat)
    print("  (hucre: yaz25 dMSE / guz25 dMSE ; * = ikisi de negatif ve TUM tohumlarda)")

    print("\n" + "=" * 92)
    print("3. MUHAFAZAKAR SECIM ve KIRPMA")
    print("=" * 92)
    if not ortak:
        print("  ORTAK BOLGE YOK -- hukum CURUDU")
        return 0
    # en muhafazakar: 1'e en yakin ortak c_yuksek DEGIL, ortadaki secilir;
    # buzulme yonunde 1'e dogru %50 geri cekilmis nokta
    c_hi_opt = min(ortak)
    c_hi_sec = round(1.0 - 0.5 * (1.0 - c_hi_opt), 2)
    print(f"  ortak bolge c_yuksek in [{min(ortak):.2f}, {max(ortak):.2f}]")
    print(f"  izgara optimumu {c_hi_opt:.2f} -> %50 BUZULMUS secim c_yuksek = {c_hi_sec:.2f}")

    for ad in ("yaz25", "guz25"):
        bl = bloklar[ad]
        for etiket, chi in (("optimum", c_hi_opt), ("buzulmus", c_hi_sec)):
            v = dmse(bl, 1.0, float(chi))
            sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
            print(
                f"\n  {ad} c_yuksek={chi:.2f} ({etiket})  dMSE {v.mean():+.5f} "
                f"SH {sh:.5f} t {v.mean() / sh:+.2f}  -> test etkisi "
                f"{P_SOGUK * v.mean():+.6f}"
            )
            # kirpma
            gi, bi, nb = bl["gi"], bl["bi"], bl["nb"]
            print(f"    {'K':>4} {'dMSE':>10} {'SH':>9} {'t':>7}  kazanan")
            for K in (0, 1, 5, 10, 25, 50):
                per, kaz = [], None
                for p in bl["parcalar"]:
                    r = p["r"]
                    duz = (chi - 1) * p["hi"][gi]
                    d = (r - duz) ** 2 - r**2
                    katki = np.bincount(bi, d, minlength=nb)
                    if kaz is None:
                        kaz = (int((katki < 0).sum()), nb)
                    at = np.argsort(katki)[:K]
                    tut = ~np.isin(bi, at) if K else np.ones(len(d), bool)
                    per.append(float(d[tut].mean()))
                vv = np.array(per)
                s2 = vv.std(ddof=1) / np.sqrt(len(vv)) if len(vv) > 1 else float("nan")
                ek = f"  {kaz[0]}/{kaz[1]} ({kaz[0] / kaz[1]:.1%})" if K == 0 else ""
                print(f"    {K:>4} {vv.mean():+10.5f} {s2:9.5f} {vv.mean() / s2:+7.2f}{ek}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
