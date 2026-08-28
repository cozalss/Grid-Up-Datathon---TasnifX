"""SICAK ADAYLAR -- 3. tur: GUN EKSENI ve TAKVIM.

Neden bu eksen: uretim sicak uzmani ``YALIN_CIKARILAN`` yuzunden
``tk_*``, ``tatil*``, ``ramazan*`` kolonlarinin HICBIRINI gormuyor
(``tuketim_model.py:1027``). Yani takvim bilgisi modelde YOK; gun
eksenindeki tek uretim mudahalesi genlik olceklemesi (c=1,3301), ki o da
modelin zaten bildigi sekli buyutuyor -- bilmedigi bir sekli yaratmiyor.

Ayrica ``teshis.py`` sunu gosterdi: bloklar arasi TASIYAN tek eksen
hafta gunu (kor guz/kis +0,989). Trafo duzeyindeki her yapi TERS tasiniyor.
O yuzden bu tur yalnizca GUN duzeyinde calisir.

Kapi ayni: uc blokta ayni isaret + test dMSE <= -0,002.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, bloklari_kur, mse, rapor, taban_r, tablo_yaz  # noqa: E402


def gun_artigi(b, r0):
    """Gun basina ORTALAMA artik (trafo bilesimi dengelenmis)."""
    e = pd.Series(b.lgy - np.maximum(r0 + b.lgc, 0.0))
    t = pd.Series(b.cerceve["tanim"].to_numpy())
    e = e - e.groupby(t).transform("mean")  # trafo etkisini cikar
    g = pd.Series(b.cerceve["tarih"].to_numpy())
    return e.groupby(g).mean()


def gun_ozellikleri(b) -> pd.DataFrame:
    c = b.cerceve
    kol = [
        "sicaklik_ort",
        "sicaklik_max",
        "cdd22",
        "cdd22_ort7",
        "cdd18",
        "gun_uzunlugu_saat",
        "yagis_toplam",
        "nem_ort",
        "ulusal_gunluk",
        "ulusal_tepe_orani",
    ]
    kol = [k for k in kol if k in c.columns]
    d = c.groupby("tarih")[kol].mean()
    tar = pd.to_datetime(d.index)
    d["hg"] = tar.dayofweek
    d["hafta_sonu"] = (tar.dayofweek >= 5).astype(float)
    d["pazar"] = (tar.dayofweek == 6).astype(float)
    d["tatil"] = c.groupby("tarih")["tatil_mi"].mean() if "tatil_mi" in c.columns else 0.0
    if "tatil_agirligi" in c.columns:
        d["tatil_ag"] = c.groupby("tarih")["tatil_agirligi"].mean()
    if "tatil_mesafe" in c.columns:
        d["tatil_mes"] = c.groupby("tarih")["tatil_mesafe"].mean().clip(-7, 7)
    if "ramazan_ayi" in c.columns:
        d["ramazan"] = c.groupby("tarih")["ramazan_ayi"].mean()
    return d.astype("float64")


def gun_duzeltmesi(kolonlar: list[str], *, alpha: float = 20.0, kirp: float = 0.25):
    """Blok-disi ogrenilmis GUN duzeyi artik regresyonu."""

    def yap(bl, taban):
        cikti = {}
        for k in BLOKLAR:
            xs, ys = [], []
            for j in BLOKLAR:
                if j == k:
                    continue
                d = gun_ozellikleri(bl[j])
                a = gun_artigi(bl[j], taban[j]).reindex(d.index)
                x = d[kolonlar].to_numpy(dtype="float64")
                xs.append(x)
                ys.append((a - a.mean()).to_numpy(dtype="float64"))
            X = np.nan_to_num(np.vstack(xs))
            y = np.nan_to_num(np.concatenate(ys))
            mu, sd = X.mean(0), X.std(0) + 1e-9
            Z = (X - mu) / sd
            w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ y)
            dk = gun_ozellikleri(bl[k])
            Xk = (np.nan_to_num(dk[kolonlar].to_numpy(dtype="float64")) - mu) / sd
            tah = pd.Series(np.clip(Xk @ w, -kirp, kirp), index=dk.index)
            v = pd.Series(bl[k].cerceve["tarih"].to_numpy()).map(tah).fillna(0.0).to_numpy()
            cikti[k] = v - v.mean()
        return cikti

    return yap


def gun_grup_ofseti(anahtar_fn, *, n0: float = 5.0, kirp: float = 0.4):
    """GUN duzeyinde grup ofseti (ornegin hafta gunu / tatil), blok-disi."""

    def yap(bl, taban):
        cikti = {}
        for k in BLOKLAR:
            pay: dict[object, list[float]] = {}
            for j in BLOKLAR:
                if j == k:
                    continue
                a = gun_artigi(bl[j], taban[j])
                a = a - a.mean()
                g = anahtar_fn(pd.to_datetime(a.index))
                agg = (
                    pd.DataFrame({"a": a.to_numpy(), "g": g})
                    .groupby("g")["a"]
                    .agg(["mean", "size"])
                )
                for i, s in agg.iterrows():
                    pay.setdefault(i, [0.0, 0.0])
                    pay[i][0] += s["mean"] * s["size"]
                    pay[i][1] += s["size"]
            harita = {i: (s / n) * (n / (n + n0)) for i, (s, n) in pay.items()}
            b = bl[k]
            g = anahtar_fn(pd.DatetimeIndex(b.cerceve["tarih"]))
            v = pd.Series(g).map(harita).fillna(0.0).to_numpy(dtype="float64")
            cikti[k] = np.clip(v - v.mean(), -kirp, kirp)
        return cikti

    return yap


def toplayici(uretici, kat: float = 1.0):
    onb: dict = {}

    def aday_yap(bl, taban):
        if "d" not in onb:
            onb["d"] = uretici(bl, taban)
        d = onb["d"]

        def aday(b, r0):
            return r0 + kat * d[b.ad]

        return aday

    return aday_yap


def main() -> int:
    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}
    print("TABAN sicak MSE:", {k: round(mse(bl[k], taban[k]), 5) for k in BLOKLAR})

    print("\n" + "=" * 96)
    print("GUN DUZEYI ARTIK -- takvim kirilimlari (trafo etkisi cikarilmis)")
    print("=" * 96)
    for k in BLOKLAR:
        a = gun_artigi(bl[k], taban[k])
        a = a - a.mean()
        tar = pd.to_datetime(a.index)
        print(f"\n-- {k}  gun sayisi {len(a)}  std {a.std():.4f} --")
        hg = (
            pd.DataFrame({"a": a.to_numpy(), "hg": tar.dayofweek})
            .groupby("hg")["a"]
            .agg(["mean", "size"])
        )
        print("   hafta gunu:", " ".join(f"{i}:{v:+.4f}" for i, v in hg["mean"].items()))
        c = bl[k].cerceve
        if "tatil_mi" in c.columns:
            t = c.groupby("tarih")["tatil_mi"].mean().reindex(a.index) > 0.5
            print(
                f"   tatil  n={int(t.sum()):>3} ort {a[t.to_numpy()].mean():+.4f} | "
                f"tatil disi n={int((~t).sum()):>3} ort {a[(~t).to_numpy()].mean():+.4f}"
            )
        if "ramazan_ayi" in c.columns:
            rm = c.groupby("tarih")["ramazan_ayi"].mean().reindex(a.index) > 0.5
            if rm.any():
                print(f"   ramazan n={int(rm.sum()):>3} ort {a[rm.to_numpy()].mean():+.4f}")

    print("\n" + "=" * 96)
    print("GUN ARTIGININ HAVA ILE KORELASYONU (blok basi)")
    print("=" * 96)
    for k in BLOKLAR:
        a = gun_artigi(bl[k], taban[k])
        d = gun_ozellikleri(bl[k]).reindex(a.index)
        kor = {c: round(float(np.corrcoef(np.nan_to_num(d[c]), a)[0, 1]), 3) for c in d.columns}
        print(f"  {k}: {kor}")

    satirlar = []
    f = toplayici(gun_grup_ofseti(lambda t: t.dayofweek))(bl, taban)
    satirlar.append(rapor(bl, f, "C1 hafta gunu (gun duzeyi)", taban))
    f = toplayici(gun_grup_ofseti(lambda t: t.dayofweek), kat=0.5)(bl, taban)
    satirlar.append(rapor(bl, f, "C1b hafta gunu x0,5", taban))
    f = toplayici(gun_grup_ofseti(lambda t: (t.dayofweek >= 5).astype(int)))(bl, taban)
    satirlar.append(rapor(bl, f, "C2 hafta sonu ikili", taban))

    for k in BLOKLAR:
        bl[k].cerceve["_tatil"] = bl[k].cerceve.get(
            "tatil_mi", pd.Series(0, index=bl[k].cerceve.index)
        )
    tat = {k: bl[k].cerceve.groupby("tarih")["tatil_mi"].mean() for k in BLOKLAR}

    def tatil_anahtar(fabrika):
        def fn(t):
            s = pd.Series(0, index=range(len(t)))
            return s.to_numpy()

        return fn

    # tatil ofseti: gun bazli, blok-disi
    def tatil_ofseti(bl_, taban_):
        cikti = {}
        for k in BLOKLAR:
            pay = [0.0, 0.0]
            for j in BLOKLAR:
                if j == k:
                    continue
                a = gun_artigi(bl_[j], taban_[j])
                a = a - a.mean()
                t = (tat[j].reindex(a.index) > 0.5).to_numpy()
                if t.any():
                    pay[0] += float(a[t].mean()) * int(t.sum())
                    pay[1] += int(t.sum())
            b_t = (pay[0] / pay[1]) * (pay[1] / (pay[1] + 3.0)) if pay[1] else 0.0
            b = bl_[k]
            t = tat[k].reindex(pd.to_datetime(sorted(set(b.cerceve["tarih"])))) > 0.5
            harita = {d: (b_t if v else 0.0) for d, v in t.items()}
            v = pd.Series(b.cerceve["tarih"].to_numpy()).map(harita).fillna(0.0).to_numpy()
            cikti[k] = v - v.mean()
        return cikti

    f = toplayici(tatil_ofseti)(bl, taban)
    satirlar.append(rapor(bl, f, "C3 resmi tatil ofseti", taban))

    hava_kol = ["sicaklik_ort", "cdd22_ort7", "gun_uzunlugu_saat", "ulusal_tepe_orani"]
    hava_kol = [k for k in hava_kol if k in gun_ozellikleri(bl["yaz25"]).columns]
    for alpha in (5.0, 50.0):
        f = toplayici(gun_duzeltmesi(hava_kol, alpha=alpha))(bl, taban)
        satirlar.append(rapor(bl, f, f"C4 gun-hava regresyonu a={alpha:.0f}", taban))

    tam_kol = hava_kol + [
        c
        for c in ("hafta_sonu", "pazar", "tatil", "tatil_mes", "yagis_toplam", "nem_ort")
        if c in gun_ozellikleri(bl["yaz25"]).columns
    ]
    for alpha in (10.0, 100.0):
        f = toplayici(gun_duzeltmesi(tam_kol, alpha=alpha))(bl, taban)
        satirlar.append(rapor(bl, f, f"C5 gun-hava+takvim a={alpha:.0f}", taban))

    tablo_yaz(satirlar)
    yol = Path(__file__).resolve().parent / "adaylar3.jsonl"
    with yol.open("w", encoding="utf-8") as f2:
        for s in satirlar:
            f2.write(pd.Series(s).to_json() + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
