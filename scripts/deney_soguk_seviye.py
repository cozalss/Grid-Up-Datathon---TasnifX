"""SOGUK UZMANIN SEVIYE YANLILIGI -- HUKUM: KAYDIRMA YOK, ama a DUSURULDU.

TESHIS (bagimsiz bir ajan taramasindan geldi)
---------------------------------------------
Soguk uzman ``maske=1,00`` ile egitiliyor. ``soguk_maskele`` maskelenen her
trafo icin ``soguk_mu=1`` yaziyor -- oran 1,00 olunca BUTUN egitim satirlari
soguk isaretleniyor ve bayrak SABITLESIYOR. Model "soguk olmak" ayrimini
yapamaz, butun nufusun kosullu ortalamasini ogrenir. Gercekten de uretim
kis26 sogukta ofset ortalamasini ~0,42 tahmin ediyor, gercek ~0,68.

Iddia: dogal soguk satirlar ayni gun + ilce + kVA kademesindeki sicaklara
gore sistematik olarak YUKARIDA, o yuzden pozitif bir sabit kaydirma
bedava kazanc verir.

OLCUM: IDDIA CURUTULDU
----------------------
Ayni hucre (gun x ilce_key x 24 kVA kovasi, hucrede >=3 sicak satir sarti):

    blok    eslesen soguk satir     fark        t
    yaz25              18.234    -0,1690    -12,8    <- NEGATIF
    guz25              34.456    +0,3316    +31,4
    kis26              57.730    +0,1844    +23,6

Uc blokta da pozitif DEGIL. Etki gercek (her uc t de buyuk) ama MEVSIMSEL:
yazin yeni trafolar ayni hucredeki yerlesiklerin ALTINDA, kis ve
sonbaharda USTUNDE. Test Nisan-Temmuz -- yani isaret ``yaz25``inki.
Pozitif kaydirma test penceresinde TERS yone calisirdi. ALINMADI.

YAN URUN: a (hucre agirligi) fazla yuksekmis
--------------------------------------------
Ayni izgara, kayma=0 sutunu:

                    a=0,30    0,40      0,47      0,55
    kis26 HAM      1,82198  1,82133   1,82117   1,82127
    TEST kVA kar.  2,01055  2,01044   2,01063   2,01111

kis26 HAM 0,47 diyor; TEST kVA karisimina agirliklandirilinca optimum
0,40'a kayiyor. Bagimsiz bir mevsim analizi de Nisan-Temmuz aylari icin
a* = 0,39-0,41 olcmus. Iki test-ilgili sinyal ayni yerde; kis26'daki
maliyet 0,00006. A_HUCRE 0,55 -> 0,40 alindi.

    python scripts/deney_soguk_seviye.py
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
import son_islem_gun as si  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

TOHUMLAR = (1000, 1001, 1002)
ONBELLEK = KOK / "data" / "interim" / "deney" / "soguk_tahmin_kis26.npz"
KAYIT = KOK / "experiments" / "soguk_seviye.jsonl"
KAYMALAR = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
A_DEGERLERI = (0.30, 0.40, 0.47, 0.55)


def kova(guc: np.ndarray, kenar: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(kenar, np.log1p(guc), side="right") - 1, 0, len(kenar) - 2)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print("SOGUK UZMANIN SEVIYE YANLILIGI")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)

    # --- A) MODEL KARISMADAN: ayni hucrede soguk - sicak farki, uc blokta ---
    print("\n--- A) Ayni gun x ilce x kVA kovasinda SOGUK - SICAK ofset farki ---")
    print("  (model yok -> ezber kanali yok -> uc blok da gecerli)")
    print(f"  {'blok':7}{'eslesen soguk satir':>21}{'fark':>10}{'SH':>9}{'t':>8}")
    farklar = {}
    for b in tm.BLOKLAR:
        _, dog, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        lg = np.log1p(dog["guc"].to_numpy(dtype="float64"))
        ofs = np.log1p(gercek) - lg
        kenar = np.linspace(lg.min(), lg.max() + 1e-9, 25)
        anahtar = (pd.to_datetime(dog["tarih"]).astype("int64").astype(str).to_numpy() + "|"
                   + pd.Series(dog["ilce_key"].to_numpy()).astype(str).to_numpy() + "|"
                   + pd.Series(
                       kova(dog["guc"].to_numpy(dtype="float64"), kenar)
                   ).astype(str).to_numpy())
        df = pd.DataFrame({"a": anahtar, "ofs": ofs, "soguk": soguk})
        g = df.groupby("a")
        sicak_ort = g.apply(lambda x: x.loc[~x["soguk"], "ofs"].mean(), include_groups=False)
        n_sicak = g.apply(lambda x: int((~x["soguk"]).sum()), include_groups=False)
        gecerli = sicak_ort.index[(n_sicak >= 3) & sicak_ort.notna()]
        m = df["soguk"].to_numpy() & df["a"].isin(set(gecerli)).to_numpy()
        fark = df.loc[m, "ofs"].to_numpy() - df.loc[m, "a"].map(sicak_ort).to_numpy()
        sh = float(fark.std(ddof=1) / np.sqrt(len(fark)))
        farklar[b.ad] = float(fark.mean())
        print(f"  {b.ad:7}{int(m.sum()):21,}{fark.mean():10.4f}{sh:9.4f}{fark.mean() / sh:8.1f}")
    print(f"  UC BLOKTA DA POZITIF: {all(v > 0 for v in farklar.values())}")
    print(f"  en kucuk {min(farklar.values()):.4f}  "
          f"medyan {np.median(list(farklar.values())):.4f}")

    # --- B) kis26'da kaydirma x a izgarasi, TEST kVA karisimina agirlikli ---
    parca, dog, gercek, soguk = di.blok_parcalari(egitim, "kis26")
    dg = dog[soguk]
    y = gercek[soguk]
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    gun = pd.to_datetime(dg["tarih"]).to_numpy()
    ay = pd.to_datetime(dg["tarih"]).dt.to_period("M").astype(str).to_numpy()
    z = np.load(ONBELLEK)

    ham = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    ham["t"] = pd.to_datetime(ham["tarih"])
    kaynak = ham[(ham["t"] >= si.TABLO_BASLANGIC) & (ham["t"] < pd.Timestamp(gun.min()))]
    hedef = pd.DataFrame({"guc": dg["guc"].to_numpy(), "lokasyon": dg["lokasyon"].to_numpy()})
    hucre = si.hucre_etkisi(kaynak, hedef)

    # TEST kVA karisimina agirlik: kis26 soguk medyani 400, testinki 630
    te = pd.read_csv(KOK / "data/raw/test.csv", usecols=["tanim", "guc"],
                     encoding="utf-8", dtype={"tanim": str})
    tr_t = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"],
                       encoding="utf-8", dtype={"tanim": str})
    te_soguk = te[~te["tanim"].isin(set(tr_t["tanim"]))]
    kenar_g = np.linspace(np.log1p(ham["guc"]).min(), np.log1p(ham["guc"]).max() + 1e-9, 25)
    kv_test = kova(te_soguk["guc"].to_numpy(dtype="float64"), kenar_g)
    kv_kis = kova(dg["guc"].to_numpy(dtype="float64"), kenar_g)
    p_test = pd.Series(kv_test).value_counts(normalize=True)
    p_kis = pd.Series(kv_kis).value_counts(normalize=True)
    w_test = (pd.Series(kv_kis).map(p_test).fillna(0.0)
              / pd.Series(kv_kis).map(p_kis)).to_numpy()
    w_test = w_test / w_test.mean()
    print(f"\n  kVA kovasi medyani  kis26 soguk {np.median(kv_kis):.0f}  TEST soguk "
          f"{np.median(kv_test):.0f}   agirlik araligi {w_test.min():.2f}-{w_test.max():.2f}")

    def gruplu(v: np.ndarray, k: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(k).transform("mean").to_numpy()

    def uygula(r: np.ndarray, a: float, kayma: float) -> np.ndarray:
        n = pd.Series(gun).groupby(gun).transform("size").to_numpy().astype("float64")
        w = n / (n + si.M_GUN)
        s = w * gruplu(r, gun) + (1 - w) * gruplu(r, ay)
        s = s - gruplu(s, ay) + gruplu(r, ay) + kayma
        hr = w * gruplu(hucre, gun) + (1 - w) * gruplu(hucre, ay)
        e = hucre - hr
        e = e - gruplu(e, ay)
        return s + a * e + si.B_MODEL * (r - s)

    def skorla(ofs: np.ndarray, agirlik: np.ndarray | None) -> float:
        hata = (ofs + lg - np.log1p(y)) ** 2
        return float(np.sqrt(np.average(hata, weights=agirlik)))

    kayitlar = []
    for etiket, agirlik in (("kis26 HAM", None), ("TEST kVA karisimi", w_test)):
        print(f"\n--- {etiket} ---")
        print("  " + f"{'a / kayma':>9}"
              + "".join(f"{k:>10.2f}" for k in KAYMALAR))
        for a in A_DEGERLERI:
            satir = []
            for kayma in KAYMALAR:
                sk = [skorla(uygula(z[f"{t}_cat"] - lg, a, kayma), agirlik) for t in TOHUMLAR]
                satir.append(float(np.mean(sk)))
                kayitlar.append({"olcut": etiket, "a": a, "kayma": kayma, "rmsle": satir[-1]})
            print(f"  {a:9.2f}" + "".join(f"{v:10.5f}" for v in satir))

    for etiket in ("kis26 HAM", "TEST kVA karisimi"):
        alt = [k for k in kayitlar if k["olcut"] == etiket]
        u = next(k for k in alt if k["a"] == 0.55 and k["kayma"] == 0.0)
        e = min(alt, key=lambda k: k["rmsle"])
        print(f"\n  {etiket}: uretim(a=0,55 kayma=0) {u['rmsle']:.5f}  ->  "
              f"en iyi a={e['a']:.2f} kayma={e['kayma']:.2f} {e['rmsle']:.5f}"
              f"  kazanc {u['rmsle'] - e['rmsle']:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
