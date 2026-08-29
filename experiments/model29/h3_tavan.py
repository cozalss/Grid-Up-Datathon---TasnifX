"""H3 -- HAFTA GUNU sinyalinin MODELSIZ TAVANI (test'in takvim ikizinde).

Q >= 0,01 kapisi icin bagimsiz bir ust sinir. Model egitmez, dogrudan
etiketlerden turer, dolayisiyla LightGBM kararsizligindan etkilenmez.

Yontem:
  ly = log1p(tuketim)
  r  = ly - trafo ortalamasi - gun etkisinin 15 gunluk merkezli hareketli ort.
       (trafo seviyesi ve mevsim cikarilir; geriye gun-ekseni sapmasi kalir)

Uc tavan:
  (a) KURESEL HAFTA GUNU : r'nin dayofweek ortalamasi
  (b) KURESEL HG + TATIL : (a) + tatil gostergeleri
  (c) TRAFO-ICI HAFTA GUNU : her trafonun kendi hafta gunu profili
      -- DURUST olcum: profil TEK haftalarda kestirilir, EGRI haftalarda
         degerlendirilir; kazanc = cov(f, r)^2 / var(f)  (optimal lambda ile)

Her tavan icin dMSE = aciklanabilen kare ortalama. Aday YONUN Q'su bu
degeri ASAMAZ (mukemmel model varsayimi). Pencere: 2025-04-01..2025-07-31,
gercek test penceresi 2026-04-01..2026-07-31'in TAKVIM IKIZI.
"""

import json
import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))

PENCERELER = {
    "yaz25_takvim_ikizi": ("2025-04-01", "2025-07-31"),
    "tum_egitim": ("2025-01-01", "2026-03-31"),
    "kis26": ("2025-12-01", "2026-03-31"),
}

tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    parse_dates=["tarih"],
    dtype={"tanim": str},
    usecols=["tanim", "tarih", "tuketim"],
)
tr["ly"] = np.log1p(tr.tuketim.clip(lower=0.0))
tr["r"] = tr.ly - tr.groupby("tanim").ly.transform("mean")

# mevsimden arindir: gun etkisinin 15 gunluk merkezli hareketli ortalamasi
gunluk = tr.groupby("tarih").r.mean().sort_index()
taban = gunluk.rolling(15, center=True, min_periods=5).mean()
tr["r"] = tr.r - tr.tarih.map(taban).to_numpy()
tr = tr.dropna(subset=["r"])

TATIL = pd.to_datetime(
    [
        "2025-01-01",
        "2025-03-30",
        "2025-03-31",
        "2025-04-01",
        "2025-04-23",
        "2025-05-01",
        "2025-05-19",
        "2025-06-06",
        "2025-06-07",
        "2025-06-08",
        "2025-06-09",
        "2025-07-15",
        "2025-08-30",
        "2025-10-29",
        "2026-01-01",
        "2026-03-20",
        "2026-03-21",
        "2026-03-22",
        "2026-04-23",
        "2026-05-01",
        "2026-05-19",
        "2026-05-27",
        "2026-05-28",
        "2026-05-29",
        "2026-06-06",
        "2026-07-15",
        "2026-08-30",
        "2026-10-29",
    ]
)
TSET = set(TATIL)


def dmse(f, r):
    """f yonunun optimal lambda ile saglayabilecegi EN BUYUK MSE dususu."""
    vf = float((f**2).mean())
    if vf <= 0:
        return 0.0, 0.0
    c = float((f * r).mean())
    return c * c / vf, vf


rap = {}
for ad, (a, b) in PENCERELER.items():
    d = tr[(tr.tarih >= a) & (tr.tarih <= b)].copy()
    r = d.r.to_numpy()
    top = float((r**2).mean())
    hg = d.tarih.dt.dayofweek.to_numpy()

    # (a) kuresel hafta gunu
    pro = pd.Series(r).groupby(hg).mean()
    fa = pro.reindex(hg).to_numpy()
    fa = fa - fa.mean()
    ka, _ = dmse(fa, r)

    # (b) kuresel hafta gunu + tatil
    tat = d.tarih.isin(TSET).to_numpy().astype(float)
    Xd = np.column_stack([np.eye(7)[hg], tat])
    beta, *_ = np.linalg.lstsq(Xd, r, rcond=None)
    fb = Xd @ beta
    fb = fb - fb.mean()
    kb, _ = dmse(fb, r)

    # (c) trafo-ici hafta gunu profili -- tek/cift hafta bolmesi ile DURUST
    hafta = d.tarih.dt.isocalendar().week.to_numpy()
    tek = (hafta % 2) == 1
    kod = pd.factorize(d.tanim.to_numpy())[0]
    anahtar = kod.astype(np.int64) * 7 + hg
    nk = int(kod.max()) + 1
    say = np.bincount(anahtar[tek], minlength=nk * 7)
    tpl = np.bincount(anahtar[tek], weights=r[tek], minlength=nk * 7)
    K = 5.0  # buzme sabiti
    prof = tpl / (say + K)  # buzulmus trafo x hafta-gunu ortalamasi
    # TRAFO SEVIYESINI CIKAR -- yoksa (c) hafta gunu degil seviye olcer
    ts = np.bincount(kod[tek], minlength=nk)
    tt = np.bincount(kod[tek], weights=r[tek], minlength=nk)
    tseviye_tek = tt / (ts + K)
    fc_tam = prof[anahtar] - tseviye_tek[kod]
    # degerlendirme yalnizca CIFT haftalarda; hedef de trafo-ici merkezlenir
    m = ~tek
    ts2 = np.bincount(kod[m], minlength=nk)
    tt2 = np.bincount(kod[m], weights=r[m], minlength=nk)
    tseviye_cift = tt2 / np.maximum(ts2, 1)
    hedef = r[m] - tseviye_cift[kod[m]]
    fc2 = fc_tam[m] - fc_tam[m].mean()
    kc, vfc = dmse(fc2, hedef - hedef.mean())

    rap[ad] = dict(
        satir=int(len(d)),
        gun_ekseni_var=top,
        a_kuresel_hg=dict(dMSE=ka, std=float(np.sqrt(ka)), R2=ka / top),
        b_kuresel_hg_tatil=dict(dMSE=kb, std=float(np.sqrt(kb)), R2=kb / top),
        c_trafo_ici_hg=dict(dMSE=kc, std=float(np.sqrt(kc)), R2=kc / float((r[m] ** 2).mean())),
        c_profil_std=float(np.sqrt(vfc)),
    )
    print(f"\n=== {ad}  ({a}..{b})  satir {len(d):,}  artik std {np.sqrt(top):.4f}")
    print(f"  (a) kuresel hafta gunu       dMSE={ka:.6f}  std={np.sqrt(ka):.4f}  R2={ka / top:.3f}")
    print(f"  (b) kuresel hg + tatil       dMSE={kb:.6f}  std={np.sqrt(kb):.4f}  R2={kb / top:.3f}")
    print(f"  (c) trafo-ici hg (durust)    dMSE={kc:.6f}  std={np.sqrt(kc):.4f}")

print("\nKAPI: aday YONUN Q'su bu dMSE degerlerini ASAMAZ (mukemmel model varsayimi).")
print("Gereken esik Q >= 0,01000")
json.dump(rap, open(os.path.join(BURA, "h3_tavan.json"), "w"), indent=1)
