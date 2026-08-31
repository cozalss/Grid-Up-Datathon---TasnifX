"""p03 tezgah: SIZINTISIZ yaz25 geri-testi.

Kural: yaz25 (2025-04-01..07-31) hedefi HICBIR yerde kullanilmaz -- ne etiket
ne de ozellik olarak. Bunun icin egitim kesimi yaz25'ten SONRA secilir ve
egitim gecmisi yaz25'i icermeyecek sekilde kirpilir.

  EGITIM  kesim 2025-11-30, gecmis 2025-09-01..11-30 (3 ay),
          hedef 2025-12-01..2026-03-31 (4 ay)
  DEGERL. kesim 2025-03-31, gecmis 2025-01-01..03-31 (3 ay),
          hedef 2025-04-01..07-31 (4 ay)  <- yaz25

Gecmis uzunlugu (3 ay) ve ufuk uzunlugu (4 ay) iki tarafta ayni.
`idnum` ozelligi ATILIR ve degerlendirmede SOGUK olan trafolar egitimden
cikarilir -- kimlik ezberi kanali kapali.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m30_ozellik import kur, yukle_ham  # noqa: E402

E_KESIM = "2025-11-30"
E_GEC_BAS = "2025-09-01"
E_HED_SON = "2026-03-31"
D_KESIM = "2025-03-31"
D_GEC_BAS = "2025-01-01"
D_HED_SON = "2025-07-31"
AT = ["idnum"]


def ortam():
    tr, te = yukle_ham()
    return tr, te


def blok(tr, kesim, gec_bas, hed_son, at_trafo=None):
    k = pd.Timestamp(kesim)
    gec = tr[(tr.tarih <= k) & (tr.tarih >= pd.Timestamp(gec_bas))]
    hed = tr[(tr.tarih > k) & (tr.tarih <= pd.Timestamp(hed_son))]
    sicak = set(gec.tanim)
    if at_trafo is not None:
        gec = gec[~gec.tanim.isin(at_trafo)]
        hed = hed[~hed.tanim.isin(at_trafo)]
    X = kur(gec, hed, kesim, sicak)
    X = X.drop(columns=[c for c in AT if c in X.columns])
    return X, np.log1p(hed.tuketim.to_numpy()), hed.reset_index(drop=True)


def veri(tr):
    """(Xe, ye, Xd, yd, hed_d) dondurur; hizalanmis kategoriler."""
    from m33_durust import hizala

    d_gec = tr[(tr.tarih <= D_KESIM) & (tr.tarih >= D_GEC_BAS)]
    d_hed = tr[(tr.tarih > D_KESIM) & (tr.tarih <= D_HED_SON)]
    d_soguk = set(d_hed.tanim) - set(d_gec.tanim)  # degerlendirmede soguk olanlar
    Xe, ye, _ = blok(tr, E_KESIM, E_GEC_BAS, E_HED_SON, at_trafo=d_soguk)
    Xd, yd, hd = blok(tr, D_KESIM, D_GEC_BAS, D_HED_SON)
    Xe, Xd = hizala(Xe, Xd)
    Xd = Xd[Xe.columns]
    return Xe, ye, Xd, yd, hd, d_soguk


def rmsle(yd, p):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(yd)) ** 2)))
