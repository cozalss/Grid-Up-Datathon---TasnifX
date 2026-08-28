"""SOGUK KOHORT -- UCUNCU TUR: KIL PAYI KACIRAN IKI ADAY.

C6d  panele girisin ILK GUNLERINDE sabit asagi kaydirma (taranmamis delta)
C14  KOMSULUK: ayni ilcedeki CANLI trafolarin o gunku seviyesi
     -> UYARI: uretimde UYGULANAMAZ, cunku test penceresinde canli
        trafolarin GERCEKLERI yok. Yalnizca sinyalin varligini olcer.
C15  ORACLE TAVANLARI: gun ekseni / ilce-gun ekseni / trafo ekseni
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ortak  # noqa: E402
from adaylar import yazdir  # noqa: E402
from ortak import BLOKLAR, SOGUK_PAY, Blok, mse, taban_r, tum_bloklar  # noqa: E402

CIKTI = Path(__file__).resolve().parent / "adaylar3.jsonl"
BAS = {"yaz25": "2025-04-01", "guz25": "2025-08-01", "kis26": "2025-12-01"}
SON = {"yaz25": "2025-07-31", "guz25": "2025-11-30", "kis26": "2026-03-31"}


def c6d(bloklar: dict[str, Blok]) -> list[dict]:
    out = []
    for ust in (0, 2, 6, 13):
        for dl in (-0.15, -0.25, -0.35, -0.50):
            s = {"aday": f"C6d yas<={ust} d={dl:+.2f}"}
            for ad in BLOKLAR:
                b = bloklar[ad]
                r0 = taban_r(b)
                s[ad] = mse(b, r0) - mse(b, r0 + np.where(b.yas <= ust, dl, 0.0))
            out.append(s)
    return out


def c14_komsuluk(bloklar: dict[str, Blok]) -> list[dict]:
    tr = pd.read_csv(ortak.KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    _, ilce = ortak._lokasyon_parcala(tr["lokasyon"])
    tr["ilce"] = ilce
    tr["lg"] = np.log1p(tr["tuketim"].clip(lower=0)) - np.log1p(tr["guc"])
    out = []
    for lam in (0.5, 1.0):
        s1 = {"aday": f"C14a canli GUN endeksi lam={lam:.1f} (UYGULANAMAZ)"}
        s2 = {"aday": f"C14b canli ILCE-GUN endeksi lam={lam:.1f} (UYGULANAMAZ)"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            w = tr[
                (tr.tarih >= BAS[ad]) & (tr.tarih <= SON[ad]) & (~tr.tanim.isin(set(b.tanim)))
            ].copy()
            w["dev"] = w["lg"] - w.groupby("tanim")["lg"].transform("mean")
            gun = w.groupby("tarih")["dev"].mean()
            gun = gun - gun.mean()
            ig = w.groupby(["ilce", "tarih"])["dev"].mean()
            ig = ig - ig.mean()
            r0 = taban_r(b)
            m0 = mse(b, r0)
            g1 = pd.Series(pd.to_datetime(b.tarih)).map(gun).fillna(0).to_numpy()
            idx = pd.MultiIndex.from_arrays([b.ilce, pd.to_datetime(b.tarih)])
            g2 = pd.Series(ig.reindex(idx).to_numpy()).fillna(0.0).to_numpy()
            s1[ad] = m0 - mse(b, r0 + lam * g1)
            s2[ad] = m0 - mse(b, r0 + lam * g2)
        out += [s1, s2]
    return out


def c15_oracle(bloklar: dict[str, Blok]) -> list[dict]:
    out = []
    for ad_e, anah in (("gun", ["gun"]), ("ilce x gun", ["ilce", "gun"]), ("trafo", ["t"])):
        s = {"aday": f"C15 ORACLE {ad_e} ekseni (SIZINTILI ust sinir)"}
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban_r(b)
            e = b.lgy - (r0 + b.lgc)
            d = pd.DataFrame({"gun": b.tarih, "ilce": b.ilce, "t": b.tanim, "e": e})
            e1 = e - d.groupby(anah)["e"].transform("mean").to_numpy()
            s[ad] = float((e * e).mean()) - float((e1 * e1).mean())
        out.append(s)
    return out


def main() -> int:
    bloklar = tum_bloklar()
    hepsi: list[dict] = []
    hepsi += yazdir("C6d -- ILK GUNLER SABIT KAYDIRMA", c6d(bloklar))
    hepsi += yazdir(
        "C14 -- KOMSULUK (canli trafo gun endeksi) -- URETIMDE UYGULANAMAZ", c14_komsuluk(bloklar)
    )
    hepsi += yazdir("C15 -- ORACLE TAVANLARI (aday DEGIL, ust sinir)", c15_oracle(bloklar))
    print(f"\nNOT: genel dMSE = ortalama(soguk dMSE) x {SOGUK_PAY:.4f}")
    with CIKTI.open("w", encoding="utf-8") as f:
        for s in hepsi:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
