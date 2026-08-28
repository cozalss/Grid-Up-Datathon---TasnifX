"""PROB TASARIMI -- AY deseni yalniz yaz25'ten (testin MEVSIM IKIZI).

Bloklarin aylari ortusmez (yaz25 4-7, guz25 8-11, kis26 12-3), bu yuzden
"blok-disi ogren" yontemi ay ekseninde TANIMSIZDIR (desen.py'de 0 cikti).
Test penceresi 2026-04-01..07-31 -> tam olarak yaz25'in aylari. Bu yuzden ay
deseni YALNIZ yaz25'ten okunur; kural 7 saglanir (olcum yaz25'i icerir) ama
bloklar arasi dogrulama YOKTUR -- bu yon en zayif kanittir.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))

from tavan import SIC, _soguk_modul  # noqa: E402


def main() -> None:
    cikti: dict[str, dict[str, float]] = {}

    bl = SIC.bloklari_kur()
    b = bl["yaz25"]
    r0 = SIC.taban_r(b)
    e = b.lgy - np.maximum(r0 + b.lgc, 0.0)
    e = e - e.mean()
    ay = b.cerceve["ay"].to_numpy()
    o = pd.Series(e).groupby(pd.Series(ay)).mean()
    print("SICAK yaz25 ay ofsetleri:", {int(k): round(float(v), 5) for k, v in o.items()})
    print("  ay basina satir:", pd.Series(ay).value_counts().sort_index().to_dict())
    cikti["sicak"] = {str(int(k)): float(v) for k, v in o.items()}
    del bl

    SOG = _soguk_modul()
    sb = SOG.tum_bloklar()
    b = sb["yaz25"]
    r0 = SOG.taban_r(b)
    e = b.lgy - (r0 + b.lgc)
    e = e - e.mean()
    ay = pd.to_datetime(b.tarih).month.to_numpy()
    o = pd.Series(e).groupby(pd.Series(ay)).mean()
    print("SOGUK yaz25 ay ofsetleri:", {int(k): round(float(v), 5) for k, v in o.items()})
    print("  ay basina satir:", pd.Series(ay).value_counts().sort_index().to_dict())
    cikti["soguk"] = {str(int(k)): float(v) for k, v in o.items()}

    (BURA / "ay_deseni.json").write_text(json.dumps(cikti, indent=2), encoding="utf-8")
    print("yazildi: ay_deseni.json")


if __name__ == "__main__":
    main()
