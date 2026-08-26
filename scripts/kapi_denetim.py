"""BUTUNLUK KAPILARI -- bir gonderim dosyasi gondermeye HAZIR mi.

Dort kapi, hepsi gecmek zorunda:
    1. satir sayisi 714.688
    2. id sirasi sample_submission.csv ile BIREBIR (set esitligi YETMEZ)
    3. NaN yok
    4. negatif yok

Ek tani (kapı degil, bilgi): ortalama log1p seviyesi, sicak/soguk kirilimi,
sifir sayisi, ve verilen bir referans dosyaya gore log1p farkinin ozeti.

    uv run python scripts/kapi_denetim.py submissions/*.csv
    uv run python scripts/kapi_denetim.py --ref submissions/A.csv submissions/B.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
BEKLENEN = 714688


def yukle_referans() -> tuple[pd.Index, set[str]]:
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    tr = pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})
    return pd.Index(ss["id"]), set(tr["tanim"].unique())


def denetle(yol: Path, ss_id: pd.Index, sicak_set: set[str], ref: pd.DataFrame | None) -> dict:
    d = pd.read_csv(yol)
    ad = yol.name
    sonuc: dict[str, object] = {"dosya": ad}

    kolonlar = list(d.columns)
    sonuc["kolonlar"] = kolonlar
    if "id" not in d.columns or "tuketim" not in d.columns:
        sonuc["KAPI"] = "KALDI: kolon adlari yanlis"
        return sonuc

    n = len(d)
    sonuc["satir"] = n
    k1 = n == BEKLENEN
    k_mukerrer = int(d["id"].duplicated().sum())
    k2 = bool((d["id"].values == ss_id.values).all()) if k1 else False
    k2_set = set(d["id"]) == set(ss_id)
    v = d["tuketim"].to_numpy(dtype="float64")
    k_nan = int(np.isnan(v).sum())
    k_neg = int((v < 0).sum())
    k_sifir = int((v == 0).sum())
    k3 = k_nan == 0
    k4 = k_neg == 0

    sonuc["mukerrer_id"] = k_mukerrer
    sonuc["id_sirasi_birebir"] = k2
    sonuc["id_kume_esit"] = k2_set
    sonuc["nan"] = k_nan
    sonuc["negatif"] = k_neg
    sonuc["sifir"] = k_sifir

    lg = np.log1p(np.clip(v, 0, None))
    sonuc["ort_log1p"] = float(lg.mean())
    sonuc["std_log1p"] = float(lg.std())

    # sicak/soguk kirilimi
    tanim = d["id"].str.rsplit("_", n=1).str[0]
    sicak = tanim.isin(sicak_set).to_numpy()
    sonuc["n_sicak"] = int(sicak.sum())
    sonuc["n_soguk"] = int((~sicak).sum())
    sonuc["ort_log1p_sicak"] = float(lg[sicak].mean())
    sonuc["ort_log1p_soguk"] = float(lg[~sicak].mean())

    if ref is not None and len(ref) == n and bool((ref["id"].values == d["id"].values).all()):
        rv = np.log1p(np.clip(ref["tuketim"].to_numpy(dtype="float64"), 0, None))
        fark = lg - rv
        sonuc["ref_fark_ort"] = float(fark.mean())
        sonuc["ref_fark_ort_sicak"] = float(fark[sicak].mean())
        sonuc["ref_fark_ort_soguk"] = float(fark[~sicak].mean())
        sonuc["ref_fark_maxabs"] = float(np.abs(fark).max())
        sonuc["ref_degisen_satir"] = int((np.abs(fark) > 1e-9).sum())

    gecti = k1 and k2 and k3 and k4 and k_mukerrer == 0
    sonuc["KAPI"] = "GECTI" if gecti else "KALDI"
    if not gecti:
        neden = []
        if not k1:
            neden.append(f"satir {n} != {BEKLENEN}")
        if not k2:
            neden.append("id sirasi birebir DEGIL" + ("" if k2_set else " (kume de farkli)"))
        if not k3:
            neden.append(f"{k_nan} NaN")
        if not k4:
            neden.append(f"{k_neg} negatif")
        if k_mukerrer:
            neden.append(f"{k_mukerrer} mukerrer id")
        sonuc["KAPI"] = "KALDI: " + " | ".join(neden)
    return sonuc


def main() -> int:
    a = argparse.ArgumentParser(description="gonderim butunluk kapilari")
    a.add_argument("dosyalar", nargs="+")
    a.add_argument("--ref", default=None, help="log1p farki icin referans gonderim")
    ar = a.parse_args()

    ss_id, sicak_set = yukle_referans()
    print(f"referans: {len(ss_id):,} id | sicak trafo {len(sicak_set):,}\n")

    ref = None
    if ar.ref:
        ref = pd.read_csv(KOK / ar.ref if not Path(ar.ref).is_absolute() else ar.ref)
        print(f"referans dosya: {ar.ref}\n")

    kotu = 0
    for yol in ar.dosyalar:
        p = Path(yol)
        if not p.is_absolute():
            p = KOK / yol
        if not p.exists():
            print(f"{yol}: DOSYA YOK")
            kotu += 1
            continue
        s = denetle(p, ss_id, sicak_set, ref)
        durum = str(s["KAPI"])
        print(f"--- {s['dosya']}  ->  {durum}")
        if durum != "GECTI":
            kotu += 1
        for k in (
            "satir",
            "mukerrer_id",
            "id_sirasi_birebir",
            "nan",
            "negatif",
            "sifir",
            "ort_log1p",
            "std_log1p",
            "n_sicak",
            "n_soguk",
            "ort_log1p_sicak",
            "ort_log1p_soguk",
            "ref_fark_ort",
            "ref_fark_ort_sicak",
            "ref_fark_ort_soguk",
            "ref_fark_maxabs",
            "ref_degisen_satir",
        ):
            if k in s:
                val = s[k]
                if isinstance(val, float):
                    print(f"      {k:22s} {val:+.6f}")
                else:
                    print(f"      {k:22s} {val}")
        print()

    print(f"TOPLAM: {len(ar.dosyalar) - kotu}/{len(ar.dosyalar)} GECTI")
    return 1 if kotu else 0


if __name__ == "__main__":
    sys.exit(main())
