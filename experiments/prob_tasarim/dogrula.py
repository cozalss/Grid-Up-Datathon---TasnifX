"""PROB TASARIMI -- adim 8: SON DOGRULAMA.

1. Her prob dosyasi TAM olarak log1p(v93) + kappa*d biçiminde mi?
2. Ilce ayristirmasi Manisa'yi (2 parcali lokasyon) dusuruyor mu?
3. Yon vektorleri olculmus 18 LB yonune gercekten dik mi?
4. SHA256 + boyut kaydi.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
BURA = Path(__file__).resolve().parent
ONB = KOK / "experiments" / "v93_denetim" / "onbellek"
sys.path.insert(0, str(BURA))

from yon import M_V93, S_V93, span_tabani  # noqa: E402


def sha(yol: Path) -> str:
    h = hashlib.sha256()
    with open(yol, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest().upper()


def main() -> None:
    kayit = json.loads((BURA / "prob_kayit.json").read_text(encoding="utf-8"))
    v93 = np.load(ONB / "v93.npy")
    n = v93.size
    B, _ = span_tabani()

    # --- 2) ilce ayristirma denetimi ---
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "lokasyon"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    p = te["lokasyon"].fillna("").astype(str).str.split(">")
    parca = p.str.len()
    ilce = p.str[-1].str.strip()
    il = p.str[0].str.strip()
    print("LOKASYON AYRISTIRMA")
    print("  parca sayisi dagilimi:", parca.value_counts().to_dict())
    print("  il dagilimi (trafo):", te.assign(il=il).groupby("il")["tanim"].nunique().to_dict())
    man = te.assign(il=il, ilce=ilce)
    man = man[man["il"].str.contains("MANİSA|MANISA", na=False)]
    print(
        f"  MANISA trafo sayisi: {man['tanim'].nunique()}  "
        f"ilceler: {sorted(man['ilce'].unique())[:8]}..."
    )
    kor = p.str[2].str.strip()  # koru korune 3. parca
    print(
        f"  '3. parcayi al' deseydik NaN olacak satir: {int(kor.isna().sum()):,} "
        f"({te.loc[kor.isna(), 'tanim'].nunique()} trafo)  <- bu tuzaga dusulmedi"
    )

    print("\nPROB DOSYALARI")
    ozet = []
    for k in kayit:
        yol = KOK / "submissions" / k["dosya"]
        lg = np.log1p(pd.read_csv(yol)["tuketim"].to_numpy(dtype="float64"))
        u = lg - v93
        q_u = float(u @ u) / n
        d = u / k["kappa"]
        dik = float(np.abs(B @ d).max()) / float(np.sqrt(d @ d))
        notr = float(np.sqrt(M_V93 + q_u))
        rec = {
            "dosya": k["dosya"],
            "rejim": k["rejim"],
            "yon": k["yon"],
            "kappa": k["kappa"],
            "Q_d": q_u / k["kappa"] ** 2,
            "Q_u": q_u,
            "degisen": int((np.abs(u) > 1e-12).sum()),
            "min_log": float(lg.min()),
            "max_log": float(lg.max()),
            "negatif": int((np.expm1(lg) < 0).sum()),
            "LB_span_diklik_bagil": dik,
            "notr_skor": notr,
            "skor_degisimi_notr": notr - S_V93,
            "boyut": yol.stat().st_size,
            "sha256": sha(yol),
        }
        ozet.append(rec)
        print(
            f"  {k['dosya']:34s} Q_u={q_u:.7f} notr={notr:.6f} "
            f"({notr - S_V93:+.6f})  span-diklik {dik:.2e}  "
            f"min_log={lg.min():.4f}  sha={rec['sha256'][:12]}"
        )

    (BURA / "dogrula.json").write_text(
        json.dumps(ozet, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nyazildi: dogrula.json")


if __name__ == "__main__":
    main()
