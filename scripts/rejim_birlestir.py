"""IKI GONDERIMI REJIM BAZINDA BIRLESTIR -- izolasyon surumleri, sifir egitim.

NEDEN
-----
``v52`` iki degisikligi BIRLIKTE tasiyor: sicak uzmana 10 kolon, soguk uzmana
8 kolon. LB'de beklenenden kotu gelirse "hangisi bozdu?" sorusu dogar ve
normalde bunu ayirmak ayri bir uretim kosusu (~90 dakika/parti) gerektirir.

Gerektirmiyor. SICAK ve SOGUK satirlar AYRIK kumelerdir (``soguk_mu``), yani
iki gonderim dosyasindan satir bazinda birlestirme yaparak her iki izolasyon
surumu de BEDAVA kurulur:

    yalniz-sicak = A'nin SICAK satirlari + B'nin SOGUK satirlari
    yalniz-soguk = B'nin SICAK satirlari + A'nin SOGUK satirlari

Boylece uc gonderim hakkiyla su uc soru ayri ayri yanitlanir:
    1. v52 butunuyle iyi mi?
    2. sicak kolonlar tek basina ne yapiyor?
    3. soguk kolonlar tek basina ne yapiyor?

BILINEN KUSUR
-------------
v50 30 tohum, v52 ~13 tohum. Birlestirilmis dosyada iki rejim FARKLI tohum
sayisindan gelir. Tohum farkinin buyuklugu OLCULMUS ve kucuktur: k=13 ile
k=30 arasi ~+0,0006 (docs/40 §4, sigma=0,15671). Yani yapilandirma sorusu
icin izolasyon temiz; mutlak skoru yorumlarken bu 0,0006 akilda tutulur.

    python scripts/rejim_birlestir.py --sicak A.csv --soguk B.csv --cikis C.csv

``--sicak`` SICAK satirlarin alinacagi dosya, ``--soguk`` SOGUK satirlarin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

TEST_PARQUET = KOK / "data" / "interim" / "deney" / "test.parquet"


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--sicak", required=True, help="SICAK satirlarin alinacagi gonderim")
    a.add_argument("--soguk", required=True, help="SOGUK satirlarin alinacagi gonderim")
    a.add_argument("--cikis", required=True)
    ar = a.parse_args()

    def oku(yol: str) -> pd.DataFrame:
        p = Path(yol)
        if not p.is_absolute() and not p.exists():
            p = KOK / "submissions" / p.name
        if not p.exists():
            raise SystemExit(f"dosya yok: {p}")
        return pd.read_csv(p)

    s_df, c_df = oku(ar.sicak), oku(ar.soguk)
    if len(s_df) != len(c_df):
        raise SystemExit(f"satir sayilari farkli: {len(s_df)} vs {len(c_df)}")
    if not (s_df["id"].to_numpy() == c_df["id"].to_numpy()).all():
        raise SystemExit("id sirasi ayni DEGIL -- birlestirme guvenli degil")

    te = pd.read_parquet(TEST_PARQUET, columns=["soguk_mu"])
    if len(te) != len(s_df):
        raise SystemExit(f"test {len(te)} satir, gonderim {len(s_df)} satir -- hizalanmiyor")
    soguk = (te["soguk_mu"] == 1).to_numpy()

    yeni = s_df.copy()
    yeni.loc[soguk, "tuketim"] = c_df.loc[soguk, "tuketim"].to_numpy()

    d_s = float(
        np.abs(
            yeni.loc[~soguk, "tuketim"].to_numpy() - s_df.loc[~soguk, "tuketim"].to_numpy()
        ).max()
    )
    d_c = float(
        np.abs(yeni.loc[soguk, "tuketim"].to_numpy() - c_df.loc[soguk, "tuketim"].to_numpy()).max()
    )
    if d_s > 0 or d_c > 0:
        raise SystemExit(f"birlestirme bozuk: sicak sapma {d_s}, soguk sapma {d_c}")

    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    yeni.to_csv(cik, index=False)

    print(f"  SICAK {int((~soguk).sum()):,} satir  <- {Path(ar.sicak).name}")
    print(f"  SOGUK {int(soguk.sum()):,} satir  <- {Path(ar.soguk).name}")
    print(f"  dogrulama: sicak sapma {d_s:.1e}, soguk sapma {d_c:.1e}  (ikisi de 0 olmali)")
    print(
        f"  min {yeni['tuketim'].min():.1f}  medyan {yeni['tuketim'].median():.1f}"
        f"  maks {yeni['tuketim'].max():.1f}"
    )
    print(f"  yazildi: {cik}  ({len(yeni):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
