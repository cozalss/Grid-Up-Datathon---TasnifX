"""p06: DELTA'yi bir uretim gonderimine uygula ve SCRATCHPAD'e yaz.

submissions/ altina HICBIR SEY YAZMAZ. Cikti scratchpad'te .npy ve .csv.

    p_yeni_log = log1p(taban_tuketim) + delta      (yalniz SOGUK satirlar)
    tuketim    = expm1(p_yeni_log), 0'a kirpilmis

Satir sirasi: data/interim/deney/test.parquet ile data/raw/test.csv BIREBIR
ayni (dogrulandi); delta dosya sirasindadir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
BURA = os.path.dirname(os.path.abspath(__file__))
CIKTI = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
TABAN = "tuketim_m6_ikiyon.csv"


def main():
    delta = np.load(os.path.join(CIKTI, "p06_test_delta_log.npy"))
    soguk = np.load(os.path.join(CIKTI, "p06_test_soguk_maske.npy"))
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    tb = pd.read_csv(os.path.join(KOK, "submissions", TABAN))
    kol = "tuketim" if "tuketim" in tb.columns else tb.columns[-1]
    assert np.array_equal(tb.id.values, te.id.values), "id sirasi uyusmuyor"
    assert len(delta) == len(tb), "delta uzunlugu uyusmuyor"
    assert np.allclose(delta[~soguk], 0.0), "delta sicak satirlarda sifir degil"

    p0 = np.log1p(tb[kol].to_numpy(dtype="float64"))
    p1 = p0 + delta
    y1 = np.clip(np.expm1(p1), 0.0, None)

    np.save(os.path.join(CIKTI, "p06_test_tahmin_log1p.npy"), p1)
    np.save(os.path.join(CIKTI, "p06_test_tahmin_tuketim.npy"), y1)
    pd.DataFrame({"id": te.id.values, "tuketim": y1}).to_csv(
        os.path.join(CIKTI, "p06_test_tahmin.csv"), index=False
    )

    R = dict(
        taban_dosya=TABAN,
        n=int(len(y1)),
        n_soguk=int(soguk.sum()),
        degisen_satir=int((np.abs(delta) > 1e-9).sum()),
        taban_log_ort=round(float(p0.mean()), 5),
        yeni_log_ort=round(float(p1.mean()), 5),
        soguk_log_ort_degisim=round(float(delta[soguk].mean()), 5),
        cikti=[
            os.path.join(CIKTI, "p06_test_tahmin_log1p.npy"),
            os.path.join(CIKTI, "p06_test_tahmin_tuketim.npy"),
            os.path.join(CIKTI, "p06_test_tahmin.csv"),
        ],
    )
    with open(os.path.join(BURA, "p06_soguk_uygula.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
