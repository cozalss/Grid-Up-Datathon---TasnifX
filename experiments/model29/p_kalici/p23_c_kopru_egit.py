"""p23-C: KOPRU YANLILIGI -- URETIM SOGUK UZMANIYLA egitim + tahmin.

Kurulum p19_a ile birebir: d.soguk_maskele(oran=1.00) [tum t_* NaN],
cat {"depth": 7}, ofset=True, di.egit_tahmin yolu.

Iki pencere:
  P1: egitim 2025-04-01..2025-12-31, degerlendirme 2026-01-01..2026-03-31
  P2: egitim 2025-04-01..2025-09-30, degerlendirme 2025-10-01..2025-12-31

TEMIZLIK: kopru (896) + kontrol trafolarinin TUM satirlari egitimden cikar.
Kontrol: parti-disi, kVA katmanli, kopru dagilimini 1:1 aynalar (tohum 42).
Degerlendirme satirlari SOGUKMUS GIBI: t_* -> NaN, soguk_mu -> 1.

Cikti (scratchpad): p23_kopru_{pencere}_{tohum}.parquet
  kolonlar: tanim, tarih, guc, ilce_key, grup(kopru/kontrol), y_log, lg
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
SCRATCH = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")
sys.path.insert(0, os.path.join(KOK, "scripts"))
sys.path.insert(0, os.path.join(KOK, "src"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

PENCERELER = {
    "P1": ("2025-04-01", "2025-12-31", "2026-01-01", "2026-03-31"),
    "P2": ("2025-04-01", "2025-09-30", "2025-10-01", "2025-12-31"),
}
T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f}dk]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pencere", nargs="+", default=["P1", "P2"])
    ap.add_argument("--tohum", type=int, nargs="+", default=[1000])
    ar = ap.parse_args()

    os.makedirs(SCRATCH, exist_ok=True)

    # --- parti kumeleri (raw csv'den, p23_a ile ayni)
    test_raw = pd.read_csv(
        os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str}, usecols=["tanim"]
    )
    m_kopru = np.load(os.path.join(AC, "p23_parti_kopru_maske.npy"))
    m_psoguk = np.load(os.path.join(AC, "p23_parti_soguk_maske.npy"))
    kopru = set(test_raw["tanim"][m_kopru].unique())
    parti_hepsi = kopru | set(test_raw["tanim"][m_psoguk].unique())
    assert len(kopru) == 896
    del test_raw

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    del test
    log(f"egitim cercevesi {egitim.shape}  kolon {len(kol)}")

    # --- kontrol grubu: parti-disi, kVA katmanli 1:1, iki pencerede de satiri olan
    tarih = egitim["tarih"]
    p1_var = set(egitim.loc[(tarih >= "2026-01-01") & (tarih <= "2026-03-31"), "tanim"].unique())
    p2_var = set(egitim.loc[(tarih >= "2025-10-01") & (tarih <= "2025-12-31"), "tanim"].unique())
    trafo_guc = egitim.groupby("tanim", observed=True)["guc"].first()
    aday = [
        t for t in trafo_guc.index
        if t not in parti_hepsi and t in p1_var and t in p2_var
    ]
    aday_guc = trafo_guc.loc[aday]
    kopru_guc = trafo_guc.loc[[t for t in kopru if t in trafo_guc.index]]
    rng = np.random.default_rng(42)
    kontrol = []
    for kva, n in kopru_guc.value_counts().items():
        havuz = list(aday_guc[aday_guc == kva].index)
        rng.shuffle(havuz)
        kontrol.extend(havuz[: min(n, len(havuz))])
    kontrol = set(kontrol)
    log(f"kopru(frame) {len(kopru_guc)}  kontrol {len(kontrol)}  (havuz {len(aday)})")

    cikarilan = kopru | kontrol
    tk = [k for k in kol if k.startswith("t_")]

    for pad in ar.pencere:
        e0, e1, v0, v1 = PENCERELER[pad]
        egitim_m = (tarih >= e0) & (tarih <= e1) & ~egitim["tanim"].isin(cikarilan)
        deger_m = (tarih >= v0) & (tarih <= v1) & egitim["tanim"].isin(cikarilan)
        parca = egitim[egitim_m]
        deger = egitim[deger_m].copy()
        deger.loc[:, tk] = np.nan
        deger.loc[:, "soguk_mu"] = 1
        n_kopru = deger["tanim"].isin(kopru).sum()
        log(f"{pad}: egitim {len(parca):,} satir  deger {len(deger):,} "
            f"(kopru {n_kopru:,} / kontrol {len(deger) - n_kopru:,})")

        for tohum in ar.tohum:
            cikti = os.path.join(SCRATCH, f"p23_kopru_{pad}_{tohum}.parquet")
            if os.path.exists(cikti):
                log(f"  {pad} t={tohum} zaten var, atlandi")
                continue
            maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
            assert all(maskeli[k].isna().all() for k in tk), "maske 1.00 bozuk"
            t1 = time.time()
            lg = di.egit_tahmin("cat", maskeli, deger, kol, tohum, ofset=True, depth=7)
            del maskeli
            cerceve = pd.DataFrame(
                {
                    "tanim": deger["tanim"].astype(str).to_numpy(),
                    "tarih": deger["tarih"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "guc": deger["guc"].to_numpy(),
                    "ilce_key": deger["ilce_key"].astype(str).to_numpy(),
                    "grup": np.where(deger["tanim"].isin(kopru), "kopru", "kontrol"),
                    "y_log": np.log1p(deger["tuketim"].clip(lower=0.0)).to_numpy(),
                    "lg": lg.astype(np.float64),
                }
            )
            cerceve.to_parquet(cikti, index=False)
            log(f"  {pad} t={tohum} egitildi+yazildi ({time.time() - t1:.0f} sn) -> {cikti}")
        del parca, deger
    log("TAMAM")


if __name__ == "__main__":
    main()
