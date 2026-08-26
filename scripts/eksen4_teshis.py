# ruff: noqa
"""EKSEN 4 (d) ON TESHIS -- artik hedefin ONCESIZ (fitsiz) buyuklukleri.

Sorular:
  1. t_log_son90 SICAK satirlarin yuzde kacinda dolu? Test'te? Egitimde?
  2. Hedef ofs = log1p(y)-log1p(guc) ile c = t_log_son90 - log1p(guc)
     arasindaki iliski: Var(ofs) -> Var(ofs - c) ne kadar duser?
     (bu, agacin cozmesi gereken problemin buyuklugunu dogrudan verir)
  3. Merkezleme katsayisi lambda: ofs ~ a + lambda*c  -- lambda 1'e yakin mi?
     1'den uzaksa TAM merkezleme (lambda=1) fazla duzeltme olur.
  4. Bunlarin TEST penceresi icin degil, her BLOK icin ayri degeri.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

SEVIYELER = ("t_log_son7", "t_log_son14", "t_log_son30", "t_log_son90", "t_log_ort")


def satirlar(cerceve: pd.DataFrame, ad: str, hedefli: bool) -> None:
    lg = np.log1p(cerceve["guc"].to_numpy(dtype="float64"))
    print(f"\n  {ad}   n={len(cerceve):,}")
    print(
        f"    {'kolon':16}{'doluluk':>10}{'kor(ofs,c)':>12}{'lambda':>9}"
        f"{'Var(ofs)':>11}{'Var(ofs-c)':>12}{'oran':>8}"
    )
    if hedefli:
        ofs = np.log1p(cerceve[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64")) - lg
    for k in SEVIYELER:
        if k not in cerceve.columns:
            continue
        dolu = cerceve[k].notna().to_numpy()
        pay = dolu.mean()
        if not hedefli:
            print(f"    {k:16}{pay:10.4f}")
            continue
        c = np.nan_to_num(cerceve[k].to_numpy(dtype="float64") - lg, nan=0.0)
        v0 = float(np.var(ofs))
        v1 = float(np.var(ofs - c))
        m = dolu
        kor = float(np.corrcoef(ofs[m], c[m])[0, 1]) if m.sum() > 10 else np.nan
        lam = float(np.polyfit(c[m], ofs[m], 1)[0]) if m.sum() > 10 else np.nan
        print(f"    {k:16}{pay:10.4f}{kor:12.4f}{lam:9.4f}{v0:11.4f}{v1:12.4f}{v1 / v0:8.3f}")


def main() -> int:
    from gridup.reporting import satir_tamponlu_cikti

    satir_tamponlu_cikti()
    egitim, test = d.cerceveleri_kur()
    print("=" * 96)
    print("EKSEN 4 (d) ON TESHIS -- artik hedef (trafonun kendi son-90-gun seviyesi)")
    print("=" * 96)

    te_s = test[test["soguk_mu"] != 1]
    satirlar(te_s, "TEST sicak (hedef yok)", hedefli=False)

    for b in tm.BLOKLAR:
        _, dog, _, soguk = di.blok_parcalari(egitim, b.ad)
        satirlar(dog[~soguk], f"{b.ad} DOGRULAMA sicak", hedefli=True)

    # Egitim tarafi (ek kokenli, maskesiz) doluluk
    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    gs = genis[genis["soguk_mu"] != 1]
    satirlar(gs, "EGITIM (ek kokenli) sicak", hedefli=True)

    # TRAFO BAZINDA: son-90 seviyesi ile gercek ofs ortalamasi arasindaki
    # kayma -- surukleme yanliligi dogrudan bu.
    print("\n" + "=" * 96)
    print("SURUKLEME: trafo bazinda  ort(ofs) - ort(c)   (c = t_log_son90 - log1p(guc))")
    print("=" * 96)
    print(f"  {'blok':8}{'trafo':>8}{'ort':>10}{'medyan':>10}{'std':>10}{'poz%':>8}")
    for b in tm.BLOKLAR:
        _, dog, _, soguk = di.blok_parcalari(egitim, b.ad)
        s = dog[~soguk]
        s = s[s["t_log_son90"].notna()]
        lg = np.log1p(s["guc"].to_numpy(dtype="float64"))
        ofs = np.log1p(s[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64")) - lg
        c = s["t_log_son90"].to_numpy(dtype="float64") - lg
        df = pd.DataFrame({"t": s[tm.GRUP].to_numpy(), "r": ofs - c})
        g = df.groupby("t", observed=True)["r"].mean()
        print(
            f"  {b.ad:8}{len(g):>8,}{g.mean():+10.4f}{g.median():+10.4f}"
            f"{g.std():10.4f}{(g > 0).mean() * 100:8.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
