"""GOREV 2 -- v83'un soy zincirini SAYISAL olarak dogrula.

Iddia edilen zincir (docs/44 §5, docs/47 §3, scripts/yarin_coz.py):

    v50 (ham) -> v66_c1335 (gun ekseni c=1,335)
              -> v67_c1335_olay (son_islem_olay, s=0.6)
              -> v80_a (son_islem_soguk_gunolcek, c=1.3301)
              -> v83 (uc rejim sabit delta: sicak 0.024860 / soguk 0.1046 / kuyruk 0.1664)

Her adimi dosyalardan yeniden kurup birebir karsilastirir. Sabit-delta
adimlari icin "betigi tekrar kos, dosya degisiyor mu" testi GECERSIZDIR
(toplamsal islem idempotent degil); onun yerine FARK VEKTORUNUN YAPISI
olculur.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from ortak import KOK, hizala, test, train

lp = lambda v: np.log1p(np.clip(np.asarray(v, dtype=float), 0.0, None))  # noqa: E731


def rejim_maskeleri(te: pd.DataFrame, tr: pd.DataFrame):
    sicak_set = set(tr["tanim"].unique())
    soguk = ~te["tanim"].isin(sicak_set).to_numpy()
    ilk = tr.groupby("tanim")["tarih"].min()
    kuyruk_set = set(ilk[ilk >= pd.Timestamp("2026-03-26")].index)
    kuyruk = te["tanim"].isin(kuyruk_set).to_numpy()
    return soguk, kuyruk, ~(soguk | kuyruk)


def ozet(ad: str, d: np.ndarray, sel: np.ndarray | None = None) -> dict:
    x = d if sel is None else d[sel]
    return {
        "ad": ad,
        "n": int(len(x)),
        "degisen": int((np.abs(x) > 1e-9).sum()),
        "ort": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> int:
    tr, te = train(), test()
    soguk, kuyruk, cekirdek = rejim_maskeleri(te, tr)
    print(f"rejimler: soguk={soguk.sum()} kuyruk={kuyruk.sum()} cekirdek={cekirdek.sum()}")

    v66 = hizala("tuketim_v66_c1335.csv", te)
    v67 = hizala("tuketim_v67_c1335_olay.csv", te)
    v80a = hizala("tuketim_v80_a.csv", te)
    v80b = hizala("tuketim_v80_b.csv", te)
    v80o = hizala("tuketim_v80_optimum.csv", te)
    v83 = hizala("tuketim_v83_sicak_optimum.csv", te)
    v56o = hizala("tuketim_v56_olay.csv", te)
    v56p = hizala("tuketim_v56_panelsinir.csv", te)
    v55 = hizala("tuketim_v55_gunolcek.csv", te)

    rap: dict = {}

    # --- ADIM 1: v66 -> v67  (olay gunu duzeltmesi)
    d = lp(v67) - lp(v66)
    rap["v67-v66"] = ozet("v67-v66 (olay)", d)
    print(f"[1] v67-v66 : degisen={rap['v67-v66']['degisen']} ort={d.mean():+.5f}")

    # --- ADIM 2: v67 -> v80_a  (soguk gun olcegi)
    d = lp(v80a) - lp(v67)
    rap["v80a-v67"] = ozet("v80a-v67 (soguk gun olcegi)", d)
    print(f"[2] v80a-v67: degisen={rap['v80a-v67']['degisen']} ort={d.mean():+.5f}")
    for ad, msk in (("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cekirdek)):
        print(f"      {ad:9s} degisen={(np.abs(d[msk]) > 1e-9).sum():7d}/{msk.sum():7d}")

    # --- ADIM 3: v80_a -> v83  (uc rejim sabit delta)
    d = lp(v83) - lp(v80a)
    rap["v83-v80a"] = ozet("v83-v80a (uc rejim delta)", d)
    detay = {}
    for ad, msk in (("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cekirdek)):
        detay[ad] = {
            "n": int(msk.sum()),
            "ort_delta": float(d[msk].mean()),
            "std_delta": float(d[msk].std()),
            "maxsapma": float(np.abs(d[msk] - d[msk].mean()).max()),
        }
        print(
            f"[3] v83-v80a {ad:9s} n={msk.sum():7d} delta={d[msk].mean():+.6f} "
            f"sapma={detay[ad]['maxsapma']:.2e}"
        )
    rap["v83-v80a_rejim"] = detay

    # --- ADIM 3b: yarin_cozum.json deltalariyla yeniden kur
    cz = json.loads((KOK / "reports/yarin_cozum.json").read_text(encoding="utf-8"))
    dd = cz["deltalar"]
    delta_vek = np.select(
        [soguk, kuyruk, cekirdek], [dd["soguk"], dd["kuyruk"], dd["sicak_cekirdek"]]
    )
    yeniden = np.maximum(np.expm1(lp(v80a) + delta_vek), 0.0)
    fark = np.abs(yeniden - v83) / np.maximum(np.abs(v83), 1e-9)
    rap["v83_yeniden_kurma"] = {
        "maxbagil": float(fark.max()),
        "farkli_satir": int((fark >= 1e-6).sum()),
    }
    print(f"[3b] v83 yeniden kurma: maxbagil={fark.max():.2e} farkli={(fark >= 1e-6).sum()}")

    # --- ADIM 4: panel sinir gunu -- v56_panelsinir ne yapiyor, v67'de var mi
    d_ps = lp(v56p) - lp(v56o)
    rap["v56panelsinir-v56olay"] = ozet("v56_panelsinir - v56_olay", d_ps)
    ps_maske = np.abs(d_ps) > 1e-9
    print(f"[4] panelsinir-olay: degisen={int(ps_maske.sum())} ort={d_ps[ps_maske].mean():+.5f}")
    # v67 icinde olay maskesi ile panelsinir maskesi ortusuyor mu
    d_olay55 = lp(v56o) - lp(v55)
    olay_maske = np.abs(d_olay55) > 1e-9
    kesisim = int((olay_maske & ps_maske).sum())
    rap["olay_panelsinir_ortusme"] = {
        "olay_satir": int(olay_maske.sum()),
        "panelsinir_satir": int(ps_maske.sum()),
        "kesisim": kesisim,
    }
    print(f"[4] olay={int(olay_maske.sum())} panelsinir={int(ps_maske.sum())} kesisim={kesisim}")
    # v67'nin olay maskesi v56'ninkiyle ayni mi
    d_olay67 = lp(v67) - lp(v66)
    o67 = np.abs(d_olay67) > 1e-9
    rap["olay_maske_v67_vs_v56"] = {
        "v67": int(o67.sum()),
        "v56": int(olay_maske.sum()),
        "kesisim": int((o67 & olay_maske).sum()),
    }
    print(
        f"[4] olay maskesi v67={int(o67.sum())} v56={int(olay_maske.sum())} "
        f"kesisim={int((o67 & olay_maske).sum())}"
    )

    # --- ADIM 5: v80_b, v80_optimum ara adimlarini dogrula
    for ad, a, b in (("v80b-v80a", v80b, v80a), ("v80o-v80b", v80o, v80b)):
        dq = lp(a) - lp(b)
        rap[ad] = ozet(ad, dq)
        for r_ad, msk in (("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cekirdek)):
            print(
                f"[5] {ad} {r_ad:9s} delta={dq[msk].mean():+.6f} "
                f"sapma={np.abs(dq[msk] - dq[msk].mean()).max():.1e}"
            )

    (KOK / "reports/g2_zincir.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: reports/g2_zincir.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
