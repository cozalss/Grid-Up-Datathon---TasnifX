"""GOREV 2 devami -- rotus 2 (panel sinir), 4 (soguk genlik), 6 (tohum) kaniti."""

from __future__ import annotations

import json

import numpy as np
from ortak import KOK, test, train

lp = lambda v: np.log1p(np.clip(np.asarray(v, dtype=float), 0.0, None))  # noqa: E731

from ortak import hizala  # noqa: E402


def main() -> int:
    tr, te = train(), test()
    sicak_set = set(tr["tanim"].unique())
    soguk = ~te["tanim"].isin(sicak_set).to_numpy()

    v50h = hizala("tuketim_v50_ham30.csv", te)
    v50n = hizala("tuketim_v50_nihai30.csv", te)
    v55 = hizala("tuketim_v55_gunolcek.csv", te)
    v56o = hizala("tuketim_v56_olay.csv", te)
    v56p = hizala("tuketim_v56_panelsinir.csv", te)
    v66 = hizala("tuketim_v66_c1335.csv", te)
    v67 = hizala("tuketim_v67_c1335_olay.csv", te)
    v51e = hizala("tuketim_v51_ek1.csv", te)
    v61 = hizala("tuketim_v61_ham35.csv", te)
    rap: dict = {}

    def m(a, b):
        d = lp(a) - lp(b)
        msk = np.abs(d) > 1e-9
        return d, msk

    # --- ROTUS 6: taban kac tohum?
    # v61_ham35 == log-uzayinda (30*v50_ham30 + 5*v51_ek1)/35 mi?
    bek = np.expm1((30 * lp(v50h) + 5 * lp(v51e)) / 35)
    f = np.abs(bek - v61) / np.maximum(np.abs(v61), 1e-9)
    rap["v61_ham35_kimlik"] = {"maxbagil": float(f.max()), "farkli": int((f >= 1e-6).sum())}
    print(f"[6] v61_ham35 = (30*v50_ham30 + 5*v51_ek1)/35 ? maxbagil={f.max():.2e}")

    # v66 hangi tabandan? v50_nihai30 mi v61 tabanli bir nihai mi
    d_66_50n, msk = m(v66, v50n)
    print(
        f"[6] v66 - v50_nihai30 : degisen={int(msk.sum())} ort={d_66_50n.mean():+.5f} "
        f"soguk_degisen={int((np.abs(d_66_50n[soguk]) > 1e-9).sum())}"
    )
    # gun ekseni ise fark GUN bazinda neredeyse sabit olmali
    gun = te["tarih"].to_numpy()
    import pandas as pd

    df = pd.DataFrame({"gun": gun, "d": d_66_50n, "soguk": soguk})
    g = df[~df["soguk"]].groupby("gun")["d"]
    rap["v66_v50n_gun_yapisi"] = {
        "degisen": int(msk.sum()),
        "soguk_degisen": int((np.abs(d_66_50n[soguk]) > 1e-9).sum()),
        "gun_ici_std_ort": float(g.std().mean()),
        "gunler_arasi_std": float(g.mean().std()),
    }
    print(
        f"[6] v66-v50n sicak: gun ICI std ort={g.std().mean():.5f} "
        f"gunler ARASI std={g.mean().std():.5f}"
    )

    # v55 - v50_nihai30 (sicak gun ekseni, ilk surum)
    d, msk = m(v55, v50n)
    print(f"[6] v55 - v50_nihai30 : degisen={int(msk.sum())}")

    # --- ROTUS 2: panel sinir maskesi
    d_ps, msk_ps = m(v56p, v55)
    d_ol, msk_ol = m(v56o, v55)
    rap["rotus2_panelsinir"] = {
        "panelsinir_vs_v55_satir": int(msk_ps.sum()),
        "olay_vs_v55_satir": int(msk_ol.sum()),
        "kesisim": int((msk_ps & msk_ol).sum()),
        "yalniz_panelsinir": int((msk_ps & ~msk_ol).sum()),
        "yalniz_olay": int((msk_ol & ~msk_ps).sum()),
        "yalniz_panelsinir_ort_delta": float(d_ps[msk_ps & ~msk_ol].mean())
        if (msk_ps & ~msk_ol).any()
        else 0.0,
        "yalniz_panelsinir_Q_tam": float((d_ps[msk_ps & ~msk_ol] ** 2).sum() / len(te)),
    }
    r2 = rap["rotus2_panelsinir"]
    print(
        f"[2] panelsinir vs v55: {r2['panelsinir_vs_v55_satir']} satir | "
        f"olay vs v55: {r2['olay_vs_v55_satir']} | kesisim {r2['kesisim']} | "
        f"YALNIZ panelsinir {r2['yalniz_panelsinir']} (ort {r2['yalniz_panelsinir_ort_delta']:+.4f},"
        f" Q_tam {r2['yalniz_panelsinir_Q_tam']:.2e})"
    )

    # v67'nin olay maskesi ile v56'nin olay maskesi ayni mi (farkli taban ama ayni kural)
    d67, m67 = m(v67, v66)
    rap["olay_maske_karsilastirma"] = {
        "v67_uzerinde": int(m67.sum()),
        "v56_uzerinde": int(msk_ol.sum()),
        "kesisim": int((m67 & msk_ol).sum()),
        "v67_ort_delta": float(d67[m67].mean()),
        "v56_ort_delta": float(d_ol[msk_ol].mean()),
    }

    # --- ROTUS 4: soguk satirlarda v83'e kadar UYGULANMIS genlik islemleri
    v80a = hizala("tuketim_v80_a.csv", te)
    v83 = hizala("tuketim_v83_sicak_optimum.csv", te)
    # (a) son_islem.py beta=0.60 buzme : v50_ham30 -> v50_nihai30, soguk satirlarda
    d, msk = m(v50n, v50h)
    r_h = lp(v50h) - np.log1p(te["guc"].to_numpy(dtype=float))
    r_n = lp(v50n) - np.log1p(te["guc"].to_numpy(dtype=float))
    olcek = float(np.std(r_n[soguk]) / np.std(r_h[soguk]))
    rap["rotus4_soguk_genlik"] = {
        "son_islem_buzme_soguk_degisen": int(msk[soguk].sum()),
        "son_islem_buzme_sicak_degisen": int(msk[~soguk].sum()),
        "soguk_ofset_std_orani_nihai_ham": olcek,
        "soguk_gun_olcegi_c": 1.3301,
        "v80a_soguk_degisen": int((np.abs(lp(v80a) - lp(v67)) > 1e-9)[soguk].sum()),
        "v83_soguk_sabit_delta": 0.1046,
    }
    print(
        f"[4] son_islem.py buzme: soguk degisen={int(msk[soguk].sum())} "
        f"sicak degisen={int(msk[~soguk].sum())} ofset std orani={olcek:.4f}"
    )
    # v83'te soguk ofset std'si
    r_83 = lp(v83) - np.log1p(te["guc"].to_numpy(dtype=float))
    rap["rotus4_soguk_genlik"]["soguk_ofset_std_v50ham"] = float(np.std(r_h[soguk]))
    rap["rotus4_soguk_genlik"]["soguk_ofset_std_v83"] = float(np.std(r_83[soguk]))
    print(
        f"[4] soguk ofset std: v50_ham={np.std(r_h[soguk]):.4f} "
        f"v50_nihai={np.std(r_n[soguk]):.4f} v83={np.std(r_83[soguk]):.4f}"
    )

    (KOK / "reports/g2b_kalanlar.json").write_text(
        json.dumps(rap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("yazildi: reports/g2b_kalanlar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
