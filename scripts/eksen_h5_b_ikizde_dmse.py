# ruff: noqa
"""H5 adim 4 -- LOKASYON YUK CAPASI soguk tarafta IKIZDE dMSE + KIRPMA TABLOSU.

Adim 1-3 (eksen_h5_a) "devir" niceliginin R2'sini ~0 buldu. Ama "devir"
URETIMDE HESAPLANAMAZ da: dogumdan SONRAKI yerlesik tuketimi ister, testte
tuketim yok. Bu betik H5'in URETILEBILIR halini olcer:

    yogunluk_L = (L lokasyonunun kesme ONCESI toplam gunluk yuku)
                 / (L'nin kesme oncesi toplam kurulu gucu)
    capa_i     = log1p( yogunluk_L * guc_i )                    (saf pay)
    seyrelme   = capa_i + log(1 - alfa * yeni_pay_L)            (YUK DEVRI)

``yeni_pay_L`` = blokta L'de dogan kapasitenin payi. Yuk devri gercekse
alfa > 0 kazandirmali (sabit yuk havuzu yeni trafolara boluniyor).

Onbelleklenmis soguk taban: data/interim/gun_ekseni/{blok}_{tohum}_taban.npy
(log1p uzayinda). Fit YOK.

KURAL 1: kirpma tablosu K = 0,1,5,10,25,50.
KURAL 3: yaz25'te 6 tohum, guz25'te 3 tohum.
KURAL 7: yaz25 ZORUNLU, guz25 isaret kontrolu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "scripts"))
import olcut  # noqa: E402

CIK = KOK / "reports" / "eksen_h5"
CIK.mkdir(parents=True, exist_ok=True)
GE = KOK / "data" / "interim" / "gun_ekseni"

BLOK = {
    "yaz25": (pd.Timestamp("2025-04-01"), (1000, 1001, 1002, 1003, 1004, 1005)),
    "guz25": (pd.Timestamp("2025-08-01"), (1000, 1001, 1002)),
}
PENCERELER = (90, 9999)
WLER = tuple(np.round(np.arange(0.0, 0.61, 0.05), 3))
ALFALAR = (0.0, 0.25, 0.5, 0.75, 1.0)
KLER = (0, 1, 5, 10, 25, 50)
P_SOGUK = 0.22159


def yukle_train():
    return pd.read_csv(
        KOK / "data/raw/train.csv",
        encoding="utf-8",
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )


def lok_capasi(tr: pd.DataFrame, kesme: pd.Timestamp, W: int):
    """Kesme ONCESI [kesme-W, kesme) penceresinden lokasyon yogunlugu."""
    bas = kesme - pd.Timedelta(days=W)
    p = tr[(tr["tarih"] >= bas) & (tr["tarih"] < kesme)]
    # trafo basi gunluk ortalama
    g = p.groupby("tanim", observed=True)
    tr_ort = g["tuketim"].mean()
    tr_guc = g["guc"].first()
    tr_lok = g["lokasyon"].first()
    d = pd.DataFrame({"ort": tr_ort, "guc": tr_guc, "lok": tr_lok})
    agg = d.groupby("lok", observed=True).agg(
        yuk=("ort", "sum"), kap=("guc", "sum"), n=("ort", "size")
    )
    agg["yogunluk"] = agg["yuk"] / agg["kap"].clip(lower=1.0)
    return agg, set(d.index)


def yeni_pay(tr: pd.DataFrame, kesme: pd.Timestamp, bit: pd.Timestamp, eski_kap: pd.Series):
    """Blokta (kesme..bit) ILK KEZ goruleni lokasyon bazinda toplam guc payi."""
    ilk = tr.groupby("tanim", observed=True)["tarih"].min()
    guc = tr.groupby("tanim", observed=True)["guc"].first()
    lok = tr.groupby("tanim", observed=True)["lokasyon"].first()
    yeni = ilk[(ilk >= kesme) & (ilk <= bit)].index
    yk = pd.DataFrame({"guc": guc.loc[yeni], "lok": lok.loc[yeni]}).groupby("lok")["guc"].sum()
    pay = (yk / (eski_kap.reindex(yk.index).fillna(0.0) + yk)).clip(0.0, 0.9)
    return pay


def agirliklar(blok: str, meta: pd.DataFrame):
    eg = pd.read_parquet(
        KOK / "data/interim/deney/egitim.parquet",
        columns=["_blok", "soguk_mu", "tanim", "tarih", "guc", "t_son_kayit_yasi", "ufuk_gun"],
    )
    d = eg[(eg["_blok"] == blok) & (eg["soguk_mu"] == 1)].reset_index(drop=True)
    assert (d["tanim"].to_numpy() == meta["tanim"].to_numpy()).all()
    te = pd.read_parquet(
        KOK / "data/interim/deney/test.parquet",
        columns=["soguk_mu", "guc", "t_son_kayit_yasi", "ufuk_gun"],
    )
    ts = te[te["soguk_mu"] == 1].reset_index(drop=True)
    gk = olcut.guc_kenarlari(ts)
    w, tani = olcut.test_agirliklari(d, ts, gk)
    return w, tani


def kirpma_tablosu(kayip_taban, kayip_yeni, w, tanim):
    """Trafo bazinda katki; en buyuk K katkiyi ATARAK dMSE."""
    fark = w * (kayip_yeni - kayip_taban)
    s = pd.Series(fark).groupby(pd.Series(tanim)).sum()
    sirali = s.sort_values()  # en NEGATIF (en cok kazandiran) basta
    toplam_w = w.sum()
    sat = []
    for K in KLER:
        at = set(sirali.index[:K])
        maske = ~pd.Series(tanim).isin(at).to_numpy()
        d = float(fark[maske].sum() / toplam_w)
        sat.append((K, d, int(maske.sum())))
    return sat, sirali


def main() -> int:
    cikti = []

    def yaz(s=""):
        print(s)
        cikti.append(s)

    tr = yukle_train()
    lok_of = tr.groupby("tanim", observed=True)["lokasyon"].first()

    sonuc = {}
    for blok, (kesme, tohumlar) in BLOK.items():
        meta = pd.read_parquet(GE / f"{blok}_meta.parquet")
        bit = meta["tarih"].max()
        ly = np.log1p(np.clip(meta["y"].to_numpy("float64"), 0, None))
        tanim = meta["tanim"].to_numpy()
        guc = meta["guc"].to_numpy("float64")
        lok = lok_of.reindex(pd.Index(tanim)).to_numpy()
        w, tani = agirliklar(blok, meta)
        yaz(
            f"\n{'=' * 78}\nBLOK {blok}  kesme {kesme.date()}  satir {len(meta):,}  trafo {len(set(tanim))}"
        )
        yaz(
            f"  olcut tani: ESS %{tani['ess_orani'] * 100:.1f}  kirpilan %{tani['kirpilan'] * 100:.2f}"
            f"  kapsanmayan %{tani['kapsanmayan'] * 100:.2f}  guvenilir={tani['guvenilir']}"
        )

        tabanlar = [np.load(GE / f"{blok}_{t}_taban.npy").astype("float64") for t in tohumlar]
        taban_ort = np.mean(tabanlar, axis=0)
        yaz(
            f"  taban MSE (agirlikli, tohum ort) {float(np.dot(w, (taban_ort - ly) ** 2) / w.sum()):.5f}"
        )

        for W in PENCERELER:
            agg, gorulen = lok_capasi(tr, kesme, W)
            eski_kap = agg["kap"]
            pay = yeni_pay(tr, kesme, bit, eski_kap)
            yog = agg["yogunluk"].reindex(pd.Index(lok)).to_numpy("float64")
            payv = pay.reindex(pd.Index(lok)).fillna(0.0).to_numpy("float64")
            kapsam = np.isfinite(yog).mean()
            capa0 = np.log1p(np.where(np.isfinite(yog), yog, np.nan) * guc)
            yaz(
                f"\n  --- pencere W={W}  lokasyon kapsami %{kapsam * 100:.1f}"
                f"  yeni_pay ort {payv.mean():.4f} (max {payv.max():.3f}) ---"
            )
            ok = np.isfinite(capa0)
            yaz(
                f"      capa ort {np.nanmean(capa0):.3f} | gercek ort {ly[ok].mean():.3f}"
                f" | taban ort {taban_ort[ok].mean():.3f}"
            )
            # ham capa ne kadar iyi?
            mse_capa = float(np.dot(w[ok], (capa0[ok] - ly[ok]) ** 2) / w[ok].sum())
            yaz(
                f"      SAF capa MSE (kapsanan satirlarda) {mse_capa:.5f}"
                f"  | taban ayni satirlarda {float(np.dot(w[ok], (taban_ort[ok] - ly[ok]) ** 2) / w[ok].sum()):.5f}"
            )

            en_iyi = None
            for alfa in ALFALAR:
                capa = capa0 + np.log(np.clip(1.0 - alfa * payv, 1e-3, None))
                capa = np.where(ok, capa, taban_ort)
                satir = []
                for wv in WLER:
                    # tohum tohum, sonra ortalama (eslenik)
                    dl = []
                    for tb in tabanlar:
                        p = (1.0 - wv) * tb + wv * capa
                        dl.append(
                            float(np.dot(w, (p - ly) ** 2) / w.sum())
                            - float(np.dot(w, (tb - ly) ** 2) / w.sum())
                        )
                    satir.append(
                        (wv, float(np.mean(dl)), float(np.std(dl, ddof=1) / np.sqrt(len(dl))))
                    )
                b = min(satir, key=lambda z: z[1])
                yaz(
                    f"      alfa={alfa:<5} en iyi w={b[0]:.2f}  dMSE_soguk {b[1]:+.5f} (sh {b[2]:.5f})"
                    f"  | w=0.10 {[s for s in satir if s[0] == 0.1][0][1]:+.5f}"
                )
                if en_iyi is None or b[1] < en_iyi[1]:
                    en_iyi = (alfa, b[1], b[0], b[2], capa)
            sonuc[(blok, W)] = en_iyi
            alfa, dm, wv, sh, capa = en_iyi
            yaz(
                f"      >> EN IYI alfa={alfa} w={wv:.2f} dMSE_soguk {dm:+.5f} (sh {sh:.5f})"
                f"  ~ toplam dMSE {dm * P_SOGUK:+.6f}"
            )
            if dm < 0 and wv > 0:
                kt_taban = (taban_ort - ly) ** 2
                p = (1.0 - wv) * taban_ort + wv * capa
                kt_yeni = (p - ly) ** 2
                sat, sirali = kirpma_tablosu(kt_taban, kt_yeni, w, tanim)
                yaz(f"      KIRPMA TABLOSU (blok {blok}, W={W}, alfa={alfa}, w={wv:.2f})")
                for K, d, n in sat:
                    yaz(
                        f"        K={K:<3} dMSE_soguk {d:+.5f}  (kalan trafo {len(set(tanim)) - K})"
                    )
                yaz(f"        en cok kazandiran 3: {list(sirali.index[:3])}")
                yaz(f"        en cok kaybettiren 3: {list(sirali.index[-3:])}")

    yaz(f"\n{'=' * 78}\nISARET TUTARLILIGI")
    for k, v in sonuc.items():
        yaz(f"  {k}  alfa={v[0]} w={v[2]:.2f}  dMSE_soguk {v[1]:+.5f}")
    (CIK / "adim4.txt").write_text("\n".join(cikti), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
