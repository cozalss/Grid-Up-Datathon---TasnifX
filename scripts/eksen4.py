# ruff: noqa
"""EKSEN 4 -- MODEL KAPASITESI VE EGITIM KURGUSU.

Dort alt eksen, tek tezgah. Hukum HER ZAMAN:
  * uretim harmani icinde (3*cat + 1*xgb + 1*lgbm + 1,4*ag) / 6,4
  * teste agirliklandirilmis olcut (olcut.py, bayatlik ekseni)
  * blok kirilimi + MEVSIMSEL IKIZ (yaz25) ayrica
  * TRAFO BAZINDA kirpma tablosu (K = 0,1,5,10,25,50)

    python scripts/eksen4.py --adim a                 # kapasite izgarasi (onbellekten)
    python scripts/eksen4.py --adim d --kollar artik90
    python scripts/eksen4.py --adim b --kollar iw_bgu
    python scripts/eksen4.py --adim c
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "aile_onbellek"
E4 = KOK / "data" / "interim" / "eksen4"
KAYIT = KOK / "experiments" / "eksen4.jsonl"

TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
MASKE = 0.15
ORTAK_CAT: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}

#: TEST karesel hatasinin SICAK tarafta duran payi.
#: Rekor defterinden: sicak 0,74263 / genel 1,07907 / sicak pay 0,7784
#:     0,7784 * 0,74263^2 / 1,07907^2 = 0,36870
#: Bugunku LB karesi 1,01591^2 = 1,03207 ile:
SICAK_KARE_PAY = 0.36870 * 1.03207  # = 0,38053
LB_KARE = 1.03207


# ------------------------------------------------------------------ olcum


def kare_hata(gercek: np.ndarray, log_tahmin: np.ndarray) -> np.ndarray:
    """Satir basina log-uzayi kare hatasi (RMSLE'nin icindeki terim)."""
    t = np.clip(np.expm1(log_tahmin), 0.0, None)
    return (np.log1p(t) - np.log1p(gercek)) ** 2


def agirlikli_mse(e2: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, e2) / w.sum())


def kirpma_tablosu(
    e2_0: np.ndarray,
    e2_1: np.ndarray,
    w: np.ndarray,
    trafo: np.ndarray,
    kler: tuple[int, ...] = (0, 1, 5, 10, 25, 50),
) -> list[dict]:
    """En cok KAZANDIRAN K trafo atilinca dMSE ne kaliyor.

    Kazanc = dMSE'nin NEGATIF olmasi. Trafo katkisi:
        katki_i = sum_{j in i} w_j (e1_j^2 - e0_j^2) / sum_j w_j
    Toplami tam olarak dMSE'ye esittir.
    """
    fark = w * (e2_1 - e2_0)
    df = pd.DataFrame({"t": trafo, "f": fark, "w": w})
    katki = df.groupby("t", observed=True)["f"].sum()
    sirali = katki.sort_values().index.to_numpy()  # en negatif (en cok kazandiran) once
    wtop = w.sum()
    satirlar = []
    for k in kler:
        at = set(sirali[:k].tolist())
        kal = ~pd.Series(trafo).isin(at).to_numpy()
        if kal.sum() == 0:
            break
        dm = float(fark[kal].sum() / w[kal].sum())
        satirlar.append(
            {
                "K": k,
                "dMSE_yerel": dm,
                "kalan_trafo": int(pd.Series(trafo[kal]).nunique()),
                "kalan_satir": int(kal.sum()),
            }
        )
    # en cok kazandiran ilk 5 trafo, katkisiyla
    ilk = [(str(t), float(katki[t] / wtop)) for t in sirali[:5]]
    return satirlar, ilk


# ------------------------------------------------------------------ veri


def tezgah():
    """(egitim, kol, genis, guc_kenar, te_s) -- uretim-sadik kurulum."""
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    return egitim, kol, genis, guc_kenar, te_s


def blok_baglami(egitim, genis, guc_kenar, te_s, blok: str, agir: bool = True):
    """Bir blogun SICAK dogrulama baglami.

    ``agir=False`` ise EGITIM parcasi (``parca``, 1,2-2,0M satir x 150 kolon)
    KURULMAZ. Bellek icin sart: uc blogun parcasi ayni anda bellekte tutulunca
    surec OOM ile oluyor ve ``tee`` cikis kodunu 0 gosterip bunu gizliyor.
    """
    _, dogrulama, _, soguk = di.blok_parcalari(egitim, blok)
    sicak = ~soguk
    parca = tm.kokenleri_ayikla(genis, blok) if agir else None
    dog_s = dogrulama[~soguk]
    w, tani = ol.test_agirliklari(dog_s, te_s, guc_kenar, eksenler=("bayatlik",))
    gercek = np.load(ONB / f"{blok}_gercek.npy")
    esler = {}
    for t in TOHUMLAR:
        for a in ("xgb", "lgbm", "sinir_agi"):
            esler[(t, a)] = np.load(ONB / f"{blok}_{t}_{a}_uretim.npy").astype("float64")
    return {
        "parca": parca,
        "dogrulama": dogrulama,
        "sicak": sicak,
        "dog_s": dog_s,
        "w": w,
        "tani": tani,
        "gercek": gercek,
        "esler": esler,
        "trafo": dog_s[tm.GRUP].to_numpy(),
    }


def harman(bag, cat_log: dict, tohumlar=TOHUMLAR) -> np.ndarray:
    """Uretim harmani, tohumlar log uzayinda torbalanmis."""
    top = sum(AGIRLIK.values())
    yig = []
    for t in tohumlar:
        pay = AGIRLIK["cat"] * cat_log[t]
        for a in ("xgb", "lgbm", "sinir_agi"):
            pay = pay + AGIRLIK[a] * bag["esler"][(t, a)]
        yig.append(pay / top)
    return np.mean(yig, axis=0)


# ------------------------------------------------------------------ fit


def cat_fit(
    maskeli: pd.DataFrame,
    dogrulama: pd.DataFrame,
    kol: list[str],
    tohum: int,
    *,
    merkez_kolon=None,
    lam: float = 1.0,
    agirlik: np.ndarray | None = None,
    ustyazim: dict | None = None,
) -> np.ndarray:
    """Bir CatBoost. Doner: TUM dogrulama satirlari icin LOG tahmin.

    ``merkez_kolon`` verilirse hedef ayrica trafonun kendi seviyesiyle
    merkezlenir:  y = log1p(tuketim) - log1p(guc) - c,
                  c = merkez_kolon - log1p(guc)   (NaN -> 0)
    """
    import catboost as cb

    p: dict[str, object] = {
        "loss_function": "RMSE",
        "iterations": 250,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 3.0,
        "rsm": 0.75,
        "random_seed": tohum,
        "verbose": 0,
        "allow_writing_files": False,
    }
    p.update(ORTAK_CAT)
    p.update(ustyazim or {})

    y = np.log1p(maskeli[tm.HEDEF].clip(lower=0.0)).to_numpy() - np.log1p(maskeli["guc"]).to_numpy()
    c_h = np.zeros(len(dogrulama))
    if merkez_kolon is not None:
        kols = (merkez_kolon,) if isinstance(merkez_kolon, str) else tuple(merkez_kolon)

        def merkez(cerceve: pd.DataFrame) -> np.ndarray:
            """Kaskad: ilk NaN olmayan kolon. Hicbiri yoksa 0 (ofset hedefine doner)."""
            s = cerceve[kols[0]].astype("float64")
            for k in kols[1:]:
                s = s.fillna(cerceve[k].astype("float64"))
            c = s.to_numpy() - np.log1p(cerceve["guc"]).to_numpy()
            return np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0) * lam

        c_e = merkez(maskeli)
        y = y - c_e
        c_h = merkez(dogrulama)

    # ``df[kol]`` zaten YENI bir cerceve dondurur; ustune ``.copy()`` cagirmak
    # 105 kolonluk ikinci bir kopya daha uretiyordu (~2 GB) ve OOM'a katkiydi.
    x_e, x_h = maskeli[kol], dogrulama[kol]
    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
    for k in kat:
        x_e[k] = x_e[k].astype(str)
        x_h[k] = x_h[k].astype(str)
    m = cb.CatBoostRegressor(**p)
    m.fit(x_e, y, sample_weight=agirlik, cat_features=kat)
    return m.predict(x_h) + c_h + np.log1p(dogrulama["guc"]).to_numpy()


# ------------------------------------------------------------------ raporlama


def rapor(ad: str, taban: dict, aday: dict, baglam: dict) -> dict:
    """taban/aday: blok -> (harman_log, cat_log). Tam hukum tablosu."""
    print("\n" + "=" * 104)
    print(f"HUKUM: {ad}")
    print("=" * 104)
    print(
        f"  {'blok':8}{'taban':>10}{'aday':>10}{'dRMSLE':>10}"
        f"{'m0':>10}{'m1':>10}{'dMSE_yerel':>12}{'cat0':>10}{'cat1':>10}"
    )
    sat = {}
    kare0 = kare1 = wtop = 0.0
    for b in [x.ad for x in tm.BLOKLAR]:
        if b not in aday:
            continue
        bag = baglam[b]
        w, g = bag["w"], bag["gercek"]
        e0 = kare_hata(g, taban[b][0])
        e1 = kare_hata(g, aday[b][0])
        m0, m1 = agirlikli_mse(e0, w), agirlikli_mse(e1, w)
        c0 = agirlikli_mse(kare_hata(g, taban[b][1]), w)
        c1 = agirlikli_mse(kare_hata(g, aday[b][1]), w)
        kare0 += m0 * w.sum()
        kare1 += m1 * w.sum()
        wtop += w.sum()
        print(
            f"  {b:8}{np.sqrt(m0):10.5f}{np.sqrt(m1):10.5f}"
            f"{np.sqrt(m1) - np.sqrt(m0):+10.5f}{m0:10.5f}{m1:10.5f}{m1 - m0:+12.5f}"
            f"{np.sqrt(c0):10.5f}{np.sqrt(c1):10.5f}"
        )
        sat[b] = {"m0": m0, "m1": m1, "cat0": c0, "cat1": c1}
    if wtop == 0:
        return {}
    M0, M1 = kare0 / wtop, kare1 / wtop
    print(
        f"  {'HAVUZ':8}{np.sqrt(M0):10.5f}{np.sqrt(M1):10.5f}"
        f"{np.sqrt(M1) - np.sqrt(M0):+10.5f}{M0:10.5f}{M1:10.5f}{M1 - M0:+12.5f}"
    )
    r = M1 / M0
    dmse_test = SICAK_KARE_PAY * (r - 1.0)
    print(
        f"\n  goreli MSE orani r = {r:.5f}   ->  TEST dMSE = {SICAK_KARE_PAY:.5f}*(r-1)"
        f" = {dmse_test:+.5f}"
    )
    print(
        f"  yeni RMSLE tahmini = sqrt({LB_KARE:.5f} {dmse_test:+.5f}) = "
        f"{np.sqrt(max(LB_KARE + dmse_test, 1e-9)):.5f}"
    )

    # tohum bazinda eslenik SH
    print("\n  ESLENIK (blok,tohum) -- cat TEK BASINA, harmansiz")
    hucre = []
    for b in [x.ad for x in tm.BLOKLAR]:
        if b not in aday or len(aday[b]) < 3:
            continue
        bag = baglam[b]
        for t in TOHUMLAR:
            f0 = agirlikli_mse(kare_hata(bag["gercek"], taban[b][2][t]), bag["w"])
            f1 = agirlikli_mse(kare_hata(bag["gercek"], aday[b][2][t]), bag["w"])
            hucre.append((b, t, np.sqrt(f0) - np.sqrt(f1)))
    if hucre:
        v = np.array([x[2] for x in hucre])
        sh = v.std(ddof=1) / np.sqrt(len(v))
        print(
            f"    fark (taban-aday, + = aday IYI) {v.mean():+.5f}  SH {sh:.5f}  "
            f"t={v.mean() / sh if sh else 0:+.2f}   {int((v > 0).sum())}/{len(v)}"
        )
        for b, t, x in hucre:
            print(f"      {b:8} tohum {t}  {x:+.5f}")

    # kirpma tablolari
    for b in [x.ad for x in tm.BLOKLAR]:
        if b not in aday:
            continue
        bag = baglam[b]
        e0 = kare_hata(bag["gercek"], taban[b][0])
        e1 = kare_hata(bag["gercek"], aday[b][0])
        tab, ilk = kirpma_tablosu(e0, e1, bag["w"], bag["trafo"])
        print(f"\n  KIRPMA -- {b}   (en cok kazandiran K trafo atildiktan sonra)")
        print(f"    {'K':>4}{'dMSE_yerel':>13}{'kalan trafo':>14}{'kalan satir':>14}")
        for r_ in tab:
            print(
                f"    {r_['K']:>4}{r_['dMSE_yerel']:+13.5f}"
                f"{r_['kalan_trafo']:>14,}{r_['kalan_satir']:>14,}"
            )
        print("    en cok kazandiran 5 trafo: " + ", ".join(f"{a} {x:+.5f}" for a, x in ilk))
        sat[b]["kirpma"] = tab
        sat[b]["ilk5"] = ilk

    kayit = {"ad": ad, "bloklar": sat, "M0": M0, "M1": M1, "r": r, "dmse_test": dmse_test}
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(kayit, ensure_ascii=False, default=float) + "\n")
    return kayit


# ------------------------------------------------------------------ adimlar


def adim_a(bags, args) -> None:
    """Kapasite izgarasi -- onbellekten, FIT YOK."""
    kollar = ["250", "500", "900", "500d7"]
    taban, catt = {}, {}
    for b, bag in bags.items():
        cl = {t: np.load(ONB / f"{b}_{t}_cat_kap250.npy").astype("float64") for t in TOHUMLAR}
        taban[b] = (harman(bag, cl), np.mean([cl[t] for t in TOHUMLAR], axis=0), cl)
    for k in kollar[1:]:
        aday = {}
        for b, bag in bags.items():
            yol = ONB / f"{b}_{TOHUMLAR[0]}_cat_kap{k}.npy"
            if not yol.exists():
                continue
            cl = {t: np.load(ONB / f"{b}_{t}_cat_kap{k}.npy").astype("float64") for t in TOHUMLAR}
            aday[b] = (harman(bag, cl), np.mean([cl[t] for t in TOHUMLAR], axis=0), cl)
        if aday:
            rapor(f"(a) kapasite {k} vs URETIM 250/d6", taban, aday, bags)


def adim_fit(egitim, genis, kol, guc_kenar, te_s, bloklar, kollar: dict, etiket: str) -> None:
    """Ortak fit dongusu -- BLOK BLOK, agir cerceve tek seferde bellekte.

    Onceki surum uc blogun ``parca``sini birden tutuyordu ve surec ucuncu
    fitte OOM ile oluyordu. Simdi: bir blogun agir parcasi kurulur, o blogun
    butun (kol, tohum) fitleri yapilir, sonra serbest birakilir. Geriye
    yalnizca TAHMINLER (birkac yuz KB) ve hafif baglam kalir.
    """
    import gc

    E4.mkdir(parents=True, exist_ok=True)
    bags, taban, aday = {}, {}, {ad: {} for ad in kollar}

    for bad in bloklar:
        eksik = [
            (ad, t) for ad in kollar for t in TOHUMLAR if not (E4 / f"{bad}_{t}_{ad}.npy").exists()
        ]
        bag = blok_baglami(egitim, genis, guc_kenar, te_s, bad, agir=bool(eksik))
        print(
            f"  {bad:8} sicak dogrulama {len(bag['dog_s']):,} satir  "
            f"{bag['dog_s'][tm.GRUP].nunique():,} trafo  ESS {bag['tani']['ess_orani']:.3f}"
            + (
                f"  egitim {len(bag['parca']):,}  eksik fit {len(eksik)}"
                if eksik
                else "  (fit yok)"
            )
        )
        for t in TOHUMLAR:
            if not any(x[1] == t for x in eksik):
                continue
            maskeli = d.soguk_maskele(bag["parca"], kol, MASKE, t)
            for ad, uretici in kollar.items():
                if (ad, t) not in eksik:
                    continue
                t1 = time.time()
                kw = uretici(maskeli, bag)
                alt = kw.pop("_alt", None)
                mk = maskeli if alt is None else maskeli[alt]
                if alt is not None:
                    print(f"      alt kume {len(mk):,} / {len(maskeli):,} satir")
                log_t = cat_fit(mk, bag["dogrulama"], kol, t, **kw)
                np.save(E4 / f"{bad}_{t}_{ad}.npy", log_t[bag["sicak"]].astype("float32"))
                del log_t, mk
                gc.collect()
                print(f"    {bad} tohum {t} {ad}  ({time.time() - t1:.0f} sn)", flush=True)
            del maskeli
            gc.collect()
        bag["parca"] = None
        gc.collect()

        cl = {t: np.load(ONB / f"{bad}_{t}_cat_kap250.npy").astype("float64") for t in TOHUMLAR}
        taban[bad] = (harman(bag, cl), np.mean([cl[t] for t in TOHUMLAR], axis=0), cl)
        for ad in kollar:
            cl2 = {t: np.load(E4 / f"{bad}_{t}_{ad}.npy").astype("float64") for t in TOHUMLAR}
            aday[ad][bad] = (harman(bag, cl2), np.mean([cl2[t] for t in TOHUMLAR], axis=0), cl2)
        bags[bad] = bag

    for ad in kollar:
        rapor(f"{etiket} {ad}", taban, aday[ad], bags)


def main() -> int:
    from gridup.reporting import satir_tamponlu_cikti

    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adim", required=True, choices=["a", "b", "c", "d"])
    ap.add_argument("--kollar", default="")
    ap.add_argument("--bloklar", default="yaz25,guz25,kis26")
    ar = ap.parse_args()

    t0 = time.time()
    egitim, kol, genis, guc_kenar, te_s = tezgah()
    bloklar = [
        b.ad for b in tm.BLOKLAR if b.ad in [x.strip() for x in ar.bloklar.split(",") if x.strip()]
    ]

    def kos(kollar, etiket):
        adim_fit(egitim, genis, kol, guc_kenar, te_s, bloklar, kollar, etiket)

    if ar.adim == "a":
        bags = {b: blok_baglami(egitim, genis, guc_kenar, te_s, b, agir=False) for b in bloklar}
        adim_a(bags, ar)
    elif ar.adim == "d":
        MERKEZLER = {
            # ad -> (kaskad kolonlari, lambda)
            "artikort": (("t_log_ort",), 1.0),
            "artik90": (("t_log_son90",), 1.0),
            "artikkas": (("t_log_son90", "t_log_ort"), 1.0),
            "artikkas30": (("t_log_son30", "t_log_son90", "t_log_ort"), 1.0),
            "artikkas_l07": (("t_log_son90", "t_log_ort"), 0.7),
            "artikkas_l05": (("t_log_son90", "t_log_ort"), 0.5),
        }
        sec = [k.strip() for k in ar.kollar.split(",") if k.strip()] or ["artikkas"]
        kollar = {
            k: (lambda mk, lv: lambda maskeli, bag: {"merkez_kolon": mk, "lam": lv})(*MERKEZLER[k])
            for k in sec
            if k in MERKEZLER
        }
        kos(kollar, "(d) ARTIK HEDEF")
    elif ar.adim == "b":

        def iw(eksenler):
            def yap(maskeli, bag):
                w, tani = ol.test_agirliklari(maskeli, te_s, guc_kenar, eksenler=eksenler)
                print(
                    f"      IW {eksenler} ESS {tani['ess_orani']:.3f} "
                    f"kirpilan {tani['kirpilan']:.4f} kapsanmayan {tani['kapsanmayan']:.4f}"
                )
                return {"agirlik": w}

            return yap

        SECIM = {
            "iw_b": ("bayatlik",),
            "iw_bg": ("bayatlik", "guc"),
            "iw_bgu": ("bayatlik", "guc", "ufuk"),
        }
        sec = [k.strip() for k in ar.kollar.split(",") if k.strip()] or ["iw_bgu"]
        kollar = {k: iw(SECIM[k]) for k in sec if k in SECIM}
        kos(kollar, "(b) ONEM AGIRLIKLANDIRMA")
    elif ar.adim == "c":

        def ufuk_kes(ust: int):
            def yap(maskeli, bag):
                return {"_alt": (maskeli["ufuk_gun"].to_numpy() <= ust)}

            return yap

        def ufuk_iw():
            """Egitim ufuk dagilimini TESTIN ufuk dagilimina tasi."""

            def yap(maskeli, bag):
                kt = np.clip(te_s["ufuk_gun"].to_numpy(dtype="float64"), 1, 122).astype(int)
                ke = np.clip(maskeli["ufuk_gun"].to_numpy(dtype="float64"), 1, 10**6).astype(int)
                pt = pd.Series(kt).value_counts(normalize=True)
                pe = pd.Series(ke).value_counts(normalize=True)
                w = pt.reindex(ke).to_numpy() / pe.reindex(ke).to_numpy()
                w = np.nan_to_num(w, nan=0.0, posinf=0.0)
                w = w / w.mean()
                w = np.minimum(w, 20.0)
                w = w / w.mean()
                ess = float(w.sum() ** 2 / np.dot(w, w)) / len(w)
                print(f"      ufuk IW  ESS {ess:.3f}  sifir agirlikli {float((w == 0).mean()):.4f}")
                return {"agirlik": w}

            return yap

        SECIM_C = {"ufuk61": ufuk_kes(61), "ufuk31": ufuk_kes(31), "ufuk_iw": ufuk_iw()}
        sec = [k.strip() for k in ar.kollar.split(",") if k.strip()] or ["ufuk61"]
        kollar = {k: SECIM_C[k] for k in sec if k in SECIM_C}
        kos(kollar, "(c) ETIKET PENCERESI")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
