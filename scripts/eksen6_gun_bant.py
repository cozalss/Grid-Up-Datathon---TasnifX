# ruff: noqa
"""H1 -- GUN EKSENI FREKANS AYRISTIRMASI  (c_dusuk / c_yuksek).

SORU
----
``son_islem_gunolcek.py`` gun faktorunu TEK kuresel genlik ``c`` ile olcekliyor
(c* = kor * sigma_gercek/sigma_model; uretimde LB-kalibreli 1,335). Hipotez:
tek genlik iki farkli fizigi ayni sayiya sikistiriyor --
  DUSUK FREKANS: Nisan->Temmuz mevsimsel rampa
  YUKSEK FREKANS: haftalik dongu + hava kaynakli gunluk salinim
ve bu ikisinin model/gercek genlik ORANI ayni olmak zorunda degil.

KURGU -- ``deney_gun_ekseni_sicak.py`` ile AYNI TEZGAH
-----------------------------------------------------
* URETIM ESLI aile onbellegi (data/interim/aile_onbellek/*_uretim.npy),
  harman agirliklari cat 3 / xgb 1 / lgbm 1 / sinir_agi 1,4.
* Olcut: ``olcut.py`` test agirliklari, eksenler=("bayatlik",), SICAK satirlar.
* Gun etkisi TRAFO ETKISI CIKARILARAK (kalici kural 6), tam olarak
  ``son_islem_gunolcek.gun_etkisi`` gibi: b_g = ort_g(r - ort_trafo(r)),
  sonra merkezlenir.
* Uygulama uretim bicimi: r' = r + sum_k (c_k - 1) * u_k[gun], u_k satir
  duzeyinde yeniden merkezlenir (genel seviye korunur).

BANT AYRIMI
-----------
u_dusuk = merkezli hareketli ortalama(b_gun, W), u_yuksek = b_gun - u_dusuk.
W birden fazla degerde taranir (kesme duyarliligi).

    python scripts/eksen6_gun_bant.py
"""

from __future__ import annotations

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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
TOHUMLAR = (1000, 1001, 1002)
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")
AGIRLIK = (3.0, 1.0, 1.0, 1.4)
ETIKET = "uretim"

#: d(genel RMSLE)/d(sicak RMSLE) -- deney_gun_ekseni_sicak.py ile ayni sabit.
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907
#: Suanki LB RMSLE; dRMSLE -> dMSE cevrimi icin (dMSE = 2*R*dR).
LB_RMSLE = 1.01591

PENCERELER = (7, 11, 15, 21, 31)
CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"


def blend(blok: str, tohum: int) -> np.ndarray:
    pay = sum(AGIRLIK)
    return (
        sum(
            w * np.load(DIZIN / f"{blok}_{tohum}_{a}_{ETIKET}.npy").astype("float64")
            for a, w in zip(AILELER, AGIRLIK, strict=True)
        )
        / pay
    )


def agirlikli_mse(e: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, e * e) / w.sum())


def gun_etkisi_kod(
    trafo_kod: np.ndarray, gun_kod: np.ndarray, r: np.ndarray, n_gun: int
) -> np.ndarray:
    """Iki yonlu ayristirma: trafo etkisi ONCE cikarilir (kalici kural 6)."""
    nt = np.bincount(trafo_kod)
    mt = np.bincount(trafo_kod, weights=r) / nt
    c = r - mt[trafo_kod]
    nd = np.bincount(gun_kod, weights=None, minlength=n_gun).astype("float64")
    b = np.bincount(gun_kod, weights=c, minlength=n_gun) / np.maximum(nd, 1.0)
    return b - b.mean()


def bantla(b: np.ndarray, pencere: int) -> tuple[np.ndarray, np.ndarray]:
    """Dusuk = merkezli hareketli ortalama, yuksek = kalan. Ikisi de merkezli."""
    s = pd.Series(b)
    dusuk = s.rolling(pencere, center=True, min_periods=1).mean().to_numpy()
    dusuk = dusuk - dusuk.mean()
    yuksek = b - dusuk
    return dusuk, yuksek - yuksek.mean()


def satir_bazi(u_gun: np.ndarray, gun_kod: np.ndarray) -> np.ndarray:
    """Gun duzeyi bileseni satirlara yay ve SATIR agirlikli yeniden merkezle."""
    x = u_gun[gun_kod]
    return x - x.mean()


def kapali_form(e: np.ndarray, U: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """min_a sum w (e - U a)^2  ->  a = (U'WU)^-1 U'We.  c_k = 1 + a_k."""
    n = len(U)
    A = np.empty((n, n), dtype="float64")
    bvec = np.empty(n, dtype="float64")
    for i in range(n):
        bvec[i] = float(np.dot(w, e * U[i]))
        for j in range(n):
            A[i, j] = float(np.dot(w, U[i] * U[j]))
    return np.linalg.solve(A, bvec)


def uygula(r: np.ndarray, U: list[np.ndarray], a: np.ndarray) -> np.ndarray:
    out = r.copy()
    for u, ak in zip(U, a, strict=True):
        out = out + ak * u
    return out


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t0 = time.time()
    print("=" * 110)
    print("H1 -- GUN EKSENI FREKANS AYRISTIRMASI (c_dusuk / c_yuksek), SICAK taraf")
    print("=" * 110)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  cerceveler kuruldu ({time.time() - t0:.0f} sn)")

    V: dict[str, dict] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        w, tani = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dg["tarih"])
        gun_kod, gunler = pd.factorize(gun, sort=True)
        trafo_kod, _ = pd.factorize(dg["tanim"].to_numpy())
        g = np.log1p(np.clip(gercek[sicak], 0.0, None)) - lg
        r = {t: blend(b.ad, t) - lg for t in TOHUMLAR}
        r["bag"] = np.mean([blend(b.ad, t) for t in TOHUMLAR], axis=0) - lg
        gunler = pd.DatetimeIndex(gunler)
        bitisik = bool(np.all(np.diff(gunler.values).astype("timedelta64[D]").astype(int) == 1))
        V[b.ad] = {
            "w": w,
            "tani": tani,
            "g": g,
            "r": r,
            "kod": gun_kod.astype("int64"),
            "trafo": trafo_kod.astype("int64"),
            "gunler": gunler,
            "n": len(dg),
            "bitisik": bitisik,
        }
        print(
            f"  {b.ad:7} sicak {len(dg):>8,}  gun {len(gunler):>4}  bitisik {bitisik}  "
            f"trafo {len(np.unique(trafo_kod)):>6,}  ESS %{100 * tani['ess_orani']:.1f}  "
            f"guvenilir {tani['guvenilir']}"
        )

    # ------------------------------------------------------ 0) BANT BETIMLEMESI
    print("\n" + "-" * 110)
    print("0) BANT BETIMLEMESI -- trafo etkisi CIKARILMIS gun ekseni, k=3 torbali")
    print("   c_bant = kor_bant * (sigma_gercek_bant / sigma_model_bant)   [tek degiskenli]")
    print("-" * 110)
    print(
        f"  {'blok':8}{'W':>4}{'sd_m_du':>9}{'sd_g_du':>9}{'kor_du':>8}{'c_du':>8}"
        f"{'sd_m_yu':>9}{'sd_g_yu':>9}{'kor_yu':>8}{'c_yu':>8}{'|kor(u_du,u_yu)|':>18}"
    )
    for b in tm.BLOKLAR:
        v = V[b.ad]
        ng = len(v["gunler"])
        bm = gun_etkisi_kod(v["trafo"], v["kod"], v["r"]["bag"], ng)
        bg = gun_etkisi_kod(v["trafo"], v["kod"], v["g"], ng)
        kor_tam = float(np.corrcoef(bm, bg)[0, 1])
        print(
            f"  {b.ad:8}{'TAM':>4}{bm.std():9.4f}{bg.std():9.4f}{kor_tam:+8.3f}"
            f"{kor_tam * bg.std() / bm.std():8.3f}" + " " * 34 + f"{'':>18}"
        )
        for W in PENCERELER:
            md, my = bantla(bm, W)
            gd, gy = bantla(bg, W)
            kd = float(np.corrcoef(md, gd)[0, 1])
            ky = float(np.corrcoef(my, gy)[0, 1])
            cd = kd * gd.std() / md.std()
            cy = ky * gy.std() / my.std()
            cross = abs(float(np.corrcoef(md, my)[0, 1]))
            print(
                f"  {'':8}{W:>4}{md.std():9.4f}{gd.std():9.4f}{kd:+8.3f}{cd:8.3f}"
                f"{my.std():9.4f}{gy.std():9.4f}{ky:+8.3f}{cy:8.3f}{cross:18.3f}"
            )

    # ------------------------------------ 1) KAPALI FORM: tek c vs iki bant c
    print("\n" + "-" * 110)
    print("1) KAPALI FORM OPTIMUM (agirlikli MSE'yi minimize eden), k=3 torbali")
    print("   TEK: c*.  IKI BANT: (c_du, c_yu) es zamanli cozum (dik olmayan taban).")
    print("-" * 110)
    print(f"  {'blok':8}{'W':>4}{'c* tek':>9}{'c_du':>9}{'c_yu':>9}")
    kapali: dict[tuple[str, int], tuple[float, float]] = {}
    tek_c: dict[str, float] = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        ng = len(v["gunler"])
        bm = gun_etkisi_kod(v["trafo"], v["kod"], v["r"]["bag"], ng)
        e = v["g"] - v["r"]["bag"]
        u_tam = satir_bazi(bm, v["kod"])
        a1 = kapali_form(e, [u_tam], v["w"])
        tek_c[b.ad] = 1.0 + float(a1[0])
        for W in PENCERELER:
            md, my = bantla(bm, W)
            U = [satir_bazi(md, v["kod"]), satir_bazi(my, v["kod"])]
            a2 = kapali_form(e, U, v["w"])
            kapali[(b.ad, W)] = (1.0 + float(a2[0]), 1.0 + float(a2[1]))
            print(
                f"  {b.ad if W == PENCERELER[0] else '':8}{W:>4}"
                f"{tek_c[b.ad] if W == PENCERELER[0] else float('nan'):9.3f}"
                f"{kapali[(b.ad, W)][0]:9.3f}{kapali[(b.ad, W)][1]:9.3f}"
            )

    # --------------------------------- 2) dMSE: ORAKUL TAVANI (blok kendi c'si)
    print("\n" + "-" * 110)
    print("2) ORAKUL TAVANI -- her blok KENDI etiketli optimumunu kullanir (ust sinir!)")
    print("   dRMSLE_sicak = sonra - once (NEGATIF = kazanc).  genele = x0,5357")
    print("-" * 110)
    print(f"  {'blok':8}{'W':>4}{'taban RMSLE':>13}{'tek c*':>11}{'iki bant':>11}{'EK KAZANC':>12}")
    ek_kazanc: dict[int, list[float]] = {W: [] for W in PENCERELER}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        ng = len(v["gunler"])
        for ti, t in enumerate(TOHUMLAR + ("bag",)):
            if t != "bag":
                continue
            r = v["r"][t]
            bm = gun_etkisi_kod(v["trafo"], v["kod"], r, ng)
            e = v["g"] - r
            taban = np.sqrt(agirlikli_mse(e, v["w"]))
            u_tam = satir_bazi(bm, v["kod"])
            a1 = kapali_form(e, [u_tam], v["w"])
            s1 = np.sqrt(agirlikli_mse(v["g"] - uygula(r, [u_tam], a1), v["w"]))
            for W in PENCERELER:
                md, my = bantla(bm, W)
                U = [satir_bazi(md, v["kod"]), satir_bazi(my, v["kod"])]
                a2 = kapali_form(e, U, v["w"])
                s2 = np.sqrt(agirlikli_mse(v["g"] - uygula(r, U, a2), v["w"]))
                ek_kazanc[W].append(s2 - s1)
                print(
                    f"  {b.ad if W == PENCERELER[0] else '':8}{W:>4}"
                    f"{taban if W == PENCERELER[0] else float('nan'):13.5f}"
                    f"{s1 - taban if W == PENCERELER[0] else float('nan'):+11.5f}"
                    f"{s2 - taban:+11.5f}{s2 - s1:+12.5f}"
                )

    # --------------------------- 3) TOHUM BAZLI ESLENIK: iki bant EK kazanci
    print("\n" + "-" * 110)
    print("3) ESLENIK SH -- (blok, tohum) ciftleri; iki bant orakulu vs tek c* orakulu")
    print("   Her ikisi de KENDI blogunun etiketli optimumunu kullanir (tavan-tavan).")
    print("-" * 110)
    print(f"  {'W':>4}{'ort ek dRMSLE':>16}{'SH':>10}{'t':>8}{'iyi/N':>9}{'genele dMSE':>14}")
    for W in PENCERELER:
        farklar = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            ng = len(v["gunler"])
            for t in TOHUMLAR:
                r = v["r"][t]
                bm = gun_etkisi_kod(v["trafo"], v["kod"], r, ng)
                e = v["g"] - r
                u_tam = satir_bazi(bm, v["kod"])
                a1 = kapali_form(e, [u_tam], v["w"])
                s1 = np.sqrt(agirlikli_mse(v["g"] - uygula(r, [u_tam], a1), v["w"]))
                md, my = bantla(bm, W)
                U = [satir_bazi(md, v["kod"]), satir_bazi(my, v["kod"])]
                a2 = kapali_form(e, U, v["w"])
                s2 = np.sqrt(agirlikli_mse(v["g"] - uygula(r, U, a2), v["w"]))
                farklar.append(s2 - s1)
        f = np.array(farklar)
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        tt = float(f.mean() / sh) if sh > 0 else 0.0
        dglobal = f.mean() * SICAK_KATSAYI
        print(
            f"  {W:>4}{f.mean():+16.6f}{sh:10.6f}{tt:+8.2f}"
            f"{int((f < 0).sum()):>5}/{len(f)}{2 * LB_RMSLE * dglobal:+14.6f}"
        )

    # ------------------- 4) GECIRGENLIK: bir blogun bant c'si digerine tasiniyor mu
    print("\n" + "-" * 110)
    print("4) GECIRGENLIK -- yaz25'in bant c'si guz25/kis26'ya, ve tersi (W=15)")
    print("   Uygulanan c KAYNAK bloktan; olculen blok HEDEF. NEGATIF = kazanc.")
    print("-" * 110)
    W = 15
    print(
        f"  {'kaynak->hedef':>20}{'c_du':>8}{'c_yu':>8}{'iki bant dRMSLE':>18}{'tek c dRMSLE':>15}"
    )
    for kay in tm.BLOKLAR:
        for hed in tm.BLOKLAR:
            if kay.ad == hed.ad:
                continue
            v = V[hed.ad]
            ng = len(v["gunler"])
            r = v["r"]["bag"]
            bm = gun_etkisi_kod(v["trafo"], v["kod"], r, ng)
            e = v["g"] - r
            taban = np.sqrt(agirlikli_mse(e, v["w"]))
            cd, cy = kapali[(kay.ad, W)]
            md, my = bantla(bm, W)
            U = [satir_bazi(md, v["kod"]), satir_bazi(my, v["kod"])]
            s2 = np.sqrt(
                agirlikli_mse(v["g"] - uygula(r, U, np.array([cd - 1.0, cy - 1.0])), v["w"])
            )
            u_tam = satir_bazi(bm, v["kod"])
            s1 = np.sqrt(
                agirlikli_mse(v["g"] - uygula(r, [u_tam], np.array([tek_c[kay.ad] - 1.0])), v["w"])
            )
            print(
                f"  {kay.ad + '->' + hed.ad:>20}{cd:8.3f}{cy:8.3f}"
                f"{s2 - taban:+18.5f}{s1 - taban:+15.5f}"
            )

    # ---------------------------- 5) URETIM DEGERI: c=1,335 tek vs bant-oranli
    print("\n" + "-" * 110)
    print("5) URETIM DEGERI -- suanki c=1,335 tek bant vs ETIKETSIZ CAPA bant orani")
    print("-" * 110)
    _capa_bolumu(V, W=15)

    print(f"\n  toplam sure {time.time() - t0:.0f} sn")
    return 0


def _capa_bolumu(V: dict, W: int) -> None:
    """ETIKETSIZ CAPA: 2025 Nis-Tem GERCEK bantlari vs 2026 Nis-Tem TAHMIN bantlari."""
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    sicak_kume = set(tr["tanim"].unique())

    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    x = pd.DataFrame({"t": g["tanim"].to_numpy(), "gg": g["tarih"].to_numpy()})
    tam = x.groupby("t")["gg"].nunique()
    tam = set(tam[tam >= 0.9 * x["gg"].nunique()].index)
    sec = np.isin(g["tanim"].to_numpy(), list(tam))
    gs = g[sec]
    rg = np.log1p(gs["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        gs["guc"].to_numpy(dtype="float64")
    )
    tk, _ = pd.factorize(gs["tanim"].to_numpy())
    gk, gunler_g = pd.factorize(gs["tarih"].to_numpy(), sort=True)
    b_gercek = gun_etkisi_kod(tk.astype("int64"), gk.astype("int64"), rg, len(gunler_g))
    doy_g = pd.to_datetime(pd.Index(gunler_g)).dayofyear

    for ad in ("tuketim_v50_ham30.csv", "tuketim_v67_c1335_olay.csv"):
        yol = KOK / "submissions" / ad
        if not yol.exists():
            print(f"  {ad}: YOK")
            continue
        sub = pd.read_csv(yol, encoding="utf-8")
        m = te.merge(sub, on="id", how="left", validate="one_to_one")
        m = m[m["tanim"].isin(sicak_kume)]
        rt = np.log1p(np.clip(m["tuketim"].to_numpy(dtype="float64"), 0.0, None)) - np.log1p(
            m["guc"].to_numpy(dtype="float64")
        )
        tk2, _ = pd.factorize(m["tanim"].to_numpy())
        gk2, gunler_t = pd.factorize(m["tarih"].to_numpy(), sort=True)
        b_tahmin = gun_etkisi_kod(tk2.astype("int64"), gk2.astype("int64"), rt, len(gunler_t))
        doy_t = pd.to_datetime(pd.Index(gunler_t)).dayofyear

        gd, gy = bantla(b_gercek, W)
        td, ty = bantla(b_tahmin, W)
        sg = pd.Series(gd, index=doy_g)
        sy = pd.Series(gy, index=doy_g)
        pg = pd.Series(td, index=doy_t)
        py = pd.Series(ty, index=doy_t)
        ortak = sg.index.intersection(pg.index)
        kd = float(np.corrcoef(sg[ortak], pg[ortak])[0, 1])
        ky = float(np.corrcoef(sy[ortak], py[ortak])[0, 1])
        cd = kd * gd.std() / td.std()
        cy = ky * gy.std() / ty.std()
        kor_tam = float(
            np.corrcoef(
                pd.Series(b_gercek, index=doy_g)[ortak], pd.Series(b_tahmin, index=doy_t)[ortak]
            )[0, 1]
        )
        c_tam = kor_tam * b_gercek.std() / b_tahmin.std()
        print(f"\n  {ad}  ({len(m):,} sicak satir, {len(ortak)} ortak gun-of-year)")
        print(
            f"    TAM   sd_gercek {b_gercek.std():.4f}  sd_tahmin {b_tahmin.std():.4f}  "
            f"kor {kor_tam:+.3f}  ->  c_capa {c_tam:.3f}"
        )
        print(
            f"    DUSUK sd_gercek {gd.std():.4f}  sd_tahmin {td.std():.4f}  "
            f"kor {kd:+.3f}  ->  c_capa {cd:.3f}"
        )
        print(
            f"    YUKSEK sd_gercek {gy.std():.4f}  sd_tahmin {ty.std():.4f}  "
            f"kor {ky:+.3f}  ->  c_capa {cy:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
