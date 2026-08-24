"""SICAK TARAFTA GUN EKSENI GENISLETMESI -- olcum, karar ve ETIKETSIZ capa.

SORU
----
Sicak satirlarda gun ortalamasini ``c`` kati genisletmek

    r' = genel + c*(gun_ort - genel) + (r - gun_ort)

agirlikli RMSLE'yi dusuruyor mu, ve ``c`` NASIL secilmeli?

``r = log1p(tahmin) - log1p(guc)`` -- kapasite ofsetli uzay, ``son_islem.py``
ile ayni uzay. Donusum satir ici (trafo ekseni) bileseni AYNEN birakir,
yalnizca gun ortalamasinin genel ortalamadan sapmasini olcekler.

NEDEN SORULUYOR
---------------
docs/41 §6c sogukta ayni ayrimi yapti: gun ekseninde model rampanin YONUNU
biliyor (korelasyon +0,865) ama GENLIGINI bilmiyor (OLS egimi +1,828).
Sicakta uretim son islemi hicbir sey yapmiyor (``son_islem.py`` yalnizca
soguk satirlara dokunuyor), yani sicak gun ekseni HIC ele alinmadi.

KALICI KURALLAR (docs/40, docs/41)
----------------------------------
* Hukum (blok, tohum) ciftleri uzerinde ESLENIK SH ile verilir.
* Kirpilmis tablo zorunlu: en buyuk K trafo ve en kotu K GUN atilarak.
* Bloklar zit ise: isaret donmesi mi (c'nin 1'in iki yaninda) yoksa yalnizca
  genlik farki mi -- sayiyla ayrilir.
* Etiketsiz kestirici (docs/39 dersi): model-disi nicelikten turetilen
  kestirim LB'ye TASINDI, tek dogrulama blogundan turetilen TASIMADI.

    python scripts/deney_gun_ekseni_sicak.py
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

#: ``d(genel)/d(sicak)`` -- docs/40; 0,01422 -> 0,00762 ile dogrulandi.
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907

C_IZGARA = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)


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


def genislet(r: np.ndarray, gun_kod: np.ndarray, c: float) -> np.ndarray:
    """``r' = genel + c*(gun_ort - genel) + (r - gun_ort)``  (AGIRLIKSIZ ortalama).

    Agirliksiz cunku gonderim aninda test satirlarinda uygulanacak sey budur;
    olcut agirliklari yalnizca DEGERLENDIRME icindir.
    """
    gun_ort = np.bincount(gun_kod, weights=r) / np.bincount(gun_kod)
    m = gun_ort[gun_kod]
    return r + (c - 1.0) * (m - r.mean())


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    t_bas = time.time()
    print("=" * 104)
    print("SICAK GUN EKSENI GENISLETMESI -- uretim esli onbellek, saf aritmetik")
    print("=" * 104)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]

    V: dict[str, dict] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        dg = dogrulama[sicak]
        w, tani = ol.test_agirliklari(dg, te_s, guc_kenar, eksenler=("bayatlik",))
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        gun = pd.to_datetime(dg["tarih"])
        kod, gunler = pd.factorize(gun, sort=True)
        g = np.log1p(np.clip(gercek[sicak], 0.0, None)) - lg
        r = {t: blend(b.ad, t) - lg for t in TOHUMLAR}
        r["bag"] = np.mean([blend(b.ad, t) for t in TOHUMLAR], axis=0) - lg
        V[b.ad] = {
            "w": w,
            "tani": tani,
            "g": g,
            "r": r,
            "kod": kod.astype("int64"),
            "gunler": pd.DatetimeIndex(gunler),
            "trafo": dg["tanim"].to_numpy(),
            "n": len(dg),
        }
        print(
            f"  {b.ad:7} sicak {len(dg):>8,}  gun {len(gunler):>4}  "
            f"ESS %{100 * tani['ess_orani']:.1f}  kirpilan %{100 * tani['kirpilan']:.2f}  "
            f"kapsanmayan %{100 * tani['kapsanmayan']:.2f}  guvenilir {tani['guvenilir']}"
        )

    # ------------------------------------------------------------- 0) betimleme
    print("\n" + "-" * 104)
    print("0) BETIMLEME (k=3 torbalanmis, AGIRLIKSIZ gun ortalamalari)")
    print("-" * 104)
    print(f"  {'blok':8}{'model std':>11}{'gercek std':>12}{'kor':>8}{'OLS egim':>10}{'gun':>7}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        n_d = np.bincount(v["kod"])
        m_d = np.bincount(v["kod"], weights=v["r"]["bag"]) / n_d
        g_d = np.bincount(v["kod"], weights=v["g"]) / n_d
        kor = float(np.corrcoef(m_d, g_d)[0, 1])
        egim = float(np.polyfit(m_d - m_d.mean(), g_d - g_d.mean(), 1)[0])
        print(f"  {b.ad:8}{m_d.std():11.4f}{g_d.std():12.4f}{kor:+8.3f}{egim:+10.3f}{len(n_d):7d}")

    # ------------------------------------------------------------- 1) c taramasi
    print("\n" + "-" * 104)
    print("1) c TARAMASI -- agirlikli RMSLE (olcut.py, eksenler=('bayatlik',))")
    print("   fark = taban - aday  (POZITIF = KAZANC).  genele = -fark * 0,5357")
    print("-" * 104)
    print(f"  {'c':>6}", end="")
    for b in tm.BLOKLAR:
        print(f"{b.ad:>11}", end="")
    print(f"{'HAVUZ':>11}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")

    tabani: dict[tuple[str, int], float] = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        for t in TOHUMLAR:
            tabani[(b.ad, t)] = agirlikli_mse(v["g"] - v["r"][t], v["w"])

    tarama = {}
    for c in C_IZGARA:
        satir, farklar = {}, []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            # blok skoru: k=3 torbalanmis (uretim gibi)
            e_t = v["g"] - genislet(v["r"]["bag"], v["kod"], c)
            e_0 = v["g"] - v["r"]["bag"]
            satir[b.ad] = np.sqrt(agirlikli_mse(e_0, v["w"])) - np.sqrt(agirlikli_mse(e_t, v["w"]))
            for t in TOHUMLAR:
                e = v["g"] - genislet(v["r"][t], v["kod"], c)
                farklar.append(np.sqrt(tabani[(b.ad, t)]) - np.sqrt(agirlikli_mse(e, v["w"])))
        f = np.array(farklar)
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        tt = float(f.mean() / sh) if sh > 0 else 0.0
        tarama[c] = {"blok": satir, "f": f, "ort": float(f.mean()), "sh": sh, "t": tt}
        print(f"  {c:6.2f}", end="")
        for b in tm.BLOKLAR:
            print(f"{satir[b.ad]:+11.5f}", end="")
        print(
            f"{f.mean():+11.5f}{sh:10.5f}{tt:+8.2f}{int((f > 0).sum()):>5}/{len(f)}"
            f"{-f.mean() * SICAK_KATSAYI:+10.5f}"
        )

    # ---------------------------------------------------- 2) c_opt: isaret mi genlik mi
    print("\n" + "-" * 104)
    print("2) KAPALI FORM c_opt  (agirlikli MSE'yi minimize eden c; blok x tohum)")
    print("   c_opt = 1 + sum(w*e*u)/sum(w*u^2),  u = gun_ort(r) - genel(r),  e = g - r")
    print("-" * 104)
    print(f"  {'blok':8}", end="")
    for t in TOHUMLAR:
        print(f"{'t' + str(t):>10}", end="")
    print(f"{'k=3':>10}{'1 uzeri?':>10}")
    copt_blok = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        print(f"  {b.ad:8}", end="")
        cs = []
        for t in list(TOHUMLAR) + ["bag"]:
            r = v["r"][t]
            n_d = np.bincount(v["kod"])
            m_d = (np.bincount(v["kod"], weights=r) / n_d)[v["kod"]]
            u = m_d - r.mean()
            e = v["g"] - r
            c = 1.0 + float(np.dot(v["w"], e * u) / np.dot(v["w"], u * u))
            cs.append(c)
            print(f"{c:10.3f}", end="")
        copt_blok[b.ad] = cs[-1]
        print(f"{'EVET' if cs[-1] > 1 else 'HAYIR':>10}")

    print("\n  ISARET TESTI: c_opt hepsi 1'in AYNI tarafinda mi?")
    ust = [b.ad for b in tm.BLOKLAR if copt_blok[b.ad] > 1]
    alt = [b.ad for b in tm.BLOKLAR if copt_blok[b.ad] <= 1]
    print(f"    c_opt > 1 : {ust}")
    print(f"    c_opt <= 1: {alt}")
    print(
        "    -> "
        + (
            "ISARET DONMESI (docs/39 sinifi)"
            if ust and alt
            else "isaret AYNI; yalnizca GENLIK degisiyor"
        )
    )

    # ------------------------------------------------------ 3) ETIKETSIZ c kestirici
    print("\n" + "-" * 104)
    print("3) ETIKETSIZ CAPA: ulusal endeks gun eksenini modelden iyi kestiriyor mu?")
    print("-" * 104)

    gunluk = {}
    for b in tm.BLOKLAR:
        v = V[b.ad]
        n_d = np.bincount(v["kod"])
        gunluk[b.ad] = pd.DataFrame(
            {
                "tarih": v["gunler"],
                "n": n_d,
                "m": np.bincount(v["kod"], weights=v["r"]["bag"]) / n_d,
                "g": np.bincount(v["kod"], weights=v["g"]) / n_d,
            }
        )

    ulus = (
        pd.concat([egitim, test], ignore_index=True)[
            ["tarih", "ulusal_gunluk", "ulusal_tepe", "ulusal_yil_once", "ulusal_yillik_buyume"]
        ]
        .drop_duplicates("tarih")
        .set_index("tarih")
    )
    ulus["z_gun"] = np.log(ulus["ulusal_gunluk"])
    ulus["z_tepe"] = np.log(ulus["ulusal_tepe"])
    ulus["z_yo"] = ulus["ulusal_yillik_buyume"]

    ADAY_Z = ("z_gun", "z_tepe", "z_yo")
    print(f"  {'blok':8}{'gun':>6}", end="")
    for z in ADAY_Z:
        print(f"{'kor(' + z + ',g)':>16}", end="")
    print(f"{'kor(model,g)':>15}")
    for b in tm.BLOKLAR:
        gf = gunluk[b.ad].join(ulus, on="tarih")
        print(f"  {b.ad:8}{len(gf):6d}", end="")
        for z in ADAY_Z:
            print(f"{gf[z].corr(gf['g']):+16.3f}", end="")
        print(f"{gf['m'].corr(gf['g']):+15.3f}")

    # blok-disi transfer: gg ~ a + b*z DIGER iki blokta uydurulur
    print("\n  BLOK-DISI TRANSFER  (z tek degisken; blok ici merkezlenmis)")
    print(
        f"  {'blok':8}{'z':>8}{'b_kaynak':>10}{'kor(ghat,g)':>13}"
        f"{'egim(ghat~m)':>14}{'c_opt(etiketli)':>17}"
    )
    for b in tm.BLOKLAR:
        for z in ADAY_Z:
            kay = pd.concat(
                [gunluk[o.ad].join(ulus, on="tarih") for o in tm.BLOKLAR if o.ad != b.ad]
            )
            kay = kay.dropna(subset=[z, "g"])
            # blok kimligini dusurmek icin her blok kendi icinde merkezlenir
            kayb = []
            for o in tm.BLOKLAR:
                if o.ad == b.ad:
                    continue
                q = gunluk[o.ad].join(ulus, on="tarih").dropna(subset=[z, "g"])
                kayb.append(pd.DataFrame({"z": q[z] - q[z].mean(), "g": q["g"] - q["g"].mean()}))
            kay = pd.concat(kayb)
            bz = float(np.polyfit(kay["z"], kay["g"], 1)[0])
            gf = gunluk[b.ad].join(ulus, on="tarih").dropna(subset=[z])
            ghat = bz * (gf[z] - gf[z].mean())
            gc = gf["g"] - gf["g"].mean()
            mc = gf["m"] - gf["m"].mean()
            egim = float(np.polyfit(mc, ghat, 1)[0])
            print(
                f"  {b.ad:8}{z:>8}{bz:10.3f}{float(np.corrcoef(ghat, gc)[0, 1]):+13.3f}"
                f"{egim:+14.3f}{copt_blok[b.ad]:17.3f}"
            )

    # TEST donemi: v50 ham30 sicak tahminlerinden gun ekseni
    ham = KOK / "submissions" / "tuketim_v50_ham30.csv"
    if ham.exists():
        sub = pd.read_csv(ham, encoding="utf-8")
        t2 = test[["id", "tarih", "guc", "soguk_mu", "tanim"]].merge(sub, on="id", how="left")
        ts = t2[t2["soguk_mu"] != 1].copy()
        ts["r"] = np.log1p(ts["tuketim"].clip(lower=0.0)) - np.log1p(ts["guc"])
        gt = ts.groupby("tarih").agg(n=("r", "size"), m=("r", "mean")).reset_index()
        gt = gt.join(ulus, on="tarih")
        print(f"\n  TEST sicak: {len(ts):,} satir, {len(gt)} gun")
        print(f"  {'z':>8}{'b_egitim':>10}{'std(ghat)':>11}{'std(m)':>10}{'egim(ghat~m)':>14}")
        egitim_gun = pd.concat([gunluk[b.ad].join(ulus, on="tarih") for b in tm.BLOKLAR])
        for z in ADAY_Z:
            kayb = []
            for o in tm.BLOKLAR:
                q = gunluk[o.ad].join(ulus, on="tarih").dropna(subset=[z, "g"])
                kayb.append(pd.DataFrame({"z": q[z] - q[z].mean(), "g": q["g"] - q["g"].mean()}))
            kay = pd.concat(kayb)
            bz = float(np.polyfit(kay["z"], kay["g"], 1)[0])
            gh = bz * (gt[z] - gt[z].mean())
            mc = gt["m"] - gt["m"].mean()
            print(
                f"  {z:>8}{bz:10.3f}{float(gh.std()):11.4f}{float(mc.std()):10.4f}"
                f"{float(np.polyfit(mc, gh, 1)[0]):+14.3f}"
            )
        _ = egitim_gun
    else:
        print(f"\n  UYARI: {ham} yok -- test gun ekseni olculemedi")

    # ------------------------------------------------------------- 4) kirpilmis
    print("\n" + "-" * 104)
    print("4) KIRPILMIS HUKUM  (blok x tohum eslenik; K trafo / K gun atilarak)")
    print("-" * 104)

    def kirpilmis(c: float, ne: str, K: int) -> tuple[float, float, float, int]:
        f = []
        for b in tm.BLOKLAR:
            v = V[b.ad]
            anahtar = v["trafo"] if ne == "trafo" else v["kod"]
            # her tohum icin ayni kirpma kumesi: k=3 torbalanmis d(MSE)'ye gore
            e0 = v["g"] - v["r"]["bag"]
            e1 = v["g"] - genislet(v["r"]["bag"], v["kod"], c)
            # POZITIF = kazanc. EN BUYUK KAZANC saglayanlar atilir (yogunlasma
            # testi). Ters yon (en cok zarar verenleri atmak) sonuca gore
            # secim yapmaktir ve hicbir sey kanitlamaz.
            d_mse = v["w"] * (e0 * e0 - e1 * e1)
            pay = pd.Series(d_mse).groupby(pd.Series(anahtar)).sum()
            kotu = set(pay.nlargest(K).index) if K else set()
            tut = ~pd.Series(anahtar).isin(kotu).to_numpy()
            for t in TOHUMLAR:
                a = v["g"] - v["r"][t]
                z = v["g"] - genislet(v["r"][t], v["kod"], c)
                f.append(
                    np.sqrt(agirlikli_mse(a[tut], v["w"][tut]))
                    - np.sqrt(agirlikli_mse(z[tut], v["w"][tut]))
                )
        fa = np.array(f)
        sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
        return float(fa.mean()), sh, float(fa.mean() / sh) if sh > 0 else 0.0, int((fa > 0).sum())

    for c in (1.25, 1.5, 1.75):
        print(f"\n  c = {c}")
        for ne, Ks in (("trafo", (0, 1, 5, 10, 25, 50)), ("gun", (0, 1, 5, 10, 25))):
            print(f"    {'K ' + ne:>10}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>8}{'genele':>10}")
            for K in Ks:
                ort, sh, tt, kaz = kirpilmis(c, ne, K)
                print(
                    f"    {K:>10}{ort:+11.5f}{sh:10.5f}{tt:+8.2f}{kaz:>5}/9"
                    f"{-ort * SICAK_KATSAYI:+10.5f}"
                )

    # yogunlasma: en buyuk trafo ve ilk5 payi
    print("\n  YOGUNLASMA (k=3, c=1,5)")
    print(f"  {'blok':8}{'trafo':>8}{'toplam d(MSE)':>16}{'EN BUYUK':>11}{'ilk5':>9}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        e0 = v["g"] - v["r"]["bag"]
        e1 = v["g"] - genislet(v["r"]["bag"], v["kod"], 1.5)
        d_mse = v["w"] * (e0 * e0 - e1 * e1)  # POZITIF = kazanc
        pay = pd.Series(d_mse).groupby(pd.Series(v["trafo"])).sum().sort_values(ascending=False)
        top = pay.sum()
        print(
            f"  {b.ad:8}{pay.size:>8}{top:16.1f}"
            f"{100 * pay.iloc[0] / top:10.1f}%{100 * pay.iloc[:5].sum() / top:8.1f}%"
        )

    # -------------------------------------- 5) KALICI KURAL 2: MEVSIM KAPSAMASI
    # Bir gun-ekseni bulgusu ancak egitimin ETIKET AYLARI dogrulamaninkileri
    # kapsiyorsa "kolon" olur; kapsamiyorsa ogrenilen sey desendir.
    print("\n" + "-" * 104)
    print("5) EGITIM ETIKET AYI KAPSAMASI  (kalici kural 2, gun eksenine uygulanmis)")
    print("-" * 104)
    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    print(f"  {'blok':8}{'dogrulama aylari':>22}{'egitimde bu aylar':>20}{'egitim satir':>14}")
    for b in tm.BLOKLAR:
        v_ay = sorted(
            pd.to_datetime(egitim.loc[egitim["_blok"] == b.ad, "tarih"]).dt.month.unique()
        )
        parca = tm.kokenleri_ayikla(genis, b.ad)
        p_ay = pd.to_datetime(parca["tarih"]).dt.month
        pay = float(p_ay.isin(v_ay).mean())
        print(
            f"  {b.ad:8}{','.join(f'{a:02d}' for a in v_ay):>22}"
            f"{'%' + f'{100 * pay:.2f}':>20}{len(parca):>14,}"
        )
    t_ay = sorted(pd.to_datetime(test["tarih"]).dt.month.unique())
    u_ay = pd.to_datetime(genis["tarih"]).dt.month
    print(
        f"  {'URETIM':8}{','.join(f'{a:02d}' for a in t_ay):>22}"
        f"{'%' + f'{100 * float(u_ay.isin(t_ay).mean()):.2f}':>20}{len(genis):>14,}   <- TEST"
    )

    # ---------------------------------- 6) DURUST PROTOKOL: c blok-disi secilir
    print("\n" + "-" * 104)
    print("6) BLOK-DISI c SECIMI  (c diger IKI bloktan, ucuncude olculur)")
    print("-" * 104)

    def c_kapali(bloklar: list[str], t) -> float:  # noqa: ANN001
        pay = payda = 0.0
        for a in bloklar:
            v = V[a]
            r = v["r"][t]
            n_d = np.bincount(v["kod"])
            u = (np.bincount(v["kod"], weights=r) / n_d)[v["kod"]] - r.mean()
            e = v["g"] - r
            pay += float(np.dot(v["w"], e * u))
            payda += float(np.dot(v["w"], u * u))
        return 1.0 + pay / payda

    print(f"  {'blok':8}{'c(kaynak)':>11}{'fark':>11}{'genele':>10}")
    dis_f = []
    for b in tm.BLOKLAR:
        kaynak = [o.ad for o in tm.BLOKLAR if o.ad != b.ad]
        c_b = c_kapali(kaynak, "bag")
        v = V[b.ad]
        e0 = v["g"] - v["r"]["bag"]
        e1 = v["g"] - genislet(v["r"]["bag"], v["kod"], c_b)
        fark = np.sqrt(agirlikli_mse(e0, v["w"])) - np.sqrt(agirlikli_mse(e1, v["w"]))
        print(f"  {b.ad:8}{c_b:11.3f}{fark:+11.5f}{-fark * SICAK_KATSAYI:+10.5f}")
        for t in TOHUMLAR:
            ct = c_kapali(kaynak, t)
            a = v["g"] - v["r"][t]
            z = v["g"] - genislet(v["r"][t], v["kod"], ct)
            dis_f.append(np.sqrt(agirlikli_mse(a, v["w"])) - np.sqrt(agirlikli_mse(z, v["w"])))
    fa = np.array(dis_f)
    sh = float(fa.std(ddof=1) / np.sqrt(len(fa)))
    print(
        f"  {'HAVUZ':8}{'':>11}{fa.mean():+11.5f}{-fa.mean() * SICAK_KATSAYI:+10.5f}"
        f"   SH {sh:.5f}  t {fa.mean() / sh:+.2f}  tohum {int((fa > 0).sum())}/9"
    )

    # ------------------------------- 7) BILESIM KONTROLU: trafo-merkezli gun ekseni
    print("\n" + "-" * 104)
    print("7) BILESIM KONTROLU -- gun ekseni trafo-merkezlendikten sonra da duruyor mu")
    print("-" * 104)
    print(f"  {'blok':8}{'ham egim':>11}{'merkezli egim':>16}{'ham kor':>10}{'merk kor':>10}")
    for b in tm.BLOKLAR:
        v = V[b.ad]
        n_d = np.bincount(v["kod"])
        m_d = np.bincount(v["kod"], weights=v["r"]["bag"]) / n_d
        g_d = np.bincount(v["kod"], weights=v["g"]) / n_d
        tr = pd.Series(v["trafo"])
        rm = v["r"]["bag"] - tr.map(pd.Series(v["r"]["bag"]).groupby(tr).mean()).to_numpy()
        gm = v["g"] - tr.map(pd.Series(v["g"]).groupby(tr).mean()).to_numpy()
        m_c = np.bincount(v["kod"], weights=rm) / n_d
        g_c = np.bincount(v["kod"], weights=gm) / n_d
        print(
            f"  {b.ad:8}"
            f"{float(np.polyfit(m_d - m_d.mean(), g_d - g_d.mean(), 1)[0]):+11.3f}"
            f"{float(np.polyfit(m_c - m_c.mean(), g_c - g_c.mean(), 1)[0]):+16.3f}"
            f"{float(np.corrcoef(m_d, g_d)[0, 1]):+10.3f}"
            f"{float(np.corrcoef(m_c, g_c)[0, 1]):+10.3f}"
        )

    # ------------------- 8) MEKANIZMA SINAMASI: kapsanan ay vs kapsanmayan ay
    # kis26 TEK blok ki egitiminde kendi aylarindan bir kismi (02,03) VAR.
    # Mekanizma dogruysa c_opt kapsanan aylarda 1'e YAKIN, kapsanmayanlarda
    # 1'in USTUNDE olmali. Uretim testin dort ayini da kapsiyor.
    print("\n" + "-" * 104)
    print("8) MEKANIZMA: c_opt, egitimde GORULEN ay ile GORULMEYEN ayda ayri (kis26)")
    print("-" * 104)
    v = V["kis26"]
    ay = V["kis26"]["gunler"].month.to_numpy()[v["kod"]]
    print(f"  {'altkume':>22}{'n':>10}{'model std':>11}{'gercek std':>12}{'egim':>9}{'c_opt':>9}")
    for ad, maske in (
        ("02,03  GORULEN", np.isin(ay, [2, 3])),
        ("12,01  GORULMEYEN", np.isin(ay, [12, 1])),
    ):
        kod2, gun2 = pd.factorize(pd.Series(v["kod"])[maske], sort=True)
        r, g, w = v["r"]["bag"][maske], v["g"][maske], v["w"][maske]
        n_d = np.bincount(kod2)
        m_d, g_d = np.bincount(kod2, weights=r) / n_d, np.bincount(kod2, weights=g) / n_d
        u = m_d[kod2] - r.mean()
        c = 1.0 + float(np.dot(w, (g - r) * u) / np.dot(w, u * u))
        print(
            f"  {ad:>22}{maske.sum():>10,}{m_d.std():11.4f}{g_d.std():12.4f}"
            f"{float(np.polyfit(m_d - m_d.mean(), g_d - g_d.mean(), 1)[0]):+9.3f}{c:9.3f}"
        )
        _ = gun2

    # 8b) UFUK KONTROLU. kis26'da "gorulen ay" ayni zamanda GEC ufuk. yaz25 ve
    # guz25'te HICBIR ay gorulmuyor, yani orada ilk/son yari farki SAF ufuktur.
    print("\n  UFUK KONTROLU -- kapsamasi %0 olan bloklarda ilk yari / son yari")
    print(
        f"  {'blok':8}{'yari':>10}{'tarih':>14}{'n':>10}"
        f"{'model std':>11}{'gercek std':>12}{'c_opt':>9}"
    )
    for b in tm.BLOKLAR:
        vv = V[b.ad]
        # KRONOLOJIK yari (ay adiyla degil): kis26 aylari 12,1,2,3 -- ay
        # numarasina gore siralamak sirayi bozar.
        orta = len(vv["gunler"]) // 2
        etik = {
            "ilk yari": f"{vv['gunler'][0]:%m-%d}..{vv['gunler'][orta - 1]:%m-%d}",
            "son yari": f"{vv['gunler'][orta]:%m-%d}..{vv['gunler'][-1]:%m-%d}",
        }
        for ad, sec in (("ilk yari", vv["kod"] < orta), ("son yari", vv["kod"] >= orta)):
            mk = sec
            kod2 = pd.factorize(pd.Series(vv["kod"])[mk], sort=True)[0]
            r, g, w = vv["r"]["bag"][mk], vv["g"][mk], vv["w"][mk]
            n_d = np.bincount(kod2)
            m_d, g_d = np.bincount(kod2, weights=r) / n_d, np.bincount(kod2, weights=g) / n_d
            u = m_d[kod2] - r.mean()
            c = 1.0 + float(np.dot(w, (g - r) * u) / np.dot(w, u * u))
            print(
                f"  {b.ad:8}{ad:>10}{etik[ad]:>14}{int(mk.sum()):>10,}"
                f"{m_d.std():11.4f}{g_d.std():12.4f}{c:9.3f}"
            )

    # -------------- 9) CAPA-GERCEK UYUMU ve YAZ25-TEK kirpilmis hukum
    print("\n" + "-" * 104)
    print("9) ULUSAL CAPA ile GERCEK c_opt UYUMU  +  yaz25-TEK kirpilmis hukum")
    print("-" * 104)
    capa = np.array([0.600, 0.923, 2.310])  # egim(ghat~m), z_gun, blok-disi
    gercek_c = np.array([copt_blok[b.ad] for b in tm.BLOKLAR])
    print(f"  capa (z_gun)      {capa}")
    print(f"  c_opt (etiketli)  {np.round(gercek_c, 3)}")
    print(f"  Pearson kor       {float(np.corrcoef(capa, gercek_c)[0, 1]):+.3f}   (3 nokta)")
    print(f"  isaret uyumu      {int(((capa > 1) == (gercek_c > 1)).sum())}/3")

    print(f"\n  {'c':>6}{'K':>5}{'ne':>7}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>8}")
    for c in (1.5, 1.75):
        for ne, Ks in (("trafo", (0, 5, 25, 50)), ("gun", (0, 5, 10, 25))):
            for K in Ks:
                f = []
                vy = V["yaz25"]
                e0b = vy["g"] - vy["r"]["bag"]
                e1b = vy["g"] - genislet(vy["r"]["bag"], vy["kod"], c)
                anah = vy["trafo"] if ne == "trafo" else vy["kod"]
                pay = pd.Series(vy["w"] * (e0b * e0b - e1b * e1b)).groupby(pd.Series(anah)).sum()
                tut = ~pd.Series(anah).isin(set(pay.nlargest(K).index) if K else set()).to_numpy()
                for t in TOHUMLAR:
                    a = vy["g"] - vy["r"][t]
                    z = vy["g"] - genislet(vy["r"][t], vy["kod"], c)
                    f.append(
                        np.sqrt(agirlikli_mse(a[tut], vy["w"][tut]))
                        - np.sqrt(agirlikli_mse(z[tut], vy["w"][tut]))
                    )
                fa = np.array(f)
                sh = float(fa.std(ddof=1) / np.sqrt(3))
                print(
                    f"  {c:6.2f}{K:>5}{ne:>7}{fa.mean():+11.5f}{sh:10.5f}"
                    f"{fa.mean() / sh:+8.2f}{int((fa > 0).sum()):>5}/3"
                )

    # ------- 10) BAGIMSIZ 2025 REFERANSI: gecen yilin GERCEKLESEN yaz genligi
    # docs/41 §6c'deki soguk teyidin sicak karsiligi. 364 gun kaydirma
    # haftagunu hizasini korur. Model-disi, ETIKETSIZ (test etiketi degil,
    # GECEN YILIN egitim etiketi) -- docs/39'un "tasiyan" sinifi.
    print("\n" + "-" * 104)
    print("10) BAGIMSIZ 2025 REFERANSI -- test gun ekseni vs 2025'in GERCEK yaz rampasi")
    print("-" * 104)
    if ham.exists():
        e_s = egitim[egitim["soguk_mu"] != 1].copy()
        e_s["a"] = np.log1p(e_s["tuketim"].clip(lower=0.0)) - np.log1p(e_s["guc"])
        # trafo-merkezli (bilesim kontrolu) ve ham
        e_s["a_m"] = e_s["a"] - e_s.groupby("tanim", observed=True)["a"].transform("mean")
        ga = e_s.groupby("tarih").agg(a=("a", "mean"), a_m=("a_m", "mean"), n=("a", "size"))
        ts["r_m"] = ts["r"] - ts.groupby("tanim", observed=True)["r"].transform("mean")
        gm = ts.groupby("tarih").agg(m=("r", "mean"), m_m=("r_m", "mean"), n=("r", "size"))
        gm = gm.reset_index()
        gm["ref_tarih"] = pd.to_datetime(gm["tarih"]) - pd.Timedelta(days=364)
        gm = gm.join(ga, on="ref_tarih", rsuffix="_ref").dropna(subset=["a"])
        print(f"  eslesen gun {len(gm)} / {len(ts.groupby('tarih'))}")
        for etk, mk, ak in (("HAM", "m", "a"), ("TRAFO-MERKEZLI", "m_m", "a_m")):
            mc, ac = gm[mk] - gm[mk].mean(), gm[ak] - gm[ak].mean()
            print(
                f"  {etk:16} std(model TEST) {float(mc.std()):.4f}   "
                f"std(gercek 2025) {float(ac.std()):.4f}   "
                f"kor {float(np.corrcoef(mc, ac)[0, 1]):+.3f}   "
                f"OLS egim(gercek~model) {float(np.polyfit(mc, ac, 1)[0]):+.3f}"
            )
        gm["_ay"] = pd.to_datetime(gm["tarih"]).dt.month
        print(f"\n  {'ay':>4}{'model TEST 2026':>18}{'gercek 2025':>14}{'fark':>9}")
        for a_, q in gm.groupby("_ay"):
            print(
                f"  {a_:>4}{q['m'].mean() - gm['m'].mean():+18.3f}"
                f"{q['a'].mean() - gm['a'].mean():+14.3f}"
                f"{q['a'].mean() - gm['a'].mean() - (q['m'].mean() - gm['m'].mean()):+9.3f}"
            )
        # AYNI referans yaz25 DOGRULAMA modeline uygulanirsa ne der?
        # (orada gercegi biliyoruz: c_opt = 1,752 -- referansin kalibresi)
        vy = V["yaz25"]
        n_d = np.bincount(vy["kod"])
        my = pd.DataFrame(
            {"tarih": vy["gunler"], "m": np.bincount(vy["kod"], weights=vy["r"]["bag"]) / n_d}
        )
        my["ref_tarih"] = pd.to_datetime(my["tarih"]) - pd.Timedelta(days=364)
        my = my.join(ga, on="ref_tarih").dropna(subset=["a"])
        if len(my) > 2:
            mc = my["m"] - my["m"].mean()
            ac = my["a"] - my["a"].mean()
            kuyruk = f", OLS egim {float(np.polyfit(mc, ac, 1)[0]):+.3f}"
        else:
            kuyruk = "  (2024 verisi YOK -- referans yaz25'te KURULAMAZ)"
        print(
            f"\n  KALIBRE: ayni referans yaz25 modeline uygulansaydi"
            f" -> eslesen gun {len(my)}{kuyruk}"
        )

    # ------- 11) ULUSAL ENDEKS: 2026 yazi 2025 yazina benziyor mu?
    # 364-gun referansinin KALIBRESI icin ayri betik:
    #     python scripts/deney_gun_ekseni_referans.py
    print("\n" + "-" * 104)
    print("11) 2026 YAZI 2025 YAZINA BENZIYOR MU (ulusal endeks, etiketsiz)")
    print("-" * 104)
    uu = ulus.reset_index()
    uu["_y"] = pd.to_datetime(uu["tarih"]).dt.year
    uu["_ay"] = pd.to_datetime(uu["tarih"]).dt.month
    print(f"  {'pencere':>18}{'gun':>6}{'std(log ulusal)':>18}{'07 - 04 farki':>16}")
    for yil in (2025, 2026):
        q = uu[(uu["_y"] == yil) & (uu["_ay"].between(4, 7))].dropna(subset=["z_gun"])
        if len(q) < 30:
            continue
        z = q["z_gun"] - q["z_gun"].mean()
        d47 = float(q.loc[q["_ay"] == 7, "z_gun"].mean() - q.loc[q["_ay"] == 4, "z_gun"].mean())
        print(f"  {f'{yil} 04-07':>18}{len(q):>6}{float(z.std()):18.4f}{d47:+16.4f}")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
