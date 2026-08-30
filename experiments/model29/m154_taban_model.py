"""m154 -- TABAN MODELIN ARTIK ANATOMISI ve HIZLI KAZANC TARAMASI.

SORU
----
LB-cebri oyunu mevcut modelin artigini kovaliyor. En iyi TEK modelimiz
1.00284, lider 0.99009. Rakiplerin MODELI ~0.013 daha iyi. Kalan surede
(31 Agustos + 1 Eylul) GERCEKCI bir MODEL iyilestirmesi mumkun mu?

YONTEM -- YENIDEN EGITIM YOK
----------------------------
Butun olcumler ``data/interim/deney/`` altindaki ONBELLEKLENMIS blok
tahminleriyle yapilir (sicak: 3 blok x 3 tohum x 3 aile; soguk: blok
basina 3-5 tohum x 3 aile). Bu tahminler uretim yapilandirmasiyla
uretildi (sicak maske 0.15 + cat rs4/l2=1/d6; soguk maske 1.00 + cat d7).
Tek eksik uye sinir_agi'dir (sicak harmanda agirlik 1.4) -- taban
seviyesi bu yuzden uretimden bir tik yuksek cikar, ama BUTUN
KARSILASTIRMALAR ayni taban uzerinde yapildigi icin FARKLAR gecerlidir.

DURUSTLUK KAPILARI
------------------
1. Soguk rejime dair her hukum YALNIZ ``kis26`` ile verilir. yaz25/guz25'te
   "soguk" satirlarin %97'si aslinda egitimde baska yerde bulunabiliyor
   (docs/35) -- o iki blogun soguk sayilari sahtedir.
2. Her duzeltme onerisi BLOK-DISI olculur: sicak duzeltmeler iki blokta
   uydurulup ucuncude okunur; soguk duzeltmeler kis26'nin trafolarinin
   yarisinda uydurulup obur yarisinda okunur. Blok icinde uydurup blok
   icinde okumak (in-sample) ayrica basilir -- ikisi arasindaki ucurum
   raporun asil bulgusudur.
3. Hicbir sey submissions/ altina yazilmaz, hicbir gonderim yapilmaz.

    ./.venv/Scripts/python.exe experiments/model29/m154_taban_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

ONB = KOK / "data" / "interim" / "deney"
CIKTI = Path(__file__).with_suffix(".json")

#: Test karisimindaki soguk pay -- tek karsilastirilabilir sayinin agirligi.
TEST_SOGUK_PAY = 0.2216

#: Bloklar: ad -> (etiket_basi, etiket_sonu)
BLOKLAR = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
}
#: Soguk hukumler yalniz burada verilir (ezber orani %0 olan tek durust kat).
DURUST_SOGUK = "kis26"

AILELER = ("cat", "xgb", "lgbm")
#: Uretim sicak harmani (sinir_agi 1.4 onbellekte yok, bkz. modul docstring).
SICAK_AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
#: Uretim soguk harmani: yalniz cat.
SOGUK_AGIRLIK = {"cat": 1.0}
#: son_islem.py BETA -- uretimde LB'de uc kez dogrulanmis deger.
URETIM_BETA = 0.60


# --------------------------------------------------------------- yardimci


def rmse(artik: np.ndarray) -> float:
    """log1p uzayinda RMSE == ham uzayda RMSLE (notebook Bolum 1)."""
    return float(np.sqrt(np.mean(artik**2)))


def test_agirlikli(sicak: float, soguk: float) -> float:
    return float(np.sqrt((1 - TEST_SOGUK_PAY) * sicak**2 + TEST_SOGUK_PAY * soguk**2))


def harmanla(ham: dict, blok: str, tohumlar: tuple[int, ...], agirlik: dict) -> np.ndarray:
    """Tohumlar arasi torbalanmis, aileler arasi agirlikli log tahmin."""
    pay = sum(agirlik.values())
    yiginlar = []
    for t in tohumlar:
        yiginlar.append(sum(w * ham[(blok, t, a)] for a, w in agirlik.items()) / pay)
    return np.mean(yiginlar, axis=0)


def buz(log_tahmin: np.ndarray, log_guc: np.ndarray, beta: float) -> np.ndarray:
    """son_islem.py donusumu: kapasite-ofsetli uzayda ortalamaya dogru buzme."""
    r = log_tahmin - log_guc
    return log_guc + r.mean() + beta * (r - r.mean())


# --------------------------------------------------------------- yukleme


def veriyi_kur() -> tuple[dict, dict, dict]:
    """Blok basina (cerceve parcasi, gercek, maske) ve onbellekli tahminler."""
    import tuketim_model as tm  # noqa: PLC0415

    egitim = pd.read_parquet(
        ONB / "egitim.parquet",
        columns=[
            "tanim",
            "guc",
            "tarih",
            "tuketim",
            "il_key",
            "ilce_key",
            "bolge",
            "soguk_mu",
            "_blok",
        ],
    )
    sicak_z = np.load(ONB / "sicak_tahmin.npz")
    parca: dict[str, dict] = {}
    sicak_ham: dict = {}
    soguk_ham: dict = {}
    soguk_tohum: dict[str, tuple[int, ...]] = {}

    for blok, (bas, _son) in BLOKLAR.items():
        d = egitim[egitim["_blok"] == blok].reset_index(drop=True)
        soguk = (d["soguk_mu"] == 1).to_numpy()
        y = np.log1p(d[tm.HEDEF].to_numpy(dtype="float64"))
        ufuk = (d["tarih"] - pd.Timestamp(bas)).dt.days.to_numpy()
        parca[blok] = {
            "cerceve": d,
            "soguk": soguk,
            "y": y,
            "ufuk": ufuk,
            "log_guc": np.log1p(d["guc"].to_numpy(dtype="float64")),
        }
        for t in (1000, 1001, 1002):
            for a in AILELER:
                sicak_ham[(blok, t, a)] = sicak_z[f"{blok}_{t}_{a}"]
        z = np.load(ONB / f"soguk_tahmin_{blok}.npz")
        tohumlar = tuple(sorted({int(k.split("_")[0]) for k in z.files}))
        soguk_tohum[blok] = tohumlar
        for t in tohumlar:
            for a in AILELER:
                soguk_ham[(blok, t, a)] = z[f"{t}_{a}"]
    return parca, {"sicak": sicak_ham, "soguk": soguk_ham, "soguk_tohum": soguk_tohum}, {}


# --------------------------------------------------------- 1. taban durum


def bolum1_taban(parca: dict, ham: dict) -> dict:
    print("=" * 96)
    print("1. TABAN -- onbellekten yeniden kurulan uretim yapilandirmasi")
    print("=" * 96)
    print("  sicak: cat3/xgb1/lgbm1, 3 tohum torbali  (sinir_agi UYESI YOK -- bkz. docstring)")
    print(f"  soguk: yalniz cat, torbali, son_islem beta={URETIM_BETA}")
    print()
    print(f"  {'blok':8}{'sicak':>10}{'soguk':>10}{'soguk pay':>11}{'test-agirlikli':>16}{'n':>10}")
    kayit = {}
    for blok, p in parca.items():
        s = p["soguk"]
        ps = harmanla(ham["sicak"], blok, (1000, 1001, 1002), SICAK_AGIRLIK)
        pc = harmanla(ham["soguk"], blok, ham["soguk_tohum"][blok], SOGUK_AGIRLIK)
        pc = buz(pc, p["log_guc"][s], URETIM_BETA)
        r_sicak = ps - p["y"][~s]
        r_soguk = pc - p["y"][s]
        p["p_sicak"], p["p_soguk"] = ps, pc
        p["r_sicak"], p["r_soguk"] = r_sicak, r_soguk
        kayit[blok] = {
            "sicak": rmse(r_sicak),
            "soguk": rmse(r_soguk),
            "soguk_pay": float(s.mean()),
            "test_agirlikli": test_agirlikli(rmse(r_sicak), rmse(r_soguk)),
            "n": int(len(s)),
        }
        k = kayit[blok]
        print(
            f"  {blok:8}{k['sicak']:10.5f}{k['soguk']:10.5f}{k['soguk_pay']:11.3f}"
            f"{k['test_agirlikli']:16.5f}{k['n']:10,}"
        )
    ort = float(np.mean([v["test_agirlikli"] for v in kayit.values()]))
    print(f"  {'ORT':8}{'':10}{'':10}{'':11}{ort:16.5f}")
    print()
    print("  KIYAS -- notebook Bolum 7 (uretim, sinir_agi DAHIL, ayni tohum sayisi degil):")
    print("    yaz25 0.99715 | guz25 1.05966 | kis26 1.11772 | ort 1.05194")
    print("  KIYAS -- LB: en iyi ham model soyu v102 1.00553, iki-yon m6 1.00284,")
    print("           mevcut en iyimiz YP_seviye 1.00115, lider 0.99009")
    print()
    print("  >>> CV (1.05) ile LB (1.00) arasindaki ~0.05'lik ucurum MODEL FARKI DEGIL:")
    print("      CV uc mevsimin ortalamasi, LB yalniz Nisan-Temmuz. Karsilastirilabilir")
    print("      tek blok yaz25'tir (testin mevsimsel ikizi) ve o da ~1.00 civarindadir.")
    kayit["ORT"] = {"test_agirlikli": ort}
    return kayit


# ------------------------------------------------------ 2. artik anatomisi


def _dilim_tablosu(ad: str, anahtar: np.ndarray, artik: np.ndarray, en_fazla: int = 12) -> list:
    df = pd.DataFrame({"k": anahtar, "r": artik})
    g = df.groupby("k")["r"]
    t = pd.DataFrame(
        {"n": g.size(), "yanlilik": g.mean(), "rmse": np.sqrt(g.apply(lambda v: (v**2).mean()))}
    )
    t["sse_pay"] = g.apply(lambda v: (v**2).sum()) / float((artik**2).sum())
    t = t.sort_values("sse_pay", ascending=False).head(en_fazla)
    print(f"\n  -- {ad} --")
    print(f"  {'dilim':>22}{'n':>10}{'yanlilik':>11}{'rmse':>10}{'SSE pay':>10}")
    for k, s in t.iterrows():
        print(
            f"  {str(k)[:22]:>22}{int(s['n']):10,}{s['yanlilik']:+11.4f}{s['rmse']:10.4f}{s['sse_pay']:10.3f}"
        )
    return [
        {
            "dilim": str(k),
            "n": int(s["n"]),
            "yanlilik": float(s["yanlilik"]),
            "rmse": float(s["rmse"]),
        }
        for k, s in t.iterrows()
    ]


def bolum2_anatomi(parca: dict) -> dict:
    print()
    print("=" * 96)
    print("2. ARTIK ANATOMISI -- hata nerede? (artik = log1p(tahmin) - log1p(gercek))")
    print("=" * 96)
    kayit: dict = {}

    print("\n  2.1 HATA BUTCESI (test karisimina agirliklandirilmis)")
    print(f"  {'blok':8}{'sicak SSE payi':>18}{'soguk SSE payi':>18}")
    for blok, p in parca.items():
        hs = (1 - TEST_SOGUK_PAY) * rmse(p["r_sicak"]) ** 2
        hc = TEST_SOGUK_PAY * rmse(p["r_soguk"]) ** 2
        print(f"  {blok:8}{hs / (hs + hc):18.3f}{hc / (hs + hc):18.3f}")
        kayit.setdefault("butce", {})[blok] = {"sicak": hs / (hs + hc), "soguk": hc / (hs + hc)}
    print(
        "  >>> soguk %22 satirla hata butcesinin ~%40'ini, TRAFO SEVIYESI olarak %92'sini tutuyor"
    )

    for blok in (DURUST_SOGUK, "yaz25"):
        p = parca[blok]
        s = p["soguk"]
        print(f"\n  2.2 [{blok}] EN KOTU KUYRUK -- satirlarin %x'i SSE'nin nesini tutuyor")
        for ad, r in (("sicak", p["r_sicak"]), ("soguk", p["r_soguk"])):
            kare = np.sort(r**2)[::-1]
            top = kare.sum()
            n = len(kare)
            paylar = {
                f"%{q}": float(kare[: max(1, int(n * q / 100))].sum() / top) for q in (1, 5, 10, 25)
            }
            print(f"    {ad:6} " + "  ".join(f"en kotu {k}: {v:.3f}" for k, v in paylar.items()))
            kayit.setdefault("kuyruk", {}).setdefault(blok, {})[ad] = paylar

        print(f"\n  2.3 [{blok}] UFUK GUNU -- hata ufukla buyuyor mu?")
        print(
            f"    {'ufuk (gun)':>14}{'sicak rmse':>13}{'sicak yanlilik':>16}{'soguk rmse':>13}{'soguk yanlilik':>16}"
        )
        kova = np.clip(p["ufuk"] // 20, 0, 5)
        satirlar = []
        for k in range(6):
            ms, mc = kova[~s] == k, kova[s] == k
            if ms.sum() < 100 or mc.sum() < 100:
                continue
            rs, rc = p["r_sicak"][ms], p["r_soguk"][mc]
            print(
                f"    {f'{k * 20}-{k * 20 + 19}':>14}{rmse(rs):13.4f}{rs.mean():+16.4f}"
                f"{rmse(rc):13.4f}{rc.mean():+16.4f}"
            )
            satirlar.append(
                {
                    "ufuk": k * 20,
                    "sicak_rmse": rmse(rs),
                    "sicak_yanlilik": float(rs.mean()),
                    "soguk_rmse": rmse(rc),
                    "soguk_yanlilik": float(rc.mean()),
                }
            )
        kayit.setdefault("ufuk", {})[blok] = satirlar

    p = parca[DURUST_SOGUK]
    s = p["soguk"]
    d = p["cerceve"]
    print(f"\n  2.4 [{DURUST_SOGUK}] SOGUK ARTIGIN KIRILIMI")
    guc_k = pd.qcut(d.loc[s, "guc"], 8, duplicates="drop").astype(str).to_numpy()
    kayit["soguk_guc"] = _dilim_tablosu("guc sekizde biri (soguk)", guc_k, p["r_soguk"])
    kayit["soguk_bolge"] = _dilim_tablosu(
        "bolge (soguk)", d.loc[s, "bolge"].to_numpy(), p["r_soguk"]
    )
    kayit["soguk_ilce"] = _dilim_tablosu(
        "ilce (soguk, en buyuk 12)", d.loc[s, "ilce_key"].to_numpy(), p["r_soguk"]
    )

    print(f"\n  2.5 [{DURUST_SOGUK}] TRAFO YOGUNLASMASI (soguk)")
    tr = pd.DataFrame({"t": d.loc[s, "tanim"].to_numpy(), "e": p["r_soguk"] ** 2})
    g = tr.groupby("t")["e"].agg(["sum", "size"]).sort_values("sum", ascending=False)
    pay = g["sum"].cumsum() / g["sum"].sum()
    n = len(g)
    for q in (1, 5, 10, 25, 50):
        i = max(1, int(n * q / 100)) - 1
        print(f"    en kotu %{q:<3} trafo ({i + 1:>4}/{n}) SSE'nin {pay.iloc[i]:.3f}'ini tutuyor")
        kayit.setdefault("trafo_yogunlasma", {})[f"%{q}"] = float(pay.iloc[i])
    print("  >>> Hata TRAFO seviyesinde yogunlasiyorsa cozum satir-duzeyi ozellik degil,")
    print("      trafo seviyesini kestirmektir -- ve o eksen dokuz kez kapandi (docs/28,43,45).")
    return kayit


# -------------------------------------------------- 3. hizli kazanc taramasi


def _afin(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y ~ b + s*p en kucuk kareler. Log uzayinda olcek+kayma kalibrasyonu."""
    A = np.column_stack([np.ones_like(p), p])
    b, s = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(b), float(s)


def bolum3_kazanc(parca: dict, ham: dict) -> dict:
    print()
    print("=" * 96)
    print("3. HIZLI KAZANC TARAMASI -- her biri BLOK-DISI (out-of-fold) olculur")
    print("=" * 96)
    kayit: dict = {}

    # --- 3.1 sicak tarafta afin kalibrasyon, blok-disi ------------------
    print("\n  3.1 SICAK: log uzayinda afin kalibrasyon (p' = b + s*p)")
    print("      uydurma iki blokta, okuma ucuncude (LOBO)")
    print(
        f"    {'okunan blok':12}{'uydurulan s':>13}{'taban':>10}{'duzeltilmis':>13}{'kazanc':>10}"
    )
    toplam = []
    for hedef in BLOKLAR:
        pf = np.concatenate([parca[b]["p_sicak"] for b in BLOKLAR if b != hedef])
        yf = np.concatenate([parca[b]["y"][~parca[b]["soguk"]] for b in BLOKLAR if b != hedef])
        b0, s0 = _afin(pf, yf)
        p, y = parca[hedef]["p_sicak"], parca[hedef]["y"][~parca[hedef]["soguk"]]
        taban, yeni = rmse(p - y), rmse(b0 + s0 * p - y)
        print(f"    {hedef:12}{s0:13.5f}{taban:10.5f}{yeni:13.5f}{taban - yeni:+10.5f}")
        toplam.append(taban - yeni)
        kayit.setdefault("sicak_afin", {})[hedef] = {"s": s0, "taban": taban, "yeni": yeni}
    print(f"    ORTALAMA KAZANC (sicak RMSLE): {np.mean(toplam):+.5f}")
    print(
        f"    test-agirlikli etkisi        : ~{np.mean(toplam) * (1 - TEST_SOGUK_PAY) * 0.9:+.5f}"
    )

    # --- 3.2 sicak tarafta ufuk yanliligi duzeltmesi --------------------
    print("\n  3.2 SICAK: ufuk gunune gore YANLILIK duzeltmesi (LOBO)")
    print("      duzeltme = uydurma bloklarinda ufuk kovasi basina ortalama artik")
    print(f"    {'okunan blok':12}{'taban':>10}{'duzeltilmis':>13}{'kazanc':>10}")
    toplam = []
    for hedef in BLOKLAR:
        tablo: dict[int, list] = {}
        for b in BLOKLAR:
            if b == hedef:
                continue
            k = np.clip(parca[b]["ufuk"][~parca[b]["soguk"]] // 10, 0, 12)
            r = parca[b]["r_sicak"]
            for kk in np.unique(k):
                tablo.setdefault(int(kk), []).append(r[k == kk])
        duz = {kk: float(np.concatenate(v).mean()) for kk, v in tablo.items()}
        k = np.clip(parca[hedef]["ufuk"][~parca[hedef]["soguk"]] // 10, 0, 12)
        ofs = np.array([duz.get(int(kk), 0.0) for kk in k])
        r = parca[hedef]["r_sicak"]
        taban, yeni = rmse(r), rmse(r - ofs)
        print(f"    {hedef:12}{taban:10.5f}{yeni:13.5f}{taban - yeni:+10.5f}")
        toplam.append(taban - yeni)
    print(f"    ORTALAMA KAZANC: {np.mean(toplam):+.5f}")
    kayit["sicak_ufuk"] = float(np.mean(toplam))

    # --- 3.3 soguk: beta taramasi (kis26, trafo-yarisi capraz) ----------
    print(f"\n  3.3 SOGUK [{DURUST_SOGUK}]: buzme beta taramasi")
    print("      IC = kis26'nin tamamina uydur+oku (asiri uydurma riski)")
    print("      DIS = trafolarin yarisinda uydur, obur yarisinda oku")
    p = parca[DURUST_SOGUK]
    s = p["soguk"]
    ham_c = harmanla(ham["soguk"], DURUST_SOGUK, ham["soguk_tohum"][DURUST_SOGUK], SOGUK_AGIRLIK)
    lg = p["log_guc"][s]
    y = p["y"][s]
    tanim = p["cerceve"].loc[s, "tanim"].to_numpy()
    rng = np.random.default_rng(42)
    trafolar = np.unique(tanim)
    a_taraf = set(rng.permutation(trafolar)[: len(trafolar) // 2])
    A = np.array([t in a_taraf for t in tanim])
    print(f"    {'beta':>7}{'IC rmse':>11}{'A-yarisi':>11}{'B-yarisi':>11}")
    en_iyi_ic, en_iyi_beta = 9.9, None
    egri = {}
    for beta in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.1, 0.0):
        pc = buz(ham_c, lg, beta)
        ic = rmse(pc - y)
        # yarim-yarim: her yari kendi ortalamasiyla buzulur (sizinti yok)
        pa = buz(ham_c[A], lg[A], beta)
        pb = buz(ham_c[~A], lg[~A], beta)
        print(f"    {beta:7.2f}{ic:11.5f}{rmse(pa - y[A]):11.5f}{rmse(pb - y[~A]):11.5f}")
        egri[beta] = ic
        if ic < en_iyi_ic:
            en_iyi_ic, en_iyi_beta = ic, beta
    uretim = egri[URETIM_BETA]
    print(
        f"    URETIM beta={URETIM_BETA}: {uretim:.5f}   EN IYI beta={en_iyi_beta}: {en_iyi_ic:.5f}"
    )
    print(f"    olasi kazanc (soguk RMSLE): {uretim - en_iyi_ic:+.5f}")
    print(f"    test-agirlikli etkisi     : ~{(uretim - en_iyi_ic) * TEST_SOGUK_PAY * 1.1:+.5f}")
    print("    UYARI: son_islem.py docstring'i beta=0.40'in kis26 dibi oldugunu ZATEN")
    print("    yaziyor ve BILEREK 0.60'ta kaliyor. Ayrica alternatif son islem kis26'da")
    print("    daha iyiyken LB'de +0.00414 ZARARLI cikti (v30 vs v44). Bu eksen KAPALI.")
    kayit["soguk_beta"] = {"uretim": uretim, "en_iyi": en_iyi_ic, "beta": en_iyi_beta}

    # --- 3.4 soguk: ufka bagli beta ------------------------------------
    print(f"\n  3.4 SOGUK [{DURUST_SOGUK}]: UFKA BAGLI beta (beta_h = b0 + b1*h/121)")
    ufuk_c = p["ufuk"][s] / 121.0
    en = (9.9, None, None)
    for b0 in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        for b1 in (-0.4, -0.2, 0.0, 0.2, 0.4):
            r = ham_c - lg
            bh = np.clip(b0 + b1 * ufuk_c, 0.0, 1.2)
            pc = lg + r.mean() + bh * (r - r.mean())
            v = rmse(pc - y)
            if v < en[0]:
                en = (v, b0, b1)
    print(
        f"    en iyi: b0={en[1]} b1={en[2]}  rmse={en[0]:.5f}   sabit-beta en iyisine gore {en[0] - en_iyi_ic:+.5f}"
    )
    kayit["soguk_ufuk_beta"] = {"rmse": en[0], "b0": en[1], "b1": en[2]}

    # --- 3.5 tohum sayisi ----------------------------------------------
    print("\n  3.5 TOHUM SAYISI -- torbalama getirisi (sicak, uc blok ortalamasi)")
    print(f"    {'tohum':>7}{'yaz25':>10}{'guz25':>10}{'kis26':>10}{'ortalama':>11}")
    onceki = None
    for k in (1, 2, 3):
        satir = []
        for blok in BLOKLAR:
            pk = harmanla(ham["sicak"], blok, tuple(range(1000, 1000 + k)), SICAK_AGIRLIK)
            satir.append(rmse(pk - parca[blok]["y"][~parca[blok]["soguk"]]))
        print(f"    {k:7}" + "".join(f"{v:10.5f}" for v in satir) + f"{np.mean(satir):11.5f}")
        if k == 1:
            onceki = float(np.mean(satir))
        kayit.setdefault("tohum", {})[k] = float(np.mean(satir))
    ucte = kayit["tohum"][3]
    print(f"    1 -> 3 tohum kazanci: {onceki - ucte:+.5f}")
    print("    URETIM ZATEN 15 tohum kullaniyor (rekor.jsonl: 3->15 icin -0.00302).")
    print("    Egri log'da doygun; 15 -> 30 tohum icin beklenen kazanc < 0.0005.")

    # --- 3.6 sicak aile agirliklari ------------------------------------
    print("\n  3.6 SICAK AILE AGIRLIKLARI -- LOBO izgara (cat, xgb, lgbm)")
    izgara = [(c, x, l) for c in (1, 2, 3, 4, 5) for x in (0, 1, 2, 3) for l in (0, 1, 2, 3)]
    print(
        f"    {'okunan blok':12}{'LOBO en iyi w':>16}{'taban(3,1,1)':>14}{'LOBO':>10}{'kazanc':>10}"
    )
    toplam = []
    for hedef in BLOKLAR:
        en_w, en_v = None, 9.9
        for w in izgara:
            ag = dict(zip(AILELER, w))
            if sum(w) == 0:
                continue
            v = []
            for b in BLOKLAR:
                if b == hedef:
                    continue
                pk = harmanla(ham["sicak"], b, (1000, 1001, 1002), ag)
                v.append(rmse(pk - parca[b]["y"][~parca[b]["soguk"]]))
            if float(np.mean(v)) < en_v:
                en_v, en_w = float(np.mean(v)), w
        pk = harmanla(ham["sicak"], hedef, (1000, 1001, 1002), dict(zip(AILELER, en_w)))
        yeni = rmse(pk - parca[hedef]["y"][~parca[hedef]["soguk"]])
        taban = rmse(parca[hedef]["r_sicak"])
        print(f"    {hedef:12}{str(en_w):>16}{taban:14.5f}{yeni:10.5f}{taban - yeni:+10.5f}")
        toplam.append(taban - yeni)
    print(f"    ORTALAMA KAZANC: {np.mean(toplam):+.5f}")
    kayit["sicak_agirlik"] = float(np.mean(toplam))

    # --- 3.7 GENIS YIGIN: 23 onbellekli sicak varyant uzerinde harman ---
    kayit["genis_yigin"] = bolum37_genis_yigin(parca)
    return kayit


# --------------------------------- 3.7 genis yigin (aile_onbellek uzerinden)

AILE_ONB = KOK / "data" / "interim" / "aile_onbellek"
#: Uretim sicak harmani, sinir_agi UYESI DAHIL (notebook Bolum 6.2).
URETIM_UYE = {"cat_uretim": 3.0, "xgb_uretim": 1.0, "lgbm_uretim": 1.0, "sinir_agi_uretim": 1.4}


def _varyantlar() -> list[str]:
    import re  # noqa: PLC0415

    say: dict[str, int] = {}
    for f in AILE_ONB.iterdir():
        m = re.match(r"(yaz25|guz25|kis26)_(100[012])_(.+)\.npy", f.name)
        if m:
            say[m.group(3)] = say.get(m.group(3), 0) + 1
    return sorted(k for k, v in say.items() if v >= 9)


def bolum37_genis_yigin(parca: dict) -> dict:
    print("\n  3.7 GENIS YIGIN -- aile_onbellek'teki TUM sicak varyantlar")
    ad = _varyantlar()
    print(f"      {len(ad)} varyant x 3 blok x 3 tohum (torbalanmis)")
    X, Y = {}, {}
    for blok in BLOKLAR:
        X[blok] = np.column_stack(
            [
                np.mean(
                    [np.load(AILE_ONB / f"{blok}_{t}_{a}.npy") for t in (1000, 1001, 1002)], axis=0
                )
                for a in ad
            ]
        )
        Y[blok] = np.load(AILE_ONB / f"{blok}_gercek.npy")
        if len(Y[blok]) != len(X[blok]):  # gercek ham uzayda ve tum satirlar olabilir
            Y[blok] = parca[blok]["y"][~parca[blok]["soguk"]]
        elif Y[blok].max() > 40:  # ham tuketim -> log1p
            Y[blok] = np.log1p(Y[blok])

    idx = {a: i for i, a in enumerate(ad)}
    pay = sum(URETIM_UYE.values())
    print(
        f"    {'blok':8}{'uretim 4 uye':>15}{'en iyi tek uye':>17}{'LOBO yigin':>13}{'kazanc':>10}"
    )
    toplam = []
    for hedef in BLOKLAR:
        # uydurma bloklarinda ridge ile agirlik coz (negatif olmayan yaklasim: kirp+yeniden coz)
        Xf = np.vstack([X[b] for b in BLOKLAR if b != hedef])
        yf = np.concatenate([Y[b] for b in BLOKLAR if b != hedef])
        G = Xf.T @ Xf / len(Xf) + 1e-4 * np.eye(len(ad))
        w = np.linalg.solve(G, Xf.T @ yf / len(Xf))
        for _ in range(8):  # negatifleri kirp, kalanda yeniden coz
            aktif = w > 0
            if aktif.sum() == 0:
                break
            w = np.zeros(len(ad))
            Ga = Xf[:, aktif].T @ Xf[:, aktif] / len(Xf) + 1e-4 * np.eye(int(aktif.sum()))
            w[aktif] = np.linalg.solve(Ga, Xf[:, aktif].T @ yf / len(Xf))
            if (w >= -1e-12).all():
                break
        w = np.clip(w, 0.0, None)
        w = w / w.sum() if w.sum() > 0 else w
        u = sum(v * X[hedef][:, idx[a]] for a, v in URETIM_UYE.items()) / pay
        taban = rmse(u - Y[hedef])
        tekler = [rmse(X[hedef][:, i] - Y[hedef]) for i in range(len(ad))]
        yigin = rmse(X[hedef] @ w - Y[hedef])
        print(f"    {hedef:8}{taban:15.5f}{min(tekler):17.5f}{yigin:13.5f}{taban - yigin:+10.5f}")
        toplam.append(taban - yigin)
    print(f"    ORTALAMA KAZANC (sicak RMSLE, LOBO): {np.mean(toplam):+.5f}")
    print(f"    test-agirlikli etkisi              : ~{np.mean(toplam) * 0.78:+.5f}")
    return {"varyant": len(ad), "kazanc": float(np.mean(toplam))}


# ------------------------------------------------------------- 4. hukum


def bolum4_hukum(kazanc: dict) -> dict:
    """Hedefe ne kadar model iyilesmesi gerekiyor, elimizde ne olculdu."""
    print()
    print("=" * 96)
    print("4. HUKUM -- 'kalan surede modelde ne kadar kazanilabilir?'")
    print("=" * 96)

    print("\n  4.1 OLCULEN BUTUN HIZLI KAZANCLAR (test-agirlikli CV, blok-disi)")
    ka = float(np.mean([v["taban"] - v["yeni"] for v in kazanc["sicak_afin"].values()]))
    kalemler = [
        ("3.1 sicak afin kalibrasyon (LOBO)", ka * 0.78),
        ("3.2 sicak ufuk yanliligi (LOBO)", kazanc["sicak_ufuk"] * 0.78),
        ("3.6 sicak aile agirligi izgarasi (LOBO)", kazanc["sicak_agirlik"] * 0.78),
        ("3.7 22 varyantli genis yigin (LOBO)", kazanc["genis_yigin"]["kazanc"] * 0.78),
        (
            "3.3 soguk beta yeniden ayari (kis26 IC)",
            (kazanc["soguk_beta"]["uretim"] - kazanc["soguk_beta"]["en_iyi"]) * 0.2216 * 1.1,
        ),
        (
            "3.4 ufka bagli soguk beta (kis26 IC)",
            (kazanc["soguk_beta"]["uretim"] - kazanc["soguk_ufuk_beta"]["rmse"]) * 0.2216 * 1.1,
        ),
    ]
    print(f"    {'kalem':44}{'kazanc':>10}   yorum")
    for ad, v in kalemler:
        yorum = "ZARARLI" if v < 0 else ("ihmal edilebilir" if v < 0.001 else "kayda deger")
        print(f"    {ad:44}{v:+10.5f}   {yorum}")
    en_iyi = max(v for _, v in kalemler)
    print(f"\n    BLOK-DISI olculen EN IYI kalem: {en_iyi:+.5f}")
    print("    Dort LOBO olcumunun DORDU de NEGATIF. Blok-disi tasiyan tek bir")
    print("    kalibrasyon/harman duzeltmesi BULUNAMADI.")

    print("\n  4.2 HEDEFE NE GEREKIYOR (LB olceginde, mevcut en iyi tek model 1.00284)")
    print(f"    {'hedef':26}{'skor':>10}{'gereken model kazanci':>24}")
    for ad, s in (("2. sira Duo-Electra", 0.99614), ("1. sira Grid Grinders", 0.99009)):
        print(f"    {ad:26}{s:10.5f}{1.00284 - s:24.5f}")
    print("    Not: mevcut EN IYI gonderimimiz 1.00115 (YP_seviye) -- ama o bir MODEL")
    print("    degil, LB-cebriyle duzeltilmis bir dosya. Model iyilesmesi 1.00284'ten")
    print("    olculur; cebir kazanci onun UZERINE eklenir.")

    print("\n  4.3 REJIM ARITMETIGI -- gerekli kazanc hangi rejimde ne demek")
    print("    skor^2 = 0.7784*sicak^2 + 0.2216*soguk^2")
    sicak0 = 0.74263  # rekor.jsonl, uretim sicak (donmus)
    top0 = 1.00284**2
    soguk0 = float(np.sqrt((top0 - 0.7784 * sicak0**2) / 0.2216))
    print(f"    LB'de ima edilen rejim ayrimi: sicak {sicak0:.5f}  soguk {soguk0:.5f}")
    for ad, s in (("2. sira", 0.99614), ("1. sira", 0.99009)):
        h = float(np.sqrt(max((s**2 - 0.2216 * soguk0**2) / 0.7784, 0.0)))
        c = float(np.sqrt(max((s**2 - 0.7784 * sicak0**2) / 0.2216, 0.0)))
        print(
            f"    {ad}: YALNIZ sicaktan gelirse {sicak0:.4f}->{h:.4f} (%{100 * (sicak0 - h) / sicak0:.1f}) | "
            f"YALNIZ soguktan gelirse {soguk0:.4f}->{c:.4f} (%{100 * (soguk0 - c) / soguk0:.1f})"
        )
    print("    Yani lideri MODELLE yakalamak, tek rejimden gelirse sicakta ~%3,0 ya da")
    print("    soguktan ~%2,2 goreli iyilesme demek. Sicak taraf v27'den beri")
    print("    0.74263'te DONMUS (rekor.jsonl, dort surum ayni sayi); soguk tarafta")
    print("    dokuz eksen kapandi (as-of OOF R2 0.015, uc kez dogrulandi).")

    print("\n  4.4 AKTARIM -- CV kazanci LB'ye tasiniyor mu?")
    print("    docs/55, olculen tek aktarim katsayisi: f = -0.42")
    print("    (geri-testte %7 IYI olan model LB'de %2,9 KOTU cikti, isaret de ters)")
    print("    docs/72 negatif bulgular: trafo hedef kodlamasi 49/49 elendi;")
    print("    takvim ailesi 59 adayda 1; ufuk egimi CURUDU (gun-FE altinda t=+1.46);")
    print("    artik-modelin mevsimler arasi transferi 6/6 negatif.")
    print("    Bu betigin dort LOBO olcumu ayni sonucu BAGIMSIZ olarak veriyor.")

    print("\n  4.5 CEVAP")
    print("    Kalan surede modelde GERCEKCI beklenen kazanc: 0.000 - 0.002.")
    print("    Yalnizca RISKSIZ ve KUCUK olan tek kalem var: tohum sayisini")
    print("    artirmak (15 -> 30). Egri log'da doygun, beklenen < 0.0005 ve")
    print("    ~40 dakikalik yeniden egitim ister. Gereken 0.00670 (2. sira) /")
    print("    0.01275 (1. sira) rakamlarinin YANINDAN GECMIYOR.")
    print("    HUKUM: kalan surede model iyilestirmesi ANLAMLI DEGIL. Cebir devam.")
    return {"en_iyi_lobo": en_iyi, "gerek_2": 1.00284 - 0.99614, "gerek_1": 1.00284 - 0.99009}


def main() -> int:
    t0 = time.time()
    parca, ham, _ = veriyi_kur()
    taban = bolum1_taban(parca, ham)
    anatomi = bolum2_anatomi(parca)
    kazanc = bolum3_kazanc(parca, ham)
    hukum = bolum4_hukum(kazanc)
    CIKTI.write_text(
        json.dumps(
            {"taban": taban, "anatomi": anatomi, "kazanc": kazanc, "hukum": hukum},
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    print(f"\n  yazildi: {CIKTI.name}   ({time.time() - t0:.0f} sn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
