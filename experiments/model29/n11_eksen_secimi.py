"""EKSEN ALT KUMESI + AGIRLIKLANDIRMA -- gerceklesen korelasyonu maksimize et.

SORU. m148 136 dik eksenden BETA = toplam KATS[i]*U[i] kuruyor
(KATS[i] = 1.95*|rho_s_i|*isaret(rho_cv_i)). Ongorulen rho = ||BETA||
buyuyor ama GERCEKLESEN rho K~25'te doyuyor. Demek ki sorun eksen SAYISI
degil, eksen SECIMI ve AGIRLIKLANDIRMASI.

OLCUM DUZENI (sizinti yok). Elimizde uc egitim blogu var: yaz25, guz25,
kis26. Bir blokta SEC + AGIRLIKLANDIR, DIGER blokta OLC. Uc katli
donusumlu dogrulama: her blok sirayla "olcum blogu" olur, digerleri
"uydurma blogu". rho_s LIDERLIK TABLOSU olcumlerinden gelir (test
uzayinda) ve bloklardan bagimsizdir -- bu yuzden her iki tarafta da
serbestce kullanilabilir. rho_cv ise BLOGA BAGLIDIR ve yalnizca uydurma
blogundan hesaplanir.

GEOMETRI. Eksenler test uzayinda span'e (V) dik hale getirilir:
    xp_i = x_i - V c_i
Gram-Schmidt bunlari ortonormal U'ya cevirir; U = T xp (T alt ucgen).
BETA = toplam k_i U_i  =>  BETA = toplam c_i xp_i  ile  c = T' k.
BLOK KARSILIGI: blokta span parcasi ZATEN artikta yok (rb = y - model),
yani xp_i'nin blok karsiligi dogrudan xb_i'dir. Dolayisiyla
    z_b = toplam c_i xb_i
ve GERCEKLESEN rho = CARPAN * <rb, z_b>_w / sqrt(m0b * <z_b,z_b>_w).
Bu yalnizca BETA'nin YONUNE baglidir, uzunluguna degil -- tam da
"gerceklesen korelasyon" kavramina karsilik gelir.

HESAP KISAYOLU. Her sey Gram matrislerine iner:
    Gt_ij = <xp_i, xp_j>          (test, Gram-Schmidt ve Q_dik icin)
    Gb_ij = <xb_i, xb_j>_w        (blok)
    gb_i  = <rb, xb_i>_w          (blok)
Boylece 714 bin x 300 matrisler bir kez taranir, gerisi 300x300'de doner.
Onyukleme (bootstrap) icin blok, trafo (tanim) gruplarina gore KUMELERE
bolunur ve her kumenin kendi Gb/gb'si saklanir; onyukleme kumeleri
yeniden orneklemekten ibarettir.

HICBIR GONDERIM YAZILMAZ. m148 DEGISTIRILMEZ.
"""

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
ARA = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
os.makedirs(ARA, exist_ok=True)

TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
RHO_S_ALT = 0.015
QS_ALT, QD_ALT = 0.02, 0.25
AZAMI_EKSEN = 40  # m148'in m121 taramasindan aldigi ust sinir
HAVUZ_AZAMI = 460
BLOKLAR = ("yaz25", "guz25", "kis26")
KUME = 80  # onyukleme kume sayisi (trafo gruplari bu kumelere dagitilir)
SIC_AILE = [(t, aa) for t in (1000, 1001, 1002) for aa in ("cat", "xgb", "lgbm")]

sys.path.insert(0, M29)


# --------------------------------------------------------------- ortak
def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}
CARP_KIP = ("x_sv", "x_soguk", "x_ufuk", "x_ay")


def parcala(ad):
    """Eksen adini kur()'un gordugu bicimde ayristirir."""
    if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
        k1, k2 = ad[2:-1].split("]x[", 1)
        return ("carp", k1, k2)
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        return ("ikili", k1, k2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    return ("tek", kol, kip)


def kolonlar(ad, acc):
    t, a, b = parcala(ad)
    if t == "carp":
        kolonlar(a, acc)
        kolonlar(b, acc)
    elif t == "ikili":
        acc.add(a)
        acc.add(b)
    else:
        acc.add(a)
    return acc


def yap_kur(df, carp, tp_ref):
    """m148'in kur()'unun TEK TARAFLI kopyasi. Esikler HER ZAMAN test
    (tp_ref) yuzdeliklerinden gelir -- m148'de de oyle."""

    def kur(ad):
        t, a, b = parcala(ad)
        if t == "carp":
            v1, v2 = kur(a), kur(b)
            return None if v1 is None or v2 is None else st(v1 * v2)
        if t == "ikili":
            if a not in df.columns or b not in df.columns:
                return None
            v1, v2 = st(df[a].to_numpy()), st(df[b].to_numpy())
            return None if v1 is None or v2 is None else st(v1 * v2)
        kol, kip = a, b
        if kol not in df.columns or kol not in tp_ref.columns:
            return None
        x = df[kol].to_numpy()
        if kip in carp:
            m = carp[kip]
            v = st(x)
            return None if v is None else st(v * m)
        if kip in ESIK:
            q, ust = ESIK[kip]
            xr = tp_ref[kol].to_numpy(dtype=np.float64)
            fv = xr[np.isfinite(xr)]
            if fv.size == 0:
                return None
            v_ = np.quantile(fv, q)
            return st((x > v_).astype(np.float64)) if ust else st((x < v_).astype(np.float64))
        if kip == "mnt75":
            xr = tp_ref[kol].to_numpy(dtype=np.float64)
            fv = xr[np.isfinite(xr)]
            if fv.size == 0:
                return None
            v_ = float(np.quantile(fv, 0.75))
            return st(np.maximum(np.asarray(x, dtype=np.float64) - v_, 0.0))
        if kip == "kare":
            v = st(x)
            return None if v is None else st(v**2)
        return st(x)

    return kur


def aday_listesi():
    """m148'in gordugu sira: once m121 taramasi, sonra m144'un yenileri."""
    with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
        tarama = json.load(fh)
    with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
        m144 = json.load(fh)["kapidan_gecen"]
    yeni = sorted(m144, key=lambda r: -abs(r["rho_s"]))
    aile = {r["eksen"]: r["aile"] for r in m144}
    ads = [(k["eksen"], False, "m121_taban") for k in tarama]
    ads += [(r["eksen"], True, aile[r["eksen"]]) for r in yeni]
    gor, ciks = set(), []
    for ad, y, al in ads:
        if ad in gor:
            continue
        gor.add(ad)
        ciks.append((ad, y, al))
    return ciks


# =========================================================== FAZ 1: TEST
def faz1():
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ids = te.id.values
    del te

    def oku(f):
        d = pd.read_csv(os.path.join(S, f))
        k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
        if not np.array_equal(d.id.values, ids):
            if len(d) != len(ids) or d.id.duplicated().any():
                return None
            pos = pd.Index(d.id).get_indexer(ids)
            if (pos < 0).any():
                return None
            d = d.iloc[pos].reset_index(drop=True)
        return np.log1p(d[k].values.astype(np.float64))

    from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat

    a0 = oku(TABAN)
    n = len(a0)
    with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
        sk = json.load(fh)
    with open(os.path.join(M29, "m112_durum.json")) as fh:
        dur = json.load(fh)
    vv, ll = [], []
    for f, pj in sk.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        v = oku(f)
        if v is None or len(v) != n:
            continue
        dd = v - a0
        vv.append(dd)
        ll.append((M0 + float((dd * dd).mean()) - pj * pj) / 2)
    for f, lj in EK_MODEL.items():
        vv.append(oku(f) - a0)
        ll.append(lj)
    for o in dur.get("olcumler", []):
        dd = oku(o["dosya"]) - a0
        vv.append(dd)
        ll.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
    V, L = np.array(vv).T, np.array(ll)
    del vv
    gc.collect()
    G = (V.T @ V) / n
    Gi = np.linalg.pinv(G, rcond=1e-6)
    GI5 = np.linalg.pinv(G, rcond=1e-5)
    r_hat, gercek, kL = buzmeli_r_hat(V, L, G, n)
    mse_opt = M0 - gercek
    print(f"saf optimum {np.sqrt(mse_opt):.6f}   V: {V.shape}", flush=True)

    adaylar = aile_listesi = aday_listesi()
    ihtiyac = set()
    for ad, _, _ in adaylar:
        kolonlar(ad, ihtiyac)
    import pyarrow.parquet as pq

    tum = set(pq.read_schema(os.path.join(DN, "test.parquet")).names)
    kols = sorted(ihtiyac & tum) + ["soguk_mu", "ufuk_gun", "tarih"]
    tp = pd.read_parquet(os.path.join(DN, "test.parquet"), columns=sorted(set(kols)))
    print(f"test sutunlari: {len(tp.columns)} / {len(tum)}", flush=True)

    carpT = {
        "x_sv": st(a0),
        "x_soguk": tp.soguk_mu.values.astype(np.float64),
        "x_ufuk": st(tp.ufuk_gun.to_numpy()),
        "x_ay": st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64)),
    }
    kurT = yap_kur(tp, carpT, tp)

    havuz, BB = [], []
    XT = np.zeros((HAVUZ_AZAMI, n), dtype=np.float32)
    n121 = 0
    for ad, yeni, al in aile_listesi:
        if len(havuz) >= HAVUZ_AZAMI:
            break
        if not yeni and n121 >= 3 * AZAMI_EKSEN:
            continue  # m148 m121'den en fazla 40 KABUL eder; 120 aday bol bir ust kume
        xt = kurT(ad)
        if xt is None:
            continue
        b = (V.T @ xt) / n
        cc = Gi @ b
        xp0 = xt - V @ cc
        qs = 1.0 - float((xp0 * xp0).mean())
        if qs < QS_ALT:
            continue
        dot = float((r_hat * xt).mean())
        rho_s = dot / np.sqrt(qs)
        if abs(rho_s) < RHO_S_ALT:
            continue
        cc5 = GI5 @ b
        xp5 = xt - V @ cc5
        qs5 = 1.0 - float((xp5 * xp5).mean())
        if qs5 < QS_ALT:
            continue
        if abs(dot / np.sqrt(qs5) - rho_s) > 0.3 * abs(rho_s):
            continue
        XT[len(havuz)] = xt
        havuz.append(dict(eksen=ad, yeni=bool(yeni), aile=al, rho_s=float(rho_s), Qs=float(qs)))
        BB.append(b)
        if not yeni:
            n121 += 1
    print(f"havuz: {len(havuz)} eksen ({n121} m121, {len(havuz) - n121} m144)", flush=True)

    XT = XT[: len(havuz)]
    BBm = np.array(BB).T  # (nV, p)
    Gt = (XT @ XT.T).astype(np.float64) / n - BBm.T @ Gi @ BBm
    Gt = (Gt + Gt.T) / 2
    del XT, BB, V, tp
    gc.collect()

    np.savez(
        os.path.join(ARA, "n11_faz1.npz"),
        Gt=Gt,
        rho_s=np.array([h["rho_s"] for h in havuz]),
        mse_opt=mse_opt,
        M0=M0,
        taban_mse=float(M0 - 2 * kL + float((r_hat * r_hat).mean())),
    )
    with open(os.path.join(ARA, "n11_havuz.json"), "w", encoding="utf-8") as fh:
        json.dump(havuz, fh, ensure_ascii=False)
    print("faz1 bitti", flush=True)


# ========================================================== FAZ 2: BLOKLAR
def faz2():
    with open(os.path.join(ARA, "n11_havuz.json"), encoding="utf-8") as fh:
        havuz = json.load(fh)
    adlar = [h["eksen"] for h in havuz]
    ihtiyac = set()
    for ad in adlar:
        kolonlar(ad, ihtiyac)
    import pyarrow.parquet as pq

    tumk = set(pq.read_schema(os.path.join(DN, "test.parquet")).names)
    ekstra = ["soguk_mu", "ufuk_gun", "tarih", "tanim", "tuketim", "_blok"]
    kols = sorted((ihtiyac & tumk) | set(ekstra))
    tp_ref = pd.read_parquet(os.path.join(DN, "test.parquet"), columns=sorted(ihtiyac & tumk))
    e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=kols)
    print(f"egitim sutunlari {len(e.columns)}; havuz {len(adlar)}", flush=True)

    cikti = {}
    for blok in BLOKLAR:
        blk = e[e._blok == blok]
        sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
        P = [
            np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
            for t, aa in SIC_AILE
            if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
        ]
        z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
        idx = np.concatenate([sic.index.values, sog.index.values])
        pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
        del P, z
        bf = e.loc[idx].copy()
        rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
        sgm = bf.soguk_mu.values.astype(np.float64)
        ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
        ww = ww / ww.mean()
        m0b = float((ww * rb * rb).mean())
        nb = len(rb)

        carpB = {
            "x_sv": st(pb),
            "x_soguk": sgm,
            "x_ufuk": st(bf.ufuk_gun.to_numpy()),
            "x_ay": st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64)),
        }
        kurB = yap_kur(bf, carpB, tp_ref)
        Xb = np.zeros((len(adlar), nb), dtype=np.float32)
        gecerli = np.zeros(len(adlar), dtype=bool)
        for i, ad in enumerate(adlar):
            v = kurB(ad)
            if v is None:
                continue
            Xb[i] = v.astype(np.float32)
            gecerli[i] = True
        print(f"{blok}: {nb:,} satir, {int(gecerli.sum())}/{len(adlar)} eksen kuruldu", flush=True)

        # plasebo gurultusu (m148'in kapisi): trafo gruplarini permute et
        rng = np.random.default_rng(5)
        tn = bf.tanim.values
        uqn = pd.unique(tn)
        gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
        perm = [
            np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
            for _ in range(20)
        ]
        wr = (ww * rb).astype(np.float32)
        kor = (Xb @ wr) / nb / np.sqrt(m0b)
        pz = np.array([(Xb @ (ww * rb[s]).astype(np.float32)) / nb / np.sqrt(m0b) for s in perm])
        gur = pz.std(axis=0)
        del pz, perm

        # onyukleme kumeleri: trafo gruplari KUME parcaya bolunur
        rng2 = np.random.default_rng(7)
        kg = rng2.integers(0, KUME, size=len(uqn))[gi]
        Gk = np.zeros((KUME, len(adlar), len(adlar)), dtype=np.float32)
        gk = np.zeros((KUME, len(adlar)), dtype=np.float64)
        mk = np.zeros(KUME, dtype=np.float64)
        for j in range(KUME):
            m = kg == j
            Xs = Xb[:, m]
            wj = ww[m].astype(np.float32)
            Gk[j] = (Xs * wj) @ Xs.T / nb
            gk[j] = Xs.astype(np.float64) @ (ww[m] * rb[m]) / nb
            mk[j] = float((ww[m] * rb[m] * rb[m]).sum()) / nb
        del Xb, Xs
        gc.collect()
        np.savez(
            os.path.join(ARA, f"n11_blok_{blok}.npz"),
            Gk=Gk,
            gk=gk,
            mk=mk,
            kor=kor.astype(np.float64),
            gur=gur.astype(np.float64),
            gecerli=gecerli,
            m0b=m0b,
            nb=nb,
        )
        cikti[blok] = dict(nb=nb, m0b=m0b, gecerli=int(gecerli.sum()))
        del Gk, gk, bf, rb, ww
        gc.collect()
    print(json.dumps(cikti, indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("faz", choices=["1", "2"])
    a = ap.parse_args()
    (faz1 if a.faz == "1" else faz2)()
