"""p05: IKI ASAMALI AYRISTIRMA -- URETIM boru hattinda, IKI BLOKTA olculmus.

p03_uretim_iki_asama.py yalnizca yaz25'te olcuyor. Burada ayni kurulus
BLOK BLOK tekrarlanir (yaz25 ve guz25), cunku p01'in dersi sudur: bir
duzeltme tek blokta kazandirip digerinde KAYBETTIREBILIR. Ayrica ilce-ici
tanim_num-komsu seviyesi ozelligi (c varyanti) eklenir.

KURULUS (her degerlendirme blogu B icin)
  taban pb(B) = uretim hattinin B blogundaki tahmini
       (aile_onbellek/B_{tohum}_{alg}_uretim.npy sicak satirlar icin,
        soguk_tahmin_B.npz soguk satirlar icin -- m148_demet_plani ile ayni)
  egitim  = DIGER iki blogun satirlari
  (a) (1-P0) * pb
  (b) (1-P0) * ppos      ppos: YALNIZ pozitif satirlarda huber regresyon
  (c) (b) + ilce-ici komsu seviyesi ozelligi
  (b2)/(c2) harman: (1-P0) * (w*ppos + (1-w)*pb)

SIZINTI KONTROLU
  * Degerlendirme blogunun (ornegin yaz25) HEDEFI hicbir modelde etiket
    olarak kullanilmaz; modeller yalnizca diger bloklarin satirlarinda
    egitilir. Erken durdurma YOK (sabit tur sayisi), kalibrasyon sabiti YOK.
  * KIMLIK sutunlari (tanim_num, tanim_on2..5) ELENIR: guz25/kis26
    satirlarinin t_* ozetleri yaz25 donemini kapsar; kimlikle birlestiginde
    model "bu trafo yaz25'te suydu" ezberleyebilirdi.
  * Komsu ozelligi de yalnizca t_log_ort (blogun KENDI kesim oncesi ozeti)
    uzerinden hesaplanir; degerlendirme blogunun hedefi girmez.
  * TEST vektorunde egitim = TUM egitim bloklari (yaz25 dahil) -- orada
    yaz25 gecmis, sizinti degil.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
ARA = os.environ.get(
    "ARA",
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
HEDEF_SOGUK = 0.222
KIMLIK = ["tanim_num", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5"]
ATLA = ["tanim", "tarih", "tuketim", "lokasyon", "_blok", "id"]
BLOKLAR = ["yaz25", "guz25", "kis26"]
TUR = 500
TOHUM = [7]
ORT = dict(
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=8,
    verbose=-1,
)
PC = dict(ORT, objective="binary", metric="binary_logloss")
PR_HUB = dict(ORT, objective="huber", alpha=2.0, lambda_l2=20.0, metric="l2")
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------- veri ----------------
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SAY = [
    c
    for c in e.columns
    if c not in ATLA and c not in KIMLIK and pd.api.types.is_numeric_dtype(e[c])
]
log(f"egitim {len(e)} satir, {len(SAY)} ozellik")


def uretim_tahmini(blok):
    """m148_demet_plani.py satir 95-113 ile AYNI birlestirme."""
    blk = e[e._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    assert len(pb) == len(idx) == len(blk), (len(pb), len(idx), len(blk))
    return idx, pb, len(P)


def komsu_seviye(df_kaynak, df_hedef, k=8):
    """Ayni ilcede tanim_num'a en yakin k komsunun kesim-oncesi t_log_ort ort.

    df_kaynak: seviyesi BILINEN (t_log_ort dolu) trafolarin blok-ici tekil kaydi.
    Hedefteki trafonun kendisi komsulardan CIKARILIR.
    """
    kay = df_kaynak.dropna(subset=["t_log_ort", "tanim_num", "ilce_key"])
    kay = kay.drop_duplicates("tanim")
    out = {}
    for ilce, gl in kay.groupby("ilce_key", observed=True):
        gh = df_hedef[df_hedef.ilce_key == ilce]
        if len(gl) < 2 or len(gh) == 0:
            continue
        o = np.argsort(gl.tanim_num.to_numpy())
        xs = gl.tanim_num.to_numpy()[o]
        vs = gl.t_log_ort.to_numpy()[o]
        ids = gl.tanim.to_numpy()[o]
        for tn, xv in zip(gh.tanim.to_numpy(), gh.tanim_num.to_numpy()):
            if not np.isfinite(xv):
                continue
            j = int(np.searchsorted(xs, xv))
            sel = np.arange(max(0, j - k), min(len(xs), j + k))
            sel = sel[ids[sel] != tn]
            if len(sel):
                out[tn] = float(vs[sel].mean())
    return out


def komsu_sutunu(df):
    """df: bir blogun (veya testin) tum satirlari. Kendi blogunun ici kaynak."""
    tek = df.drop_duplicates("tanim")[["tanim", "tanim_num", "ilce_key", "t_log_ort"]]
    m = komsu_seviye(tek, tek)
    return df.tanim.map(m).astype(np.float32).to_numpy()


def egit_tahmin(pk, X, y, Xh, tohumlar=TOHUM, tur=TUR):
    return np.mean(
        [
            lgb.train(dict(pk, seed=s), lgb.Dataset(X, y), num_boost_round=tur).predict(Xh)
            for s in tohumlar
        ],
        axis=0,
    )


def olcucu(yv, sgm):
    ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
    ww = ww / ww.mean()
    s = sgm == 1

    def olc(p):
        r = np.asarray(p, dtype=np.float64) - yv
        return {
            "duz": float(np.sqrt(np.mean(r * r))),
            "test_agirlikli": float(np.sqrt(np.mean(ww * r * r))),
            "soguk": float(np.sqrt(np.mean(r[s] ** 2))),
            "sicak": float(np.sqrt(np.mean(r[~s] ** 2))),
        }

    return olc


def blok_kosusu(hedef_blok):
    idx, pb, nmod = uretim_tahmini(hedef_blok)
    bf = e.loc[idx]
    yv = np.log1p(bf.tuketim.to_numpy(dtype=np.float64))
    sgm = bf.soguk_mu.to_numpy(dtype=np.float64)
    olc = olcucu(yv, sgm)
    taban = olc(pb)
    log(f"{hedef_blok}: {len(pb)} satir, {nmod} sicak model, soguk {sgm.mean():.4f}")
    log(f"{hedef_blok} URETIM TABANI {json.dumps(taban)}")

    egt = e[~e._blok.eq(hedef_blok)]
    ye = np.log1p(egt.tuketim.to_numpy(dtype=np.float64))
    ze = (egt.tuketim.to_numpy() == 0).astype(np.int8)
    poz = ye > 0
    Xe = egt[SAY].astype(np.float32)
    Xh = bf[SAY].astype(np.float32)

    P0 = egit_tahmin(PC, Xe, ze, Xh)
    log(
        f"{hedef_blok} P0 ort {P0.mean():.4f} (gercek sifir "
        f"{float((bf.tuketim.to_numpy() == 0).mean()):.4f}), egitim sifir {ze.mean():.4f}"
    )
    ppos = egit_tahmin(PR_HUB, Xe[poz], ye[poz], Xh)
    log(f"{hedef_blok} ppos hazir")

    # AYNI model sinifiyla "tum satir" ikizi: pozitif-altkume etkisini
    # uretim demetine TASIYAN yapisal fark (sabit kaydirma DEGIL, iki modelin
    # satir-satir farki). p01'in dis-blok sabit kalibrasyon tuzagina dusmez.
    pall = egit_tahmin(PR_HUB, Xe, ye, Xh)
    delta = ppos - pall
    log(f"{hedef_blok} delta ort {delta.mean():.4f} std {delta.std():.4f}")

    R = {
        "n": int(len(pb)),
        "soguk_pay": float(sgm.mean()),
        "uretim_tabani": taban,
        "P0_ort": float(P0.mean()),
        "gercek_sifir": float((bf.tuketim.to_numpy() == 0).mean()),
        "egitim_sifir": float(ze.mean()),
        "a_carpim": olc((1 - P0) * pb),
        "b_huber": olc((1 - P0) * ppos),
        "b_ham_ppos": olc(ppos),
    }
    for w in (0.25, 0.5, 0.75):
        R[f"b2_harman_w{w}"] = olc((1 - P0) * (w * ppos + (1 - w) * pb))
    R["d_delta_ham"] = olc(pb + delta)
    R["d_delta_carpim"] = olc((1 - P0) * (pb + delta))
    for w in (0.5, 1.0):
        R[f"d_delta_carpim_w{w}"] = olc((1 - P0) * (pb + w * delta))
    R["delta_ort"] = float(delta.mean())
    R["delta_std"] = float(delta.std())
    log(f"{hedef_blok} d) {json.dumps(R['d_delta_carpim'])}")
    log(f"{hedef_blok} a) {json.dumps(R['a_carpim'])}")
    log(f"{hedef_blok} b) {json.dumps(R['b_huber'])}")

    # ---- (c) ilce-ici komsu seviyesi ----
    kom_h = komsu_sutunu(bf)
    # egt satir sirasi ile hizali olsun diye indeks uzerinden esle
    kom_map = {}
    for b in BLOKLAR:
        if b == hedef_blok:
            continue
        sub = e[e._blok == b]
        kom_map.update(dict(zip(sub.index.to_numpy(), komsu_sutunu(sub))))
    kom_e = np.array([kom_map[i] for i in egt.index.to_numpy()], dtype=np.float32)
    Xe2 = Xe.assign(k_komsu=kom_e)
    Xh2 = Xh.assign(k_komsu=kom_h)
    log(
        f"{hedef_blok} komsu NaN: egitim {np.isnan(kom_e).mean():.3f} "
        f"hedef {np.isnan(kom_h).mean():.3f} "
        f"hedef-soguk {np.isnan(kom_h[sgm == 1]).mean():.3f}"
    )
    P0c = egit_tahmin(PC, Xe2, ze, Xh2)
    pposc = egit_tahmin(PR_HUB, Xe2[poz], ye[poz], Xh2)
    R["c_komsu"] = olc((1 - P0c) * pposc)
    R["c_ham_ppos"] = olc(pposc)
    for w in (0.5,):
        R[f"c2_harman_w{w}"] = olc((1 - P0c) * (w * pposc + (1 - w) * pb))
    log(f"{hedef_blok} c) {json.dumps(R['c_komsu'])}")

    def kaz(d):
        return {k: taban[k] - d[k] for k in taban}

    R["kazanclar"] = {
        k: kaz(v)
        for k, v in R.items()
        if isinstance(v, dict) and "duz" in v and k != "uretim_tabani"
    }
    np.save(os.path.join(ARA, f"p05_P0_{hedef_blok}.npy"), P0)
    np.save(os.path.join(ARA, f"p05_ppos_{hedef_blok}.npy"), ppos)
    np.save(os.path.join(ARA, f"p05_pb_{hedef_blok}.npy"), pb)
    return R


if __name__ == "__main__":
    hedefler = sys.argv[1:] or ["yaz25", "guz25"]
    yol = os.path.join(BURA, "p05_uretim_iki_asama.json")
    R = json.load(open(yol, encoding="utf-8")) if os.path.exists(yol) else {}
    R["sizinti_kontrolu"] = __doc__.strip()
    for b in hedefler:
        R[b] = blok_kosusu(b)
        json.dump(R, open(yol, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(json.dumps(R[b]["kazanclar"], indent=1, ensure_ascii=False))
    log("bitti")
