"""p23-D: KOPRU YANLILIGI ANALIZI (p23_c parquetlerinden, egitim gerektirmez).

Olcumler (pencere x tohum):
  * ham yanlilik: ort(y_log - lg) kopru / kontrol, fark
  * TARIH-ESLESMELI fark (birincil): gun bazinda kopru-kontrol farki,
    kopru satir sayisiyla agirlikli -- kopru satirlari mart sonunda
    yigildigi icin sart
  * tarih x kVA-bandi eslesmeli (duyarlilik)
  * alt-kohort: "kiymik" (train ilk tarihi >= 2026-03-01, 2-6 gunluk gecmis)
    vs "uzun" kopru -- dogum gunu etkisi ile kalici seviye ayrimi
  * onyukleme (500, trafo kumeli, kopru+kontrol bagimsiz) GA95
  * capraz dogrulama: kopru trafolarini A/B ikiye bol, kaymayi A'dan
    (tarih-eslesmeli) tuket, B satirlarinda dMSE ve RMSLE olc; tersi de
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
SCRATCH = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
PK = os.path.join(KOK, "experiments/model29/p_kalici")
JSON_YOL = os.path.join(PK, "p23_parti.json")

BANTLAR = [0, 125, 200, 300, 500, 700, 1100, 1e9]


def bant(guc):
    return np.digitize(guc, BANTLAR)


def tarih_eslesmeli_fark(df, kopru_m):
    """Gun bazinda (kopru_ort - kontrol_ort), kopru satir sayisi agirlikli."""
    k = df[kopru_m].groupby("tarih")["r"].agg(["mean", "size"])
    c = df[~kopru_m].groupby("tarih")["r"].mean()
    ortak = k.index.intersection(c.index)
    k = k.loc[ortak]
    delta = k["mean"] - c.loc[ortak]
    w = k["size"]
    return float((delta * w).sum() / w.sum()), int(w.sum())


def tarih_bant_fark(df, kopru_m, min_kontrol=5):
    df = df.copy()
    df["bant"] = bant(df["guc"].to_numpy())
    k = df[kopru_m].groupby(["tarih", "bant"])["r"].agg(["mean", "size"])
    c = df[~kopru_m].groupby(["tarih", "bant"])["r"].agg(["mean", "size"])
    c = c[c["size"] >= min_kontrol]
    ortak = k.index.intersection(c.index)
    if len(ortak) == 0:
        return None, 0
    delta = k.loc[ortak, "mean"] - c.loc[ortak, "mean"]
    w = k.loc[ortak, "size"]
    return float((delta * w).sum() / w.sum()), int(w.sum())


def onyukleme_fark(df, kopru_m, n=500, tohum=99):
    rng = np.random.default_rng(tohum)
    kt = df.loc[kopru_m, "tanim"].unique()
    ct = df.loc[~kopru_m, "tanim"].unique()
    kg = {t: g for t, g in df[kopru_m].groupby("tanim")}
    cg = {t: g for t, g in df[~kopru_m].groupby("tanim")}
    out = []
    for _ in range(n):
        ks = rng.choice(kt, size=len(kt), replace=True)
        cs = rng.choice(ct, size=len(ct), replace=True)
        b = pd.concat([kg[t] for t in ks] + [cg[t] for t in cs], ignore_index=True)
        bm = np.zeros(len(b), dtype=bool)
        bm[: sum(len(kg[t]) for t in ks)] = True
        f, _ = tarih_eslesmeli_fark(b, bm)
        out.append(f)
    q = np.percentile(out, [2.5, 97.5])
    return [round(float(q[0]), 4), round(float(q[1]), 4)]


def capraz(df, kopru_m, n_boot=500, tohum=7):
    """A/B capraz dogrulama + B tarafinda trafo-kumeli onyukleme GA95."""
    rng = np.random.default_rng(tohum)
    kt = np.array(sorted(df.loc[kopru_m, "tanim"].unique()))
    rng.shuffle(kt)
    yarim = len(kt) // 2
    A, B = set(kt[:yarim]), set(kt[yarim:])
    sonuc = {}
    for ad, kay_kum, olc_kum in [("A_to_B", A, B), ("B_to_A", B, A)]:
        kay_m = kopru_m & df["tanim"].isin(kay_kum).to_numpy()
        sec = kay_m | ~kopru_m
        alt = df[sec]
        kayma, _ = tarih_eslesmeli_fark(alt, kay_m[sec])
        olc = df[kopru_m & df["tanim"].isin(olc_kum).to_numpy()]
        r = olc["r"].to_numpy()
        dmse = float(np.mean(r**2) - np.mean((r - kayma) ** 2))
        rmsle0 = float(np.sqrt(np.mean(r**2)))
        rmsle1 = float(np.sqrt(np.mean((r - kayma) ** 2)))
        # onyukleme: olcum kumesinin trafolari
        gg = {t: g["r"].to_numpy() for t, g in olc.groupby("tanim")}
        tt = np.array(sorted(gg))
        bt = np.random.default_rng(tohum + 1)
        dags = []
        for _ in range(n_boot):
            sec = bt.choice(tt, size=len(tt), replace=True)
            rr = np.concatenate([gg[t] for t in sec])
            dags.append(float(np.mean(rr**2) - np.mean((rr - kayma) ** 2)))
        q = np.percentile(dags, [2.5, 97.5])
        sonuc[ad] = {
            "kayma": round(kayma, 4),
            "olcum_trafo": len(olc_kum & set(df.loc[kopru_m, "tanim"])),
            "olcum_satir": len(olc),
            "dMSE": round(dmse, 4),
            "dMSE_GA95": [round(float(q[0]), 4), round(float(q[1]), 4)],
            "P_pozitif": round(float(np.mean(np.array(dags) > 0)), 3),
            "RMSLE_once_sonra": [round(rmsle0, 4), round(rmsle1, 4)],
        }
    return sonuc


def pencere_analiz(pad, tohumlar, kiymik_kume):
    dfs = []
    for t in tohumlar:
        yol = os.path.join(SCRATCH, f"p23_kopru_{pad}_{t}.parquet")
        if not os.path.exists(yol):
            return None
        d = pd.read_parquet(yol)
        d["tohum"] = t
        dfs.append(d)
    # tohum ortalamasi (ayni satirlar, lg ortalanir)
    d0 = dfs[0].copy()
    lg = np.mean([x["lg"].to_numpy() for x in dfs], axis=0)
    d0["lg"] = lg
    d0["r"] = d0["y_log"] - d0["lg"]
    kopru_m = (d0["grup"] == "kopru").to_numpy()

    S = {}
    S["satir"] = {"kopru": int(kopru_m.sum()), "kontrol": int((~kopru_m).sum())}
    S["trafo"] = {
        "kopru": int(d0.loc[kopru_m, "tanim"].nunique()),
        "kontrol": int(d0.loc[~kopru_m, "tanim"].nunique()),
    }
    S["ham_yanlilik"] = {
        "kopru": round(float(d0.loc[kopru_m, "r"].mean()), 4),
        "kontrol": round(float(d0.loc[~kopru_m, "r"].mean()), 4),
        "fark": round(float(d0.loc[kopru_m, "r"].mean() - d0.loc[~kopru_m, "r"].mean()), 4),
    }
    f, ns = tarih_eslesmeli_fark(d0, kopru_m)
    S["tarih_eslesmeli_fark"] = {"deger": round(f, 4), "kapsanan_kopru_satiri": ns}
    fb, nb = tarih_bant_fark(d0, kopru_m)
    S["tarih_bant_eslesmeli_fark"] = {
        "deger": round(fb, 4) if fb is not None else None,
        "kapsanan_kopru_satiri": nb,
    }
    # tohum bazinda tutarlilik
    tek = {}
    for x in dfs:
        x = x.copy()
        x["r"] = x["y_log"] - x["lg"]
        km = (x["grup"] == "kopru").to_numpy()
        ff, _ = tarih_eslesmeli_fark(x, km)
        tek[str(x["tohum"].iloc[0])] = round(ff, 4)
    S["tohum_bazinda"] = tek
    # alt-kohortlar
    kiymik_m = kopru_m & d0["tanim"].isin(kiymik_kume).to_numpy()
    uzun_m = kopru_m & ~d0["tanim"].isin(kiymik_kume).to_numpy()
    for ad, m in [("kiymik", kiymik_m), ("uzun", uzun_m)]:
        if m.sum() == 0:
            S[f"altkohort_{ad}"] = None
            continue
        alt = d0[m | ~kopru_m]
        fa, na = tarih_eslesmeli_fark(alt, m[m | ~kopru_m])
        S[f"altkohort_{ad}"] = {
            "trafo": int(d0.loc[m, "tanim"].nunique()),
            "satir": int(m.sum()),
            "tarih_eslesmeli_fark": round(fa, 4),
        }
    S["onyukleme_GA95_tarih_eslesmeli"] = onyukleme_fark(d0, kopru_m)
    S["capraz_dogrulama"] = capraz(d0, kopru_m)
    return S


def main():
    # kiymik kumesi: train ilk tarihi >= 2026-03-01 olan kopru trafolari
    tr = pd.read_csv(
        os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str}, usecols=["tanim", "tarih"]
    )
    test = pd.read_csv(
        os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str}, usecols=["tanim"]
    )
    mk = np.load(os.path.join(PK, "aday_csv/p23_parti_kopru_maske.npy"))
    kopru = set(test["tanim"][mk].unique())
    ilk = tr[tr["tanim"].isin(kopru)].groupby("tanim")["tarih"].min()
    kiymik = set(ilk[ilk >= "2026-03-01"].index)
    print(f"kiymik kopru trafo: {len(kiymik)} / {len(kopru)}")

    tohumlar = [1000, 1001]
    R = {}
    if os.path.exists(JSON_YOL):
        with open(JSON_YOL, encoding="utf-8") as fh:
            R = json.load(fh)
    A = R.setdefault("adim3_kopru_yanliligi", {})
    A["_kurulum"] = {
        "uzman": "cat maske=1.00 depth=7 ofset (p19/uretim yolu)",
        "tohumlar": tohumlar,
        "kontrol": "parti-disi, kVA katmanli 1:1, egitimden cikarildi (tohum 42)",
        "birincil_olcut": "tarih-eslesmeli kopru-kontrol yanlilik farki",
        "kiymik_tanimi": "train ilk tarihi >= 2026-03-01 (2-6 gun gecmisli kopru)",
        "kiymik_trafo": len(kiymik),
    }
    for pad in ["P1", "P2"]:
        s = pencere_analiz(pad, tohumlar, kiymik)
        if s is None:
            print(f"{pad}: parquet eksik, atlandi")
            continue
        A[pad] = s
        print(f"\n=== {pad} ===")
        print(json.dumps(s, indent=1, ensure_ascii=False))
    with open(JSON_YOL, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
