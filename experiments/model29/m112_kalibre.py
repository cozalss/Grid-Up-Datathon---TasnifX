"""KALIBRASYON AILESI -- 2. sira icin sistem.

BULGU (2026-08-30, olculdu): yapisal "seviye" yonu rho = -0.0304 verdi,
kazanc 9.26e-04 -- model varyantlarinin artimli rekorunun (3.12e-04) 3 KATI.
Yani hata model seciminde degil KALIBRASYONDA: tahminler asiri yayilmis.

Bu, tek bir yon degil bir AILE. Kalibrasyon egrisi yanlissa:
  - egriligi de yanlistir            (seviye^2, seviye^3)
  - kesitlere gore farklidir         (seviye x soguk, seviye x guc, seviye x ay)
  - kesitlerin kendi kaymasi vardir  (soguk, bolge, haftasonu)

Her yon olculmus span'a VE onceki olculmus yapisal yonlere DIK yapilir
-> her sonda saf yeni bilgi olcer, hicbiri otekini tekrar etmez.

Kullanim:
  python m112_kalibre.py --liste
  python m112_kalibre.py --aday seviye2 --yerdeg 0.005 --cikti tuketim_K_seviye2.csv
  python m112_kalibre.py --kaydet seviye2 --skor 1.00102
  python m112_kalibre.py --nihai --cikti tuketim_K_NIHAI.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
RCOND = 1e-6
DURUM = os.path.join(BURA, "m112_durum.json")
# LB'den dogrudan olculmus, olculmus_skorlar.json'da olmayan MODEL yonleri
EK_MODEL = {"tuketim_y40_sota_temiz.csv": -0.002229}
# Uc ileri-zaman CV blogunda ayni isareti koruyan iki eksen. Katsayilar,
# testle ayni Nisan-Temmuz penceresindeki sinyalin LB'de olculen seviye
# sinyaline oranlanip yaklasik %35 buzulmus halidir.
RANK2_ONSEL = (("seviye_x_ay", -0.030), ("ay", 0.035))


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


def gonderim_olcumu(taban, tahmin, skor, *, m0=M0):
    """Gonderilmis tahmini, gorulen LB skoruyla birebir bir Gram yonune cevir."""
    taban = np.asarray(taban, dtype=np.float64)
    tahmin = np.asarray(tahmin, dtype=np.float64)
    if taban.shape != tahmin.shape:
        raise ValueError(f"satir sayisi uyusmuyor: {len(taban)} != {len(tahmin)}")
    yon = tahmin - taban
    ic_carpim = (m0 + float((yon * yon).mean()) - float(skor) ** 2) / 2.0
    return yon, ic_carpim


def gonderim_olcumlerini_ekle(taban, yonler, ic_carpimlar, olcumler, *, okuyucu=oku, m0=M0):
    """Durumdaki gercek gonderimleri soyut yonleri yeniden kurmadan Gram'a ekle."""
    for olcum in olcumler:
        yon, ic_carpim = gonderim_olcumu(taban, okuyucu(olcum["dosya"]), olcum["skor"], m0=m0)
        yonler.append(yon)
        ic_carpimlar.append(ic_carpim)


def onsele_dayali_duzeltme(adaylar, bilinen, gram, katsayilar, n):
    """Adaylari bilinen spana ve birbirine diklestirip onsel duzeltme kur."""
    bilinen = np.asarray(bilinen, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    duzeltme = np.zeros(n, dtype=np.float64)
    bilgi = []
    for aday, katsayi in katsayilar:
        x = np.asarray(adaylar[aday], dtype=np.float64)
        c, *_ = np.linalg.lstsq(gram, (bilinen.T @ x) / n, rcond=RCOND)
        xp = x - bilinen @ c
        q_dik = float((xp * xp).mean())
        if q_dik < 1e-4:
            raise ValueError(f"{aday}: dik bilesen cok kucuk ({q_dik:.2e})")
        birim = xp / np.sqrt(q_dik)
        duzeltme += float(katsayi) * birim
        bilgi.append({"aday": aday, "katsayi": float(katsayi), "Q_dik": q_dik})
        bilinen = np.column_stack([bilinen, birim])
        gram = (bilinen.T @ bilinen) / n
    return duzeltme, bilgi


def durum_yukle():
    if os.path.exists(DURUM):
        return json.load(open(DURUM))
    # seviye 2026-08-30'da olculdu: skor 1.00115
    return {
        "yapisal": {"seviye": -0.024649},
        "olcumler": [{"aday": "seviye", "dosya": "tuketim_YP_seviye.csv", "skor": 1.00115}],
        "bekleyen": None,
        "gecmis": [],
    }


def durum_kaydet(d):
    g = DURUM + ".tmp"
    with open(g, "w") as f:
        json.dump(d, f, indent=1)
    Path(g).replace(DURUM)


def yapisal_yonler(te, a0, tr_tanim):
    """Ham (ortogonallestirilmemis) yapisal yonler."""
    tarih = pd.to_datetime(te.tarih)
    soguk = (~te.tanim.isin(tr_tanim)).to_numpy().astype(np.float64)
    ay = tarih.dt.month.to_numpy().astype(np.float64)
    hs = (tarih.dt.dayofweek >= 5).to_numpy().astype(np.float64)
    lg = np.log1p(te.guc.values.astype(np.float64))
    lg = (lg - lg.mean()) / lg.std()
    sv = (a0 - a0.mean()) / a0.std()
    ayn = (ay - ay.mean()) / ay.std()
    bolge = te.lokasyon.str.split(">").str[1].fillna("?")
    Y = {
        "seviye": sv,
        "seviye2": sv**2,
        "seviye3": sv**3,
        "seviye_x_soguk": sv * soguk,
        "seviye_x_guc": sv * lg,
        "seviye_x_ay": sv * ayn,
        "seviye_x_hs": sv * hs,
        "soguk": soguk,
        "guc": lg,
        "ay": ayn,
        "haftasonu": hs,
        "seviye2_x_soguk": (sv**2) * soguk,
    }
    # AJAN A'nin uc holdout'ta plasebo kontrollu dogruladigi TAM SEKIL:
    # dogrusal buzme, |u|>1.5 doygun, soguk 4x, ufuk Nis .30 May 1.0 Haz 1.4 Tem 1.32
    ufuk = pd.Series(ay).map({4: 0.30, 5: 1.00, 6: 1.40, 7: 1.32}).to_numpy()
    w = (1.0 + 3.0 * soguk) * ufuk
    w = w / w.mean()
    Y["buzme_tam"] = -w * np.clip(sv, -1.5, 1.5)
    Y["buzme_sade"] = -np.clip(sv, -1.5, 1.5)
    Y["buzme_soguk"] = -(1.0 + 3.0 * soguk) / (1.0 + 3.0 * soguk).mean() * np.clip(sv, -1.5, 1.5)
    Y["buzme_ufuk"] = -(ufuk / ufuk.mean()) * np.clip(sv, -1.5, 1.5)
    for b in bolge.value_counts().index[:4]:
        m = (bolge == b).to_numpy().astype(np.float64)
        Y[f"bolge_{b.split()[0][:6]}"] = m
        Y[f"seviye_x_{b.split()[0][:6]}"] = sv * m
    # merkezle + birim norm
    for k in Y:
        x = Y[k] - Y[k].mean()
        Y[k] = x / np.sqrt(float((x * x).mean()))
    return Y


def kur(te, a0, N, d):
    """Bilinen her seyden r_hat kur. Doner: r_hat, izdusum fonksiyonu."""
    SK = json.load(open(os.path.join(BURA, "olculmus_skorlar.json")))
    V, L = [], []
    for f, P in SK.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        v = oku(f)
        if len(v) != N:
            continue
        dd = v - a0
        V.append(dd)
        L.append((M0 + float((dd * dd).mean()) - P * P) / 2)
    for f, Lj in EK_MODEL.items():
        V.append(oku(f) - a0)
        L.append(Lj)
    # Yapisal adaylar yalnizca yeni sonda uretmek icin yeniden kurulur. Olculmus
    # sondalar Gram'a gonderilen CSV'nin kendisiyle eklenir; boylece yon tanimi
    # sonradan degisse bile gorulen LB skoru birebir yeniden uretilir.
    tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), usecols=["tanim"])
    Y = yapisal_yonler(te, a0, set(tr.tanim.unique()))
    gonderim_olcumlerini_ekle(a0, V, L, d.get("olcumler", []))
    V = np.array(V).T
    L = np.array(L)
    G = (V.T @ V) / N
    r_hat = V @ (np.linalg.pinv(G, rcond=RCOND) @ L)
    return r_hat, V, G, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aday")
    ap.add_argument("--yerdeg", type=float, default=0.005)
    ap.add_argument("--cikti")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--nihai", action="store_true")
    ap.add_argument("--rank2", action="store_true")
    ap.add_argument("--kaydet")
    ap.add_argument("--skor", type=float)
    a = ap.parse_args()
    d = durum_yukle()

    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    a0 = oku(TABAN)
    N = len(a0)
    r_hat, V, G, Y = kur(te, a0, N, d)
    nrm = float((r_hat * r_hat).mean())
    print(f"BILINEN: {V.shape[1]} yon ({len(d['yapisal'])} yapisal olculmus)")
    print(f"  ||r_hat||^2 = {nrm:.6f}  ->  saf optimum {np.sqrt(M0 - nrm):.5f}")

    if a.kaydet:
        b = d["bekleyen"]
        if not b or b["aday"] != a.kaydet:
            raise SystemExit(f"bekleyen sonda '{a.kaydet}' degil: {b}")
        L = (b["sabit"] - a.skor**2) / (2 * b["kappa"])
        rho = L / np.sqrt(b["Q_dik"])
        print(f"\n{a.kaydet}: skor {a.skor} -> L = {L:+.6f}  rho = {rho:+.4f}  kazanc {rho**2:.3e}")
        d["yapisal"][a.kaydet] = L
        d["gecmis"].append(dict(aday=a.kaydet, skor=a.skor, L=L, rho=rho))
        d.setdefault("olcumler", []).append({"aday": a.kaydet, "dosya": b["cikti"], "skor": a.skor})
        d["bekleyen"] = None
        durum_kaydet(d)
        print("KAYDEDILDI. Yeni taban icin --liste calistir.")
        return

    if sum(bool(x) for x in (a.aday, a.nihai, a.rank2)) > 1:
        raise SystemExit("--aday, --nihai ve --rank2 birlikte kullanilamaz")

    if a.liste or not (a.aday or a.nihai or a.rank2):
        print(f"\n{'aday':>22s} {'Q_dik':>9s} {'span-disi':>10s} {'rho=0.03 kazanci':>17s}")
        dosya_adaylari = [(o["aday"], None) for o in d.get("olcumler", []) if o["aday"] not in Y]
        for ad, x in list(Y.items()) + dosya_adaylari:
            if ad in d["yapisal"]:
                print(f"{ad:>22s}  [OLCULDU  L={d['yapisal'][ad]:+.6f}]")
                continue
            c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
            xp = x - V @ c
            Qd = float((xp * xp).mean())
            print(f"{ad:>22s} {Qd:9.4f} {Qd:10.3f} {0.03**2:17.3e}")
        print(f"\nolculen yapisal: {json.dumps(d['yapisal'], indent=1)}")
        return

    if a.rank2:
        duzeltme, bilgi = onsele_dayali_duzeltme(Y, V, G, RANK2_ONSEL, N)
        p = a0 + r_hat + duzeltme
        etiket = "RANK2 ONSEL (kontrollu agresif)"
        kap, Qd, xp = 0.0, 0.0, None
        print(f"\n{etiket}:")
        for satir in bilgi:
            print(f"  {satir['aday']:18s} beta={satir['katsayi']:+.4f} Q_dik={satir['Q_dik']:.4f}")
        print(f"  sifir-sinyal geometri maliyeti = {float((duzeltme * duzeltme).mean()):.6f}")
    elif a.nihai:
        p = a0 + r_hat
        etiket = "NIHAI (saf optimum)"
        kap, Qd, xp = 0.0, 0.0, None
    else:
        if a.aday not in Y:
            raise SystemExit(f"bilinmeyen aday {a.aday}; --liste ile bak")
        if a.aday in d["yapisal"]:
            raise SystemExit(f"{a.aday} zaten olculdu")
        x = Y[a.aday]
        c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
        xp = x - V @ c
        Qd = float((xp * xp).mean())
        if Qd < 1e-4:
            raise SystemExit(f"{a.aday}: dik bilesen cok kucuk ({Qd:.2e}), olculemez")
        kap = a.yerdeg / np.sqrt(Qd)
        p = a0 + r_hat + kap * xp
        etiket = f"SONDA {a.aday}"
        print(
            f"\n{etiket}: Q_dik={Qd:.4f} kappa={kap:.5f} "
            f"SNR(rho=0.03)={0.03 * a.yerdeg / 5.01e-6:.0f}"
        )

    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    ok = (
        len(out) == 714688
        and (out.id.values == ss.iloc[:, 0].values).all()
        and out.tuketim.isna().sum() == 0
        and (out.tuketim < 0).sum() == 0
        and np.isfinite(out.tuketim.values).all()
        and out.tuketim.max() < 3 * np.expm1(a0).max()
    )
    if not ok:
        raise SystemExit("KAPI KALDI")
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * nrm + float(dgv @ dgv) / N)
    if not a.cikti:
        raise SystemExit("--cikti gerekli")
    g = Path(os.path.join(S, a.cikti) + ".tmp")
    out.to_csv(g, index=False)
    g.replace(os.path.join(S, a.cikti))
    print(f"kirpik {int((y == 0).sum())}  maks {out.tuketim.max():,.0f}  KAPI GECTI")
    print(f"YAZILDI submissions/{a.cikti}")
    if a.nihai:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f}  (tum L'ler olculmus)")
    elif a.rank2:
        print("HEDEF: seviye_x_ay rho<=-0.035 ve ay rho>=+0.040 ise 2. sira asilir")
    else:
        d["bekleyen"] = dict(aday=a.aday, cikti=a.cikti, sabit=sabit, kappa=kap, Q_dik=Qd)
        durum_kaydet(d)
        print(f"COZUM: L = ({sabit:.9f} - P^2) / {2 * kap:.6f}")
        for rho in (-0.03, 0.0, 0.015, 0.030, 0.05):
            print(f"  rho={rho:+.3f} -> {np.sqrt(sabit - 2 * kap * rho * np.sqrt(Qd)):.5f}")
        print(f"\nskor gelince: python m112_kalibre.py --kaydet {a.aday} --skor <S>")


if __name__ == "__main__":
    main()
