"""m148 BORU HATTI DENETIMI -- dusmanca, bagimsiz.

m148_demet_plani.py yarinki tum yarismayi tasiyacak. Bu betik onu ucdan uca
sinar: kodun kendi cikardigi sayilara GUVENMEZ, her seyi yeniden hesaplar.

Asamalar (ayri ayri kosulur, her m148 kosusu ~2 dk):
  --asama 1   m148'i surec icinde kosar, tum ic nesneleri yakalar, statik
              cebir ve D1 dosyasi denetimi yapar. Sonuclari onbellege yazar.
  --asama 2   SENTETIK GERCEK ile kumulatif akisi ucdan uca simule eder:
              D1 -> P1 -> D2 -> P2 -> D3 -> P3 -> D4 -> P4 -> Z_NIHAI
              ve GERI COZULEN rho_k'yi GERCEK rho_k ile karsilastirir.
  --asama 3   uc durumlar: yanlis yazilmis skor, negatif/dev rho.
  --temizlik  m148_olcumler.json'u siler, D2..D4/Z_NIHAI'yi siler,
              m148_demet.json'u git'ten geri alir. D1'e DOKUNMAZ.

Sentetik dunya kurulusu (asama 2'nin cekirdegi):
    r_syn = (kL/nrm)*r_hat + toplam_k rho_k G_k + s*w
burada w, r_hat'a ve tum G_k'lara dik birim yon; s ise mean(r_syn^2) = M0
olacak sekilde secilir. Bu dunyada m148'in VARSAYIMLARI tam saglanir
(<r_hat,r> = kL, <G_k,r> = rho_k). Dolayisiyla geri cozulen rho_k gercekten
sapiyorsa sapmanin tek kaynagi CEBIR HATASIDIR, gurultu degil.
"""

import argparse
import hashlib
import io
import json
import os
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
HEDEF = os.path.join(M29, "m148_demet_plani.py")
OLC_YOL = os.path.join(M29, "m148_olcumler.json")
GECMIS_YOL = os.path.join(M29, "m148_demet.json")
SCRATCH = os.path.join(
    r"C:/Users/Cem/AppData/Local/Temp/claude",
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX",
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
ONBELLEK = os.path.join(SCRATCH, "m153_onbellek.npz")
PY = os.path.join(KOK, ".venv/Scripts/python.exe")
#: asama 2'de kullanilan GERCEK rho_k. Negatif ve kucuk degerler bilerek var.
RHO_GERCEK = np.array([0.050, -0.030, 0.020, 0.010])

SONUC = []


def yaz(ad, gecti, aciklama=""):
    SONUC.append((ad, gecti, aciklama))
    print(f"  [{'GECTI' if gecti else 'KALDI'}] {ad}" + (f" -- {aciklama}" if aciklama else ""))


def ort(a, b=None):
    """<a,b>/N -- m148'in her yerde kullandigi ic carpim."""
    return float((a * a).mean()) if b is None else float((a * b).mean())


def m148_kos_ici():
    """m148'i bu surecte kosar ve global sozlugunu dondurur."""
    tampon = io.StringIO()
    eski = sys.argv[:]
    sys.argv = [HEDEF]
    try:
        with redirect_stdout(tampon):
            g = runpy.run_path(HEDEF, run_name="__main__")
    finally:
        sys.argv = eski
    return g, tampon.getvalue()


def m148_kos_dis():
    p = subprocess.run(
        [PY, HEDEF],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=KOK,
        timeout=1800,
        check=False,
    )
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit(f"m148 cikis kodu {p.returncode}")
    return p.stdout


def md5(yol):
    h = hashlib.md5()
    with open(yol, "rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def csv_oku_log(dosya, idler):
    d = pd.read_csv(os.path.join(S, dosya))
    if not np.array_equal(d.id.values, idler):
        raise SystemExit(f"{dosya}: id sirasi test.csv ile ayni degil")
    return np.log1p(d.tuketim.values.astype(np.float64)), d


# ---------------------------------------------------------------------------
# ASAMA 1 -- statik cebir + D1
# ---------------------------------------------------------------------------
def asama1():
    print("=== ASAMA 1: kod okumasi ve statik cebir ===")
    g, _ = m148_kos_ici()
    A = {k: g[k] for k in ("a0", "r_hat", "kL", "M0", "TABAN_MSE", "RHO", "N")}
    G = np.asarray(g["G"], dtype=np.float64)
    U = np.asarray(g["U"], dtype=np.float64)
    KATS = np.asarray(g["KATS"], dtype=np.float64)
    RHO_CV = np.asarray(g["RHO_CV_LISTE"], dtype=np.float64)
    kul = list(g["kul"])
    BETA = np.asarray(g["BETA"], dtype=np.float64)
    KAPPA_K = np.asarray(g["KAPPA_K"], dtype=np.float64)
    RHO_K = np.asarray(g["RHO_K"], dtype=np.float64)
    ETIKET = list(g["ETIKET"])
    N = int(A["N"])
    r_hat = A["r_hat"]
    nrm = ort(r_hat)

    # 1.1/1.2 -- listelerin AYNI SIRADA olmasi
    n_ok = len(kul) == len(KATS) == len(RHO_CV) == U.shape[0]
    yaz(
        "1.1 kul/KATS/RHO_CV/U ayni uzunlukta",
        n_ok,
        f"{len(kul)}/{len(KATS)}/{len(RHO_CV)}/{U.shape[0]}",
    )
    isaret_ok = bool(np.all(np.sign(KATS) == np.sign(RHO_CV)))
    yaz(
        "1.2 KATS ve RHO_CV isaretleri birebir ortusuyor (kayma yok)",
        isaret_ok,
        "kaymis liste rastgele isaret uyusmazligi verirdi",
    )

    # 1.3 -- U ortonormal mi (bagimsiz hesap)
    UU = U @ U.T / N
    sap_u = float(np.abs(UU - np.eye(len(U))).max())
    yaz("1.3 U (eksen dik birim yonleri) ortonormal", sap_u < 1e-9, f"en buyuk sapma {sap_u:.2e}")

    # 1.4 -- G ortonormal mi (bagimsiz hesap)
    GG = G @ G.T / N
    sap_g = float(np.abs(GG - np.eye(len(G))).max())
    yaz("1.4 G (demet yonleri) ortonormal", sap_g < 1e-9, f"en buyuk sapma {sap_g:.2e}")

    # 1.5 -- G'yi SIFIRDAN yeniden kur (CIFT gecisli Gram-Schmidt) ve karsilastir
    HIP = g["HIPOTEZ"]
    ISR = np.sign(KATS)
    Gy, Ey = [], []
    for ad, ag in HIP.items():
        v = (ISR * ag) @ U
        n0 = np.sqrt(ort(v))
        if n0 < 1e-12:
            continue
        v = v / n0
        for _ in range(2):  # CIFT gecis -- m148 TEK gecis yapiyor
            for q in Gy:
                v = v - ort(v, q) * q
        n1 = np.sqrt(ort(v))
        if n1 < 0.05:
            continue
        Gy.append(v / n1)
        Ey.append(ad)
    Gy = np.array(Gy)
    ayni_etiket = Ey == ETIKET
    fark = (
        float(np.abs(np.abs(np.diag(Gy @ G.T / N)) - 1.0).max())
        if ayni_etiket and Gy.shape == G.shape
        else np.inf
    )
    yaz(
        "1.5 G bagimsiz cift-gecisli Gram-Schmidt ile birebir yeniden kuruldu",
        ayni_etiket and fark < 1e-9,
        f"etiketler {Ey}, birim-ic-carpim sapmasi {fark:.2e}",
    )

    # 1.6 -- G_k'lar r_hat'a dik mi? ("olculen yon risksizdir" iddiasinin temeli)
    dik_r = float(np.abs(G @ r_hat / N).max())
    yaz(
        "1.6 tum G_k, r_hat'a dik (span sinyali bozulmuyor)",
        dik_r < 1e-12,
        f"en buyuk |<G_k,r_hat>| = {dik_r:.2e}",
    )

    # 1.7-1.9 -- BETA / RHO / RHO_K
    beta_y = KATS @ U
    yaz("1.7 BETA = toplam KATS_i U_i", float(np.abs(beta_y - BETA).max()) < 1e-12)
    yaz(
        "1.8 RHO = ||BETA|| ve G_1 = BETA/||BETA||",
        abs(np.sqrt(ort(BETA)) - A["RHO"]) < 1e-12 and abs(ort(BETA, G[0]) - A["RHO"]) < 1e-9,
        f"RHO={A['RHO']:.6f}",
    )
    yaz(
        "1.9 RHO_K[2:] sifir (BETA tamamen G_1 boyunca; hedge yonlerinde ongoru YOK)",
        float(np.abs(RHO_K[1:]).max()) < 1e-9,
        f"maks {float(np.abs(RHO_K[1:]).max()):.1e}",
    )

    # 1.10 -- TABAN_MSE = M0 - 2kL + ||r_hat||^2
    tm = A["M0"] - 2 * A["kL"] + nrm
    yaz(
        "1.10 TABAN_MSE = M0 - 2*kL + ||r_hat||^2 (nrm DEGIL kL kullanilmis)",
        abs(tm - A["TABAN_MSE"]) < 1e-12,
        f"{A['TABAN_MSE']:.9f}; ||r_hat||^2={nrm:.6f} kL={A['kL']:.6f} (esit DEGIL, dogrusu bu)",
    )

    # 1.11 -- KAPPA_K
    kk = (0.8 / 1.95) * A["RHO"] / np.sqrt(len(G))
    yaz(
        "1.11 KAPPA_K = (0.8/1.95)*RHO/sqrt(DEMET), tum yonlerde esit",
        float(np.abs(KAPPA_K - kk).max()) < 1e-12,
        f"kappa={kk:.7f}",
    )

    # --- D1 DOSYASI, BAGIMSIZ ---
    print("\n--- D1 dosyasi bagimsiz denetim ---")
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    d1log, d1 = csv_oku_log("tuketim_D1_demet.csv", te.id.values)
    yaz("5.1 satir sayisi 714688", len(d1) == 714688, f"{len(d1)}")
    yaz("5.2 id sirasi test.csv ile birebir", bool((d1.id.values == te.id.values).all()))
    yaz(
        "5.3 id sirasi sample_submission ile birebir",
        bool((d1.id.values == ss.iloc[:, 0].values).all()),
    )
    yaz("5.4 NaN yok", int(d1.tuketim.isna().sum()) == 0)
    yaz("5.5 negatif yok", int((d1.tuketim < 0).sum()) == 0)
    yaz("5.6 hepsi sonlu", bool(np.isfinite(d1.tuketim.values).all()))
    a0 = A["a0"]
    tab = np.expm1(a0)
    yaz(
        "5.7 deger araligi makul",
        float(d1.tuketim.max()) < 3 * float(tab.max()),
        f"maks {d1.tuketim.max():,.1f} (taban maks {tab.max():,.1f}) min {d1.tuketim.min():.4f} "
        f"sifir sayisi {int((d1.tuketim.values == 0).sum())}",
    )
    dg = d1log - a0
    taban_log = a0 + r_hat
    ek = dg - (np.log1p(np.clip(np.expm1(taban_log), 0.0, None)) - a0)
    ketkin = np.sqrt(ort(ek))
    # D1'in a0'dan farkinin normu kappa_etkin DEGILDIR: r_hat da icindedir.
    # Dogru ozdeslik ||d||^2 = ||r_hat||^2 + kappa_etkin^2 (G_1 dik oldugu icin).
    yaz(
        "5.8 ||D1-a0||^2 = ||r_hat||^2 + kappa_etkin^2 (kappa_etkin TEK BASINA degil)",
        abs(ort(dg) - (nrm + ketkin**2)) < 1e-6,
        f"olculen {ort(dg):.9f} beklenen {nrm + ketkin**2:.9f}; "
        f"ham kappa {KAPPA_K[0]:.6f} -> kirpma sonrasi {ketkin:.6f}",
    )
    with open(GECMIS_YOL) as fh:
        GEC = {d["sonda"]: d for d in json.load(fh)["sondalar"]}
    yaz(
        "5.9 kappa_etkin = ||D1 - taban|| kayitla birebir",
        abs(ketkin - GEC[1]["kappa_etkin"]) < 1e-12,
        f"{ketkin:.9f}",
    )
    hiza = ort(ek, G[0]) / ketkin
    yaz(
        "5.10 ek yonu G_1 ile ayni dogrultuda (kirpma sapmasi kucuk)",
        hiza > 0.999,
        f"cos = {hiza:.9f}; ||ek||/kappa = {ketkin / KAPPA_K[0]:.6f}",
    )
    sabit = A["M0"] - 2 * A["kL"] + ort(dg)
    yaz(
        "5.11 sabit = M0 - 2kL + ||d||^2 kayitla birebir",
        abs(sabit - GEC[1]["sabit"]) < 1e-12,
        f"{sabit:.9f}",
    )

    np.savez(
        ONBELLEK,
        a0=a0,
        r_hat=r_hat,
        G=G,
        kL=A["kL"],
        M0=A["M0"],
        nrm=nrm,
        TABAN_MSE=A["TABAN_MSE"],
        KAPPA_K=KAPPA_K,
        idler=te.id.values,
    )
    print(f"\nonbellek yazildi: {ONBELLEK}")


# ---------------------------------------------------------------------------
# ASAMA 2 -- kumulatif akis, sentetik gercek
# ---------------------------------------------------------------------------
def sentetik_gercek(z):
    """m148'in varsayimlarini TAM saglayan bir gercek artik vektoru kur."""
    r_hat, G = z["r_hat"], z["G"]
    kL, nrm, M0 = float(z["kL"]), float(z["nrm"]), float(z["M0"])
    rng = np.random.default_rng(11)
    w = rng.standard_normal(len(r_hat))
    w -= w.mean()
    w -= ort(w, r_hat) / nrm * r_hat
    for k in range(len(G)):
        w -= ort(w, G[k]) * G[k]
    w /= np.sqrt(ort(w))
    s2 = M0 - kL**2 / nrm - float((RHO_GERCEK**2).sum())
    return (kL / nrm) * r_hat + RHO_GERCEK @ G + np.sqrt(s2) * w


def asama2():
    print("=== ASAMA 2: kumulatif akis, sentetik gercek ile ucdan uca ===")
    z = np.load(ONBELLEK, allow_pickle=True)
    r = sentetik_gercek(z)
    a0, G = z["a0"], z["G"]
    kL, M0 = float(z["kL"]), float(z["M0"])
    idler = z["idler"]
    print(
        f"  sentetik dunya: mean(r^2)={ort(r):.9f} (M0={M0:.9f})  "
        f"<r_hat,r>={ort(z['r_hat'], r):.9f} (kL={kL:.9f})"
    )
    for k in range(len(G)):
        print(f"    <G_{k + 1}, r> = {ort(G[k], r):+.6f}  (hedef {RHO_GERCEK[k]:+.3f})")

    olcum, ozet, son_cikti = {}, [], ""
    for k in range(1, len(G) + 1):
        dosya = f"tuketim_D{k}_demet.csv"
        if not os.path.exists(os.path.join(S, dosya)):
            yaz(f"2.{k} D{k} var", False, "dosya yok -- akis koptu")
            return
        dlog, _ = csv_oku_log(dosya, idler)
        d = dlog - a0
        P = np.sqrt(M0 - 2 * ort(d, r) + ort(d))
        olcum[str(k)] = float(P)
        with open(OLC_YOL, "w") as fh:
            json.dump(olcum, fh)
        son_cikti = m148_kos_dis()
        with open(GECMIS_YOL) as fh:
            gec = json.load(fh)
        gk = {q["sonda"]: q for q in gec["sondalar"]}[k]
        rho_geri = (gk["sabit"] - P * P) / (2 * gk["kappa_etkin"])
        rho_ger = ort(G[k - 1], r)
        ozet.append((k, P, rho_geri, rho_ger))
        yaz(
            f"2.{k} sonda {k}: P={P:.6f} -> geri cozulen rho_{k} GERCEK ile ayni",
            abs(rho_geri - rho_ger) < 1e-3,
            f"geri={rho_geri:+.6f} GERCEK={rho_ger:+.6f} hata={rho_geri - rho_ger:+.6f}",
        )
        if k < len(G):
            sonraki = f"tuketim_D{k + 1}_demet.csv"
            yaz(f"2.{k}b D{k + 1} uretildi", os.path.exists(os.path.join(S, sonraki)), sonraki)

    if "OLCULEN rho_k" in son_cikti:
        print("\n--- m148'in son kosudaki kendi raporu ---")
        print(son_cikti[son_cikti.find("OLCULEN rho_k") :][:900])

    zyol = os.path.join(S, "tuketim_Z_NIHAI.csv")
    yaz("2.Z Z_NIHAI uretildi", os.path.exists(zyol))
    if not os.path.exists(zyol):
        return
    zlog, zdf = csv_oku_log("tuketim_Z_NIHAI.csv", idler)
    d = zlog - a0
    Pz = np.sqrt(M0 - 2 * ort(d, r) + ort(d))
    tm = float(z["TABAN_MSE"])
    geri = np.array([o[2] for o in ozet])
    bek = np.sqrt(max(tm - float((geri**2).sum()), 1e-9))
    eniyi = np.sqrt(max(tm - float((RHO_GERCEK**2).sum()), 1e-9))
    yaz(
        "2.Z1 Z_NIHAI gecerli (satir/id/NaN/negatif/sonlu)",
        len(zdf) == 714688
        and bool((zdf.id.values == idler).all())
        and int(zdf.tuketim.isna().sum()) == 0
        and int((zdf.tuketim < 0).sum()) == 0
        and bool(np.isfinite(zdf.tuketim.values).all()),
    )
    yaz(
        "2.Z2 Z_NIHAI'nin GERCEK skoru betigin bastigi beklentiye esit",
        abs(Pz - bek) < 5e-5,
        f"gercek {Pz:.6f}  betigin beklentisi {bek:.6f}  kusursuz olsaydi {eniyi:.6f}  "
        f"kayip {Pz - eniyi:+.6f}",
    )


# ---------------------------------------------------------------------------
# ASAMA 4 -- ONERILEN DUZELTMENIN DOGRULANMASI (saf cebir, m148 kosulmaz)
#
# Asama 2 kanitladi: rho_k = (sabit - P^2)/(2*kappa_etkin) capraz terimi
# dusuruyor. Burada duzeltilmis boru hattini vektor uzayinda bastan sonra
# simule ederiz -- CSV uretmeden, ayni sentetik gercekle. Boylece duzeltmenin
# gercekten tam geri kazanim verdigini m148'e DOKUNMADAN gosteririz.
# ---------------------------------------------------------------------------
def asama4():
    print("=== ASAMA 4: onerilen duzeltmenin dogrulanmasi (m148'e dokunulmadi) ===")
    z = np.load(ONBELLEK, allow_pickle=True)
    r = sentetik_gercek(z)
    r_hat, G = z["r_hat"], z["G"]
    kL, M0, tm = float(z["kL"]), float(z["M0"]), float(z["TABAN_MSE"])
    KAP = z["KAPPA_K"]
    n = len(G)

    for etiket, duzelt in (("SU ANKI m148", False), ("DUZELTILMIS", True)):
        rr, taban_ek = [], np.zeros(len(r_hat))
        capraz_kayit = []
        for k in range(n):
            d = r_hat + taban_ek + KAP[k] * G[k]
            P2 = M0 - 2 * ort(d, r) + ort(d)
            sabit = M0 - 2 * kL + ort(d)
            # capraz = toplam_{j<k} r_j * rho_j_GERCEK -- m148 bunu atliyor
            capraz = sum(rr[j] * ort(G[j], r) for j in range(k))
            capraz_kayit.append(capraz)
            duz = 2 * sum(rr[j] * rr[j] for j in range(k)) if duzelt else 0.0
            rr.append((sabit - duz - P2) / (2 * KAP[k]))
            taban_ek = taban_ek + rr[k] * G[k]
        rr = np.array(rr)
        ger = np.array([ort(G[k], r) for k in range(n)])
        mse = tm - sum(2 * rr[k] * ger[k] - rr[k] ** 2 for k in range(n))
        yaz(
            f"4.{'B' if duzelt else 'A'} {etiket}: geri cozulen rho == GERCEK rho",
            float(np.abs(rr - ger).max()) < 1e-9,
            f"geri={np.round(rr, 6).tolist()} gercek={np.round(ger, 4).tolist()} "
            f"nihai GERCEK skor={np.sqrt(max(mse, 1e-12)):.6f}",
        )
        if duzelt:
            print(
                "    NOT: 'toplam r_j^2' ancak r_j == rho_j_gercek iken dogrudur.\n"
                "    Daha saglami: her sondanin kaydina taban'a giren r_j'leri yaz\n"
                "    ve capraz = toplam(r_j_kullanilan * rho_j_cozulen) ile coz.\n"
                f"    (bu kosuda gercek capraz terimler: {np.round(capraz_kayit, 6).tolist()})"
            )
    print(f"\n  kusursuz olabilecek en iyi skor = {np.sqrt(tm - float((ger**2).sum())):.6f}")


# ---------------------------------------------------------------------------
# ASAMA 3 -- uc durumlar (cebir, m148 kosulmaz)
# ---------------------------------------------------------------------------
def asama3():
    print("=== ASAMA 3: uc durumlar ===")
    with open(os.path.join(SCRATCH, "demet_ORIJINAL.json")) as fh:
        gec = json.load(fh)
    g1 = gec["sondalar"][0]
    sab, ket, tm = g1["sabit"], g1["kappa_etkin"], gec["taban_mse"]
    print(f"  D1: sabit={sab:.9f} kappa_etkin={ket:.7f} TABAN_MSE={tm:.7f}")
    print(f"\n{'girilen P':>12s} {'rho_1':>10s} {'nihai skor tahmini':>20s}  yorum")
    for P, yorum in [
        (1.00235, "rho=0 hali (risksiz taban)"),
        (0.99967, "tahmin tuttu"),
        (1.00500, "rho NEGATIF -- isaret ters"),
        (0.97000, "beklenmedik iyi -> dev rho"),
        (1.00000, "'1.00235' yerine '1' yazilmis (YAZIM HATASI)"),
        (0.10000, "ondalik kaymis (YAZIM HATASI)"),
    ]:
        rho = (sab - P * P) / (2 * ket)
        sk = np.sqrt(max(tm - rho * rho, 1e-9))
        print(f"{P:12.5f} {rho:+10.5f} {sk:20.5f}  {yorum}")
    print(
        "\n  m148'de P icin de rho_k icin de HICBIR aralik kontrolu yok:\n"
        "  yukaridaki her satir sessizce kabul edilir ve taban'a eklenir."
    )
    # kumulatif capraz terim -- asama 2'de olculen sapmanin cebirsel kaynagi
    print("\n  KUMULATIF CAPRAZ TERIM (sonda k>=2 icin):")
    print("    P^2 = sabit_k - 2*toplam_{j<k}(r_j*rho_j) - 2*kappa_etkin_k*rho_k")
    print("    m148 ilk terimi ATLIYOR -> rho_k tahmini +toplam(rho_j^2)/kappa_etkin kadar sisiyor")
    for r1 in (0.02, 0.05, 0.10):
        print(f"      rho_1={r1:.2f} ise sonda 2'de yanlilik = {r1 * r1 / ket:+.4f}")


def temizlik():
    print("=== TEMIZLIK ===")
    if os.path.exists(OLC_YOL):
        Path(OLC_YOL).unlink()
        print("  silindi m148_olcumler.json")
    for f in (
        "tuketim_D2_demet.csv",
        "tuketim_D3_demet.csv",
        "tuketim_D4_demet.csv",
        "tuketim_Z_NIHAI.csv",
    ):
        y = os.path.join(S, f)
        if os.path.exists(y):
            Path(y).unlink()
            print(f"  silindi submissions/{f}")
    subprocess.run(
        ["git", "checkout", "--", "experiments/model29/m148_demet.json"], cwd=KOK, check=True
    )
    print("  m148_demet.json git'ten geri alindi")
    print(f"  D1 md5 = {md5(os.path.join(S, 'tuketim_D1_demet.csv'))}  (DOKUNULMADI)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", type=int)
    ap.add_argument("--temizlik", action="store_true")
    a = ap.parse_args()
    if a.temizlik:
        temizlik()
        return
    {1: asama1, 2: asama2, 3: asama3, 4: asama4}[a.asama]()
    if SONUC:
        k = sum(1 for _, g, _ in SONUC if not g)
        print(f"\nOZET: {len(SONUC) - k}/{len(SONUC)} GECTI, {k} KALDI")


if __name__ == "__main__":
    main()
