"""m148 KABUL TESTI -- 83d9faf'teki on bir duzeltmeden SONRA.

m153_boruhatti_denetimi.py BAYAT: K9 duzeltmesi demet yonlerini G -> GD diye
yeniden adlandirdi; m153 hala g["G"]'yi okuyor ve artik oraya GRAM MATRISI
(V.T@V/N, 28x28) dusuyor. m153 --asama 1/2 bu yuzden COKUYOR; --asama 2'nin
geri cozme formulu de capraz terimsiz (duzeltme oncesi) surumdur. Bu betik
m153'un isini DUZELTILMIS boru hatti icin bastan yapar.

Asamalar:
  --asama 1  m148'i surec icinde kosar, ic nesneleri yakalar, statik cebir +
             D1 dosyasi denetimi. Onbellek yazar.
  --asama 2  SENTETIK GERCEK ile ucdan uca: D1 -> P1 -> D2 -> ... -> Z_NIHAI.
             Geri cozulen rho_k GERCEK rho_k ile karsilastirilir.
  --asama 3  K1-K11 duzeltmelerinin tek tek dogrulanmasi (kaynak + davranis).
  --asama 4  YENI KUSUR AVI: sirasiz sonda, sonradan duzeltilen skor,
             D1'de eksik onceki_r, KAPPA_K[0] ile D1 yeniden uretimi.
  --temizlik olcumler.json + D2/D3/D4/Z_NIHAI silinir, demet.json geri alinir.
             D1'e DOKUNULMAZ.
"""

import argparse
import ast
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
ONBELLEK = os.path.join(SCRATCH, "m158_onbellek.npz")
D1_MD5 = "6995cfdf8adedabebd8af5721a7b915e"
PY = os.path.join(KOK, ".venv/Scripts/python.exe")
RHO_GERCEK = np.array([0.050, -0.030, 0.020, 0.010])

SONUC = []


def yaz(ad, gecti, aciklama=""):
    SONUC.append((ad, bool(gecti), aciklama))
    print(f"  [{'GECTI' if gecti else 'KALDI'}] {ad}" + (f" -- {aciklama}" if aciklama else ""))


def ort(a, b=None):
    return float((a * a).mean()) if b is None else float((a * b).mean())


def md5(yol):
    h = hashlib.md5()
    with open(yol, "rb") as fh:
        for blok in iter(lambda: fh.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def m148_ici():
    tampon = io.StringIO()
    eski = sys.argv[:]
    sys.argv = [HEDEF]
    try:
        with redirect_stdout(tampon):
            g = runpy.run_path(HEDEF, run_name="__main__")
    finally:
        sys.argv = eski
    return g, tampon.getvalue()


def m148_dis(bekle_hata=False):
    p = subprocess.run(
        [PY, HEDEF],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=KOK,
        timeout=1800,
        check=False,
    )
    if p.returncode != 0 and not bekle_hata:
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"m148 cikis kodu {p.returncode}")
    return p.returncode, p.stdout, p.stderr


def olcum_yaz(d):
    with open(OLC_YOL, "w") as fh:
        json.dump({str(k): float(v) for k, v in d.items()}, fh)


def gecmis_oku():
    with open(GECMIS_YOL) as fh:
        j = json.load(fh)
    return j, {d["sonda"]: d for d in j["sondalar"]}


def csv_log(dosya, idler):
    d = pd.read_csv(os.path.join(S, dosya))
    if not np.array_equal(d.id.values, idler):
        raise SystemExit(f"{dosya}: id sirasi test.csv ile ayni degil")
    return np.log1p(d.tuketim.values.astype(np.float64)), d


def coz(gk, P, RHO):
    """m148'in DUZELTILMIS cozum formulunun bagimsiz kopyasi."""
    onc = gk.get("onceki_r", {})
    capraz = sum(float(onc.get(str(j), 0.0)) * RHO[j] for j in RHO if j < gk["sonda"])
    return (gk["sabit"] - 2.0 * capraz - P * P) / (2 * gk["kappa_etkin"]), capraz


# ---------------------------------------------------------------- ASAMA 1
def asama1():
    print("=== ASAMA 1: statik cebir + D1 ===")
    g, _cikti = m148_ici()
    GD = np.asarray(g["GD"], dtype=np.float64)
    U = np.asarray(g["U"], dtype=np.float64)
    KATS = np.asarray(g["KATS"], dtype=np.float64)
    RHO_CV = np.asarray(g["RHO_CV_LISTE"], dtype=np.float64)
    BETA = np.asarray(g["BETA"], dtype=np.float64)
    KAPPA_K = np.asarray(g["KAPPA_K"], dtype=np.float64)
    RHO_K = np.asarray(g["RHO_K"], dtype=np.float64)
    ETIKET = list(g["ETIKET"])
    kul = list(g["kul"])
    N, r_hat, a0 = int(g["N"]), g["r_hat"], g["a0"]
    kL, M0, TABAN_MSE, RHO = g["kL"], g["M0"], g["TABAN_MSE"], g["RHO"]
    nrm = ort(r_hat)

    yaz(
        "1.0 m148 hatasiz kostu; GD/KAPPA_K/ETIKET uretildi",
        GD.ndim == 2 and len(GD) == len(KAPPA_K) == len(ETIKET),
        f"DEMET={len(GD)} etiketler={ETIKET}",
    )
    yaz(
        "1.1 kul/KATS/RHO_CV/U ayni uzunlukta",
        len(kul) == len(KATS) == len(RHO_CV) == U.shape[0],
        f"{len(kul)}",
    )
    yaz("1.2 KATS ve RHO_CV isaretleri birebir", bool(np.all(np.sign(KATS) == np.sign(RHO_CV))))
    sap_u = float(np.abs(U @ U.T / N - np.eye(len(U))).max())
    yaz("1.3 U ortonormal", sap_u < 1e-9, f"sapma {sap_u:.2e}")
    sap_g = float(np.abs(GD @ GD.T / N - np.eye(len(GD))).max())
    yaz("1.4 GD (demet yonleri) ortonormal", sap_g < 1e-9, f"sapma {sap_g:.2e}")

    HIP, ISR = g["HIPOTEZ"], np.sign(KATS)
    Gy, Ey = [], []
    for ad, ag in HIP.items():
        v = (ISR * ag) @ U
        n0 = np.sqrt(ort(v))
        if n0 < 1e-12:
            continue
        v = v / n0
        for _ in range(2):
            for q in Gy:
                v = v - ort(v, q) * q
        n1 = np.sqrt(ort(v))
        if n1 < 0.05:
            continue
        Gy.append(v / n1)
        Ey.append(ad)
    Gy = np.array(Gy)
    ayni = Ey == ETIKET and Gy.shape == GD.shape
    fark = float(np.abs(np.abs(np.diag(Gy @ GD.T / N)) - 1.0).max()) if ayni else np.inf
    yaz(
        "1.5 GD bagimsiz cift-gecisli Gram-Schmidt ile yeniden kuruldu",
        ayni and fark < 1e-9,
        f"etiket {Ey} sapma {fark:.2e}",
    )
    dik_r = float(np.abs(GD @ r_hat / N).max())
    yaz("1.6 tum GD_k r_hat'a dik", dik_r < 1e-12, f"maks |<GD,r_hat>| {dik_r:.2e}")
    yaz("1.7 BETA = toplam KATS_i U_i", float(np.abs(KATS @ U - BETA).max()) < 1e-12)
    yaz(
        "1.8 RHO = ||BETA|| ve GD_1 = BETA/||BETA||",
        abs(np.sqrt(ort(BETA)) - RHO) < 1e-12 and abs(ort(BETA, GD[0]) - RHO) < 1e-9,
        f"RHO={RHO:.6f}",
    )
    yaz(
        "1.9 RHO_K[1:] ~ 0 (BETA tamamen GD_1 boyunca)",
        float(np.abs(RHO_K[1:]).max()) < 1e-9,
        f"maks {float(np.abs(RHO_K[1:]).max()):.1e}",
    )
    yaz(
        "1.10 TABAN_MSE = M0 - 2kL + ||r_hat||^2",
        abs(M0 - 2 * kL + nrm - TABAN_MSE) < 1e-12,
        f"{TABAN_MSE:.9f}",
    )
    yaz(
        "1.11 KAPPA_K[0] D1'e cakili, digerleri 0.0125",
        abs(KAPPA_K[0] - 0.05174190699701174) < 1e-15
        and float(np.abs(KAPPA_K[1:] - 0.0125).max()) < 1e-15,
        f"{np.round(KAPPA_K, 6).tolist()}",
    )

    print("\n--- D1 dosyasi ---")
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    d1log, d1 = csv_log("tuketim_D1_demet.csv", te.id.values)
    yaz(
        "5.0 D1 md5 degismedi",
        md5(os.path.join(S, "tuketim_D1_demet.csv")) == D1_MD5,
        md5(os.path.join(S, "tuketim_D1_demet.csv")),
    )
    yaz("5.1 satir 714688", len(d1) == 714688)
    yaz("5.2 id sirasi test.csv ile birebir", bool((d1.id.values == te.id.values).all()))
    yaz(
        "5.3 id sirasi sample_submission ile birebir",
        bool((d1.id.values == ss.iloc[:, 0].values).all()),
    )
    yaz(
        "5.4-5.6 NaN/negatif/sonsuz yok",
        int(d1.tuketim.isna().sum()) == 0
        and int((d1.tuketim < 0).sum()) == 0
        and bool(np.isfinite(d1.tuketim.values).all()),
    )
    dg = d1log - a0
    ek = dg - (np.log1p(np.clip(np.expm1(a0 + r_hat), 0.0, None)) - a0)
    ketkin = np.sqrt(ort(ek))
    _, GEC = gecmis_oku()
    yaz(
        "5.8 ||D1-a0||^2 = ||r_hat||^2 + kappa_etkin^2",
        abs(ort(dg) - (nrm + ketkin**2)) < 1e-6,
        f"{ort(dg):.9f} vs {nrm + ketkin**2:.9f}",
    )
    yaz(
        "5.9 kappa_etkin kayitla birebir",
        abs(ketkin - GEC[1]["kappa_etkin"]) < 1e-12,
        f"{ketkin:.9f}",
    )
    sabit = M0 - 2 * kL + ort(dg)
    yaz(
        "5.11 sabit kayitla birebir",
        abs(sabit - GEC[1]["sabit"]) < 1e-12,
        f"olculen {sabit:.12f} kayit {GEC[1]['sabit']:.12f} fark {sabit - GEC[1]['sabit']:.2e}",
    )
    j, _ = gecmis_oku()
    yaz(
        "5.12 kayitli taban_mse bu kosunun TABAN_MSE'siyle ayni (r_hat kaymadi)",
        abs(j["taban_mse"] - TABAN_MSE) < 1e-12,
        f"kayit {j['taban_mse']:.12f} kosu {TABAN_MSE:.12f}",
    )

    np.savez(
        ONBELLEK,
        a0=a0,
        r_hat=r_hat,
        GD=GD,
        kL=kL,
        M0=M0,
        nrm=nrm,
        TABAN_MSE=TABAN_MSE,
        KAPPA_K=KAPPA_K,
        idler=te.id.values,
    )
    print(f"\nonbellek: {ONBELLEK}")


# ---------------------------------------------------------------- SENTETIK
def sentetik(z):
    r_hat, GD = z["r_hat"], z["GD"]
    kL, nrm, M0 = float(z["kL"]), float(z["nrm"]), float(z["M0"])
    rng = np.random.default_rng(11)
    w = rng.standard_normal(len(r_hat))
    w -= w.mean()
    w -= ort(w, r_hat) / nrm * r_hat
    for k in range(len(GD)):
        w -= ort(w, GD[k]) * GD[k]
    w /= np.sqrt(ort(w))
    s2 = M0 - kL**2 / nrm - float((RHO_GERCEK**2).sum())
    return (kL / nrm) * r_hat + RHO_GERCEK @ GD + np.sqrt(s2) * w


def gercek_skor(dosya, z, r):
    dlog, df = csv_log(dosya, z["idler"])
    d = dlog - z["a0"]
    return float(np.sqrt(float(z["M0"]) - 2 * ort(d, r) + ort(d))), df


# ---------------------------------------------------------------- ASAMA 2
def asama2():
    print("=== ASAMA 2: ucdan uca sentetik gercek ===")
    z = np.load(ONBELLEK, allow_pickle=True)
    r = sentetik(z)
    GD = z["GD"]
    print(
        f"  sentetik dunya: mean(r^2)={ort(r):.9f} (M0={float(z['M0']):.9f})  "
        f"<r_hat,r>={ort(z['r_hat'], r):.9f} (kL={float(z['kL']):.9f})"
    )
    for k in range(len(GD)):
        print(f"    <GD_{k + 1},r> = {ort(GD[k], r):+.6f}  (hedef {RHO_GERCEK[k]:+.3f})")

    olcum, RHO, son = {}, {}, ""
    for k in range(1, len(GD) + 1):
        dosya = f"tuketim_D{k}_demet.csv"
        if not os.path.exists(os.path.join(S, dosya)):
            yaz(f"2.{k} D{k} var", False, "dosya yok -- akis koptu")
            return
        P, _ = gercek_skor(dosya, z, r)
        olcum[k] = P
        olcum_yaz(olcum)
        _, son, _ = m148_dis()
        _, gec = gecmis_oku()
        rgeri, capraz = coz(gec[k], P, RHO)
        RHO[k] = rgeri
        rger = ort(GD[k - 1], r)
        # TOLERANS. Cebir TAM olsa bile expm1/clip/log1p gidis-donusu d'yi
        # r_hat + kappa*GD_k'dan bir parca saptirir (negatif log-tahminler
        # sifira kirpilir). Bu, kappa_etkin ile normalize edilince rho'da
        # ~1e-4 sistematik yanlilik birakir ve hedge sondalarinda kappa 4 kat
        # kucuk oldugu icin 4 kat buyur. Betigin KENDI kabul ettigi olcum
        # hatasi 1.72e-4/(2*0.0125) = 6.9e-3; tolerans onun ALTINDA tutuldu.
        yaz(
            f"2.{k} sonda {k}: geri cozulen rho_{k} == GERCEK (tol 2e-3)",
            abs(rgeri - rger) < 2e-3,
            f"P={P:.6f} capraz={capraz:+.9f} geri={rgeri:+.6f} gercek={rger:+.6f} hata={rgeri - rger:+.2e}",
        )
        satir = [x for x in son.splitlines() if f"demet {k} (" in x]
        if satir:
            kendi = float(satir[0].split("rho_k =")[1].split()[0])
            yaz(
                f"2.{k}m m148'in KENDI bastigi rho_{k} de GERCEK ile ayni",
                abs(kendi - rger) < 2e-3,
                f"m148={kendi:+.6f} gercek={rger:+.6f}",
            )
        else:
            yaz(f"2.{k}m m148 rho_{k}'yi bastirdi", False, "cikti satiri bulunamadi")
        if k < len(GD):
            yaz(
                f"2.{k}b D{k + 1} uretildi",
                os.path.exists(os.path.join(S, f"tuketim_D{k + 1}_demet.csv")),
            )
            _, gec2 = gecmis_oku()
            bek = {str(j): RHO[j] for j in RHO}
            got = gec2[k + 1].get("onceki_r", {})
            yaz(
                f"2.{k}c D{k + 1} kaydinda onceki_r dogru",
                set(got) == set(bek) and all(abs(got[q] - bek[q]) < 1e-12 for q in bek),
                f"{got}",
            )

    if "OLCULEN rho_k" in son:
        print("\n--- m148'in son raporu ---")
        print(son[son.find("OLCULEN rho_k") :][:900])

    zyol = os.path.join(S, "tuketim_Z_NIHAI.csv")
    yaz("2.Z Z_NIHAI uretildi", os.path.exists(zyol))
    if not os.path.exists(zyol):
        return
    Pz, zdf = gercek_skor("tuketim_Z_NIHAI.csv", z, r)
    yaz(
        "2.Z1 Z_NIHAI gecerli (satir/id/NaN/negatif/sonlu)",
        len(zdf) == 714688
        and bool((zdf.id.values == z["idler"]).all())
        and int(zdf.tuketim.isna().sum()) == 0
        and int((zdf.tuketim < 0).sum()) == 0
        and bool(np.isfinite(zdf.tuketim.values).all()),
    )
    tm = float(z["TABAN_MSE"])
    geri = np.array([RHO[k] for k in sorted(RHO)])
    bek = np.sqrt(max(tm - float((geri**2).sum()), 1e-9))
    eniyi = np.sqrt(max(tm - float((RHO_GERCEK**2).sum()), 1e-9))
    basilan = None
    for x in son.splitlines():
        if "beklenen skor" in x:
            basilan = float(x.split("beklenen skor")[1].split()[0])
    yaz(
        "2.Z2 Z_NIHAI'nin GERCEK skoru betigin bastigi beklentiye esit",
        basilan is not None and abs(Pz - basilan) < 5e-5,
        f"gercek {Pz:.6f} betigin bastigi {basilan} cebirsel {bek:.6f} kusursuz {eniyi:.6f} kayip {Pz - eniyi:+.2e}",
    )


# ---------------------------------------------------------------- ASAMA 3
def asama3():
    print("=== ASAMA 3: K1-K11 tek tek ===")
    src = Path(HEDEF).read_text(encoding="utf-8")
    agac = ast.parse(src)

    yaz(
        "K1 capraz terim cozum formulunde",
        "2.0 * capraz" in src and 'g.get("onceki_r"' in src and "onceki_r=dict(ONCEKI_R)" in src,
        "RHO_OLC[k] = (sabit - 2*capraz - P*P)/(2*kappa_etkin), capraz onceki_r'den",
    )

    kapi_cagri = [
        n
        for n in ast.walk(agac)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "kapilar"
    ]
    yaz(
        "K2 kapilar() hem sonda hem Z_NIHAI icin cagriliyor",
        len(kapi_cagri) == 2 and src.count("KAPI KALDI") == 2 and "Z_NIHAI KAPI KALDI" in src,
        f"{len(kapi_cagri)} cagri; kapilar: satir/id/NaN/negatif/sonlu/maks",
    )

    yaz(
        "K3 m148_demet.json atomik yaziliyor",
        "def yaz_atomik" in src
        and 'yol + ".tmp"' in src
        and "yaz_atomik(" in src
        and 'open(GECMIS_YOL, "w")' not in src,
        ".tmp + Path.replace",
    )

    yaz("K4 P ve rho araliklari var", "0.90 < P < 1.20" in src and "abs(RHO_OLC[k]) > 0.20" in src)

    yaz(
        "K5 kayit+dosya varsa yeniden uretilmiyor",
        "if kayit and os.path.exists(yol):" in src and "ZATEN VAR" in src,
    )

    yaz(
        "K6 olcum hatasi kalibre sabiti de iceriyor",
        "SABIT_HATA = 1.72e-4" in src and "np.sqrt(YUV**2 + SABIT_HATA**2)" in src,
    )

    yaz("K7 'rho=kappa ise' etiketi dogru", "=kappa ise" in src and "tahmin tutarsa" not in src)

    yaz(
        "K8 sifira bolme korumasi",
        'if abs(tah) > 1e-9 else "  -  "' in src
        and "max(TABAN_MSE - _t2, 1e-9)" in src
        and "max(sabit, 1e-9)" in src,
    )

    yaz(
        "K9 demet yonleri GD; G yalniz Gram matrisi",
        "GD @ GD.T / N" in src and src.count("G = (V.T @ V) / N") == 1 and "GD[k]" in src,
        "m148 icinde tutarli -- ama m153 hala eski adi okuyor (asagiya bak)",
    )

    cikislar = src.count("raise SystemExit")
    yaz(
        "K10 sessiz basarisizliklar SystemExit oldu",
        cikislar >= 6 and "UYARI: sonda" not in src,
        f"{cikislar} SystemExit",
    )

    olu = []
    if "gruplar" in src:
        olu.append("gruplar")
    if 'os.environ.get("DEMET"' in src:
        olu.append('satir 276 DEMET=os.environ.get("DEMET") -- satir 361 len(GD) ile eziliyor')
    yaz("K11 olu kod temizlendi", not olu, ("KALAN: " + "; ".join(olu)) if olu else "temiz")

    # --- m153'un bayatligi ---
    m153 = Path(os.path.join(M29, "m153_boruhatti_denetimi.py")).read_text(encoding="utf-8")
    yaz(
        "K9b m153 denetim betigi K9'dan SONRA guncellenmemis",
        False,
        'm153 g["G"] okuyor (artik Gram matrisi) ve capraz terimsiz formul kullaniyor'
        if 'g["G"]' in m153 and "capraz" not in m153.split("def asama2")[1].split("def asama4")[0]
        else "beklenmeyen",
    )

    print("\n--- K4 davranissal sinama (m148 iki kez kosulur) ---")
    yedek = Path(OLC_YOL).read_text() if os.path.exists(OLC_YOL) else None
    try:
        olcum_yaz({1: 1.5})
        rc, so, se = m148_dis(bekle_hata=True)
        yaz(
            "K4a P=1.5 SystemExit ile durduruyor",
            rc != 0 and "makul araligin" in (so + se),
            str([x for x in (so + se).splitlines() if "DUR:" in x][:1]),
        )
        olcum_yaz({1: 0.95})
        rc, so, se = m148_dis(bekle_hata=True)
        yaz(
            "K4b dev rho (|rho|>0.20) SystemExit ile durduruyor",
            rc != 0 and "|rho| > 0.20" in (so + se),
            str([x for x in (so + se).splitlines() if "DUR:" in x][:1]),
        )
    finally:
        if yedek is not None:
            Path(OLC_YOL).write_text(yedek)
        elif os.path.exists(OLC_YOL):
            Path(OLC_YOL).unlink()


# ---------------------------------------------------------------- ASAMA 4
def asama4():
    """YENI KUSUR AVI. asama2'den SONRA, temizlikten ONCE kosulmali."""
    print("=== ASAMA 4: yeni kusur avi ===")
    z = np.load(ONBELLEK, allow_pickle=True)
    r = sentetik(z)
    GD = z["GD"]
    _, gec = gecmis_oku()
    n = len(GD)
    var = [k for k in range(1, n + 1) if os.path.exists(os.path.join(S, f"tuketim_D{k}_demet.csv"))]
    print(f"  elde D{var} ve kayitlar {sorted(gec)}")

    # --- 4.1 D1 kaydinda onceki_r YOK -- zararsiz mi? ---
    yaz(
        "4.1 D1 kaydinda onceki_r alani yok (eski surumden kalma)",
        "onceki_r" not in gec[1],
        f"alanlar {sorted(gec[1])}",
    )
    P1, _ = gercek_skor("tuketim_D1_demet.csv", z, r)
    r1, cap1 = coz(gec[1], P1, {})
    yaz(
        "4.1b k=1 icin capraz toplami BOS -> onceki_r eksikligi zararsiz",
        cap1 == 0.0 and abs(r1 - ort(GD[0], r)) < 2e-3,
        f"capraz={cap1} rho_1={r1:+.6f} gercek={ort(GD[0], r):+.6f}",
    )

    # --- 4.2 SIRASIZ sonda ---
    print("\n--- 4.2 sirasiz sonda (yalniz sonda 2'nin skoru girilir) ---")
    if 2 in gec and 2 in var:
        P2, _ = gercek_skor("tuketim_D2_demet.csv", z, r)
        olcum_yaz({2: P2})
        rc, so, se = m148_dis(bekle_hata=True)
        satir = [x for x in so.splitlines() if "demet 2 (" in x]
        rger2 = ort(GD[1], r)
        if rc != 0:
            yaz("4.2 eksik onceki sonda tespit ediliyor", True, "SystemExit ile durdu")
        elif satir:
            kendi = float(satir[0].split("rho_k =")[1].split()[0])
            cg = float(gec[2]["onceki_r"]["1"]) * ort(GD[0], r)
            yaz(
                "4.2 sonda 1 atlanirsa capraz terim SESSIZCE 0 aliniyor",
                False,
                f"m148 rho_2={kendi:+.6f} GERCEK={rger2:+.6f} dusen capraz={cg:+.6f} "
                f"-> yanlilik {cg / gec[2]['kappa_etkin']:+.6f}",
            )
        else:
            yaz("4.2 sirasiz sonda", False, "beklenmeyen cikti: " + (so + se)[-300:])
    else:
        yaz("4.2 sirasiz sonda sinanamadi (D2 yok)", False, "once --asama 2 kos")


def asama5():
    """4.3 -- sonradan duzeltilen skor. Kendi kendine yeter."""
    print("=== ASAMA 5 (4.3): sonradan duzeltilen skor ===")
    z = np.load(ONBELLEK, allow_pickle=True)
    r = sentetik(z)
    GD = z["GD"]
    n = len(GD)
    P1, _ = gercek_skor("tuketim_D1_demet.csv", z, r)
    for k in range(2, n + 1):
        f = os.path.join(S, f"tuketim_D{k}_demet.csv")
        if os.path.exists(f):
            Path(f).unlink()
    zn = os.path.join(S, "tuketim_Z_NIHAI.csv")
    if os.path.exists(zn):
        Path(zn).unlink()
    j, _ = gecmis_oku()
    j["sondalar"] = [d for d in j["sondalar"] if d["sonda"] == 1]
    with open(GECMIS_YOL, "w") as fh:
        json.dump(j, fh, indent=1)
    P1_hatali = P1 + 0.0030
    olcum_yaz({1: P1_hatali})
    m148_dis()
    _, gec = gecmis_oku()
    if 2 not in gec or not os.path.exists(os.path.join(S, "tuketim_D2_demet.csv")):
        raise SystemExit(
            f"DUR: D2 uretilemedi (kayitlar {sorted(gec)}) -- paralel oturum karismis olabilir"
        )
    r1_hatali = float(gec[2]["onceki_r"]["1"])
    print(f"  D2, HATALI r_1={r1_hatali:+.6f} ile uretildi (dogrusu {ort(GD[0], r):+.6f})")
    P2b, _ = gercek_skor("tuketim_D2_demet.csv", z, r)
    olcum_yaz({1: P1, 2: P2b})
    rc, so, se = m148_dis(bekle_hata=True)
    satir = [x for x in so.splitlines() if "demet 2 (" in x]
    rger2 = ort(GD[1], r)
    if rc == 0 and satir:
        kendi = float(satir[0].split("rho_k =")[1].split()[0])
        yaz(
            "4.3 skor duzeltilince capraz DOGRU yeniden hesaplaniyor",
            abs(kendi - rger2) < 1e-5,
            f"m148 rho_2={kendi:+.6f} GERCEK={rger2:+.6f} hata={kendi - rger2:+.2e} "
            f"(D2 tabani {r1_hatali:+.6f} ile kurulmustu)",
        )
    else:
        yaz("4.3 skor duzeltilme senaryosu", False, (so + se)[-400:])


def asama6():
    """4.4 -- KAPPA_K[0] cakili. D1 yeniden uretilse BIT BIT ayni cikar mi?

    D1'i SILMEDEN sinar: m148'i surec icinde kosar, uretim satirlarinin tipatip
    ayni kod yolunu (taban + KAPPA_K[0]*GD[0] -> expm1 -> clip -> to_csv)
    scratchpad'de tekrarlar ve md5'leri karsilastirir.
    """
    print("=== ASAMA 6 (4.4): D1 yeniden uretimi (KAPPA_K[0] cakili) ===")
    if os.path.exists(OLC_YOL):
        Path(OLC_YOL).unlink()
    g, _ = m148_ici()
    a0, r_hat, GD, KAPPA_K = g["a0"], g["r_hat"], np.asarray(g["GD"]), np.asarray(g["KAPPA_K"])
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    taban = a0 + r_hat.copy()  # RHO_OLC bos -> m148'in taban'i budur
    kap = float(KAPPA_K[0])
    y = np.clip(np.expm1(taban + kap * GD[0]), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kopya = os.path.join(SCRATCH, "m158_D1_yeniden.csv")
    out.to_csv(kopya, index=False)
    h = md5(kopya)
    yaz(
        "4.4 D1 yeniden uretilse BIT BIT ayni dosya cikar (r_hat/GD/KAPPA_K[0] kaymamis)",
        h == D1_MD5,
        f"yeniden uretim md5 {h}  beklenen {D1_MD5}  kappa={kap:.17g}",
    )
    d1yol = os.path.join(S, "tuketim_D1_demet.csv")
    yaz("4.4b gercek D1 dosyasina DOKUNULMADI", md5(d1yol) == D1_MD5, md5(d1yol))
    Path(kopya).unlink()


# ---------------------------------------------------------------- TEMIZLIK
def temizlik():
    print("=== TEMIZLIK ===")
    if os.path.exists(OLC_YOL):
        Path(OLC_YOL).unlink()
        print("  [1] SILINDI experiments/model29/m148_olcumler.json")
    else:
        print("  [1] m148_olcumler.json zaten yok")
    for f in (
        "tuketim_D2_demet.csv",
        "tuketim_D3_demet.csv",
        "tuketim_D4_demet.csv",
        "tuketim_Z_NIHAI.csv",
    ):
        y = os.path.join(S, f)
        if os.path.exists(y):
            Path(y).unlink()
            print(f"  [2] SILINDI submissions/{f}")
        else:
            print(f"  [2] submissions/{f} zaten yok")
    t = GECMIS_YOL + ".tmp"
    if os.path.exists(t):
        Path(t).unlink()
        print(f"  [3] SILINDI {t}")
    else:
        print("  [3] m148_demet.json.tmp yok")
    subprocess.run(
        ["git", "checkout", "--", "experiments/model29/m148_demet.json"], cwd=KOK, check=True
    )
    print("  [4] m148_demet.json git checkout ile geri alindi")
    d1 = os.path.join(S, "tuketim_D1_demet.csv")
    h = md5(d1)
    print(f"  [5] D1 md5 = {h}  {'DOGRU' if h == D1_MD5 else 'YANLIS!!!'}")
    p = subprocess.run(
        ["git", "status", "--porcelain"], cwd=KOK, capture_output=True, text=True, check=False
    )
    print("  [6] git status --porcelain:")
    print("\n".join("      " + x for x in p.stdout.splitlines()) or "      (temiz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asama", type=int)
    ap.add_argument("--temizlik", action="store_true")
    a = ap.parse_args()
    if a.temizlik:
        temizlik()
        return
    {1: asama1, 2: asama2, 3: asama3, 4: asama4, 5: asama5, 6: asama6}[a.asama]()
    if SONUC:
        k = sum(1 for _, g, _ in SONUC if not g)
        print(f"\nOZET: {len(SONUC) - k}/{len(SONUC)} GECTI, {k} KALDI")
        for ad, g, ac in SONUC:
            if not g:
                print(f"  KALDI -> {ad}: {ac}")


if __name__ == "__main__":
    main()
