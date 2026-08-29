"""GUNLUK SURUCU -- her hak icin TEK komut.

Akis:
  1) python m108_gun.py --baslat            -> ilk sondayi uretir, kaggle komutunu basar
  2) gonder, skoru oku
  3) python m108_gun.py --skor 0.99921      -> L'yi cozer, kaydeder, SONRAKI sondayi uretir
  4) 2-3'u tekrarla
  5) python m108_gun.py --bitir             -> saf ortak optimum (sonda terimi YOK), son gonderim

Durum m108_durum.json'da tutulur; oturum/laptop kapanip acilsa da kaldigi yerden devam eder.
"""

import argparse
import json
import os
import subprocess
import sys

DURUM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m108_durum.json")
VARSAYILAN_SIRA = ["y40", "z2", "sul", "y46", "y45", "q1c", "t3", "h1"]


def yukle():
    if os.path.exists(DURUM):
        return json.load(open(DURUM))
    return {
        "olculen": {"g7": 0.002728},
        "sira": VARSAYILAN_SIRA,
        "adim": 0,
        "bekleyen": None,
        "gecmis": [],
    }


def kaydet(d):
    json.dump(d, open(DURUM, "w"), indent=1)


def sonda_uret(d, aday, cikti, ek=()):
    bil = ",".join(f"{k}={v:.9f}" for k, v in d["olculen"].items())
    cmd = [
        sys.executable,
        "m107_sonda3.py",
        "--bilinen",
        bil,
        "--aday",
        aday,
        "--cikti",
        cikti,
        *ek,
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(DURUM),
        check=False,
    )
    print(r.stdout)
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit(f"SONDA URETILEMEDI (cikis {r.returncode}) -- yukaridaki korkuluga bak")
    rap = json.load(open(os.path.join(os.path.dirname(DURUM), f"m107_{aday}.json")))
    return rap


def komut_bas(cikti, aday, skor_l0):
    print("=" * 72)
    print("GONDER:")
    print(
        f"  kaggle competitions submit -c grid-up-datathon -f submissions/{cikti} "
        f'-m "sonda {aday}: ortak optimum + {aday} olcumu"'
    )
    print("SONRA MUTLAKA LISTEYI OKU:")
    print("  kaggle competitions submissions -c grid-up-datathon")
    print(f"\nen kotu durumda (L_{aday}=0) beklenen skor: {skor_l0:.5f}")
    print("skor gelince:  python m108_gun.py --skor <SKOR>")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baslat", action="store_true")
    ap.add_argument("--skor", type=float)
    ap.add_argument("--bitir", action="store_true")
    ap.add_argument("--atla", action="store_true", help="bekleyen adayi olcmeden gec")
    ap.add_argument("--durum", action="store_true")
    a = ap.parse_args()
    d = yukle()

    if a.durum:
        print(json.dumps(d, indent=1))
        return

    if a.baslat:
        if d["bekleyen"]:
            raise SystemExit(
                f"Zaten bekleyen sonda var: {d['bekleyen']['aday']} "
                f"({d['bekleyen']['cikti']}). Skoru gir: --skor <S>"
            )
        aday = d["sira"][d["adim"]]
        cikti = f"tuketim_s3{aday}.csv"
        rap = sonda_uret(d, aday, cikti)
        d["bekleyen"] = dict(
            aday=aday,
            cikti=cikti,
            sabit=rap["sabit"],
            k_yeni=rap["k_yeni"],
            Q=rap["Q_yeni"],
            skor_L0=rap["skor_L0"],
        )
        kaydet(d)
        komut_bas(cikti, aday, rap["skor_L0"])
        return

    if a.atla:
        if not d["bekleyen"]:
            raise SystemExit("bekleyen sonda yok")
        print(f"ATLANDI: {d['bekleyen']['aday']}")
        d["sira"] = [x for x in d["sira"] if x != d["bekleyen"]["aday"]]
        d["bekleyen"] = None
        kaydet(d)
        return

    if a.skor is not None:
        b = d["bekleyen"]
        if not b:
            raise SystemExit("bekleyen sonda yok -- once --baslat")
        L = (b["sabit"] - a.skor * a.skor) / (2 * b["k_yeni"])
        rho = L / (b["Q"] ** 0.5)
        print(f"\n{b['aday']}: skor {a.skor}  ->  L = {L:+.6f}   rho = {rho:+.4f}")
        print("  (esik rho = 0.0137 | g7'nin rho'su = 0.0546 | gecmis ortanca 0.028)")
        if abs(rho) < 0.005:
            print("  UYARI: bu yon neredeyse bilgisiz; yine de zarar vermez (k kucuk cikar)")
        d["olculen"][b["aday"]] = L
        d["gecmis"].append(dict(aday=b["aday"], skor=a.skor, L=L, rho=rho, cikti=b["cikti"]))
        d["adim"] += 1
        d["bekleyen"] = None
        kaydet(d)
        if d["adim"] >= len(d["sira"]):
            print("\nTum adaylar olculdu -> python m108_gun.py --bitir")
            return
        aday = d["sira"][d["adim"]]
        cikti = f"tuketim_s3{aday}.csv"
        print(f"\n--- SONRAKI SONDA: {aday} ---")
        rap = sonda_uret(d, aday, cikti)
        d["bekleyen"] = dict(
            aday=aday,
            cikti=cikti,
            sabit=rap["sabit"],
            k_yeni=rap["k_yeni"],
            Q=rap["Q_yeni"],
            skor_L0=rap["skor_L0"],
        )
        kaydet(d)
        komut_bas(cikti, aday, rap["skor_L0"])
        return

    if a.bitir:
        # saf ortak optimum: sonda terimi yok. m107'yi son olculen yonu "aday" yapip
        # olcek=0 ile cagirmak yerine dogrudan m99'u kullan.
        bil = d["olculen"]
        print("SON GONDERIM -- saf ortak optimum, olculen L'ler:")
        for k, v in bil.items():
            print(f"  {k:5s} L={v:+.6f}")
        print("\nCalistir:")
        arg = ",".join(f"{k}={v:.9f}" for k, v in bil.items())
        print(f"  python m107_sonda3.py --bilinen {arg} --cikti tuketim_NIHAI.csv")
        print("  (--aday verilmez -> sonda terimi yok, saf ortak optimum)")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
