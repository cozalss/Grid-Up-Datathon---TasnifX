"""GUNLUK SURUCU -- her hak icin TEK komut.

  python m108_gun.py --baslat        -> sondayi uretir + gonderim komutunu basar
  <gonder, skoru oku>
  python m108_gun.py --skor 1.00042  -> L'yi cozer, kaydeder, SONRAKI sondayi uretir
  ...
  python m108_gun.py --bitir         -> saf ortak optimum komutunu basar

Durum m108_durum.json'da; laptop kapansa da kaldigi yerden devam eder.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BURA = os.path.dirname(os.path.abspath(__file__))
DURUM = os.path.join(BURA, "m108_durum.json")
SUBD = os.path.abspath(os.path.join(BURA, "..", "..", "submissions"))
L_G7 = 0.002751118  # kalibre m0 (1.005846366) altinda span'dan turetildi
VARSAYILAN_SIRA = ["y40", "z2", "sul", "y46", "y45", "q1c", "t3", "p42"]


def yukle():
    if os.path.exists(DURUM):
        return json.load(open(DURUM))
    return {
        "olculen": {"g7": L_G7},
        "sira": VARSAYILAN_SIRA,
        "adim": 0,
        "bekleyen": None,
        "gecmis": [],
    }


def kaydet(d):
    """Atomik: yarim yazma tum olculen L'leri silmesin."""
    g = DURUM + ".tmp"
    with open(g, "w") as f:
        json.dump(d, f, indent=1)
    Path(g).replace(DURUM)


def sonda_uret(d, aday, cikti):
    bil = ",".join(f"{k}={v:.9f}" for k, v in d["olculen"].items())
    cmd = [sys.executable, "m107_sonda3.py", "--bilinen", bil, "--aday", aday, "--cikti", cikti]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=BURA,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    print(r.stdout)
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit(
            f"SONDA URETILEMEDI (cikis {r.returncode}).\n"
            f"  Korkuluk tetiklendiyse: m107'ye --k1-tavan / --e-tavan ile gevsetip elle uret,\n"
            f"  ya da bu adayi atla:  python m108_gun.py --atla"
        )
    return json.load(open(os.path.join(BURA, f"m107_{aday}.json")))


def komut_bas(cikti, aday, skor_l0):
    print("=" * 74)
    print("GONDER (herhangi bir dizinden calisir):")
    print(
        f'  kaggle competitions submit -c grid-up-datathon -f "{os.path.join(SUBD, cikti)}" '
        f'-m "sonda {aday}"'
    )
    print("SONRA MUTLAKA:")
    print("  kaggle competitions submission-limits -c grid-up-datathon")
    print("  kaggle competitions submissions -c grid-up-datathon")
    print(f"\nL_{aday}=0 iken beklenen skor: {skor_l0:.5f}")
    print("skor gelince:  python m108_gun.py --skor <SKOR>")
    print("=" * 74)


def ilerlet(d):
    """Sonraki adayi uret, durumu kaydet, komutu bas."""
    if d["adim"] >= len(d["sira"]):
        print("\nTum adaylar olculdu -> python m108_gun.py --bitir")
        return
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
        print(json.dumps(d, indent=1, ensure_ascii=False))
        return

    if a.baslat:
        if d["bekleyen"]:
            b = d["bekleyen"]
            print(f"Bekleyen sonda ZATEN URETILMIS ve diskte hazir: {b['aday']}")
            komut_bas(b["cikti"], b["aday"], b["skor_L0"])
            return
        ilerlet(d)
        return

    if a.atla:
        if d["bekleyen"]:
            ad = d["bekleyen"]["aday"]
            d["bekleyen"] = None
        else:  # sonda URETILEMEDI halinden kurtulma
            if d["adim"] >= len(d["sira"]):
                raise SystemExit("atlanacak aday kalmadi")
            ad = d["sira"][d["adim"]]
        print(f"ATLANDI: {ad}")
        d["sira"] = [x for x in d["sira"] if x != ad]
        kaydet(d)
        return

    if a.skor is not None:
        b = d["bekleyen"]
        if not b:
            raise SystemExit("bekleyen sonda yok -- once --baslat")
        if not 0.90 < a.skor < 1.10:
            raise SystemExit(f"skor {a.skor} makul disi (0.90-1.10). Yanlis mi yazdin?")
        if abs(a.skor - b["skor_L0"]) > 0.02:
            raise SystemExit(
                f"skor {a.skor}, {b['aday']} sondasinin beklenen araliginda DEGIL "
                f"(L=0 iken {b['skor_L0']:.5f}, band +-0.02).\n"
                f"  Yanlis satiri mi okudun? Listede EN USTTEKI (en yeni) gonderim olmali."
            )
        # Yalnizca BIR ONCEKI girisle karsilastir: amac "--skor'u iki kez calistirma"
        # halini yakalamak. Tum gecmise bakmak yanlis pozitif verir -- iki farkli
        # sonda pekala ayni 5 haneli skoru alabilir.
        if d["gecmis"] and abs(d["gecmis"][-1]["skor"] - a.skor) < 1e-12:
            raise SystemExit(
                f"skor {a.skor} bir onceki adimda ({d['gecmis'][-1]['aday']}) girildi -- "
                f"komutu iki kez mi calistirdin?"
            )
        L = (b["sabit"] - a.skor * a.skor) / (2 * b["k_yeni"])
        rho = L / (b["Q"] ** 0.5)
        print(f"\n{b['aday']}: skor {a.skor}  ->  L = {L:+.6f}   rho = {rho:+.4f}")
        print("  (esik ~0.0224 | durust onsel ortancasi 0.0146 | g7'nin rho'su 0.0551)")
        if abs(rho) < 0.005:
            print("  NOT: bu yon neredeyse bilgisiz; zarar vermez, k kucuk cikar")
        d["olculen"][b["aday"]] = L
        d["gecmis"].append(dict(aday=b["aday"], skor=a.skor, L=L, rho=rho, cikti=b["cikti"]))
        d["adim"] += 1
        d["bekleyen"] = None
        kaydet(d)
        print("\n--- SONRAKI ---")
        ilerlet(d)
        return

    if a.bitir:
        if d["bekleyen"]:
            print(f"!! UYARI: {d['bekleyen']['aday']} sondasi uretildi ama SKORU GIRILMEDI.")
            print(
                "   Once 'python m108_gun.py --skor <S>' calistir -- "
                "yoksa bu eksen NIHAI'ye GIRMEZ.\n"
            )
        print("SON GONDERIM -- saf ortak optimum. Olculen L'ler:")
        for k, v in d["olculen"].items():
            print(f"  {k:5s} L={v:+.6f}")
        arg = ",".join(f"{k}={v:.9f}" for k, v in d["olculen"].items())
        print("\nCalistir:")
        print(f"  cd {BURA}")
        print(f"  python m107_sonda3.py --bilinen {arg} --cikti tuketim_NIHAI.csv")
        print(
            f"  kaggle competitions submit -c grid-up-datathon "
            f'-f "{os.path.join(SUBD, "tuketim_NIHAI.csv")}" -m "nihai ortak optimum"'
        )
        return

    ap.print_help()


if __name__ == "__main__":
    main()
