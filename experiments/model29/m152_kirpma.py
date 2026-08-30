"""KIRPMA VE YUVARLAMA DENETIMI -- sabit ile kappa_etkin^2 arasindaki fark.

SORU. Sonda dosyasi log uzayinda kurulur:
    p = taban + kappa*G,   taban = a0 + r_hat  (ilk sondada)
CSV'ye ise  y = clip(expm1(p), 0, None)  olarak yazilir. G hem span'a hem
r_hat'a DIK oldugu icin sunun BIREBIR tutmasi beklenirdi:
    sabit  ==  TABAN_MSE + kappa_etkin^2
Baska bir oturum bu esitligin +8.5e-05 tutmadigini bildirdi.

BU BETIK farki kalem kalem ayristirir:
    (a) expm1/log1p gidis-donusu           (kayan nokta)
    (b) 0'a kirpma                          (yonu KISALTIR)
    (c) MSE_OPT'un yanlis sabit secilmesi   (M0-gercek vs M0-2kL+||r_hat||^2)
    (d) pandas to_csv ondalik kaybi         (diskteki deger != bellektekI)
ve dogru cozum formulunu, D1 icin diskten bagimsiz yeniden hesapla birlikte
verir.

HICBIR GONDERIM YAPILMAZ, submissions/ altina HICBIR SEY YAZILMAZ.
m148_demet_plani.py orijinal haliyle kosturulur ama tum yazma cagrilari
gecici dizine yonlendirilir.
"""

import builtins
import io
import json
import os
import pathlib
import runpy
import sys
import tempfile

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
GECICI = os.path.join(tempfile.gettempdir(), "m152_kirpma")
os.makedirs(GECICI, exist_ok=True)
YAKALANAN = {}


def _gecici_ad(yol):
    return os.path.join(GECICI, os.path.basename(str(yol)))


def _koru(yol):
    """submissions/ altina veya m148_demet.json ustune yazmayi engelle."""
    y = str(yol).replace("\\", "/")
    return "/submissions/" in y or y.endswith("m148_demet.json")


def m148_calistir():
    """m148_demet_plani.py'yi HIC DEGISTIRMEDEN kosturur, yazmalarini yakalar."""
    ozgun_to_csv = pd.DataFrame.to_csv
    ozgun_replace = pathlib.Path.replace
    ozgun_open = builtins.open
    ozgun_exists = os.path.exists

    def yeni_exists(yol):
        # D1 SONDASINI yeniden kurmak icin m148'i "hic olcum yok" halinde
        # kostururuz. Paralel bir oturum m148_olcumler.json'a (simule) bir
        # skor yazmis olabilir; o zaman m148 D2 uretir ve bu denetim D1'i
        # goremez. Dosyayi gorunmez kilmak analizi belirlenimci yapar.
        if str(yol).replace("\\", "/").endswith("m148_olcumler.json"):
            return False
        return ozgun_exists(yol)

    def yeni_to_csv(self, path_or_buf=None, *a, **k):
        if isinstance(path_or_buf, (str, pathlib.Path)) and _koru(path_or_buf):
            hedef = _gecici_ad(path_or_buf)
            ad = os.path.basename(str(path_or_buf))
            YAKALANAN[ad[:-4] if ad.endswith(".tmp") else ad] = self.copy()
            return ozgun_to_csv(self, hedef, *a, **k)
        return ozgun_to_csv(self, path_or_buf, *a, **k)

    def yeni_replace(self, hedef):
        if _koru(hedef) or _koru(self):
            return pathlib.Path(_gecici_ad(hedef))
        return ozgun_replace(self, hedef)

    def yeni_open(dosya, kip="r", *a, **k):
        if isinstance(dosya, (str, pathlib.Path)) and "w" in kip and _koru(dosya):
            return ozgun_open(_gecici_ad(dosya), kip, *a, **k)
        return ozgun_open(dosya, kip, *a, **k)

    pd.DataFrame.to_csv = yeni_to_csv
    pathlib.Path.replace = yeni_replace
    builtins.open = yeni_open
    os.path.exists = yeni_exists
    yedek = sys.stdout
    sys.stdout = io.StringIO()
    try:
        g = runpy.run_path(os.path.join(M29, "m148_demet_plani.py"), run_name="m148_izole")
    finally:
        gunluk = sys.stdout.getvalue()
        sys.stdout = yedek
        pd.DataFrame.to_csv = ozgun_to_csv
        pathlib.Path.replace = ozgun_replace
        builtins.open = ozgun_open
        os.path.exists = ozgun_exists
    return g, gunluk


def bas(baslik):
    print("\n" + "=" * 78)
    print(baslik)
    print("=" * 78)


def rms2(x):
    return float(np.mean(np.asarray(x, dtype=np.float64) ** 2))


def main():
    print("m148_demet_plani.py izole kosturuluyor (yazma yok)...")
    g, gunluk = m148_calistir()
    son = [s for s in gunluk.splitlines() if s.strip()][-14:]
    print("  [m148 son satirlari]")
    for s in son:
        print("   |", s)

    a0 = g["a0"]
    r_hat = g["r_hat"]
    kL = float(g["kL"])
    M0 = float(g["M0"])
    N = int(g["N"])
    TABAN_MSE = float(g["TABAN_MSE"])
    MSE_OPT = float(g["MSE_OPT"])
    G = np.asarray(g["G"])
    KAPPA_K = np.asarray(g["KAPPA_K"])
    taban = np.asarray(g["taban"])
    ETIKET = g["ETIKET"]
    kap = float(KAPPA_K[0])
    u = G[0]

    assert np.allclose(taban, a0 + r_hat), "taban = a0 + r_hat degil (olcum girilmis?)"
    df_bellek = YAKALANAN["tuketim_D1_demet.csv"]
    y_bellek = df_bellek.tuketim.to_numpy(dtype=np.float64)

    # ---------------------------------------------------------------- 0
    bas("0. GEOMETRI -- ideal (kirpmasiz, yuvarlamasiz) durumda esitlik")
    print(f"N = {N},  M0 = {M0:.9f},  kL = {kL:.9f}")
    print(f"||r_hat||^2 = {rms2(r_hat):.9f}   gercek(=kazanc) = {float(g['gercek']):.9f}")
    print(f"TABAN_MSE = M0 - 2kL + ||r_hat||^2 = {TABAN_MSE:.9f}")
    print(f"MSE_OPT   = M0 - gercek            = {MSE_OPT:.9f}")
    print(f"IKISININ FARKI = {MSE_OPT - TABAN_MSE:+.9f}   <-- (c) kalemi")
    print(f"kappa = {kap:.9f}   yon = {ETIKET[0]}")
    print(f"<u,u>  = {rms2(u):.12f}  (1 olmali)")
    print(f"<u,r_hat> = {float(np.mean(u * r_hat)):+.3e}  (0 olmali)")
    p = taban + kap * u  # log uzayinda GERCEK hedef
    d_ideal = p - a0
    sabit_ideal = M0 - 2 * kL + rms2(d_ideal)
    print(f"\nIDEAL sabit = M0 - 2kL + ||r_hat + kappa*u||^2 = {sabit_ideal:.10f}")
    print(f"TABAN_MSE + kappa^2                            = {TABAN_MSE + kap**2:.10f}")
    print(f"  fark = {sabit_ideal - (TABAN_MSE + kap**2):+.3e}  (sirf kayan nokta)")

    # ---------------------------------------------------------------- 1
    bas("1. KIRPMA -- kac satir etkileniyor")
    neg_p = p < 0.0
    neg_t = taban < 0.0
    print(f"expm1(p) < 0 olan satir      : {int(neg_p.sum())}  ({neg_p.mean() * 100:.4f}%)")
    print(f"expm1(taban) < 0 olan satir  : {int(neg_t.sum())}  ({neg_t.mean() * 100:.4f}%)")
    if neg_p.any():
        print(f"  en negatif p = {p[neg_p].min():.6f} -> expm1 = {np.expm1(p[neg_p].min()):.6f}")
        print(f"  kirpilan satirlarda |p| ortalamasi = {np.abs(p[neg_p]).mean():.6f}")
    y_kirpsiz = np.expm1(p)
    print(f"expm1(p) en kucuk deger      : {y_kirpsiz.min():.6e}")
    print(f"y_bellek icinde tam 0 sayisi : {int((y_bellek == 0).sum())}")
    kayip = float(np.sum(np.clip(-y_kirpsiz, 0, None)))
    print(f"kirpmayla silinen toplam (negatif) kutle: {kayip:.6e}")

    # ---------------------------------------------------------------- 2
    bas("2. GIDIS-DONUS -- (a) kayan nokta, (b) kirpma AYRI AYRI")
    d_rt = np.log1p(np.expm1(p)) - a0  # kirpmasiz gidis-donus
    d_bellek = np.log1p(y_bellek) - a0  # kirpmali, bellekteki dizi
    e_a = d_rt - d_ideal
    e_b = d_bellek - d_rt
    print(
        f"(a) log1p(expm1(p)) - p        : maks |.| = {np.abs(e_a).max():.3e}  "
        f"rms = {np.sqrt(rms2(e_a)):.3e}"
    )
    print(
        f"(b) kirpmanin ekledigi         : maks |.| = {np.abs(e_b).max():.3e}  "
        f"rms = {np.sqrt(rms2(e_b)):.3e}  sifirdan farkli satir = {int((e_b != 0).sum())}"
    )

    # ---------------------------------------------------------------- 3
    bas("3. (d) pandas to_csv -- diskteki deger bellektekiyle AYNI MI?")
    D1 = os.path.join(S, "tuketim_D1_demet.csv")
    disk = pd.read_csv(D1)
    y_disk = disk.tuketim.to_numpy(dtype=np.float64)
    ayni = bool(np.array_equal(y_disk, y_bellek))
    print(
        f"diskteki D1 satir sayisi: {len(disk)}   id sirasi ayni: "
        f"{bool(np.array_equal(disk.id.to_numpy(), df_bellek.id.to_numpy()))}"
    )
    print(f"y_disk == y_bellek (birebir): {ayni}")
    frk = np.abs(y_disk - y_bellek)
    print(f"  farkli satir = {int((frk > 0).sum())}  maks mutlak fark = {frk.max():.3e}")
    bag = np.where(y_bellek > 0, frk / np.maximum(y_bellek, 1e-300), 0.0)
    print(f"  maks BAGIL fark = {bag.max():.3e}  (1 ulp = 2.2e-16)")
    d_disk = np.log1p(y_disk) - a0
    e_d = d_disk - d_bellek
    print(
        f"(d) log uzayinda etki          : maks |.| = {np.abs(e_d).max():.3e}  "
        f"rms = {np.sqrt(rms2(e_d)):.3e}"
    )

    # ---------------------------------------------------------------- 4
    bas("4. HER KALEMIN 'sabit'E KATKISI (sabit = M0 - 2kL + ||d||^2)")
    adim = [
        ("ideal      d = r_hat + kappa*u", d_ideal),
        ("(a) +gidis-donus              ", d_rt),
        ("(b) +kirpma  (= bellek)       ", d_bellek),
        ("(d) +to_csv  (= disk)         ", d_disk),
    ]
    onceki = None
    for ad, dd in adim:
        sb = M0 - 2 * kL + rms2(dd)
        art = "" if onceki is None else f"  artis {sb - onceki:+.3e}"
        print(f"  {ad}  sabit = {sb:.10f}{art}")
        onceki = sb
    sabit_disk = M0 - 2 * kL + rms2(d_disk)
    sabit_bellek = M0 - 2 * kL + rms2(d_bellek)

    # ---------------------------------------------------------------- 5
    bas("5. kappa_etkin ve BILDIRILEN TUTARSIZLIK")
    b_bellek = np.log1p(np.clip(np.expm1(taban), 0.0, None)) - a0  # m148'in kullandigi
    ek_bellek = d_bellek - b_bellek
    ket_bellek = float(np.sqrt(rms2(ek_bellek)))
    ek_disk = d_disk - b_bellek
    ket_disk = float(np.sqrt(rms2(ek_disk)))
    print(f"kappa            = {kap:.9f}")
    print(f"kappa_etkin (bellek, m148'in yazdigi) = {ket_bellek:.9f}")
    print(f"kappa_etkin (disk)                    = {ket_disk:.9f}")
    print(f"kappa - kappa_etkin = {kap - ket_bellek:.3e}  (kirpmanin KISALTMASI)")

    for ad, sb, ket in (("bellek", sabit_bellek, ket_bellek), ("disk", sabit_disk, ket_disk)):
        print(f"\n  [{ad}]  sabit = {sb:.10f}")
        for etiket, taban_ in (("TABAN_MSE", TABAN_MSE), ("MSE_OPT", MSE_OPT)):
            fark = sb - (taban_ + ket**2)
            print(
                f"    sabit - ({etiket} + kappa_etkin^2) = {fark:+.3e}"
                f"   -> rho sapmasi {fark / (2 * ket):+.3e}"
            )

    # farkin TAM ayrisimi:  ||d||^2 - ||r_hat||^2 - ||ek||^2 = (||b||^2-||r_hat||^2) + 2<b,ek>
    taban_bozulma = rms2(b_bellek) - rms2(r_hat)
    capraz = 2.0 * float(np.mean(b_bellek * ek_bellek))
    print("\n  TAM AYRISIM (bellek):")
    print(f"    ||b||^2 - ||r_hat||^2 (tabanin kendi kirpilmasi) = {taban_bozulma:+.3e}")
    print(f"    2*<b, ek>             (kirpma dikligi bozdu)     = {capraz:+.3e}")
    print(f"    toplam                                           = {taban_bozulma + capraz:+.3e}")
    print(
        f"    olcum : sabit - (TABAN_MSE + ket^2)              = "
        f"{sabit_bellek - (TABAN_MSE + ket_bellek**2):+.3e}"
    )

    # kirpilan satirlarin OLCUM yonune etkisi
    kos = float(np.mean(ek_bellek * u)) / max(ket_bellek, 1e-30)
    pay = float(np.sum(ek_bellek[neg_p] ** 2)) / float(np.sum(ek_bellek**2))
    print("\n  KIRPILAN 530 SATIRIN OLCUM YONUNE ETKISI:")
    print(f"    cos(ek, u) = {kos:.9f}   (1 olsa kirpma yonu hic egmemis olurdu)")
    print(f"    aci = {np.degrees(np.arccos(min(kos, 1.0))):.3f} derece")
    print(f"    ||ek||^2'nin kirpilan satirlardaki payi = {pay * 100:.4f}%")
    print("    NOT: rho tanim geregi ek/||ek|| yonunde olculur, dolayisiyla")
    print("    kirpma OLCUMU BOZMAZ; sadece olculen yonu u'dan cok az egiyor.")

    # ---------------------------------------------------------------- 6
    bas("6. DOGRU COZUM FORMULU")
    print("Kaggle'in gordugu vektor DISKTEKI d_disk'tir. Ozdeslik:")
    print("    P^2 = M0 - 2<d_disk, r> + ||d_disk||^2")
    print("d_disk = b + ek  (b = kirpilmis taban, ek = sondanin ek parcasi)")
    print("    <d_disk, r> = <b, r> + kappa_etkin * rho")
    print("    =>  rho = (M0 - 2<b,r> + ||d_disk||^2 - P^2) / (2*kappa_etkin)")
    print("           = (sabit - P^2) / (2*kappa_etkin)      [sabit <b,r> ~ kL ile]")
    print("\nYani m148'in FORMU DOGRU. Sartlar:")
    print("  1) sabit ve kappa_etkin AYNI vektorden gelmeli (ikisi de disk ya da")
    print("     ikisi de bellek). m148 ikisini de bellekten aliyor -> tutarli.")
    print("  2) sabit'te MSE_OPT DEGIL, M0-2kL+||d||^2 kullanilmali. m148 boyle")
    print("     yapiyor; tutarsizligi bildiren oturum yanlis referans kullanmis.")
    b_hata = float(np.sqrt(rms2(b_bellek - r_hat)))
    sinir = b_hata * np.sqrt(M0)
    print("\nGERI KALAN TEK SISTEMATIK: <b,r> yerine kL kullanmak.")
    print(f"  ||b - r_hat|| = {b_hata:.3e}")
    print(f"  |<b-r_hat, r>| <= ||b-r_hat||*sqrt(M0) = {sinir:.3e}")
    print(
        f"  rho'daki EN KOTU sapma = {sinir / (2 * ket_bellek):.3e}"
        f"   (rho ~ 0.25 beklendiginde onemsiz)"
    )

    # ---------------------------------------------------------------- 7
    bas("7. D1 -- DISKTEN BAGIMSIZ YENIDEN HESAP vs m148_demet.json")
    with open(os.path.join(M29, "m148_demet.json")) as fh:
        JS = json.load(fh)
    s1 = next(q for q in JS["sondalar"] if q["sonda"] == 1)
    print(f"{'kalem':>16s} {'json':>16s} {'bellek':>16s} {'disk':>16s} {'json-disk':>12s}")
    for ad, jv, bv, dv in (
        ("kappa", s1["kappa"], kap, kap),
        ("kappa_etkin", s1["kappa_etkin"], ket_bellek, ket_disk),
        ("sabit", s1["sabit"], sabit_bellek, sabit_disk),
        ("taban_mse", JS["taban_mse"], TABAN_MSE, TABAN_MSE),
    ):
        print(f"{ad:>16s} {jv:16.10f} {bv:16.10f} {dv:16.10f} {jv - dv:+12.3e}")
    for P in (0.99614, 0.99927, 1.00115, 1.00235):
        rj = (s1["sabit"] - P * P) / (2 * s1["kappa_etkin"])
        rd = (sabit_disk - P * P) / (2 * ket_disk)
        print(f"  P={P:.5f}: rho_json = {rj:+.6f}   rho_disk = {rd:+.6f}   fark {rj - rd:+.2e}")

    # ---------------------------------------------------------------- 8
    bas("8. KIRPMAYI ONLEMEK -- kucuk pozitif taban")
    print("RMSLE: skor = sqrt(mean((log1p(yhat)-log1p(y))^2)). yhat=0 GECERLIDIR")
    print("(log1p(0)=0); yalniz NEGATIF yhat tanimsizdir. Yani 0'a kirpmak")
    print("kural geregi ZORUNLU, tek secim tabanin nereye konuldugu.")
    for taban_deger in (0.0, 1e-9, 1e-6, 1e-3, 1e-2):
        yy = np.clip(np.expm1(p), taban_deger, None)
        dd = np.log1p(yy) - a0
        ekk = dd - b_bellek
        kk = float(np.sqrt(rms2(ekk)))
        sb = M0 - 2 * kL + rms2(dd)
        print(
            f"  taban={taban_deger:.0e}: "
            f"etkilenen satir={int((np.expm1(p) < taban_deger).sum()):6d}"
            f"  kappa_etkin={kk:.9f}  sabit={sb:.10f}  "
            f"tutarsizlik={sb - (TABAN_MSE + kk**2):+.2e}"
        )
    print("\nSONUC: kirpma tabanini degistirmek TUTARSIZLIGI KAPATMAZ (kaynagi")
    print("kirpma degil), sadece skoru bozar. Dogru cozum sabit'i GONDERILEN")
    print("vektorden hesaplamak -- m148 zaten oyle yapiyor.")

    # ---------------------------------------------------------------- 9
    bas("9. HUKUM")
    fark_dogru = sabit_bellek - (TABAN_MSE + ket_bellek**2)
    fark_yanlis = sabit_bellek - (MSE_OPT + ket_bellek**2)
    print(f"dogru referansla tutarsizlik  : {fark_dogru:+.3e}")
    print(f"MSE_OPT referansiyla          : {fark_yanlis:+.3e}  <-- bildirilen kalem")
    print(f"(c) kaleminin tek basina payi : {TABAN_MSE - MSE_OPT:+.3e}")
    print(f"(a) kalemi (kayan nokta)      : ~{abs(rms2(d_rt) - rms2(d_ideal)):.3e}")
    print(f"(b) kalemi (kirpma)           : ~{abs(rms2(d_bellek) - rms2(d_rt)):.3e}")
    print(f"(d) kalemi (to_csv)           : ~{abs(rms2(d_disk) - rms2(d_bellek)):.3e}")
    print("\nHICBIR GONDERIM YAPILMADI; submissions/ altina yazilmadi.")


if __name__ == "__main__":
    main()
