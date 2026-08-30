# ruff: noqa  -- YARIM KALDI: ajan kesildi, betik HIC KOSMADI, sonuc uretmedi.
# Sordugu soru sonradan baska olcumlerle cevaplandi (bkz. commit mesaji).
"""IKINCI NESIL CARPIM AILESI.

m144'un en verimli ailesi H_carpim40'ti: mevcut 40 eksenin IKILI carpimlari,
780 adaydan 285'i kapidan gecti. Ama m144 YALNIZCA eski 40 ekseni carpti.
m148 su an 136 ekseni kabul ediyor (40 m121 + 96 m144 ekseni) ve yeni
aileler (D_hava_gecikme, E_ufuk, F_guc_yas, G_mentese, B_lokasyon,
C_takvim) HIC carpima girmedi.

BU BETIK IKI ADAY KUMESI TARAR:
  P1  m148'in kabul ettigi 136 eksenin |rho_s| en buyuk ~60'inin ikili
      carpimlari (1770 aday). Yani "carpimin carpimi" dahil.
  P2  m144'un H DISI kapidan gecen 44 ekseninin (D/E/F/G/B/C) birbiriyle
      carpimlari (946 aday). Bunlar m144'ta hic carpilmadi.

KURULUS ve KAPILAR m144'ten AYNEN alinir: m144_yeni_aileler.py'nin
"3) TARAMA" oncesi tum govdesi exec edilir, boylece st/kur/olc/V/G/r_hat/
PERM/blok kurulusu BIREBIR aynidir. Kapilar: |rho_s| >= 0.015, Qs >= 0.02,
rcond kararliligi (1e-5 vs 1e-6, %30), Q_dik >= 0.25, plasebo |z| >= 3,
tavan |rho_cv| >= 1.95*|rho_s|.

TABAN: m148_demet_plani.py'nin kabul dongusunun KOPYASI kurulur (m148
DEGISTIRILMEZ, hic okunmaz disinda dokunulmaz). 136 eksen ve bunlarin dik
birim yonleri elde edilir; yeni adaylar BUNLARA diklestirilerek olculur.

CIKTI: n03_ikinci_nesil.json. HICBIR GONDERIM YAZILMAZ.
"""

import gc
import json
import os
import sys

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
UST_N = 60  # P1'de carpilacak eksen sayisi (60*59/2 = 1770 aday)
AZAMI_YENI = 200  # bellek/zaman siniri; kesim asil KAPIDAN gelmeli

# --------------------------------------------------------------- m144 govde
M144 = os.path.join(M29, "m144_yeni_aileler.py")
with open(M144, encoding="utf-8") as fh:
    KAYNAK = fh.read()
AYIRAC = "# ================================================== 3) TARAMA"
assert KAYNAK.count(AYIRAC) == 1, "m144 ayiraci bulunamadi"
ONEK = KAYNAK.split(AYIRAC)[0]
assert "def olc(" in ONEK and "def kur(" in ONEK and "MEVCUT_ADLAR" in ONEK
NS = {"__name__": "m144_onek", "__file__": M144}
print("m144 govdesi (kurulus + aile ureticileri) exec ediliyor...")
exec(compile(ONEK, M144, "exec"), NS)  # noqa: S102 -- kendi repomuzdaki betik

st = NS["st"]
olc = NS["olc"]
kur144 = NS["kur"]
V, G, Gi, r_hat = NS["V"], NS["G"], NS["Gi"], NS["r_hat"]
MEVCUT = NS["MEVCUT"]
ONCEKI40 = NS["ONCEKI"]
A_MEVCUT = NS["A_MEVCUT"]
MSE_OPT, TAVAN = NS["MSE_OPT"], NS["TAVAN"]
assert len(MEVCUT) == 40, f"m121 tabani 40 degil: {len(MEVCUT)}"


# ---------------------------------------------- m148'in kur()'u (mnt75 + M[])
def kur(ad):
    """m148_demet_plani.py'nin kur()'u: m144'unkine mnt75 ve M[a]x[b] eklenmis."""
    if ad.startswith("M[") and "]x[" in ad and ad.endswith("]"):
        k1, k2 = ad[2:-1].split("]x[", 1)
        a1, b1 = kur(k1)
        a2, b2 = kur(k2)
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
    if ":" in ad and ad.split(":", 1)[1] == "mnt75":
        kol = ad.split(":", 1)[0]
        if kol not in NS["tp"].columns or kol not in NS["bf"].columns:
            return None, None
        xt, xb = NS["tp"][kol].to_numpy(), NS["bf"][kol].to_numpy()
        fv = np.asarray(xt, dtype=np.float64)
        fv = fv[np.isfinite(fv)]
        if fv.size == 0:
            return None, None
        v_ = float(np.quantile(fv, 0.75))
        return st(np.maximum(xt - v_, 0.0)), st(np.maximum(xb - v_, 0.0))
    return kur144(ad)


# ============================================ 1) m148'in 136 ekseni (KOPYA)
with open(os.path.join(M29, "m144_yeni_aileler.json"), encoding="utf-8") as fh:
    _M144J = json.load(fh)
GECEN144 = _M144J["kapidan_gecen"]
YENI_ADLAR = [r["eksen"] for r in sorted(GECEN144, key=lambda r: -abs(r["rho_s"]))]
AILE144 = {r["eksen"]: r["aile"] for r in GECEN144}

ONCEKI = list(ONCEKI40)
ALAR = list(A_MEVCUT)
TABAN_EKSEN = [
    dict(eksen=m["eksen"], aile="m121_taban", rho_s=m["rho_s"], rho_cv=m["rho_cv"], Qd=m["Qd"])
    for m in MEVCUT
]
for ad in YENI_ADLAR:
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    s, _ = olc(xt, xb, ONCEKI)
    if s is None:
        continue
    ONCEKI.append(s["birim"])
    ALAR.append(s["a"])
    TABAN_EKSEN.append(
        dict(eksen=ad, aile=AILE144.get(ad, "?"), rho_s=s["rho_s"], rho_cv=s["rho_cv"], Qd=s["Qd"])
    )
S2_TABAN = float(sum(t["rho_s"] ** 2 for t in TABAN_EKSEN))
RHO_TABAN = float(np.sqrt(S2_TABAN))
print(f"\nTABAN (m148 kopyasi): {len(TABAN_EKSEN)} eksen")
print(f"  sqrt(sum rho_s^2) = {RHO_TABAN:.4f}   rho_pred = 1.95* = {TAVAN * RHO_TABAN:.4f}")

# ============================================ 2) ADAY KUMELERI
# --- P1: kabul edilen 136 eksenin en guclu UST_N tanesi
sirali = sorted(TABAN_EKSEN, key=lambda t: -abs(t["rho_s"]))[:UST_N]
P1 = []
for t in sirali:
    xt, xb = kur(t["eksen"])
    if xt is None or xb is None:
        continue
    P1.append((t["eksen"], xt.astype(np.float32), xb.astype(np.float32)))
print(f"P1 tabani: {len(P1)} eksen -> {len(P1) * (len(P1) - 1) // 2} carpim adayi")

# --- P2: m144'un H DISI kapidan gecen eksenleri, ORIJINAL ureticilerinden
ISTENEN = {r["eksen"] for r in GECEN144 if r["aile"] != "H_carpim40"}
URETICILER = [
    ("B_lokasyon", NS["aile_b_lokasyon"]),
    ("C_takvim", NS["aile_c_takvim"]),
    ("D_hava_gecikme", NS["aile_d_hava"]),
    ("E_ufuk", NS["aile_e_ufuk"]),
    ("F_guc_yas", NS["aile_f_guc"]),
    ("G_mentese", NS["aile_g_mentese"]),
]
P2 = []
for aile_ad, uret in URETICILER:
    for ad, fn in uret():
        if ad not in ISTENEN:
            continue
        try:
            xt, xb = fn()
        except Exception as ex:  # noqa: BLE001
            print(f"  ! {ad}: {type(ex).__name__} {ex}")
            continue
        if xt is None or xb is None:
            continue
        if not (np.isfinite(xt).all() and np.isfinite(xb).all()):
            continue
        P2.append((ad, xt.astype(np.float32), xb.astype(np.float32)))
bulunan = {a for a, _, _ in P2}
print(f"P2 tabani: {len(P2)}/{len(ISTENEN)} H-disi eksen yeniden uretildi")
if ISTENEN - bulunan:
    print(f"  uretilemeyen: {sorted(ISTENEN - bulunan)}")
print(f"  -> {len(P2) * (len(P2) - 1) // 2} carpim adayi")
gc.collect()


def carpim_tara(taban, etiket):
    """Taban listesindeki eksenlerin ikili carpimlarini kapilardan gecirir."""
    n_aday = 0
    sb = {}
    gecti = []
    for i, (a1, t1, b1) in enumerate(taban):
        for a2, t2, b2 in taban[i + 1 :]:
            n_aday += 1
            xt, xb = st(t1 * t2), st(b1 * b2)
            if xt is None or xb is None:
                sb["dejenere"] = sb.get("dejenere", 0) + 1
                continue
            if not (np.isfinite(xt).all() and np.isfinite(xb).all()):
                sb["dejenere"] = sb.get("dejenere", 0) + 1
                continue
            s, sebep = olc(xt, xb, ONCEKI)
            if s is None:
                sb[sebep] = sb.get(sebep, 0) + 1
                continue
            gecti.append(
                dict(
                    aile=etiket,
                    eksen=f"N[{a1}]x[{a2}]",
                    p1=a1,
                    p2=a2,
                    rho_s=s["rho_s"],
                    rho_cv=s["rho_cv"],
                    Qd=s["Qd"],
                    z=s["z"],
                )
            )
        if (i + 1) % 10 == 0:
            print(f"    {etiket} {i + 1}/{len(taban)} ... {n_aday} aday, {len(gecti)} gecti")
    ilk = sorted(sb.items(), key=lambda t: -t[1])[:4]
    print(f"  {etiket}: {n_aday} aday -> {len(gecti)} GECTI  [elenme: {ilk}]")
    return gecti, n_aday, sb


print("\nIKINCI NESIL TARAMA (kapilar m144 ile AYNI, 136 eksene dik)")
G1, N1, SB1 = carpim_tara(P1, "P1_136carpim")
gc.collect()
G2, N2, SB2 = carpim_tara(P2, "P2_yeniaile_carpim")
gc.collect()
del P1, P2
gc.collect()

TUM = sorted(G1 + G2, key=lambda d: -(d["rho_s"] ** 2))
print(f"\ntoplam {N1 + N2} aday, {len(TUM)} kapiyi gecti")


# ============================================ 3) DIK EKLEME
def dik_ekle(adaylar, azami):
    onc = list(ONCEKI)
    alar = list(ALAR)
    sec, s2 = [], S2_TABAN
    print(
        f"\n{'eksen':>52s} {'aile':>18s} {'rho_cv':>8s} {'rho_s':>8s} "
        f"{'Q_dik':>6s} {'z':>6s} {'kum.rho':>8s}"
    )
    for k in adaylar:
        if len(sec) >= azami:
            print(f"  ... AZAMI {azami} sinirina dayandi (kesim kapidan gelmedi)")
            break
        # p1/p2 P2'de kur() ile kurulamayabilir -> parca onbelleginden uret
        xt, xb = yeniden(k)
        if xt is None:
            continue
        s, _ = olc(xt, xb, onc)
        if s is None:
            continue
        onc.append(s["birim"])
        alar.append(s["a"])
        s2 += s["rho_s"] ** 2
        sec.append(
            dict(
                eksen=k["eksen"],
                aile=k["aile"],
                rho_s=s["rho_s"],
                rho_cv=s["rho_cv"],
                Qd=s["Qd"],
                z=s["z"],
                kum_rho_s=float(np.sqrt(s2)),
            )
        )
        if len(sec) <= 25 or len(sec) % 10 == 0:
            print(
                f"{k['eksen'][:52]:>52s} {k['aile'][:18]:>18s} {s['rho_cv']:+8.4f} "
                f"{s['rho_s']:+8.4f} {s['Qd']:6.3f} {s['z']:6.1f} {np.sqrt(s2):8.4f}"
            )
    return sec, float(np.sqrt(s2)), alar


# parca onbellegi: ad -> (xt32, xb32); dik ekleme sirasinda yeniden uretim icin
PARCA = {}


def parca(ad):
    if ad in PARCA:
        return PARCA[ad]
    xt, xb = kur(ad)
    if xt is None or xb is None:
        return None
    PARCA[ad] = (xt.astype(np.float32), xb.astype(np.float32))
    return PARCA[ad]


def yeniden(k):
    p1, p2 = parca(k["p1"]), parca(k["p2"])
    if p1 is None or p2 is None:
        return None, None
    return st(p1[0] * p2[0]), st(p1[1] * p2[1])


# P2 eksenleri kur() ile kurulamaz -> ureticilerden gelen vektorleri onbellege koy
for aile_ad, uret in URETICILER:
    for ad, fn in uret():
        if ad not in ISTENEN or ad in PARCA:
            continue
        try:
            xt, xb = fn()
        except Exception:  # noqa: BLE001, S112
            continue
        if xt is None or xb is None:
            continue
        PARCA[ad] = (xt.astype(np.float32), xb.astype(np.float32))

secilen, RHO_YENI, A_TUM = dik_ekle(TUM, AZAMI_YENI)

# ---------------------------------------------- KANIT TABANI (m144 ile ayni)
LR = (V.T @ r_hat) / len(r_hat)


def kanit_tabani(alar):
    A = np.array(alar).T
    GA = A.T @ G @ A
    LA = A.T @ LR
    return float(np.sqrt(max(float(LA @ np.linalg.pinv(GA, rcond=1e-8) @ LA), 0.0)))


NRM = float(np.sqrt(float((r_hat * r_hat).mean())))
K_TABAN, K_YENI = kanit_tabani(ALAR), kanit_tabani(A_TUM)

print()
print("KANIT TABANI ||P_A r_hat||  (tavan ||r_hat||)")
print(f"  ||r_hat||                   = {NRM:.4f}   <- ASILAMAZ TAVAN")
print(f"  taban {len(ALAR):3d} eksen            = {K_TABAN:.4f}")
print(f"  + {len(secilen):3d} ikinci nesil       = {K_YENI:.4f}   ({K_YENI - K_TABAN:+.4f})")

print(f"\nTABAN  {len(TABAN_EKSEN):3d} eksen : sqrt(sum rho_s^2) = {RHO_TABAN:.4f}")
print(f"+ YENI {len(secilen):3d} eksen : sqrt(sum rho_s^2) = {RHO_YENI:.4f}")
print(f"ARTIS: {RHO_YENI - RHO_TABAN:+.4f}  ({100 * (RHO_YENI / RHO_TABAN - 1):+.1f}%)")
print(f"rho_pred = 1.95 * {RHO_YENI:.4f} = {TAVAN * RHO_YENI:.4f}  (taban {TAVAN * RHO_TABAN:.4f})")
for ad, h in [("3. sira", 0.99927), ("2. sira", 0.99614), ("1. sira", 0.99009)]:
    kap = float(np.sqrt(max(MSE_OPT - h * h, 1e-12)))
    print(
        f"  {ad}: gereken rho {kap:.4f}  ->  f_taban = {kap / (TAVAN * RHO_TABAN):.3f}"
        f"   f_yeni = {kap / (TAVAN * RHO_YENI):.3f}"
    )

aile_dagilim = {}
for s in secilen:
    aile_dagilim[s["aile"]] = aile_dagilim.get(s["aile"], 0) + 1

with open(os.path.join(BURA, "n03_ikinci_nesil.json"), "w", encoding="utf-8") as fh:
    json.dump(
        dict(
            taban_eksen=len(TABAN_EKSEN),
            taban_rho_s=RHO_TABAN,
            taban_rho_pred=TAVAN * RHO_TABAN,
            yeni_eksen=len(secilen),
            birlesik_rho_s=RHO_YENI,
            birlesik_rho_pred=TAVAN * RHO_YENI,
            artis_rho_s=RHO_YENI - RHO_TABAN,
            ust_n=UST_N,
            aday_sayisi=dict(P1=N1, P2=N2),
            gecen_sayisi=dict(P1=len(G1), P2=len(G2)),
            elenme=dict(P1=SB1, P2=SB2),
            aile_dagilim=aile_dagilim,
            kanit_tabani=dict(r_hat_normu=NRM, taban=K_TABAN, yeni=K_YENI),
            secilen=secilen,
            kapidan_gecen=[
                dict(
                    aile=g["aile"], eksen=g["eksen"], rho_s=g["rho_s"], rho_cv=g["rho_cv"], z=g["z"]
                )
                for g in TUM[:500]
            ],
            taban_liste=TABAN_EKSEN,
        ),
        fh,
        indent=1,
    )
print("\n-> n03_ikinci_nesil.json yazildi (GONDERIM YOK)")
sys.stdout.flush()
