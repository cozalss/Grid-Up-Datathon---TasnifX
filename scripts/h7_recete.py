"""H7-c -- TASINABILIR SABITI turet ve --lb-kalibre'nin UYGULAMA BICIMINI sina.

c* = kor * sigma_gercek / sigma_model  =>  c* TABANA baglidir ama
     c* x sigma_model = kor x sigma_gercek  TABANDAN BAGIMSIZDIR.
Yani tasinabilir sabit ULASILACAK GENLIK'tir, olcek degil.

Ayrica: son_islem_gunolcek.py `c = 1 + k(c_formul - 1)` uyguluyor (AFFIN).
c* dogrudan sigma_gercek ile ORANTILI oldugu icin dogru bicim CARPIMSAL
(c = k * c_formul). Iki bicim LB'nin cozdugu 1,335'e karsi sinanir.
"""

from __future__ import annotations

import json
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
CIK = KOK / "reports" / "h7_cstar"

ozet = json.loads((CIK / "ozet.json").read_text(encoding="utf-8"))
dog = json.loads((CIK / "dogrudan.json").read_text(encoding="utf-8"))
T = {r["taban"]: r for r in ozet["tabanlar"]}

sg = ozet["sigma_gercek"]
p_s, p_c = ozet["p_sicak"], ozet["p_soguk"]
v50, v55, v66, v67 = (
    T["tuketim_v50_nihai30.csv"],
    T["tuketim_v55_gunolcek.csv"],
    T["tuketim_v66_c1335.csv"],
    T["tuketim_v67_c1335_olay.csv"],
)

print("=== 1. NOMINAL vs ULASILAN OLCEK (kirpma kaybi) ===")
for ad, r, nom in [
    ("v55", v55, 1.492),
    ("v66", v66, 1.335),
    ("v57", T.get("tuketim_v57_gunolcek175.csv"), 1.75),
]:
    if r is None:
        continue
    ula = r["sigma_sicak"] / v50["sigma_sicak"]
    print(f"  {ad}: istenen c={nom:.3f}  ULASILAN {ula:.4f}  kayip {(ula / nom - 1) * 100:+.2f}%")

print("\n=== 2. TASINABILIR SABIT: hedef ULASILAN gun-ekseni genligi ===")
m_y, m67 = dog["m_yildiz"], dog["m67"]
# v67'nin ulasilan sigma'si optimum konumun m67 katinda; optimum genlik:
sig_opt = v67["sigma_sicak"] * (m_y / m67)
kor = v67["kor_sicak"]
sig_gercek_2026 = sig_opt / kor
print(f"  v67 ulasilan sicak gun sigma      = {v67['sigma_sicak']:.4f}")
print(f"  LB-optimum ULASILAN sigma  S*     = {sig_opt:.4f}   <== TASINABILIR SABIT")
print(f"  kor (taban-bagimsiz, 0,920-0,922) = {kor:.4f}")
print(f"  ima edilen 2026 GERCEK gun sigma  = {sig_gercek_2026:.4f}")
print(f"  2025 olculen gercek sigma         = {sg:.4f}")
print(f"  2026/2025 genlik orani            = {sig_gercek_2026 / sg:.4f}  <== LB kalibresi")

k_dogru = sig_gercek_2026 / sg
print("\n=== 3. --lb-kalibre: AFFIN mi CARPIMSAL mi? (LB cozumu 1,3242 ulasilan) ===")
cf = v50["c_formul_sicak"]
ula_hedef = sig_opt / v50["sigma_sicak"]
print(f"  v50 formul c = {cf:.4f} (nominal) ;  LB-optimum ULASILAN c = {ula_hedef:.4f}")
for ad, deger in [
    ("AFFIN  1+k(c-1), k=0,893", 1.0 + 0.893 * (cf - 1.0)),
    ("AFFIN  1+k(c-1), k=%.4f" % k_dogru, 1.0 + k_dogru * (cf - 1.0)),
    ("CARPIMSAL k*c,   k=0,893", 0.893 * cf),
    ("CARPIMSAL k*c,   k=%.4f" % k_dogru, k_dogru * cf),
]:
    ula = deger * (v55["sigma_sicak"] / v50["sigma_sicak"]) / 1.492  # nominal->ulasilan shrink
    hata = ula - ula_hedef
    B = dog["Q55"] / (v55["sigma_sicak"] / v50["sigma_sicak"] - 1.0) ** 2  # birim (c-1)^2 basina
    print(
        f"  {ad:34s} -> nominal {deger:.4f}  ulasilan {ula:.4f}  "
        f"hata {hata:+.4f}  dMSE maliyeti {B * hata**2:+.6f}"
    )

print("\n=== 4. SOGUK GUN EKSENI -- duzeltilmis recete (YENI HIPOTEZ) ===")
kor_c = v67["kor_soguk"]
sig_c = v67["sigma_soguk"]
rms_c = v67["rms_satir_soguk"]
B_c = p_c * rms_c**2
c_ham = kor_c * sg / sig_c
c_kal = k_dogru * c_ham  # CARPIMSAL kalibre (dogru bicim)
c_affin_hatali = 1.0 + 0.893 * (c_ham - 1.0)  # v58'de KULLANILAN (hatali bicim)
c_panel = 1.337  # kompozisyon-temiz 17-trafo paneli
print(f"  v67 soguk: sigma {sig_c:.4f}  satir-RMS {rms_c:.4f}  kor {kor_c:+.4f}  B_soguk {B_c:.6f}")
print(f"  ham formul c        = {c_ham:.4f}")
print(f"  CARPIMSAL kalibreli = {c_kal:.4f}   <== ONERILEN")
print(f"  v58'de kullanilan   = {c_affin_hatali:.4f} (affin, ulasilan 1,411) -- ASIRI")
print(f"  kompozisyon-temiz panel bagimsiz kestirim = {c_panel:.3f}")
print(f"\n  {'c_soguk':>9} {'dMSE (c*=%.3f ise)' % c_kal:>22}")
for c in [1.00, 1.10, 1.20, 1.30, c_kal, 1.40, 1.411, 1.45, 1.507]:
    d = B_c * ((c - 1.0) ** 2 - 2 * (c_kal - 1.0) * (c - 1.0))
    print(f"  {c:9.4f} {d:+22.6f}")
print(f"\n  optimumda beklenen kazanc = {-B_c * (c_kal - 1.0) ** 2:+.6f} MSE")
print(
    f"  (v58'in 1,411'i uygulansaydi: {B_c * ((1.411 - 1) ** 2 - 2 * (c_kal - 1) * (1.411 - 1)):+.6f})"
)

print("\n=== 5. RECETE: sampiyon degisirse c* nasil yeniden turetilir ===")
print(f"""
  1) yeni taban icin SICAK satirlarda gun etkisi b_g olcul (trafo etkisi
     ONCE cikarilir), sigma_model = b_g.std()
  2) c_nominal = S* / sigma_model  ,  S* = {sig_opt:.4f}   (LB ile kilitli)
  3) betigi --c c_nominal ile calistir; yazdirdigi `uygulanan olcek` x
     sigma_model = {sig_opt:.4f} +- %1 OLMALI. Degilse c_nominal'i
     {sig_opt:.4f}/(uygulanan olcek x sigma_model) ile carp ve TEKRARLA.
  4) formul yolu (LB yoksa): c = {k_dogru:.4f} x kor x {sg:.4f} / sigma_model
     -- CARPIMSAL kalibre; `--lb-kalibre` bayragi AFFIN uyguluyor, KULLANMA.
""")

json.dump(
    {
        "S_yildiz_ulasilan_sicak_gun_sigma": sig_opt,
        "kor_taban_bagimsiz": kor,
        "sigma_gercek_2025": sg,
        "sigma_gercek_2026_ima": sig_gercek_2026,
        "lb_kalibre_carpimsal": k_dogru,
        "soguk_c_onerilen": c_kal,
        "soguk_B": B_c,
        "soguk_beklenen_dMSE": -B_c * (c_kal - 1.0) ** 2,
    },
    (CIK / "recete.json").open("w", encoding="utf-8"),
    ensure_ascii=False,
    indent=2,
)
print(f"yazildi: {CIK / 'recete.json'}")
