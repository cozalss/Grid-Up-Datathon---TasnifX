# -*- coding: utf-8 -*-
"""NIHAI HARITA -- tek ekran."""
import os, json
import numpy as np
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
J = lambda n: json.load(open(os.path.join(CIK, n), encoding="utf-8")) if os.path.exists(os.path.join(CIK, n)) else None
K1, K2, K3, K4, K5, K7, K9, K10 = (J("k01_kahin.json"), J("k02_ulasilabilir.json"),
                                   J("k03_birlesik.json"), J("k04_span_disi.json"),
                                   J("k05_gozlenebilir.json"), J("k07_sizintili.json"),
                                   J("k09_soguk.json"), J("k10_teshis.json"))
G = 0.12298 / np.sqrt(1.0013719)
bl = ("yaz25", "guz25", "kis26")
idx = {b: {c["ad"]: c for c in K1[b]["cepler"]} for b in bl}

print("=" * 104)
print("ESIK: ortusme orani >= %.5f (T=0.99310) | kabul %.5f (T=0.99413) | <=> MSE payi >= %%%.3f"
      % (G, 0.11436 / np.sqrt(1.0013719), 100 * G ** 2))
print("=" * 104)
h = "%-32s %6s %7s | %7s %7s %7s | %8s %8s %8s %5s" % (
    "CEP", "sat%", "MSE%", "KAHIN_y", "KAHIN_g", "KAHIN_k", "BAYRAK_y", "BAYRAK_g", "BAYRAK_k", "isrt")
print(h); print("-" * len(h))
for a in [c["ad"] for c in K1["yaz25"]["cepler"]]:
    v = [idx[b].get(a) for b in bl]
    if v[0] is None:
        continue
    bay = [x["oran_bayrak"] if x else float("nan") for x in v]
    ss = "AYNI" if all(not np.isnan(x) for x in bay) and (bay[0] > 0) == (bay[1] > 0) == (bay[2] > 0) else "ZIT"
    print("%-32s %6.2f %7.2f | %7.4f %7.4f %7.4f | %+8.4f %+8.4f %+8.4f %5s" % (
        a[:32], 100 * v[0]["satir_payi"], 100 * v[0]["mse_payi"],
        v[0]["oran_max"], v[1]["oran_max"] if v[1] else np.nan, v[2]["oran_max"] if v[2] else np.nan,
        bay[0], bay[1], bay[2], ss))

print("\nULASILABILIRLIK -- ayni cepler, UYELIK gozlenebilirlerden kestirilerek (blok-disi)")
h2 = "%-38s %8s %9s %9s %9s %8s" % ("YON", "AUC_y", "yaz25", "guz25", "kis26", "KAHIN_y")
print(h2); print("-" * len(h2))
B = K2["bloklar"]


def g(b, k, f="oran"):
    v = B[b].get(k)
    return v[f] if isinstance(v, dict) else v


satir = [("sifir bayragi (tahmini uyelik)", "B_sifir_auc", "B_sifir_sert_bayrak", "B_sifir_KAHIN_bayrak"),
         ("olu trafo bayragi (tahmini)", "C_olu_auc", "C_olu_sert_bayrak", "C_olu_KAHIN_bayrak"),
         ("buyuk asagi kacirma r<-2 (tahmini)", "D_asagi_auc", "D_asagi_bayrak_e0.5", "D_asagi_KAHIN_bayrak")]
for ad, ak, bk, kk in satir:
    print("%-38s %8.4f %+9.4f %+9.4f %+9.4f %+8.4f" % (
        ad, B["yaz25"][ak], g("yaz25", bk), g("guz25", bk), g("kis26", bk), g("yaz25", kk)))
print("%-38s %8.4f %+9.4f %+9.4f %+9.4f %+8.4f" % (
    "soguk sifir/pozitif ayrimi (tahmini)", K9["yaz25"]["soguk_auc_sifir"],
    K9["yaz25"]["yumusak_q_soguk"]["oran"], K9["guz25"]["yumusak_q_soguk"]["oran"],
    K9["kis26"]["yumusak_q_soguk"]["oran"], K9["yaz25"]["KAHIN soguk +-1 (sifir/pozitif ayrimi)"]["oran"]))
print("%-38s %8s %+9.4f %+9.4f %+9.4f %+8.4f" % (
    "TUM gozlenebilirler: artik regresyonu", "-",
    g("yaz25", "A_artik_regresyon"), g("guz25", "A_artik_regresyon"), g("kis26", "A_artik_regresyon"), 1.0))
print("%-38s %8s %+9.4f %+9.4f %+9.4f" % (
    "   ... sabit(seviye) cikarilmis", "-",
    g("yaz25", "A_artik_regresyon_merkezli"), g("guz25", "A_artik_regresyon_merkezli"),
    g("kis26", "A_artik_regresyon_merkezli")))
print("%-38s %8s %+9.4f %+9.4f %+9.4f" % (
    "   SABIT yon tek basina (seviye)", "-",
    g("yaz25", "A0_sabit"), g("guz25", "A0_sabit"), g("kis26", "A0_sabit")))

if K7:
    print("\nSIZINTILI UST SINIR (yaz25 ici trafo-gruplu; MEVSIM BILINIYOR -- ulasilamaz)")
    for k in ("SIZINTILI artik regresyonu", "SIZINTILI sifir olasiligi (yumusak)",
              "SIZINTILI sifir bayrak e=0.5"):
        v = K7[k]
        print("   %-42s oran=%+.4f  t=%.1f" % (k, v["oran"], v["t"]))
    print("   %-42s %.4f" % ("blok-ici sifir AUC", K7["auc_sifir"]))

if K10:
    print("\nA YONU TESHISI (yaz25): takvim-yapaylik altuzayina dik yapinca ne kalir?")
    for k, v in K10.items():
        if isinstance(v, dict) and "oran" in v:
            print("   %-42s oran=%+.4f t=%5.1f  kalan_norm=%s" % (
                k, v["oran"], v["t"], ("%.3f" % v["kalan_norm_payi"]) if "kalan_norm_payi" in v else "-"))
    print("   CEP KATKISI:", {k: round(x, 4) for k, x in K10["cep_katkisi"].items()})
    print("   TEST span-disi carpan (yenilik payi) = %.4f" % K10["TEST_span_disi_carpan"])

print("\nSPAN-DISI CARPAN (yenilik payi; 30 gonderim + 3 dik yon):")
for x in K4:
    print("   %-34s %.4f" % (x["ad"][:34], x["carpan"]))

print("\nBIRLESIK KAHIN TAVANLAR (yaz25) -- birlesim / cok-bayrak izdusumu:")
for k in K3["yaz25"]["birlesim_kahin"]:
    print("   %-34s %.4f / %.4f" % (k, K3["yaz25"]["birlesim_kahin"][k],
                                    K3["yaz25"]["birlesim_cok_bayrak"][k]))
print("   9 bayragin hepsi (izdusum): %.4f" % K3["yaz25"]["dokuz_bayrak_hepsi"])
