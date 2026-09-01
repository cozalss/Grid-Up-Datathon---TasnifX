"""p36-h: KUTLE-KESINLIGI ve 0.1230'a ulasmak icin gereken iyilesme."""
import os, sys, pickle, json
import numpy as np, pandas as pd
KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
sys.path.insert(0, os.path.join(KOK, "experiments/model29/p_kalici"))
from p27_ortak import agirlik, blok  # noqa
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"
D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in ("yaz25","guz25","kis26")}
W = {b: agirlik(D[b]) for b in D}
Q = pickle.load(open(os.path.join(CIK, "p36_q.pkl"), "rb"))
GEREK = 0.12298
rows = []
for b in D:
    d, w = D[b], W[b]; p, r = d.p.values, d.r.values
    z = d.tuketim.values <= 0; q = Q[b]["q"]; sw = w.sum()
    for e in (0.1, 0.3, 0.5, 0.7, 0.9):
        f = q > e
        Mdp = float(np.sum(w[f & z] * p[f & z]**2) / sw)
        Mfp = float(np.sum(w[f & ~z] * p[f & ~z]**2) / sw)
        Y   = float(np.sum(w[f & ~z] * p[f & ~z] * r[f & ~z]) / sw)   # YP surukleme payi
        # YP kutlesinin (1-phi) kadari ELENSE (YP'ler duzgun kaldirilsa) rho ne olur
        gerek_phi = np.nan
        for phi in np.linspace(0, 1, 2001):        # phi = elenen YP orani
            rho = (Mdp - (1-phi)*Y) / np.sqrt(max(Mdp + (1-phi)*Mfp, 1e-12))
            if rho >= GEREK: gerek_phi = phi; break
        rows.append(dict(blok=b, esik=e,
            satir_kesinlik=round(float(np.sum(w[f]*z[f])/np.sum(w[f])), 4),
            KUTLE_kesinligi=round(Mdp/(Mdp+Mfp), 4),
            DP_kutle=round(Mdp, 5), YP_kutle=round(Mfp, 5),
            rho_net=round((Mdp - Y)/np.sqrt(Mdp+Mfp), 5),
            rho_tavan_YP_sifir=round(Mdp/np.sqrt(Mdp), 5),
            gereken_YP_eleme_orani=(round(float(gerek_phi),3) if gerek_phi==gerek_phi
                                    else "ULASILAMAZ(phi=1'de bile)")))
T = pd.DataFrame(rows)
print(T.to_string(index=False))
print("\nNOT: gereken_YP_eleme_orani = mevcut yanlis pozitiflerin yuzde kaci")
print("     tamamen elenirse rho >= 0.12298 olur.  'ULASILAMAZ' = YP'lerin")
print("     TAMAMI elense bile yakalanan kutle yetmiyor.")
json.dump(rows, open(os.path.join(CIK,"p36_h.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
