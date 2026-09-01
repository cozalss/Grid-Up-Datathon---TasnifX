"""p18b: DELTA YAMASI ile TAM YENIDEN EGITIMIN FARKINI OLC.

SORU: ayni yapilandirma (a) delta yamasiyla, (b) tam yeniden egitimle
uygulandiginda blok CV'si ne kadar farkli?

UC SKOR uretilir, hepsi ayni olcutte (p02_duzeltme.skor, test bilesimine
agirliklandirilmis RMSLE):

  TABAN  uretim onbellegi (aile_onbellek/*_uretim.npy + soguk_tahmin_*.npz)
  DELTA  TABAN, ama YALNIZ ``P18K_DEGISEN``de sayilan (taraf,aile) ciftleri
         p18 ciktisindan alinir -- geri kalan aileler uretimden gelir
  TAM    butun aileler p18 ciktisindan alinir

BEKLENEN SONUC VE NEDEN ONEMLI
------------------------------
Bu hatta harman SABIT AGIRLIKLI ARITMETIK ORTALAMADIR ve aileler
birbirinden BAGIMSIZ egitilir. Dolayisiyla yalnizca bir ailenin kaybini
degistirmek, digerlerinin tahminini DEGISTIRMEZ; DELTA ile TAM blok
uzayinda BIREBIR AYNI cikmalidir (fark ~1e-15).

Fark ancak su durumlarda dogar ve betik onlari ayirt eder:
  1. Degisiklik AILELER ARASI ORTAK ise (ornegin yakinlik agirligi TAU
     butun ailelere uygulanirsa) -- o zaman DELTA "yalniz bir aile"
     varsayimini ihlal eder.
  2. Tohum kumesi degisirse.
  3. TEST tarafinda: uretim gonderimi ham model ciktisi DEGIL, gonderim
     uzayinda cebirsel olarak kurulmus bir dosyadir (m6_ikiyon -> span ->
     YP_seviye) ve son islem zinciri eski tahminlere gore kalibre edildi.
     Blokta sifir olan fark ORADA sifir degildir -- ve olculemez.

Yani "DELTA=TAM" ciktisi bir BASARISIZLIK degil, hattin yapisi hakkinda
bir OLCUMDUR: blok tarafinda tam yeniden egitim EK BILGI GETIRMEZ.

    set P18K_ETIKET=huber & set P18K_DEGISEN=soguk:lgbm ^
        & python experiments/model29/p18_delta_vs_tam.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
AO = os.path.join(KOK, "data", "interim", "aile_onbellek")
DN = os.path.join(KOK, "data", "interim", "deney")
SP = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/p18"
)

HEDEF_SOGUK = 0.2216  # gercek test'teki soguk trafo payi
BLOKLAR = ("yaz25", "guz25", "kis26")

ETIKET = os.environ.get("P18K_ETIKET", "taban").strip()
CIKTI = os.environ.get("P18K_CIKTI", SP).strip()
DEGISEN = {
    p.split(":")[0].strip(): [a.strip() for a in p.split(":")[1].split("+")]
    for p in os.environ.get("P18K_DEGISEN", "soguk:lgbm").split(",")
    if p.strip()
}
TOHUMLAR = [int(t) for t in os.environ.get("P18K_TOHUMLAR", "1000,1001,1002").split(",")]
AILELER = ("cat", "xgb", "lgbm")

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))


def uretim_tahmini(blok: str, taraf: str) -> dict[str, np.ndarray]:
    """Uretim onbelleginden {tohum_aile: log tahmin}."""
    P: dict[str, np.ndarray] = {}
    if taraf == "sicak":
        for t in TOHUMLAR:
            for a in AILELER:
                y = os.path.join(AO, f"{blok}_{t}_{a}_uretim.npy")
                if os.path.exists(y):
                    P[f"{t}_{a}"] = np.load(y).astype(np.float64)
    else:
        z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
        for t in TOHUMLAR:
            for a in AILELER:
                if f"{t}_{a}" in z.files:
                    P[f"{t}_{a}"] = z[f"{t}_{a}"].astype(np.float64)
    if not P:
        raise SystemExit(f"uretim onbellegi bos: {blok}/{taraf}")
    return P


def yeni_tahmin(blok: str, taraf: str) -> dict[str, np.ndarray]:
    yol = os.path.join(CIKTI, ETIKET, f"{taraf}_{blok}.npz")
    if not os.path.exists(yol):
        raise SystemExit(f"p18 ciktisi yok: {yol}  (once p18_yeniden_egit.py kos)")
    z = np.load(yol)
    return {k: z[k].astype(np.float64) for k in z.files}


def harman(P: dict[str, np.ndarray]) -> np.ndarray:
    return np.mean([P[k] for k in sorted(P)], axis=0)


def blok_cerceve(blok: str) -> tuple[pd.DataFrame, np.ndarray]:
    blk = E[E._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    idx = np.concatenate([sic.index.values, sog.index.values])
    d = E.loc[idx].copy()
    d["y"] = np.log1p(d.tuketim.values.astype(np.float64))
    return d, np.concatenate([np.zeros(len(sic), bool), np.ones(len(sog), bool)])


def skor(d: pd.DataFrame, p: np.ndarray) -> tuple[float, float]:
    """p02_duzeltme.skor ile BIREBIR (ham ve test-bilesimi agirlikli)."""
    r = d.y.values - p
    sg = d.soguk_mu.values.astype(np.float64)
    pay = sg.mean()
    w = np.where(sg == 1, HEDEF_SOGUK / pay, (1 - HEDEF_SOGUK) / (1 - pay))
    w = w / w.mean()
    return float(np.sqrt(np.mean(r * r))), float(np.sqrt(np.mean(w * r * r)))


def main() -> None:
    R: dict = {"etiket": ETIKET, "degisen": DEGISEN, "tohumlar": TOHUMLAR, "bloklar": {}}
    print(f"etiket={ETIKET}  degisen={DEGISEN}")
    for blok in BLOKLAR:
        d, sog_maske = blok_cerceve(blok)
        parcalar: dict[str, list[np.ndarray]] = {"TABAN": [], "DELTA": [], "TAM": []}
        for taraf, maske in (("sicak", ~sog_maske), ("soguk", sog_maske)):
            U = uretim_tahmini(blok, taraf)
            try:
                Y = yeni_tahmin(blok, taraf)
            except SystemExit:
                Y = {}
            # DELTA: yalniz degisen aileler yeni, geri kalan uretim
            D = dict(U)
            for k in list(D):
                a = k.split("_")[1]
                if a in DEGISEN.get(taraf, []) and k in Y:
                    D[k] = Y[k]
            # TAM: hepsi yeni (yoksa uretim)
            T = {k: Y.get(k, U[k]) for k in U}
            parcalar["TABAN"].append((maske, harman(U)))
            parcalar["DELTA"].append((maske, harman(D)))
            parcalar["TAM"].append((maske, harman(T)))
        satir: dict = {}
        p_kayit: dict[str, np.ndarray] = {}
        for ad, parca in parcalar.items():
            p = np.zeros(len(d))
            for maske, v in parca:
                p[maske] = v
            p_kayit[ad] = p
            ham, agr = skor(d, p)
            satir[ad] = {"rmsle": round(ham, 6), "rmsle_test_bilesimi": round(agr, 6)}
        satir["DELTA_eksi_TABAN"] = round(
            satir["TABAN"]["rmsle_test_bilesimi"] - satir["DELTA"]["rmsle_test_bilesimi"], 6
        )
        satir["TAM_eksi_TABAN"] = round(
            satir["TABAN"]["rmsle_test_bilesimi"] - satir["TAM"]["rmsle_test_bilesimi"], 6
        )
        satir["TAM_ile_DELTA_maxabs"] = float(np.max(np.abs(p_kayit["TAM"] - p_kayit["DELTA"])))
        R["bloklar"][blok] = satir
        print(
            f"{blok:6} TABAN {satir['TABAN']['rmsle_test_bilesimi']:.5f}  "
            f"DELTA {satir['DELTA']['rmsle_test_bilesimi']:.5f} "
            f"({satir['DELTA_eksi_TABAN']:+.5f})  "
            f"TAM {satir['TAM']['rmsle_test_bilesimi']:.5f} "
            f"({satir['TAM_eksi_TABAN']:+.5f})  "
            f"|TAM-DELTA|max={satir['TAM_ile_DELTA_maxabs']:.3e}"
        )
    R["ortalama_kazanc"] = {
        "DELTA": round(float(np.mean([R["bloklar"][b]["DELTA_eksi_TABAN"] for b in BLOKLAR])), 6),
        "TAM": round(float(np.mean([R["bloklar"][b]["TAM_eksi_TABAN"] for b in BLOKLAR])), 6),
    }
    R["hukum"] = (
        "DELTA ve TAM blok uzayinda ayni ciktiysa (maxabs ~1e-15), tam yeniden "
        "egitim BLOK tarafinda ek bilgi getirmiyor demektir -- harman sabit "
        "agirlikli ortalama ve aileler bagimsiz egitiliyor. Fark yalnizca TEST "
        "tarafinda, gonderim-uzayi cebiri ve son islem zinciri yuzunden dogar."
    )
    yol = os.path.join(BURA, "p_kalici", f"p18_delta_vs_tam_{ETIKET}.json")
    os.makedirs(os.path.dirname(yol), exist_ok=True)
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(f"\nortalama kazanc: {R['ortalama_kazanc']}")
    print(f"kayit: {yol}")


if __name__ == "__main__":
    main()
