"""SICAK: OLU KUYRUKLU trafolara UFKA GORE AZALAN carpimsal buzme.

FIKIR (bir ajan taramasindan geldi, burada BAGIMSIZ dogrulaniyor)
------------------------------------------------------------------
Egitim satirlarinin %4,7'si tam SIFIR ve bunlar toplam kare hatanin yarisindan
fazlasini tasiyor. Sifirlar gun kesintisi degil TRAFO DURUMU: kismi sifirlarin
%47,7'si 120 gunden uzun kosularda. Model bunu zaten biliyor (kuyruk>=7 ->
P(sifir) 0,86-0,95) ve sert sifirlama her esikte kaybettiriyor.

Ama kesim aninda kuyrugu OLU olan bir trafo icin, tahmin ufku kisaysa hala
olu olma olasiligi yuksek; ufuk uzadikca dirilme olasiligi artiyor. Bu yuzden
SABIT degil UFKA GORE AZALAN bir buzme onerildi:

    alfa(ufuk) = 1 - delta * (1 - ufuk / 122)
    log1p(tahmin)' = alfa * log1p(tahmin)     (yalniz olu kuyruklu satirlarda)

ufuk=1'de en guclu (alfa ~ 1-delta), ufuk=122'de kapaniyor (alfa=1).

KRITIK UYARI (ajanin kendi notu): kis26 dirilme riskini ~3 KAT kucuk
gosteriyor -- 4. ayda P(sifir kalma) yaz25 0,657 / guz25 0,658 / kis26 0,885.
Test satirlarinin %58'i son iki ayda. Yani YALNIZ kis26'ya bakarak alinan
duz bir sifir kurali testte isaret degistirir. Bu yuzden hukum UC BLOKTA
birden aranir ve 3/3 sarti konur.

Onbelleklenmis sicak tahminler kullanilir (fit yok).

HUKUM: REDDET -- olculen kazanc SINIR AGININ YOKLUGUNDAN geliyor
----------------------------------------------------------------
Onbellek harmani cat/xgb/lgbm; uretim ise buna %21,9 agirlikla sinir agini
ekliyor ve agin ayri bir SIFIR BASLIGI var. Cebir:

    uretim_log = 0,781 * onbellek_log + 0,219 * ag_log

Ag olu satirlarda ~0 verdigi icin uretim ZATEN ~0,78'lik sabit bir buzme
uyguluyor. Ustelik ufka gore azalan yapiyi da zaten uretiyor -- uretimin
TEST olu satirlarindaki log1p tahmini:

    ufuk   1- 30   0,2391        ufuk  61- 90   0,3995
    ufuk  31- 60   0,2854        ufuk  91-122   0,5896

Yani onerilen duzeltmenin eklemek istedigi sey zaten var. Onbellekte
olculen optimal SABIT alfa ise:

    yaz25 0,9755   guz25 0,7381   kis26 0,8257

Test'in mevsimsel ikizi yaz25'te 0,976 -- neredeyse hic buzme gerekmiyor.
Uretimin 0,78'i uzerine delta=0,30 (ortalama alfa 0,85) eklemek birlesik
0,66 ederdi: ASIRI BUZME. ALINMADI.

Olcum yine de kayitli: onbellek harmani tek basina kullanilacak olsaydi
esik=3, delta=0,20-0,30 uc blokta da kazandiriyordu.

    python scripts/deney_olu_kuyruk.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
KAYIT = KOK / "experiments" / "olu_kuyruk.jsonl"
DELTALAR = (0.0, 0.10, 0.20, 0.30, 0.40)
#: Olu kuyruk esikleri: kesimde son N gunun tamami sifir.
ESIKLER = (3, 7, 14)
UFUK_ENB = 122.0


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print("SICAK: olu kuyruklu satirlarda ufka gore azalan buzme")
    print("=" * 96)

    if not ONBELLEK.exists():
        raise RuntimeError(f"onbellek yok: {ONBELLEK}")
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(ONBELLEK)

    print(f"\n  {'blok':7}{'sicak satir':>13}{'kuyruk>=7':>11}{'pay %':>8}{'kare hata payi %':>18}")
    veri = {}
    for b in tm.BLOKLAR:
        _, dog, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        dg = dog[~soguk]
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_t = np.mean(loglar, axis=0)
        y = gercek[~soguk]
        kuyruk = dg["t_kuyruk_sifir"].to_numpy(dtype="float64")
        ufuk = dg["ufuk_gun"].to_numpy(dtype="float64")
        hata = (log_t - np.log1p(y)) ** 2
        m7 = np.nan_to_num(kuyruk, nan=0.0) >= 7
        veri[b.ad] = {"log_t": log_t, "y": y, "lg": lg, "kuyruk": kuyruk, "ufuk": ufuk}
        print(
            f"  {b.ad:7}{len(y):13,}{int(m7.sum()):11,}{100 * m7.mean():8.2f}"
            f"{100 * hata[m7].sum() / hata.sum():18.2f}"
        )

    print("\n  ESLENIK ETKI (uc blok, uretim harmani, 3 tohum torbalanmis)")
    kayitlar = []
    for esik in ESIKLER:
        print(f"\n  --- olu kuyruk esigi: son {esik} gun tamamen sifir ---")
        print(
            f"  {'delta':>7}"
            + "".join(f"{b.ad:>12}" for b in tm.BLOKLAR)
            + f"{'kazanan':>10}{'ortalama':>11}"
        )
        for delta in DELTALAR:
            farklar = []
            for b in tm.BLOKLAR:
                v = veri[b.ad]
                m = np.nan_to_num(v["kuyruk"], nan=0.0) >= esik
                alfa = np.ones(len(m))
                alfa[m] = 1.0 - delta * (1.0 - np.clip(v["ufuk"][m], 0, UFUK_ENB) / UFUK_ENB)
                yeni = alfa * v["log_t"]
                onceki = tm.rmsle(v["y"], np.clip(np.expm1(v["log_t"]), 0.0, None))
                sonraki = tm.rmsle(v["y"], np.clip(np.expm1(yeni), 0.0, None))
                farklar.append(sonraki - onceki)
            kazanan = sum(1 for f in farklar if f < 0)
            print(
                f"  {delta:7.2f}"
                + "".join(f"{f:+12.5f}" for f in farklar)
                + f"{kazanan:>7}/3{np.mean(farklar):+11.5f}"
            )
            kayitlar.append(
                {
                    "esik": esik,
                    "delta": delta,
                    "bloklar": [float(f) for f in farklar],
                    "kazanan": kazanan,
                    "ortalama": float(np.mean(farklar)),
                }
            )

    uygun = [k for k in kayitlar if k["kazanan"] == 3 and k["delta"] > 0]
    if uygun:
        en = min(uygun, key=lambda k: k["ortalama"])
        print(
            f"\n  3/3 KAZANAN VAR: esik={en['esik']} delta={en['delta']:.2f}"
            f"  ortalama {en['ortalama']:+.5f}"
        )
        print(f"  genel skora tahmini etki {en['ortalama'] * 0.528:+.5f}")
        print("  HUKUM: AL" if en["ortalama"] < -0.001 else "  HUKUM: esik alti")
    else:
        print("\n  3/3 kazanan YOK -> HUKUM: REDDET")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
