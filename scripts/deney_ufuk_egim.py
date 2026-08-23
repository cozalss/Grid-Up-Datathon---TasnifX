"""SICAK TAHMINLERE UFKA GORE AFIN KALIBRASYON -- capraz blok dogrulamali.

NEDEN
-----
Sicakta GLOBAL afin egim olculdu ve ~1 cikti (yaz25 +1,0365 / guz25 +1,0628
/ kis26 +0,9671), yani global buzme degersiz. Ama egim UFKA GORE hic
ayristirilmadi, oysa modelin bilgisi ufukla azaliyor: ``t_log_son14``
ufuk 1'de neredeyse kesin, ufuk 122'de tahmin.

Bilgi azaldikca tahmin daha gurultulu olur ve L2 altinda optimal egim
1'in ALTINA duser. Uzun ufuk diliminde 0,10'luk bir egim sapmasi bile
Var(r_tahmin) ~ 1,5 ile ~0,004 MSE eder.

REDDEDILMIS "ufuk yanliligi duzeltmesi"nden farki onemli: o TOPLAMSAL bir
kaydirmaydi ve bloga ozgu bir sabiti tasimaya calisiyordu (mevsim kurgusu
yuzunden tasinmadi -- capraz blokta kosegen disi HER hucre pozitifti).
Egim bir VARYANS ORANI: blok yanliligindan bagimsiz bir nesne.

KURGU -- capraz blok, sizintisiz
    hedef blok C icin (c, b) katsayilari A ve B bloklarinda uydurulur,
    C'ye uygulanir. Uc blogun UCUNDE de kazanmiyorsa REDDEDILIR.

    r' = c(ufuk) + b(ufuk) * r        r = log1p(tahmin) - log1p(guc)

Onbelleklenmis sicak tahminler kullanilir (``deney_sicak_agirlik.py``
uretti), yani fit YOK -- saniyeler.

    python scripts/deney_ufuk_egim.py
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
KAYIT = KOK / "experiments" / "ufuk_egim.jsonl"

#: Ufuk dilim kenarlari (gun). Test ufku 1..122.
KENARLAR = (0, 15, 30, 60, 90, 10_000)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("SICAK: ufka gore afin kalibrasyon  --  capraz blok")
    print("=" * 92)

    if not ONBELLEK.exists():
        raise RuntimeError(f"onbellek yok: {ONBELLEK} -- once deney_sicak_agirlik.py")
    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    z = np.load(ONBELLEK)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    veri: dict[str, dict[str, np.ndarray]] = {}
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = parcalar[b.ad]
        pay = sum(AGIRLIK)
        loglar = [
            sum(AGIRLIK[i] * z[f"{b.ad}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in di.TOHUMLAR
        ]
        log_t = np.mean(loglar, axis=0)  # 3 tohum torbalanmis (gonderim gibi)
        dg = dogrulama[~soguk]
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        veri[b.ad] = {
            "r": log_t - lg,
            "gercek": np.log1p(gercek[~soguk]) - lg,
            "lg": lg,
            "y": gercek[~soguk],
            "ufuk": dg["ufuk_gun"].to_numpy(dtype="float64"),
        }
        print(f"  {b.ad}: {len(lg):,} sicak satir  ufuk {dg['ufuk_gun'].min():.0f}"
              f"-{dg['ufuk_gun'].max():.0f}")
    te_ufuk = test["ufuk_gun"].to_numpy(dtype="float64")
    print(f"  TEST ufku {te_ufuk.min():.0f}-{te_ufuk.max():.0f}")

    def dilim(u: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(np.array(KENARLAR[1:-1]), u, side="right"), 0,
                       len(KENARLAR) - 2)

    print(f"\n  {'blok':7}{'dilim':>7}{'n':>9}{'egim':>8}{'kesme':>9}")
    kayitlar = []
    for b in tm.BLOKLAR:
        kaynak = [o.ad for o in tm.BLOKLAR if o.ad != b.ad]
        r_k = np.concatenate([veri[k]["r"] for k in kaynak])
        g_k = np.concatenate([veri[k]["gercek"] for k in kaynak])
        d_k = dilim(np.concatenate([veri[k]["ufuk"] for k in kaynak]))
        hedef = veri[b.ad]
        d_h = dilim(hedef["ufuk"])
        yeni = hedef["r"].copy()
        for j in range(len(KENARLAR) - 1):
            mk = d_k == j
            mh = d_h == j
            if mk.sum() < 5_000 or not mh.any():
                continue
            egim, kesme = np.polyfit(r_k[mk], g_k[mk], 1)
            yeni[mh] = kesme + egim * hedef["r"][mh]
            print(f"  {b.ad:7}{j:7d}{int(mk.sum()):9,}{egim:8.4f}{kesme:+9.4f}")
        onceki = tm.rmsle(hedef["y"], np.clip(np.expm1(hedef["r"] + hedef["lg"]), 0.0, None))
        sonraki = tm.rmsle(hedef["y"], np.clip(np.expm1(yeni + hedef["lg"]), 0.0, None))
        print(f"     -> {b.ad} sicak {onceki:.5f} -> {sonraki:.5f}   {sonraki - onceki:+.5f}")
        kayitlar.append({"blok": b.ad, "once": onceki, "sonra": sonraki,
                         "fark": sonraki - onceki})

    kazanan = sum(1 for k in kayitlar if k["fark"] < 0)
    ort = float(np.mean([k["fark"] for k in kayitlar]))
    hukum = "AL" if kazanan == len(kayitlar) and ort < -0.001 else "REDDET"
    print(f"\n  {kazanan}/{len(kayitlar)} blokta kazanc, ortalama {ort:+.5f}   HUKUM: {hukum}")
    print(f"  genel skora tahmini etki {ort * 0.528:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
