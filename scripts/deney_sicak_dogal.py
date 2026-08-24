"""SICAK UZMAN: dogal soguk satirlar egitimden CIKARILSIN mi?

NEDEN
-----
Sicak uzman yalnizca gecmisi OLAN satirlara hizmet ediyor, ama egitim
setinin %13,4'u dogal olarak soguk (o trafonun o tarihte hic gecmisi yok).
Uzerine yapay maske 0,15 gelince soguk-benzeri pay ~%26,4'e cikiyor.

Iki somut bedel:
  1) Soguk satirlarin RMSLE'si 1,86 vs sicak 0,74 -- KARESEL hatada 6,3 kat
     agirlik. Gradyani esir aliyorlar.
  2) CatBoost oblivious agac kullaniyor: her derinlik seviyesinde TEK bolme.
     ``soguk_mu`` uzerinde bir bolme, derinlik 6'da kapasitenin 1/6'sini
     yiyor.

Maske orani taramasi YALNIZCA yapay maske oranini degistirdi (0,00 -> 0,15
-> ... -> 1,00). Dogal soguk satirlarin ATILMASI hic olculmedi.

Yapay maske ile dogal soguk ayni sey DEGIL: yapay maske DropoutNet
duzenlileyicisi (model servis anindaki girdi dagilimini egitimde gorsun),
dogal soguk ise sicak uzmanin ASLA karsilasmayacagi bir dagilim. Ilkini
tutup ikincisini atmak tutarli bir tasarim.

Uc kol:
    TABAN            uretim (dogal soguk icerde, maske 0,15)
    -DOGAL_SOGUK     dogal soguk satirlar atilir, maske 0,15 kalir
    -DOGAL maske0    dogal soguk atilir, yapay maske de kapatilir

Ucuncu kol su soruyu yanitlar: dogal soguk gidince yapay maske hala
gerekli mi, yoksa 0,15 sadece onlarin varligini telafi ediyor muydu?

    python scripts/deney_sicak_dogal.py
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

SICAK_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
KOLLAR = (
    ("TABAN", False, 0.15),
    ("-DOGAL_SOGUK", True, 0.15),
    ("-DOGAL maske0", True, 0.00),
)
KAYIT = KOK / "experiments" / "sicak_dogal.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("SICAK UZMAN: dogal soguk satirlar egitimden cikarilsin mi?")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    p0 = parcalar[tm.BLOKLAR[0].ad][0]
    dogal = (p0["soguk_mu"] == 1).mean()
    print(f"  egitim parcasinda dogal soguk pay %{100 * dogal:.2f}")

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, at_dogal, maske in KOLLAR:
        t0 = time.time()
        tekil[ad] = {}
        blok = {}
        for b in tm.BLOKLAR:
            parca, dogrulama, gercek, soguk = parcalar[b.ad]
            if at_dogal:
                parca = parca[parca["soguk_mu"] != 1]
            sicak = ~soguk
            loglar = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(parca, kol, maske, tohum)
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, tohum, **SICAK_USTYAZIM)
                loglar.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sicak], tek[sicak])
            harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
            blok[b.ad] = tm.rmsle(gercek[sicak], harman[sicak])
        ort = float(np.mean(list(blok.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok.items())
        print(f"  {ad:16} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    kayitlar = []
    for ad, _, _ in KOLLAR[1:]:
        f = np.array([tekil["TABAN"][k] - tekil[ad][k] for k in tekil["TABAN"]])
        o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = o / sh if sh > 0 else 0.0
        hukum = "AL" if t_d >= 2 else ("REDDET" if t_d <= -2 else "esik alti")
        print(f"\n  {ad}: ESLENIK FARK {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
        for b in tm.BLOKLAR:
            bb = np.array([tekil["TABAN"][(b.ad, t)] - tekil[ad][(b.ad, t)] for t in di.TOHUMLAR])
            print(f"     {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum kazanc)")
        print(f"     genel skora tahmini etki {-o * 0.528:+.5f}")
        kayitlar.append({"kol": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
