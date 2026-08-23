"""SOGUK UZMAN: harman sadelestirme + ofset uzayinda BUZME. Karar mercii kis26.

NEDEN
-----
kis26 (ezber orani %0, TEK DURUST kat) soguk uzmani ONEMSIZ bir tabandan geride:

    uretim soguk 1,86509   |   duz kVA kovasi ortalamasi 1,8162

Bir tahminci onemsiz bir tabandan kotuyse en olasi sebep ASIRI YAYILMA:
tahmin dagilimi gercekten genis. RMSLE log uzayinda kareli hata oldugu icin
tahmini ortalamaya dogru buzmek (James-Stein mantigi) beklenen hatayi
DUSURUR -- yanlilik ekler ama varyanstan daha cok kazandirir.

Iki bagimsiz mudahale olculuyor:
  A) HARMAN: soguk harmani cat/xgb/lgbm 3/1/1 yerine YALNIZ cat
     (eski koddaki "sogukta cat EN KOTU aile" yorumu KIRLI bloklarin
      ortalamasindan geliyordu -- kis26 tek basina bakildiginda hukum degisebilir)
  B) BUZME: ofset uzayindaki tahmini kendi ortalamasina dogru beta ile buz:
     yeni = ort + beta * (tahmin - ort)

Ikisi de kis26 soguk satirlarinda, uc tohumla olculur. yaz25/guz25 BAKILMAZ --
o bloklarda ezber kanali acik ve hukumleri gecersiz (docs/35).

    python scripts/deney_soguk_buzme.py
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

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
USTYAZIM: dict[str, object] = {"depth": 7}
HARMANLAR = {"TABAN 3/1/1": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}, "YALNIZ cat": {"cat": 1.0}}
BETALAR = (1.00, 0.90, 0.80, 0.70, 0.66, 0.60, 0.50, 0.40)
KAYIT = KOK / "experiments" / "soguk_buzme.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SOGUK UZMAN: harman + ofset buzmesi  --  {BLOK} (tek durust kat)")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    y = gercek[soguk]
    log_guc = np.log1p(dogrulama["guc"].to_numpy())[soguk]
    print(f"  {BLOK} soguk {int(soguk.sum()):,} satir | {len(kol)} kolon")

    # aile tahminlerini bir kez uret, harmanlari uzerinde hesapla
    ham: dict[tuple[int, str], np.ndarray] = {}
    for tohum in TOHUMLAR:
        maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
        for aile in ("cat", "xgb", "lgbm"):
            ust = USTYAZIM if aile == "cat" else {}
            ham[(tohum, aile)] = di.egit_tahmin(aile, maskeli, dogrulama, kol, tohum, **ust)[soguk]
        print(f"  tohum {tohum} tahminleri hazir  ({time.time() - t0:.0f} sn)")

    sonuc: dict[tuple[str, float], list[float]] = {}
    for h_ad, w in HARMANLAR.items():
        for beta in BETALAR:
            skor = []
            for tohum in TOHUMLAR:
                pay = sum(w.values())
                log_t = sum(w[a] * ham[(tohum, a)] for a in w) / pay
                ofs = log_t - log_guc  # ofset uzayina gec
                ofs = ofs.mean() + beta * (ofs - ofs.mean())
                tah = np.clip(np.expm1(ofs + log_guc), 0.0, None)
                skor.append(tm.rmsle(y, tah))
            sonuc[(h_ad, beta)] = skor

    taban = float(np.mean(sonuc[("TABAN 3/1/1", 1.00)]))
    print(f"\n  TABAN (3/1/1, buzme yok): {taban:.5f}")
    print("\n  harman        beta    RMSLE      tabana gore     t")
    kayitlar = []
    for (h_ad, beta), skor in sonuc.items():
        o = float(np.mean(skor))
        f = np.array(sonuc[("TABAN 3/1/1", 1.00)]) - np.array(skor)
        sh = float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = float(f.mean() / sh) if sh > 0 else 0.0
        yildiz = "  <<<" if o < taban - 0.005 else ""
        print(f"  {h_ad:13} {beta:.2f}  {o:.5f}   {o - taban:+.5f}   {t_d:+7.2f}{yildiz}")
        kayitlar.append({"harman": h_ad, "beta": beta, "rmsle": o, "fark": o - taban, "t": t_d})

    en_iyi = min(sonuc.items(), key=lambda kv: np.mean(kv[1]))
    ad, beta = en_iyi[0]
    kazanc = taban - float(np.mean(en_iyi[1]))
    print(f"\n  EN IYI: {ad}  beta={beta:.2f}   kazanc {kazanc:+.5f}")
    print(f"  genel skora etkisi {-kazanc * 0.350:+.5f}   (d(genel)/d(soguk) = 0,350)")
    print("  kiyas: duz kVA kovasi ortalamasi 1,8162")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
