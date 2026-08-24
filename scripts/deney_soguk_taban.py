"""SOGUK UZMAN: kosullu taban + buzme hedefi. Karar mercii kis26.

NEDEN
-----
Bugunku buzme (``son_islem.py``) soguk tahmini ofset uzayinda KENDI GENEL
ORTALAMASINA dogru cekiyor. Genel ortalama, bir tahminin cekilebilecegi en
kaba hedef: trafonun ilcesini ve kVA kademesini yok sayiyor.

James-Stein'in kendi literaturunde bunun adi var -- "shrinkage toward a
grand mean" yerine "shrinkage toward a **conditional** mean". Efron-Morris
(1975) tam bu genellemeyi yapar: hedef ne kadar bilgiliyse buzmenin
maliyeti (yanlilik) o kadar kucuk, kazanci (varyans) o kadar buyuk kalir.

Elimizde kosullu hedef icin iki dogal degisken var ve IKISI DE servis
aninda biliniyor:
    guc      -> kVA kademesi (log-varyansin %26,2'si)
    ilce_key -> konum        (%15,5)

Bu deney UC seyi ayni anda olcer, hepsi TEK bir fit kumesi uzerinde
aritmetikle (fit ~3 dakika, geri kalani saniyeler):

  A) KOSULLU TABANLAR tek baslarina ne yapiyor (model olmadan)
  B) MODEL -> TABAN convex harmani (alfa)
  C) TABANA DOGRU BUZME (beta) -- genel ortalamaya dogru buzmenin yerine

DURUSTLUK
---------
Tabanlar YALNIZCA ``parca`` (kis26 disindaki bloklar) uzerinden kuruluyor.
kis26'nin kendi etiketleri hicbir tabanda kullanilmiyor. Hucre ortalamalari
ampirik-Bayes ile ust seviyeye dogru buzuluyor (n azsa ebeveyne yakin).

kis26 ezber orani %0 olan TEK durust kattir (docs/35); yaz25/guz25'e
BAKILMAZ.

    python scripts/deney_soguk_taban.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
KAYIT = KOK / "experiments" / "soguk_taban.jsonl"

#: Ampirik-Bayes buzme sabiti: hucrede n satir varsa agirlik n/(n+M).
#: M = 200 -> 200 satirlik bir hucre ebeveyniyle yari yariya harmanlanir.
M_ONCE = 200.0

#: Taranan buzme katsayilari. 0,00 = saf taban (model tamamen atilir).
BETALAR = (0.80, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.00)

#: Taranan buzme hedefleri -- hepsi YALNIZ egitim parcasindan tureyor.
HEDEFLER = (
    "genel",
    "ilce",
    "ilcexkova_M1000",
    "ilcexkova_M2000",
    "ilcexkova_M5000",
    "ilcexkova_M20000",
)


def _tahminleri_getir(
    egitim: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Aile x tohum log tahminlerini uretir ya da onbellekten okur."""
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)

    if ONBELLEK.exists():
        z = np.load(ONBELLEK)
        ham = {(int(t), a): z[f"{t}_{a}"] for t in TOHUMLAR for a in ("cat", "xgb", "lgbm")}
        print(f"  tahminler onbellekten: {ONBELLEK.name}")
        return ham, dogrulama, gercek, soguk

    t0 = time.time()
    ham: dict[tuple[int, str], np.ndarray] = {}
    for tohum in TOHUMLAR:
        maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
        for aile in ("cat", "xgb", "lgbm"):
            ust = USTYAZIM if aile == "cat" else {}
            ham[(tohum, aile)] = di.egit_tahmin(aile, maskeli, dogrulama, kol, tohum, **ust)[soguk]
        print(f"  tohum {tohum} hazir ({time.time() - t0:.0f} sn)")
    ONBELLEK.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ONBELLEK, **{f"{t}_{a}": v for (t, a), v in ham.items()})
    print(f"  onbellege yazildi: {ONBELLEK.name}")
    return ham, dogrulama, gercek, soguk


#: kVA kademesi sayisi. Kenarlar YALNIZ egitim parcasindan turetilir ve
#: hedefe AYNI kenarlarla uygulanir -- iki tarafta ayri ayri hesaplamak
#: kovalari kaydirir ve hucre eslesmesini sessizce bozar.
KOVA_SAYISI = 24


def _kova_kenarlari(guc: np.ndarray) -> np.ndarray:
    lg = np.log1p(np.clip(guc, 0.0, None))
    return np.linspace(float(lg.min()), float(lg.max()) + 1e-9, KOVA_SAYISI + 1)


def _kova(guc: np.ndarray, kenar: np.ndarray) -> np.ndarray:
    lg = np.log1p(np.clip(guc, 0.0, None))
    return np.clip(np.searchsorted(kenar, lg, side="right") - 1, 0, KOVA_SAYISI - 1)


def _hucre_ortalamasi(
    anahtar_egitim: np.ndarray,
    ofset_egitim: np.ndarray,
    anahtar_hedef: np.ndarray,
    ebeveyn: np.ndarray,
    m_once: float = M_ONCE,
) -> np.ndarray:
    """Ampirik-Bayes hucre ortalamasi: (n*hucre + M*ebeveyn) / (n + M)."""
    s = pd.Series(ofset_egitim).groupby(anahtar_egitim).agg(["sum", "count"])
    top = pd.Series(s["sum"]).reindex(anahtar_hedef).to_numpy(dtype="float64")
    n = pd.Series(s["count"]).reindex(anahtar_hedef).to_numpy(dtype="float64")
    top = np.nan_to_num(top, nan=0.0)
    n = np.nan_to_num(n, nan=0.0)
    return (top + m_once * ebeveyn) / (n + m_once)


def tabanlari_kur(
    parca: pd.DataFrame, dogrulama: pd.DataFrame, soguk: np.ndarray
) -> dict[str, np.ndarray]:
    """OFSET uzayinda kosullu tabanlar. Kaynak YALNIZ ``parca``.

    Hucre yapisi ``deney_taban_ince.py`` ile UC BLOKTA tarandi. Kazanan:
    ilce x kova, EBEVEYN ``ilce`` (``kova`` degil -- ilce tek basina kovadan
    iyi ve seyrek hucre oraya dusmeli), kova sayisi 24, M ~ 1000.
    """
    of_e = np.log1p(parca[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64")) - np.log1p(
        parca["guc"].to_numpy(dtype="float64")
    )
    kenar = _kova_kenarlari(parca["guc"].to_numpy(dtype="float64"))
    kova_e = _kova(parca["guc"].to_numpy(dtype="float64"), kenar)
    ilce_e = parca["ilce_key"].to_numpy()

    dg = dogrulama[soguk]
    kova_h = _kova(dg["guc"].to_numpy(dtype="float64"), kenar)
    ilce_h = dg["ilce_key"].to_numpy()

    n = int(soguk.sum())
    genel = np.full(n, float(of_e.mean()))
    kova = _hucre_ortalamasi(kova_e, of_e, kova_h, genel, 200.0)
    ilce = _hucre_ortalamasi(ilce_e, of_e, ilce_h, genel, 200.0)
    anahtar_e = _metin(ilce_e) + "|" + _metin(kova_e)
    anahtar_h = _metin(ilce_h) + "|" + _metin(kova_h)

    t: dict[str, np.ndarray] = {"genel": genel, "kova": kova, "ilce": ilce}
    for m_once in (1000.0, 2000.0, 5000.0, 20000.0):
        t[f"ilcexkova_M{m_once:.0f}"] = _hucre_ortalamasi(anahtar_e, of_e, anahtar_h, ilce, m_once)
    return t


def _metin(v: np.ndarray) -> np.ndarray:
    """Anahtar birlestirmek icin metne cevir."""
    return pd.Series(v).astype(str).to_numpy()


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SOGUK UZMAN: kosullu taban + buzme hedefi  --  {BLOK}")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    ham, dogrulama, gercek, soguk = _tahminleri_getir(egitim, test)
    parca = egitim[egitim["_blok"] != BLOK]

    y = gercek[soguk]
    log_guc = np.log1p(dogrulama["guc"].to_numpy(dtype="float64"))[soguk]
    print(f"  {BLOK} soguk {len(y):,} satir")

    tabanlar = tabanlari_kur(parca, dogrulama, soguk)

    def skorla(ofs: np.ndarray) -> float:
        return tm.rmsle(y, np.clip(np.expm1(ofs + log_guc), 0.0, None))

    # tohum basina harmanlanmis model ofseti
    pay = sum(HARMAN.values())
    model_ofs = [sum(HARMAN[a] * ham[(t, a)] for a in HARMAN) / pay - log_guc for t in TOHUMLAR]

    kayitlar: list[dict] = []

    def yaz(etiket: str, skorlar: list[float], taban_skorlar: list[float] | None = None) -> None:
        o = float(np.mean(skorlar))
        satir = f"  {etiket:34} {o:.5f}"
        if taban_skorlar is not None:
            f = np.array(taban_skorlar) - np.array(skorlar)
            sh = float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0
            t_d = float(f.mean() / sh) if sh > 0 else 0.0
            satir += f"   {o - float(np.mean(taban_skorlar)):+.5f}   t={t_d:+7.2f}"
            kayitlar.append(
                {"etiket": etiket, "rmsle": o, "fark": o - float(np.mean(taban_skorlar)), "t": t_d}
            )
        else:
            kayitlar.append({"etiket": etiket, "rmsle": o})
        print(satir)

    print("\n--- A) KOSULLU TABANLAR (model yok) ---")
    for ad, v in tabanlar.items():
        print(f"  taban:{ad:28} {skorla(v):.5f}")

    ham_skor = [skorla(m) for m in model_ofs]
    print("\n--- MODEL (3/1/1, buzmesiz) ---")
    yaz("MODEL", ham_skor)
    ref = ham_skor

    print("\n--- B) URETIMDEKI BUZME: hedef = TAHMININ KENDI ortalamasi ---")
    print("  (son_islem.py bunu yapiyor -- C blogunun karsilastirma tabani)")
    print(f"  {'':34} {'RMSLE':>7}   {'fark':>8}   t")
    for beta in BETALAR:
        sk = [skorla(m.mean() + beta * (m - m.mean())) for m in model_ofs]
        yaz(f"URETIM kendi_ort    beta={beta:.2f}", sk, ref)

    print("\n--- C) EGITIMDEN TUREYEN TABANA DOGRU BUZME ---")
    print("  ofs' = taban + beta*(ofs - taban);  beta=0,00 -> saf taban, model atiliyor")
    print(f"  {'':34} {'RMSLE':>7}   {'fark':>8}   t")
    for ad in HEDEFLER:
        for beta in BETALAR:
            sk = [skorla(tabanlar[ad] + beta * (m - tabanlar[ad])) for m in model_ofs]
            yaz(f"buzme  {ad:14} beta={beta:.2f}", sk, ref)

    print("\n--- C2) TABAN YAPISI + MODEL SEVIYESI ---")
    print("  taban' = taban - ort(taban) + ort(model_ofs)")
    print("  (hucre YAPISI egitimden, SEVIYE modelden -- model tarihi ve trendi biliyor)")
    gercek_ofs = np.log1p(y) - log_guc
    print(f"  gercek kis26 soguk ofset ortalamasi   {gercek_ofs.mean():+.5f}")
    print(f"  egitim parcasi genel ofset ortalamasi {tabanlar['genel'][0]:+.5f}")
    print(f"  modelin tahmin ettigi ofset ortalamasi {np.mean([m.mean() for m in model_ofs]):+.5f}")
    print(f"  {'':34} {'RMSLE':>7}   {'fark':>8}   t")
    for ad in HEDEFLER:
        for beta in (0.40, 0.30, 0.25, 0.20, 0.10, 0.00):
            sk = []
            for m in model_ofs:
                tb = tabanlar[ad] - tabanlar[ad].mean() + m.mean()
                sk.append(skorla(tb + beta * (m - tb)))
            yaz(f"MSEV   {ad:14} beta={beta:.2f}", sk, ref)

    print("\n--- C3) ZAMAN EKSENI KORUNARAK BUZME (test icin dogru kurgu) ---")
    print("  taban = GUNLUK model ortalamasi + (hucre etkisi - ortalamasi)")
    print("  Gerekce: ofsetin zaman ekseni test penceresinde (Nis-Tem) kis26'dakinden")
    print("  136 KAT guclu. Zamana sabit bir hedefe buzmek mevsim rampasini ezer;")
    print("  kis26'da bu bedava gorunur cunku kis ofseti duz. Gun ortalamasi modelden")
    print("  alinir -- model tarihi ve trendi biliyor; hucre yapisi egitimden gelir.")
    gun = pd.to_datetime(dogrulama["tarih"]).to_numpy()[soguk]
    gun_kod = pd.factorize(gun)[0]
    print(f"  kis26 soguk: {len(np.unique(gun_kod))} farkli gun")
    print(f"  {'':34} {'RMSLE':>7}   {'fark':>8}   t")

    def gun_ortalamasi(v: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(gun_kod).transform("mean").to_numpy()

    for beta in (0.60, 0.40, 0.30, 0.25, 0.20, 0.10):
        sk = [skorla(gun_ortalamasi(m) + beta * (m - gun_ortalamasi(m))) for m in model_ofs]
        yaz(f"GUN    yalniz_gun     beta={beta:.2f}", sk, ref)
    for ad in HEDEFLER:
        if ad == "genel":
            continue
        etki = tabanlar[ad] - tabanlar[ad].mean()
        for beta in (0.40, 0.30, 0.25, 0.20, 0.10, 0.00):
            sk = [
                skorla((gun_ortalamasi(m) + etki) + beta * (m - (gun_ortalamasi(m) + etki)))
                for m in model_ofs
            ]
            yaz(f"GUN+   {ad:14} beta={beta:.2f}", sk, ref)

    print("\n--- D) UST SINIR: kis26 uzerinde OLS ile uydurulmus afin kalibrasyon ---")
    print("  (bilerek asiri uydurma; sadece 'masada ne kadar var' sorusunu yanitlar)")
    of_gercek = np.log1p(y) - log_guc
    for i, m in enumerate(model_ofs, 1):
        egim, kesme = np.polyfit(m, of_gercek, 1)
        print(
            f"  tohum{i}: egim={egim:.4f} kesme={kesme:+.4f}  RMSLE={skorla(kesme + egim * m):.5f}"
        )

    en_iyi = min((k for k in kayitlar if "fark" in k), key=lambda k: k["rmsle"])
    kazanc = float(np.mean(ref)) - en_iyi["rmsle"]
    print(f"\n  EN IYI: {en_iyi['etiket']}  {en_iyi['rmsle']:.5f}  kazanc {kazanc:+.5f}")
    print(f"  genel skora tahmini etki {-kazanc * 0.350:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
