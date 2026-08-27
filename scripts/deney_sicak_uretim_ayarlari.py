"""Üretim-eşli sıcak CatBoost ayarlarını gerçek harman içinde tara.

Taban CatBoost ile sabit XGB/LightGBM/sinir ağı tahminleri daha önce üç
blok ve üç tohum için üretim yolundan önbelleklenmiştir. Bu tezgâh tabanı
yeniden eğitmez; yalnız aday CatBoost kolonunu eğitir ve onu üretimdeki
3/1/1/1.4 log-harmanına koyar. Böylece eleme sekiz, doğrulama dokuz yeni
fit ile sınırlıdır.

Örnekler::

    uv run python scripts/deney_sicak_uretim_ayarlari.py --asama eleme
    uv run python scripts/deney_sicak_uretim_ayarlari.py --asama dogrula --aday lr003_i400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402
from deney_uretim_ayarlari import Aday, eslestirilmis_ozet, uretim_kolonlari  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

ONBELLEK = KOK / "data" / "interim" / "aile_onbellek"
CIKTI = KOK / "experiments" / "sicak_uretim_ayarlari.jsonl"
MASKE = 0.15
AILELER = ("cat", "xgb", "lgbm", "sinir_agi")


def sicak_adaylar() -> tuple[Aday, ...]:
    """Üretim tabanı ve eşit yaklaşık öğrenme bütçeli dört aday."""
    taban: dict[str, object] = {
        "depth": 6,
        "l2_leaf_reg": 1.0,
        "random_strength": 4.0,
    }
    return (
        Aday("TABAN", taban),
        Aday("lr003_i400", {**taban, "learning_rate": 0.03, "iterations": 400}),
        Aday("lr0075_i170", {**taban, "learning_rate": 0.075, "iterations": 170}),
        Aday("rsm_100", {**taban, "rsm": 1.0}),
        Aday(
            "bernoulli_080",
            {**taban, "bootstrap_type": "Bernoulli", "subsample": 0.8},
        ),
    )


def log_harmani(tahminler: dict[str, np.ndarray], agirliklar: dict[str, float]) -> np.ndarray:
    """Üretimin aile tahminlerini log uzayında ağırlıklı ortala."""
    if set(tahminler) != set(agirliklar):
        raise ValueError("tahmin ve ağırlık aileleri eşleşmiyor")
    uzunluklar = {np.asarray(v).shape for v in tahminler.values()}
    if len(uzunluklar) != 1:
        raise ValueError(f"aile tahmin boyutları eşleşmiyor: {uzunluklar}")
    toplam = float(sum(agirliklar.values()))
    if toplam <= 0:
        raise ValueError("ağırlık toplamı pozitif olmalı")
    return sum(agirliklar[a] * np.asarray(tahminler[a]) for a in tahminler) / toplam


def torbali_log_harmani(
    tohum_tahminleri: list[dict[str, np.ndarray]], agirliklar: dict[str, float]
) -> np.ndarray:
    """Üretim gibi önce aileleri, sonra tohumları log uzayında ortala."""
    if not tohum_tahminleri:
        raise ValueError("torbalama için en az bir tohum tahmini gerekli")
    return np.mean([log_harmani(t, agirliklar) for t in tohum_tahminleri], axis=0)


def blok_hukmu(blok_kazanclari: dict[str, float]) -> bool:
    """Aday yalnız ölçülen her blokta MSE kazancı varsa yön kapısını geçer."""
    return bool(blok_kazanclari) and all(v > 0 for v in blok_kazanclari.values())


def devam_etmeli(asama: str, blok: str, ilk_blok: str, ilk_blok_farklari: list[float]) -> bool:
    """Elemede ilk blok kaybedildiyse sonraki pahalı fitleri atla."""
    if asama != "eleme" or blok == ilk_blok:
        return True
    return bool(ilk_blok_farklari) and float(np.mean(ilk_blok_farklari)) > 0


def kuadratik_optimum(
    gercek_log: np.ndarray,
    taban_log: np.ndarray,
    aday_log: np.ndarray,
    agirlik: np.ndarray,
) -> tuple[float, float]:
    """Tabandan adaya uzanan doğru üzerindeki optimum ölçeği ve MSE farkını bul."""
    gercek = np.asarray(gercek_log, dtype="float64")
    taban = np.asarray(taban_log, dtype="float64")
    aday = np.asarray(aday_log, dtype="float64")
    w = np.asarray(agirlik, dtype="float64")
    if not (gercek.shape == taban.shape == aday.shape == w.shape):
        raise ValueError("kuadratik optimum dizilerinin boyutları eşleşmiyor")
    if not np.isfinite(np.stack((gercek, taban, aday, w))).all() or np.any(w < 0):
        raise ValueError("kuadratik optimum girdileri sonlu ve ağırlıklar negatif olmamalı")
    toplam = float(w.sum())
    if toplam <= 0:
        raise ValueError("ağırlık toplamı pozitif olmalı")
    yon = aday - taban
    q = float(np.sum(w * yon * yon) / toplam)
    if q <= 0:
        raise ValueError("aday yönünün normu pozitif olmalı")
    capraz = float(np.sum(w * (taban - gercek) * yon) / toplam)
    kappa = -capraz / q
    mse_farki = -(capraz * capraz) / q
    return kappa, mse_farki


def _program(asama: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if asama == "eleme":
        return ("yaz25", "kis26"), (1000,)
    return tuple(b.ad for b in tm.BLOKLAR), (1000, 1001, 1002)


def _aday_sec(asama: str, aday_adi: str | None) -> tuple[Aday, ...]:
    adaylar = sicak_adaylar()[1:]
    if asama == "eleme" and aday_adi is None:
        return adaylar
    if not aday_adi:
        raise ValueError("dogrula aşaması --aday ister")
    secilen = tuple(a for a in adaylar if a.ad == aday_adi)
    if not secilen:
        raise ValueError(f"aday bulunamadı: {aday_adi}")
    return secilen


def _aday_yolu(blok: str, tohum: int, aday: Aday) -> Path:
    return ONBELLEK / f"{blok}_{tohum}_cat_ayar_{aday.ad}.npy"


def _uretim_agirliklari() -> dict[str, float]:
    ayarlar = tm.REJIM_AYARLARI
    if ayarlar is None:
        raise RuntimeError("üretimde rejim uzmanları kapalı")
    sicak = ayarlar["sicak"]
    if sicak.get("ek_koken") is not True or float(sicak["maske"]) != MASKE:
        raise RuntimeError("sıcak üretim sözleşmesi değişmiş; tezgâhı güncelle")
    agirlik = {k: float(v) for k, v in dict(sicak["agirlik"]).items()}
    if set(agirlik) != set(AILELER):
        raise RuntimeError(f"beklenmeyen üretim aileleri: {agirlik}")
    return agirlik


def _genis_kaynak(dar: pd.DataFrame) -> pd.DataFrame:
    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([ad for ad, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in dar.columns if k in ek.columns]
    genis = pd.concat([dar[ortak], ek[ortak]], ignore_index=True)
    for kolon in tm.KATEGORIK:
        genis[kolon] = pd.Categorical(genis[kolon], categories=dar[kolon].cat.categories)
    return genis


def _dizi_yukle(yol: Path, beklenen: int) -> np.ndarray:
    if not yol.exists():
        raise FileNotFoundError(f"üretim önbelleği eksik: {yol}")
    dizi = np.load(yol).astype("float64")
    if dizi.shape != (beklenen,) or not np.isfinite(dizi).all():
        raise ValueError(f"geçersiz önbellek {yol}: {dizi.shape}, beklenen {(beklenen,)}")
    return dizi


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asama", choices=("eleme", "dogrula"), default="eleme")
    ap.add_argument("--aday", help="yalnız bu adayı tara")
    ap.add_argument("--en-fazla-fit", type=int, default=9)
    ap.add_argument("--cikti", type=Path, default=CIKTI)
    args = ap.parse_args()

    adaylar = _aday_sec(args.asama, args.aday)
    bloklar, tohumlar = _program(args.asama)
    eksik = [
        (blok, tohum, aday)
        for blok in bloklar
        for tohum in tohumlar
        for aday in adaylar
        if not _aday_yolu(blok, tohum, aday).exists()
    ]
    if len(eksik) > args.en_fazla_fit:
        raise RuntimeError(f"fit bütçesi aşıldı: {len(eksik)} > {args.en_fazla_fit}")

    print("=" * 100)
    print("ÜRETİM-EŞLİ SICAK CATBOOST AYARLARI — GERÇEK HARMAN İÇİNDE")
    print("=" * 100)
    print(
        f"  aşama={args.asama} aday={[a.ad for a in adaylar]} blok={bloklar} "
        f"tohum={tohumlar} yeni_fit={len(eksik)}/{args.en_fazla_fit}"
    )

    t0 = time.time()
    dar, test = d.cerceveleri_kur()
    kolonlar = uretim_kolonlari(dar, test)
    tm.kategorik_kodla(dar, test)
    agirliklar = _uretim_agirliklari()
    guc_kenar = ol.guc_kenarlari(test)
    test_sicak = test[test["soguk_mu"] != 1]
    genis = _genis_kaynak(dar) if eksik else None
    if genis is not None:
        print(f"  ana={len(dar):,} geniş={len(genis):,} kolon={len(kolonlar)}")
    else:
        print(f"  ana={len(dar):,} kolon={len(kolonlar)}; bütün adaylar önbellekte")

    taban_skor: dict[tuple[str, int], float] = {}
    aday_skor: dict[str, dict[tuple[str, int], float]] = {a.ad: {} for a in adaylar}
    blok_fark: dict[str, dict[str, list[float]]] = {a.ad: {b: [] for b in bloklar} for a in adaylar}
    torbali_blok_fark: dict[str, dict[str, float]] = {a.ad: {} for a in adaylar}

    for blok in bloklar:
        etkin_adaylar = tuple(
            aday
            for aday in adaylar
            if devam_etmeli(args.asama, blok, bloklar[0], blok_fark[aday.ad][bloklar[0]])
        )
        atlanan = [aday.ad for aday in adaylar if aday not in etkin_adaylar]
        if atlanan:
            print(f"  ERKEN KES {blok}: ilk blokta kaybedenler {atlanan}")
        if not etkin_adaylar:
            continue
        dogrulama = dar[dar["_blok"] == blok]
        gercek = dogrulama[tm.HEDEF].to_numpy()
        soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
        sicak = ~soguk
        w, _ = ol.test_agirliklari(dogrulama[sicak], test_sicak, guc_kenar, eksenler=("bayatlik",))
        parca = tm.kokenleri_ayikla(genis, blok) if genis is not None else None
        torba_taban: list[dict[str, np.ndarray]] = []
        torba_aday: dict[str, list[dict[str, np.ndarray]]] = {aday.ad: [] for aday in etkin_adaylar}
        for tohum in tohumlar:
            sabit = {
                aile: _dizi_yukle(ONBELLEK / f"{blok}_{tohum}_{aile}_uretim.npy", int(sicak.sum()))
                for aile in AILELER
            }
            torba_taban.append(sabit)
            taban_log = log_harmani(sabit, agirliklar)
            taban_t = np.clip(np.expm1(taban_log), 0.0, None)
            taban = ol.agirlikli_rmsle(gercek[sicak], taban_t, w)
            taban_skor[(blok, tohum)] = taban

            maskeli = None
            if any(not _aday_yolu(blok, tohum, a).exists() for a in etkin_adaylar):
                assert parca is not None
                maskeli = d.soguk_maskele(parca, kolonlar, MASKE, tohum)
            for aday in etkin_adaylar:
                yol = _aday_yolu(blok, tohum, aday)
                if not yol.exists():
                    assert maskeli is not None
                    bas = time.time()
                    log_t = di.egit_tahmin(
                        "cat", maskeli, dogrulama, kolonlar, tohum, **aday.parametreler
                    )
                    np.save(yol, log_t[sicak].astype("float32"))
                    print(f"  FIT {blok} t={tohum} {aday.ad:16} {time.time() - bas:.0f} sn")
                cat = _dizi_yukle(yol, int(sicak.sum()))
                harman = {**sabit, "cat": cat}
                torba_aday[aday.ad].append(harman)
                tahmin = np.clip(np.expm1(log_harmani(harman, agirliklar)), 0.0, None)
                skor = ol.agirlikli_rmsle(gercek[sicak], tahmin, w)
                aday_skor[aday.ad][(blok, tohum)] = skor
                blok_fark[aday.ad][blok].append(taban**2 - skor**2)
                print(
                    f"    {blok} t={tohum} {aday.ad:16} taban={taban:.6f} "
                    f"aday={skor:.6f} dMSE={taban**2 - skor**2:+.6f}",
                    flush=True,
                )

        taban_torbali = np.clip(np.expm1(torbali_log_harmani(torba_taban, agirliklar)), 0.0, None)
        taban_torbali_skor = ol.agirlikli_rmsle(gercek[sicak], taban_torbali, w)
        for aday in etkin_adaylar:
            if len(torba_aday[aday.ad]) != len(tohumlar):
                continue
            aday_torbali = np.clip(
                np.expm1(torbali_log_harmani(torba_aday[aday.ad], agirliklar)),
                0.0,
                None,
            )
            aday_torbali_skor = ol.agirlikli_rmsle(gercek[sicak], aday_torbali, w)
            fark = taban_torbali_skor**2 - aday_torbali_skor**2
            torbali_blok_fark[aday.ad][blok] = fark
            print(
                f"  TORBALI {blok} {aday.ad:16} taban={taban_torbali_skor:.6f} "
                f"aday={aday_torbali_skor:.6f} dMSE={fark:+.6f}"
            )

    args.cikti.parent.mkdir(parents=True, exist_ok=True)
    print("\nSONUÇ — pozitif dMSE kazançtır")
    for aday in adaylar:
        olculen = tuple(aday_skor[aday.ad])
        taban_dizi = np.array([taban_skor[k] for k in olculen])
        skor_dizi = np.array([aday_skor[aday.ad][k] for k in olculen])
        ozet = eslestirilmis_ozet(taban_dizi, skor_dizi)
        bloklar_ozet = {b: float(np.mean(v)) for b, v in blok_fark[aday.ad].items() if v}
        yon = blok_hukmu(bloklar_ozet)
        torbali_ozet = torbali_blok_fark[aday.ad]
        torbali_yon = blok_hukmu(torbali_ozet)
        hukum = "ADAY" if yon and ozet.kazanan_cift == ozet.toplam_cift else "REDDET"
        print(
            f"  {aday.ad:16} dMSE={ozet.kazanc_mse:+.6f} "
            f"SH={ozet.standart_hata:.6f} t={ozet.t_degeri:+.2f} "
            f"kazanan={ozet.kazanan_cift}/{ozet.toplam_cift} "
            f"bloklar={bloklar_ozet} yön={yon} {hukum}"
        )
        if torbali_ozet:
            print(f"    TORBALI bloklar={torbali_ozet} yön={torbali_yon}")
        kayit = {
            "zaman": pd.Timestamp.now(tz="UTC").isoformat(),
            "asama": args.asama,
            "aday": aday.ad,
            "parametreler": aday.parametreler,
            "bloklar": list(bloklar),
            "tohumlar": list(tohumlar),
            "ozet": asdict(ozet),
            "blok_kazanclari": bloklar_ozet,
            "yon_kapisi": yon,
            "torbali_blok_kazanclari": torbali_ozet,
            "torbali_yon_kapisi": torbali_yon,
            "hukum": hukum,
        }
        with args.cikti.open("a", encoding="utf-8") as f:
            f.write(json.dumps(kayit, ensure_ascii=False, allow_nan=True) + "\n")

    print(f"\nTAMAM {(time.time() - t0) / 60:.1f} dakika | {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
