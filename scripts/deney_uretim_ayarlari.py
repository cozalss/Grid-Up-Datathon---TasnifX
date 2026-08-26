"""Üretim-sadık soğuk uzman CatBoost ayar taraması.

Eski ayar tezgâhları üretimde bulunmayan 151 kolonu kullanıyordu. Bu tezgâh
üretimin gerçek 105 kolonluk yalın setini, soğuk uzmanın ek-kökensiz eğitim
nüfusunu, maske=1.00 sözleşmesini ve sabit ağaç sayısını aynen uygular.

İki aşama vardır:

* ``eleme``: yaz25 + kis26, tek eşleştirilmiş tohum, en fazla 24 fit.
* ``dogrula``: seçilen tek aday + taban, üç blok, beş tohum, 30 fit.

Örnek::

    uv run python scripts/deney_uretim_ayarlari.py --asama eleme
    uv run python scripts/deney_uretim_ayarlari.py --asama dogrula --aday lr003_i400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
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

VARSAYILAN_CIKTI = KOK / "experiments" / "uretim_ayarlari.jsonl"


@dataclass(frozen=True)
class Aday:
    ad: str
    parametreler: dict[str, object]


@dataclass(frozen=True)
class EslesmisOzet:
    kazanc_mse: float
    standart_hata: float
    t_degeri: float
    kazanan_cift: int
    toplam_cift: int


def uretim_kolonlari(egitim: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Üretimin ``YALIN_CIKARILAN`` filtresini birebir uygula."""
    ham = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    return [k for k in ham if not k.startswith(tm.YALIN_CIKARILAN)]


def rejim_kaynagi(rejim: str, genis: object, dar: object) -> object:
    """Üretimde ilgili uzmanın gördüğü eğitim nüfusunu seç."""
    if rejim not in ("sicak", "soguk"):
        raise ValueError(f"bilinmeyen rejim: {rejim}")
    ayarlar = tm.REJIM_AYARLARI
    if ayarlar is None:
        raise RuntimeError("üretimde rejim uzmanları kapalı")
    return genis if ayarlar[rejim].get("ek_koken", True) else dar


def eslestirilmis_ozet(taban_rmsle: np.ndarray, aday_rmsle: np.ndarray) -> EslesmisOzet:
    """Aynı (blok, tohum) çiftlerinde MSE kazancını özetle."""
    taban = np.asarray(taban_rmsle, dtype="float64")
    aday = np.asarray(aday_rmsle, dtype="float64")
    if taban.shape != aday.shape or taban.ndim != 1 or taban.size == 0:
        raise ValueError(f"eşleşmeyen skor dizileri: {taban.shape} / {aday.shape}")
    fark = taban**2 - aday**2
    sh = float(fark.std(ddof=1) / np.sqrt(fark.size)) if fark.size > 1 else float("nan")
    ort = float(fark.mean())
    t = ort / sh if np.isfinite(sh) and sh > 0 else float("nan")
    return EslesmisOzet(
        kazanc_mse=ort,
        standart_hata=sh,
        t_degeri=float(t),
        kazanan_cift=int((fark > 0).sum()),
        toplam_cift=int(fark.size),
    )


def soguk_adaylar() -> tuple[Aday, ...]:
    """Eşit yaklaşık öğrenme bütçeli, tek-eksenli adaylar."""
    return (
        Aday("TABAN", {"depth": 7}),
        Aday("lr003_i400", {"depth": 7, "learning_rate": 0.03, "iterations": 400}),
        Aday("lr0075_i170", {"depth": 7, "learning_rate": 0.075, "iterations": 170}),
        Aday("lr010_i125", {"depth": 7, "learning_rate": 0.10, "iterations": 125}),
        Aday("l2_1", {"depth": 7, "l2_leaf_reg": 1.0}),
        Aday("l2_10", {"depth": 7, "l2_leaf_reg": 10.0}),
        Aday("rsm_050", {"depth": 7, "rsm": 0.50}),
        Aday("rsm_100", {"depth": 7, "rsm": 1.00}),
        Aday("random_0", {"depth": 7, "random_strength": 0.0}),
        Aday("random_4", {"depth": 7, "random_strength": 4.0}),
        Aday(
            "lr003_random4",
            {
                "depth": 7,
                "learning_rate": 0.03,
                "iterations": 400,
                "random_strength": 4.0,
            },
        ),
        Aday("depth_6", {"depth": 6}),
        Aday("depth_8", {"depth": 8}),
    )


def _aday_sec(asama: str, aday_adi: str | None) -> tuple[Aday, ...]:
    tumu = soguk_adaylar()
    if asama == "eleme":
        if aday_adi:
            secilen = [a for a in tumu if a.ad in {"TABAN", aday_adi}]
            if len(secilen) != 2:
                raise ValueError(f"aday bulunamadı: {aday_adi}")
            return tuple(secilen)
        return tumu
    if not aday_adi or aday_adi == "TABAN":
        raise ValueError("dogrula aşaması tek bir taban-dışı --aday ister")
    secilen = [a for a in tumu if a.ad in {"TABAN", aday_adi}]
    if len(secilen) != 2:
        raise ValueError(f"aday bulunamadı: {aday_adi}")
    return tuple(secilen)


def _ayarlar(asama: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if asama == "eleme":
        return ("yaz25", "kis26"), (1000,)
    return tuple(b.ad for b in tm.BLOKLAR), (1000, 1001, 1002, 1003, 1004)


def _skorla(gercek: np.ndarray, soguk: np.ndarray, log_tahmin: np.ndarray) -> float:
    tahmin = np.clip(np.expm1(log_tahmin), 0.0, None)
    return tm.rmsle(gercek[soguk], tahmin[soguk])


def _kaydet(
    yol: Path,
    *,
    asama: str,
    aday: Aday,
    bloklar: tuple[str, ...],
    tohumlar: tuple[int, ...],
    skorlar: list[float],
    ozet: EslesmisOzet | None,
) -> None:
    yol.parent.mkdir(parents=True, exist_ok=True)
    kayit = {
        "zaman": pd.Timestamp.now(tz="UTC").isoformat(),
        "asama": asama,
        "rejim": "soguk",
        "aday": aday.ad,
        "parametreler": aday.parametreler,
        "bloklar": list(bloklar),
        "tohumlar": list(tohumlar),
        "skorlar": skorlar,
        "ozet": asdict(ozet) if ozet else None,
    }
    with yol.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False, allow_nan=True) + "\n")


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asama", choices=("eleme", "dogrula"), default="eleme")
    ap.add_argument("--aday", help="yalnız bu adayı TABAN ile karşılaştır")
    ap.add_argument("--en-fazla-fit", type=int, default=30)
    ap.add_argument("--cikti", type=Path, default=VARSAYILAN_CIKTI)
    args = ap.parse_args()

    adaylar = _aday_sec(args.asama, args.aday)
    bloklar, tohumlar = _ayarlar(args.asama)
    fit_sayisi = len(adaylar) * len(bloklar) * len(tohumlar)
    if fit_sayisi > args.en_fazla_fit:
        raise RuntimeError(f"fit bütçesi aşıldı: {fit_sayisi} > {args.en_fazla_fit}")

    print("=" * 96)
    print("ÜRETİM-SADIK SOĞUK UZMAN AYARLARI")
    print("=" * 96)
    print(
        f"  aşama={args.asama} aday={len(adaylar)} blok={bloklar} "
        f"tohum={tohumlar} fit={fit_sayisi}/{args.en_fazla_fit}"
    )

    t0 = time.time()
    dar, test = d.cerceveleri_kur()
    genis = dar  # Soğuk uzman ek köken görmez; bu aşamada geniş sete ihtiyaç yok.
    kaynak = rejim_kaynagi("soguk", genis, dar)
    assert kaynak is dar
    kolonlar = uretim_kolonlari(dar, test)
    tm.kategorik_kodla(dar, test)
    print(f"  eğitim={len(dar):,} kolon={len(kolonlar)} (üretim yalın seti)")

    skorlar: dict[str, list[float]] = {a.ad: [] for a in adaylar}
    blok_skorlari: dict[str, dict[str, list[float]]] = {
        a.ad: {b: [] for b in bloklar} for a in adaylar
    }
    for blok in bloklar:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(dar, blok)
        if not soguk.any():
            raise RuntimeError(f"{blok} doğrulamasında soğuk satır yok")
        for tohum in tohumlar:
            t_fit = time.time()
            maskeli = d.soguk_maskele(kalan, kolonlar, 1.0, tohum)
            for aday in adaylar:
                log_tahmin = di.egit_tahmin(
                    "cat", maskeli, dogrulama, kolonlar, tohum, **aday.parametreler
                )
                skor = _skorla(gercek, soguk, log_tahmin)
                skorlar[aday.ad].append(skor)
                blok_skorlari[aday.ad][blok].append(skor)
                print(
                    f"  {blok} t={tohum} {aday.ad:16} RMSLE={skor:.6f}",
                    flush=True,
                )
            del maskeli
            print(f"    çift süresi {time.time() - t_fit:.0f} sn", flush=True)

    taban = np.asarray(skorlar["TABAN"])
    print("\nSONUÇ — pozitif değer MSE kazancıdır")
    for aday in adaylar:
        if aday.ad == "TABAN":
            print(f"  TABAN ortalama RMSLE {taban.mean():.6f}")
            _kaydet(
                args.cikti,
                asama=args.asama,
                aday=aday,
                bloklar=bloklar,
                tohumlar=tohumlar,
                skorlar=skorlar[aday.ad],
                ozet=None,
            )
            continue
        ozet = eslestirilmis_ozet(taban, np.asarray(skorlar[aday.ad]))
        blok_farklari = {
            b: float(
                np.mean(np.asarray(blok_skorlari["TABAN"][b]) ** 2)
                - np.mean(np.asarray(blok_skorlari[aday.ad][b]) ** 2)
            )
            for b in bloklar
        }
        yon = all(v > 0 for v in blok_farklari.values())
        print(
            f"  {aday.ad:16} dMSE={ozet.kazanc_mse:+.6f} "
            f"SH={ozet.standart_hata:.6f} t={ozet.t_degeri:+.2f} "
            f"kazanan={ozet.kazanan_cift}/{ozet.toplam_cift} "
            f"bloklar={blok_farklari} aynı_yön={yon}"
        )
        _kaydet(
            args.cikti,
            asama=args.asama,
            aday=aday,
            bloklar=bloklar,
            tohumlar=tohumlar,
            skorlar=skorlar[aday.ad],
            ozet=ozet,
        )

    print(f"\nTAMAM {(time.time() - t0) / 60:.1f} dakika | {args.cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
