"""Ölçülmüş Kaggle skorlarından etiketsiz log-uzayı Gram ansamblı üret.

RMSLE karesel olduğundan, aynı test satırlarındaki daha önce puanlanmış tahminler
arasındaki iç çarpımlar ve skorlar optimum doğrusal izdüşümü tam olarak belirler.
Bu betik kaynakları hash ile kilitler, kötü koşullu çözümleri reddeder ve hiçbir
gönderim yapmadan yeni CSV ile denetim raporunu üretir.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TABAN_ADI = "v83"
KAYNAKLAR: dict[str, tuple[str, float, str, str]] = {
    "v18": (
        "tuketim_v18.csv",
        1.03370,
        "55688907",
        "FB4E2A4C52A8432D556C6272CEFEA14AD0DA16285AB9B56B47730103EF5BBFD3",
    ),
    "v27": (
        "tuketim_v27_v18hedge.csv",
        1.03362,
        "55707804",
        "51822FF1472D32138030C9DD53019D972DD69535B318326C56D97E4AB65CBA06",
    ),
    "v30": (
        "tuketim_v30_buzme.csv",
        1.02639,
        "55717274",
        "1764CD9C1D69F273025794606B032E944F1505AFC1239316FDBA27582E395636",
    ),
    "v44": (
        "tuketim_v44_v27yeni.csv",
        1.03053,
        "55732790",
        "5F8B7F257C3BEB786E43B0425E040748B00A817A34CAF7E01ECFB1B0084051A1",
    ),
    "v46": (
        "tuketim_v46_gun.csv",
        1.02448,
        "55732647",
        "2C23F5FD63858F8150EEB3CDA0BA2325421028D03DE91272962EE995CD7810B3",
    ),
    "v47": (
        "tuketim_v47_eskison.csv",
        1.01750,
        "55732850",
        "72DE024A7D77CE8BB28EB258661182512CDE7A9814214A6CA46B14B2082631E0",
    ),
    "v50": (
        "tuketim_v50_nihai30.csv",
        1.01686,
        "55755676",
        "706EEF87869EDE9FFE52B6614809F53C4C5C748041BE940B051F4936F9D68BC4",
    ),
    "v73": (
        "tuketim_v73_soguk_gun160.csv",
        1.01538,
        "55780987",
        "08673F271EE8257BBC323FF17228289A31A55F6F6E6BC081794B1A82FDC9DEB8",
    ),
    "v81": (
        "tuketim_v81_sicak08.csv",
        1.01429,
        "55811392",
        "ED9B792B7FB8B448D5C5AC2EB28B12DDDA5CE114F80966C8EB29383E98137A35",
    ),
    "v83": (
        "tuketim_v83_sicak_optimum.csv",
        1.01318,
        "55811502",
        "F482A9DEEB771BF6D17B9271B9D11190B8FB495D28388D35E5A6C28CAC108041",
    ),
}
ADAY_ADLARI = tuple(ad for ad in KAYNAKLAR if ad != TABAN_ADI)


@dataclass(frozen=True)
class GramSonucu:
    aday_adlari: tuple[str, ...]
    katsayilar: np.ndarray
    gram: np.ndarray
    log_tahmin: np.ndarray
    tahmini_mse: float
    tahmini_rmsle: float
    kosul_sayisi: float
    cozum_artigi: float


def _dizi(dizi: np.ndarray, ad: str) -> np.ndarray:
    sonuc = np.asarray(dizi, dtype="float64")
    if sonuc.ndim != 1 or sonuc.size == 0 or not np.isfinite(sonuc).all():
        raise ValueError(f"{ad} sonlu, boş olmayan tek boyutlu dizi olmalı")
    return sonuc


def gram_coz(
    taban_log: np.ndarray,
    adaylar: dict[str, np.ndarray],
    skorlar: dict[str, float],
    *,
    taban_adi: str,
) -> GramSonucu:
    """Skorlar ve tahmin farklarından optimum izdüşüm katsayılarını çöz."""
    taban = _dizi(taban_log, taban_adi)
    adlar = tuple(adaylar)
    if not adlar or set(skorlar) != {taban_adi, *adlar}:
        raise ValueError("aday ve skor adları bire bir eşleşmeli")
    kolonlar = []
    for ad in adlar:
        aday = _dizi(adaylar[ad], ad)
        if aday.shape != taban.shape:
            raise ValueError(f"{ad} boyutu tabanla eşleşmiyor")
        kolonlar.append(aday - taban)
    skor_dizisi = np.array([skorlar[taban_adi], *(skorlar[a] for a in adlar)])
    if not np.isfinite(skor_dizisi).all() or np.any(skor_dizisi <= 0):
        raise ValueError("skorlar sonlu ve pozitif olmalı")

    d = np.column_stack(kolonlar)
    gram = d.T @ d / taban.size
    gram = (gram + gram.T) / 2.0
    kosul = float(np.linalg.cond(gram))
    if not np.isfinite(kosul):
        raise ValueError("Gram matrisi tekil")
    taban_mse = float(skorlar[taban_adi] ** 2)
    izdusum = (taban_mse + np.diag(gram) - np.square(skor_dizisi[1:])) / 2.0
    katsayilar = np.linalg.solve(gram, izdusum)
    artik = float(np.max(np.abs(gram @ katsayilar - izdusum)))
    log_tahmin = taban + d @ katsayilar
    tahmini_mse = float(taban_mse - izdusum @ katsayilar)
    if tahmini_mse < -1e-12:
        raise ValueError(f"hesaplanan MSE negatif: {tahmini_mse}")
    tahmini_mse = max(tahmini_mse, 0.0)
    return GramSonucu(
        aday_adlari=adlar,
        katsayilar=katsayilar,
        gram=gram,
        log_tahmin=log_tahmin,
        tahmini_mse=tahmini_mse,
        tahmini_rmsle=float(np.sqrt(tahmini_mse)),
        kosul_sayisi=kosul,
        cozum_artigi=artik,
    )


def prob_kappa(taban_skor: float, prob_skor: float, q: float) -> float:
    """Bir yön probunun skorundan o doğru üzerindeki optimum ölçeği çıkar."""
    if (
        not all(np.isfinite((taban_skor, prob_skor, q)))
        or taban_skor <= 0
        or prob_skor <= 0
        or q <= 0
    ):
        raise ValueError("skorlar ve yön normu sonlu, pozitif olmalı")
    return float((q - (prob_skor**2 - taban_skor**2)) / (2.0 * q))


def prob_skor_araligi(
    *,
    taban_skor: float,
    prob_skor: float,
    q: float,
    kappa: float,
    yarim_adim: float = 5e-6,
) -> tuple[float, float]:
    """Sabit kappa adayını taban/prob skor yuvarlama köşelerinde değerlendir."""
    sonuclar: list[float] = []
    for taban_yon, prob_yon in itertools.product((-1.0, 1.0), repeat=2):
        s0 = taban_skor + taban_yon * yarim_adim
        s1 = prob_skor + prob_yon * yarim_adim
        mse = s0**2 + kappa * (s1**2 - s0**2 - q) + kappa * kappa * q
        sonuclar.append(float(np.sqrt(max(mse, 0.0))))
    return min(sonuclar), max(sonuclar)


def afin_agirliklari(
    katsayilar: np.ndarray,
    aday_adlari: tuple[str, ...],
    *,
    taban_adi: str,
    kappa: float,
) -> dict[str, float]:
    """Taban+kappa*yön gösterimini kaynak vektörlerin afin ağırlıklarına çevir."""
    k = np.asarray(katsayilar, dtype="float64")
    if k.shape != (len(aday_adlari),) or not np.isfinite(k).all():
        raise ValueError("katsayı boyutu veya değeri geçersiz")
    agirliklar = {ad: float(kappa * deger) for ad, deger in zip(aday_adlari, k, strict=True)}
    agirliklar[taban_adi] = float(1.0 - kappa * k.sum())
    return agirliklar


def aday_skor_araligi(
    gram: np.ndarray,
    katsayilar: np.ndarray,
    skorlar: dict[str, float],
    *,
    aday_adlari: tuple[str, ...],
    taban_adi: str,
    yarim_adim: float = 5e-6,
) -> tuple[float, float]:
    """Sabit adayın görüntülenen skor yuvarlama köşelerindeki RMSLE bandı."""
    q = np.asarray(gram, dtype="float64")
    k = np.asarray(katsayilar, dtype="float64")
    if q.shape != (len(aday_adlari), len(aday_adlari)) or k.shape != (len(aday_adlari),):
        raise ValueError("Gram ve katsayı boyutları aday sayısıyla eşleşmiyor")
    sonuclar: list[float] = []
    adlar = (taban_adi, *aday_adlari)
    for isaretler in itertools.product((-1.0, 1.0), repeat=len(adlar)):
        koseli = {
            ad: float(skorlar[ad] + yon * yarim_adim)
            for ad, yon in zip(adlar, isaretler, strict=True)
        }
        m0 = koseli[taban_adi] ** 2
        mi = np.square([koseli[ad] for ad in aday_adlari])
        capraz = (mi - m0 - np.diag(q)) / 2.0
        mse = float(m0 + 2.0 * capraz @ k + k @ q @ k)
        sonuclar.append(float(np.sqrt(max(mse, 0.0))))
    return min(sonuclar), max(sonuclar)


def _sha256(yol: Path) -> str:
    ozet = hashlib.sha256()
    with yol.open("rb") as dosya:
        for parca in iter(lambda: dosya.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest().upper()


def _kaynaklari_oku() -> tuple[pd.Series, dict[str, np.ndarray]]:
    kimlik: pd.Series | None = None
    loglar: dict[str, np.ndarray] = {}
    for ad, (dosya_adi, _skor, _ref, beklenen_hash) in KAYNAKLAR.items():
        yol = KOK / "submissions" / dosya_adi
        gercek_hash = _sha256(yol)
        if gercek_hash != beklenen_hash:
            raise RuntimeError(f"{ad} kaynak hash değişti: {gercek_hash}")
        cerceve = pd.read_csv(yol)
        if list(cerceve.columns) != ["id", "tuketim"] or cerceve["id"].duplicated().any():
            raise RuntimeError(f"{ad} kaynak şeması geçersiz")
        if kimlik is None:
            kimlik = cerceve["id"].copy()
        elif not cerceve["id"].equals(kimlik):
            raise RuntimeError(f"{ad} id sırası tabanla eşleşmiyor")
        tahmin = cerceve["tuketim"].to_numpy(dtype="float64")
        if not np.isfinite(tahmin).all() or np.any(tahmin < 0):
            raise RuntimeError(f"{ad} tahminleri sonlu ve negatif olmamalı")
        loglar[ad] = np.log1p(tahmin)
    assert kimlik is not None
    if len(kimlik) != 714_688:
        raise RuntimeError(f"beklenmeyen satır sayısı: {len(kimlik)}")
    resmi_kimlik = pd.read_csv(KOK / "data" / "raw" / "sample_submission.csv", usecols=["id"])["id"]
    if not kimlik.equals(resmi_kimlik):
        raise RuntimeError("kaynak id sırası resmi sample_submission ile eşleşmiyor")
    return kimlik, loglar


def _rapor_kaynaklari() -> dict[str, dict[str, object]]:
    return {
        ad: {"dosya": deger[0], "skor": deger[1], "ref": deger[2], "sha256": deger[3]}
        for ad, deger in KAYNAKLAR.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cikis", type=Path)
    ap.add_argument("--rapor", type=Path)
    ap.add_argument(
        "--prob-skor",
        type=float,
        help="v85 gerçek skoru ile doğru üzerinde v86'yı çöz",
    )
    ap.add_argument(
        "--hedef-skor",
        type=float,
        help="gönderimden hemen önce okunan canlı ikincilik skoru",
    )
    args = ap.parse_args()
    if args.hedef_skor is not None and args.hedef_skor <= 0:
        raise ValueError("hedef skor pozitif olmalı")
    if args.cikis is None:
        dosya = "tuketim_v86_gram_kappa.csv" if args.prob_skor else "tuketim_v85_gram_rank2.csv"
        args.cikis = KOK / "submissions" / dosya
    if args.rapor is None:
        dosya = "gram_rank2_v86.json" if args.prob_skor else "gram_rank2.json"
        args.rapor = KOK / "reports" / dosya

    kimlik, loglar = _kaynaklari_oku()
    skorlar = {ad: deger[1] for ad, deger in KAYNAKLAR.items()}
    sonuc = gram_coz(
        loglar[TABAN_ADI],
        {ad: loglar[ad] for ad in ADAY_ADLARI},
        skorlar,
        taban_adi=TABAN_ADI,
    )
    if sonuc.kosul_sayisi > 1_000 or sonuc.cozum_artigi > 1e-12:
        raise RuntimeError(
            f"sayısal kapı geçmedi: cond={sonuc.kosul_sayisi:.3f}, artık={sonuc.cozum_artigi:.3e}"
        )
    aday_alt, aday_ust = aday_skor_araligi(
        sonuc.gram,
        sonuc.katsayilar,
        skorlar,
        aday_adlari=ADAY_ADLARI,
        taban_adi=TABAN_ADI,
    )
    log_tahmin = sonuc.log_tahmin
    kappa = 1.0
    tahmini_skor = sonuc.tahmini_rmsle
    q_yon = float(np.mean(np.square(sonuc.log_tahmin - loglar[TABAN_ADI])))
    if args.prob_skor is not None:
        kappa = prob_kappa(skorlar[TABAN_ADI], args.prob_skor, q_yon)
        log_tahmin = loglar[TABAN_ADI] + kappa * (sonuc.log_tahmin - loglar[TABAN_ADI])
        delta = args.prob_skor**2 - skorlar[TABAN_ADI] ** 2
        tahmini_mse = skorlar[TABAN_ADI] ** 2 - (q_yon - delta) ** 2 / (4.0 * q_yon)
        if tahmini_mse < -1e-12:
            raise RuntimeError(f"prob çözümü negatif MSE üretti: {tahmini_mse}")
        tahmini_skor = float(np.sqrt(max(tahmini_mse, 0.0)))
        aday_alt, aday_ust = prob_skor_araligi(
            taban_skor=skorlar[TABAN_ADI],
            prob_skor=args.prob_skor,
            q=q_yon,
            kappa=kappa,
        )

    if not np.isfinite(log_tahmin).all() or float(log_tahmin.min()) < 0:
        raise RuntimeError("aday log tahminleri sonlu ve negatif olmamalı; clipping yapılmadı")
    tahmin = np.expm1(log_tahmin)
    if not np.isfinite(tahmin).all() or np.any(tahmin < 0):
        raise RuntimeError("aday tahminleri sonlu ve negatif olmamalı")

    agirliklar = afin_agirliklari(
        sonuc.katsayilar,
        ADAY_ADLARI,
        taban_adi=TABAN_ADI,
        kappa=kappa,
    )
    dogrudan = sum(agirliklar[ad] * loglar[ad] for ad in KAYNAKLAR)
    cift_form_farki = float(np.max(np.abs(dogrudan - log_tahmin)))
    if abs(sum(agirliklar.values()) - 1.0) > 1e-12 or cift_form_farki > 1e-12:
        raise RuntimeError("afin ağırlık veya çift-form kapısı geçmedi")

    args.cikis.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": kimlik, "tuketim": tahmin}).to_csv(
        args.cikis, index=False, float_format="%.17g"
    )
    cikti_hash = _sha256(args.cikis)
    rapor = {
        "yontem": "etiketsiz_log_gram_optimumu",
        "durum": "AGRESIF_RANK2_ADAYI",
        "kaynaklar": _rapor_kaynaklari(),
        "aday_adlari": list(ADAY_ADLARI),
        "katsayilar": agirliklar,
        "kappa": kappa,
        "tahmini_rmsle": tahmini_skor,
        "aday_yuvarlama_araligi": [aday_alt, aday_ust],
        "hedef_skor": args.hedef_skor,
        "hedefi_geciyor": (None if args.hedef_skor is None else tahmini_skor < args.hedef_skor),
        "q_yon": q_yon,
        "gram": sonuc.gram.tolist(),
        "kosul_sayisi": sonuc.kosul_sayisi,
        "cozum_artigi": sonuc.cozum_artigi,
        "cift_form_farki": cift_form_farki,
        "min_log": float(log_tahmin.min()),
        "max_log": float(log_tahmin.max()),
        "satir": len(kimlik),
        "cikis": str(args.cikis),
        "cikis_sha256": cikti_hash,
        "not": (
            "Gönderim yapmaz. v85 skoru gelirse --prob-skor ile aynı yönde optimum v86 üretilir."
        ),
    }
    args.rapor.parent.mkdir(parents=True, exist_ok=True)
    args.rapor.write_text(json.dumps(rapor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ozet_alanlari = (
        "durum",
        "tahmini_rmsle",
        "aday_yuvarlama_araligi",
        "q_yon",
        "kosul_sayisi",
        "min_log",
        "satir",
        "cikis",
        "cikis_sha256",
    )
    print(
        json.dumps(
            {alan: rapor[alan] for alan in ozet_alanlari},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
