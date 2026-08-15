"""EPIAS Seffaflik: Turkiye geneli saatlik TUKETIM ve URETIM cekicisi.

NEDEN BU VERI
-------------
2023 GDZ Elektrik Datathon **birincisinin** iki model asamasindan birinin
tamami buna dayaniyordu. Kendi ifadesiyle::

    "Nihai Cozumde Kullanilan Dis Veriler: ...
     EPIAS Seffaflik Real Time Generation - Minimum 24 lag
     (Turkiye'nin toplam enerji uretimi)"

Ulke capindaki tuketim/uretim, tek bir ilcenin verisinde GORUNMEYEN ortak
sinyali tasir: ekonomik aktivite, ulke capinda hava dalgasi, tatil
davranisi, sanayi duruslari. Yerel yukun aciklanamayan kisminin buyuk
bolumu buradadir.

Uretim tarafinda YAKIT KIRILIMI ham toplamdan daha bilgilendiricidir::

    ruzgar + gunes payi yuksek   -> hava acik ve ruzgarli
    dogalgaz + ithal komur payi  -> tepe yuk karsilaniyor
    barajli hidro payi           -> su rejimi / kuraklik sinyali

SIZINTI UYARISI -- BU EN ONEMLI KISIM
--------------------------------------
Bu seriler GECMIS GOZLEMDIR, tahmin degildir. Tahmin aninda yalnizca o ana
kadar YAYIMLANMIS degerler elindedir.

Birinci lag(24) kullandi cunku ufku 24 SAATTI. Ufkun bir AY ise lag(24)
SIZINTIDIR: 20 Agustos'u tahmin ederken 19 Agustos'un uretimi henuz
olusmamistir.

Kural: **lag >= tahmin ufku.** ``features.temporal.add_lag_features``
``horizon`` parametresiyle bunu zorlar -- oradan gec, elle shift YAPMA.

KULLANIM
--------
::

    python scripts/fetch_epias_load.py                      # 2020-2026
    python scripts/fetch_epias_load.py --start 2024-01-01   # kisa aralik
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridup.epias import EpiasClient, EpiasRequestError, load_env_file  # noqa: E402

#: API tek istekte cok uzun araligi reddediyor; aylik parcalar guvenli.
CHUNK_DAYS = 31

#: Istekler arasi bekleme. EPIAS agresif istekte 429 donuyor.
PAUSE_SECONDS = 1.0

#: Bir parca basarisiz olursa kac kez denenecek.
MAX_RETRY = 3


def _iso(gun: date, saat: str = "00:00:00") -> str:
    """EPIAS'in bekledigi ISO + TR saat dilimi bicimi."""
    return f"{gun.isoformat()}T{saat}+03:00"


def _parcalar(baslangic: date, bitis: date) -> list[tuple[date, date]]:
    """Araligi ayliğa yakin parcalara boler (uc uca, cakismasiz)."""
    parcalar = []
    imlec = baslangic
    while imlec < bitis:
        parca_sonu = min(imlec + timedelta(days=CHUNK_DAYS), bitis)
        parcalar.append((imlec, parca_sonu))
        imlec = parca_sonu
    return parcalar


def _cek(istemci: EpiasClient, ad: str, fn, baslangic: date, bitis: date) -> pd.DataFrame:
    """Bir seriyi parca parca ceker ve birlestirir."""
    parcalar = _parcalar(baslangic, bitis)
    print(f"\n{ad}: {len(parcalar)} parca ({baslangic} -> {bitis})")

    toplanan: list[pd.DataFrame] = []
    basarisiz: list[str] = []

    for indeks, (bas, son) in enumerate(parcalar, start=1):
        for deneme in range(1, MAX_RETRY + 1):
            try:
                frame = fn(start=_iso(bas), end=_iso(son, "23:00:00"))
                if not frame.empty:
                    toplanan.append(frame)
                print(f"  [{indeks:>3}/{len(parcalar)}] {bas} {len(frame):>4} satir")
                break
            except EpiasRequestError as hata:
                if deneme == MAX_RETRY:
                    basarisiz.append(f"{bas}: {str(hata)[:80]}")
                    print(f"  [{indeks:>3}/{len(parcalar)}] {bas} BASARISIZ")
                else:
                    time.sleep(PAUSE_SECONDS * 2 * deneme)
        time.sleep(PAUSE_SECONDS)

    if basarisiz:
        # SESSIZ ATLAMA YOK: eksik parca, veride fark edilmeyen bir delik
        # birakir ve lag/rolling feature'lari o delikte sessizce yanlis olur.
        print(f"  UYARI: {len(basarisiz)} parca alinamadi:")
        for satir in basarisiz[:5]:
            print(f"    {satir}")

    if not toplanan:
        return pd.DataFrame()
    return pd.concat(toplanan, ignore_index=True)


def _zaman_kolonu_kur(frame: pd.DataFrame) -> pd.DataFrame:
    """``date`` + ``time``/``hour`` kolonlarindan tek bir ``zaman`` uretir."""
    if frame.empty:
        return frame
    sonuc = frame.copy()
    tarih = pd.to_datetime(sonuc["date"], errors="coerce", format="mixed", utc=True)
    # EPIAS 'date' zaten tam zaman damgasi tasiyor; ayri saat kolonu varsa
    # dogrulama icin kullaniyoruz ama zamani date'ten aliyoruz.
    sonuc.insert(0, "zaman", tarih.dt.tz_convert("Europe/Istanbul").dt.tz_localize(None))
    return sonuc.drop(columns=[c for c in ("date", "time", "hour") if c in sonuc.columns])


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--start", default="2020-01-01")
    ayristirici.add_argument("--end", default="2026-09-01")
    ayristirici.add_argument("--out", default="data/external/epias")
    args = ayristirici.parse_args()

    baslangic = date.fromisoformat(args.start)
    bitis = date.fromisoformat(args.end)

    ortam = load_env_file(ROOT / ".env")
    if not ortam:
        print("HATA: .env okunamadi. EPIAS_USERNAME ve EPIAS_PASSWORD gerekli.")
        return 1
    os.environ.update(ortam)

    istemci = EpiasClient.from_env()
    cikti_dizini = ROOT / args.out
    cikti_dizini.mkdir(parents=True, exist_ok=True)

    isler = (
        ("tuketim", istemci.realtime_consumption, "tuketim_saatlik.parquet"),
        ("uretim", istemci.realtime_generation, "uretim_saatlik.parquet"),
    )

    ozet: list[str] = []
    for ad, fn, dosya in isler:
        frame = _zaman_kolonu_kur(_cek(istemci, ad, fn, baslangic, bitis))
        if frame.empty:
            ozet.append(f"  {ad:10} BOS -- hicbir parca alinamadi")
            continue
        # Parca sinirlarinda tekrar olabilir; zaman anahtarinda tekille.
        onceki = len(frame)
        frame = frame.drop_duplicates(subset="zaman").sort_values("zaman").reset_index(drop=True)
        yol = cikti_dizini / dosya
        frame.to_parquet(yol, index=False)

        bosluk = ""
        if len(frame) > 1:
            beklenen = int((frame.zaman.max() - frame.zaman.min()).total_seconds() // 3600) + 1
            eksik = beklenen - len(frame)
            bosluk = f" | {eksik} saat EKSIK" if eksik > 0 else " | bosluksuz"
        ozet.append(
            f"  {ad:10} {len(frame):>6,} satir x {frame.shape[1]:>2} kolon"
            f" ({onceki - len(frame)} tekrar atildi){bosluk}"
        )

    print("\n" + "=" * 62)
    print("OZET")
    print("=" * 62)
    for satir in ozet:
        print(satir)
    print(f"\nKaydedildi: {cikti_dizini}")
    print("\nHATIRLATMA: bunlar GECMIS gozlemdir. Feature'a cevirirken")
    print("add_lag_features(horizon=TAHMIN_UFKU) kullan -- elle shift YAPMA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
