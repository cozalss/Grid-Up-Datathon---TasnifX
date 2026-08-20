"""EPIAS plansiz/planli kesinti gecmisini indirir -- GDZ + ADM bolgesi.

NEDEN BU EN DEGERLI VERI
------------------------
Sema (olculdu)::

    province  district  date  distributionCompanyName  reason
    startTime endTime   effectedNeighbourhoods  effectedSubscribers  hourlyLoadAvg

Bu, yarismanin hedef degiskeninin GECMISE DONUK GERCEK HALIDIR -- ustelik
ariza sebebi ve etkilenen abone sayisiyla birlikte. Dort ise yarar:

1. **Sifir orani ve asiri yayilimi OLCEBILIRIZ.** Faz 2 arastirmasi "sifir
   orani muhtemelen %30'un altinda" dedi ama bu bir CIKARIMDI. Simdi
   gercek panelde sayabiliriz. Bu tek sayi, iki asamali modelin gerekip
   gerekmedigini belirliyor.

2. **Ilce adlarinin RESMI YAZIMI.** Yarisma verisi ayni kurumsal kaynaklardan
   gelecek; join anahtarini bugun kurabiliriz.

3. **Ariza sebebi dagilimi.** Hangi sebep hangi mevsimde/hava kosulunda
   artiyor -- feature hipotezlerini gercek veriyle test edebiliriz.

4. **Tum pipeline'in provasi.** Panel kurma, ufuk-farkindalikli lag, hava
   birlestirme, CV -- hepsi 21 Agustos'tan ONCE gercek veride denenir.

API NOTLARI (olculdu)
---------------------
* ``period`` bir TEK GUNDUR, ay degil. Tum Turkiye icin gunde ~2000-3000 kayit.
* Yanit ``{"items": [...], "page": null}`` bicimindedir; ``page`` verilmezse
  tum kayitlar doner (sayfalama siniri yok).
* Cok gunluk aralik icin gun gun donmek gerekir.

Calistirma::

    python scripts/fetch_epias_outages.py --start 2022-01-01 --end 2026-08-01
    python scripts/fetch_epias_outages.py --start 2025-06-01 --end 2025-06-30 --planned
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_weather import rate_limit_beklemesi  # noqa: E402

from gridup.epias import EpiasClient, EpiasRequestError, load_env_file  # noqa: E402
from gridup.io_utils import publish_dataframe, validate_published_dataframe  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "external" / "epias"

# GDZ (Izmir, Manisa) + ADM (Aydin, Denizli, Mugla). join_key ile
# karsilastirilir; EPIAS il adlarini BUYUK HARF yazar (olculdu: 'YOZGAT').
TARGET_PROVINCES = {"izmir", "manisa", "aydin", "denizli", "mugla"}

UNPLANNED_PATH = "consumption/data/unplanned-power-outage-info"
PLANNED_PATH = "consumption/data/planned-power-outage-info"

# API'ye nazik ol. Open-Meteo'da 429 yedik; burada bastan yavas gidiyoruz.
DEFAULT_PAUSE = 0.35
RATE_LIMIT_BACKOFF = (30, 90, 240)


def daterange(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_day(
    client: EpiasClient, path: str, day: date, *, retries: int = 3, pause: float
) -> pd.DataFrame:
    """Tek gunun kesinti kayitlarini ceker ve hedef illere filtreler.

    Raises:
        EpiasRequestError: Tum denemeler basarisiz olursa. Sessizce bos frame
            dondurmez -- eksik gun, farkedilmeden paneli bozar.
    """
    body = {"period": f"{day.isoformat()}T00:00:00+03:00"}
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            frame = client.to_frame(client.post(path, body))
            break
        except EpiasRequestError as error:
            last_error = error
            if "429" in str(error) and attempt < retries:
                # EPIAS istemcisi ham yanit dondurmez; govde metni yerine
                # hata metnini veriyoruz -- pencere ipucu varsa yakalanir.
                wait, gerekce = rate_limit_beklemesi(
                    SimpleNamespace(text=str(error), headers={}),
                    attempt,
                    merdiven=RATE_LIMIT_BACKOFF,
                )
                print(f"    hiz siniri; {wait} sn bekleniyor ({gerekce})")
                time.sleep(wait)
                continue
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise EpiasRequestError(f"{day}: {retries} denemede alinamadi -- {error}") from error
    else:  # pragma: no cover
        raise EpiasRequestError(f"{day}: alinamadi -- {last_error}")

    if frame.empty or "province" not in frame.columns:
        return frame

    # join_key ile filtrele: EPIAS 'MUĞLA' yazar, biz 'mugla' ariyoruz.
    # Ham .lower() burada SESSIZCE bos donerdi -- 'İZMİR'.lower() != 'izmir'.
    frame = frame.assign(
        il_key=frame["province"].astype(str).map(join_key),
        ilce_key=frame["district"].astype(str).map(join_key),
    )
    return frame[frame["il_key"].isin(TARGET_PROVINCES)].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument(
        "--planned", action="store_true", help="Plansiz yerine PLANLI kesintileri cek"
    )
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE)
    parser.add_argument("--fresh", action="store_true", help="Mevcut dosyayi yok say")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    client = EpiasClient.from_env()

    path = PLANNED_PATH if args.planned else UNPLANNED_PATH
    label = "planli" if args.planned else "plansiz"
    output = OUTPUT_DIR / f"kesinti_{label}.parquet"
    source = f"epias://{path}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    all_days = list(daterange(start, end))

    # DEVAM ETME: mevcut dosyadaki gunleri atla.
    existing: pd.DataFrame | None = None
    done: set[str] = set()
    if output.exists() and not args.fresh:
        try:
            existing = validate_published_dataframe(
                output,
                required_columns=("date", "province", "district"),
                min_rows=1,
                source=source,
            )
        except (OSError, ValueError) as error:
            print(f"Mevcut kesinti cache'i dogrulanamadi; kullanilmayacak: {error}")
            existing = None
        if existing is not None and "date" in existing.columns:
            done = set(existing["date"].astype(str).str[:10].unique())
        print(f"Mevcut dosyada {len(done)} gun var -- atlanacak.")

    pending = [day for day in all_days if day.isoformat() not in done]
    print(f"{len(pending)} gun indirilecek ({label} kesinti, 5 il).")
    if not pending:
        print("Yapilacak is yok.")
        return 0

    collected: list[pd.DataFrame] = []
    failures: list[str] = []
    started = time.perf_counter()

    for index, day in enumerate(pending, start=1):
        try:
            frame = fetch_day(client, path, day, pause=args.pause)
        except EpiasRequestError as error:
            print(f"  [{index}/{len(pending)}] {day} BASARISIZ -- {str(error)[:90]}")
            failures.append(day.isoformat())
            continue

        collected.append(frame)

        if index % 25 == 0 or index == len(pending):
            rows = sum(len(item) for item in collected)
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0
            remaining = (len(pending) - index) / rate if rate else 0
            print(
                f"  [{index}/{len(pending)}] {day}  toplam {rows:,} satir  "
                f"~{remaining / 60:.0f} dk kaldi"
            )

        # Ara kayit: uzun indirmede coksek bile ilerleme kaybolmasin.
        if index % 200 == 0:
            _save(collected, existing, output, source=source)

        time.sleep(args.pause)

    combined = _save(collected, existing, output, source=source)

    print(f"\nYazildi: {output}")
    print(f"  {len(combined):,} satir x {combined.shape[1]} kolon")
    if "date" in combined.columns:
        days = combined["date"].astype(str).str[:10]
        print(f"  Tarih araligi: {days.min()} -> {days.max()}  ({days.nunique()} gun)")
    if "il_key" in combined.columns:
        print(f"  Il dagilimi:\n{combined['il_key'].value_counts().to_string()}")
    if "ilce_key" in combined.columns:
        print(f"  Benzersiz ilce: {combined['ilce_key'].nunique()}")

    if failures:
        print(f"\n  UYARI: {len(failures)} gun alinamadi. Ornekler: {failures[:5]}")
        print("  Ayni komutu tekrar calistir -- eksik gunler tamamlanir.")
        return 1
    return 0


def _save(
    collected: list[pd.DataFrame],
    existing: pd.DataFrame | None,
    output: Path,
    *,
    source: str,
) -> pd.DataFrame:
    """Toplananlari mevcutla birlestirip yazar ve birlesigi dondurur."""
    parts = [frame for frame in collected if not frame.empty]
    if existing is not None and not existing.empty:
        parts.insert(0, existing)
    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset=["id"])
    publish_dataframe(
        combined,
        output,
        required_columns=("date", "province", "district"),
        min_rows=1,
        source=source,
    )
    return combined


if __name__ == "__main__":
    raise SystemExit(main())
