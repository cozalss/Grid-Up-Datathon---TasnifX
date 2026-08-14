"""Hafif deney takibi -- 12 gunde 100+ deney yapacaksin, hangisinin ne oldugunu unutma.

NEDEN MLFLOW/W&B DEGIL: 12 gunluk bir yarismada kurulum ve ogrenme maliyeti
geri donmez. Gereken sey tek bir JSONL dosyasi ve bir tablo.

NEDEN HIC YOKTAN IYI: 8. gunde "en iyi skorumu hangi feature setiyle almistim?"
sorusuna cevap veremezsen, o skoru bir daha uretemezsin. Bu, yarismalarda
kaybedilen puanlarin sessiz kaynagidir. Ayrica juri notebook'u okurken deney
gecmisi "sistematik calistik" kanitidir.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["ExperimentRecord", "ExperimentLog", "current_git_sha"]


def current_git_sha() -> str | None:
    """Mevcut git commit SHA'si. Repo yoksa ``None``.

    Deneyi koda baglar: bir skoru yeniden uretmek icin hangi commit'e donecegini
    bilirsin.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


@dataclass
class ExperimentRecord:
    """Tek bir deney kaydi."""

    name: str
    cv_score: float
    metric: str
    model_kind: str
    n_features: int
    fold_scores: list[float] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    features: list[str] = field(default_factory=list)
    notes: str = ""
    lb_score: float | None = None
    submission_path: str | None = None
    timestamp: str = ""
    git_sha: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.git_sha is None:
            self.git_sha = current_git_sha()
        if not self.environment:
            self.environment = {
                "python": sys.version.split()[0],
                "platform": platform.system(),
            }

    @property
    def fold_std(self) -> float:
        if not self.fold_scores:
            return 0.0
        mean = sum(self.fold_scores) / len(self.fold_scores)
        variance = sum((score - mean) ** 2 for score in self.fold_scores) / len(self.fold_scores)
        return variance**0.5


class ExperimentLog:
    """JSONL tabanli deney defteri.

    Kullanim::

        log = ExperimentLog("experiments/deneyler.jsonl")
        log.add(ExperimentRecord(
            name="lgbm_takvim_lag",
            cv_score=result.overall_score,
            metric="rmse",
            model_kind="lightgbm",
            n_features=len(features.columns),
            fold_scores=result.fold_scores,
            notes="lag[1,7,28] + TR tatil eklendi",
        ))
        print(log.leaderboard())

    Submission gonderdikten SONRA leaderboard skorunu geri yaz::

        log.record_lb("lgbm_takvim_lag", 12.3456)

    Bu, CV-LB korelasyonunu izlemeni saglar -- shakeup'tan korunmanin tek yolu.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, record: ExperimentRecord) -> ExperimentRecord:
        """Kaydi ekler ve geri dondurur."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def load(self) -> list[dict[str, Any]]:
        """Tum kayitlari okur. Dosya yoksa bos liste."""
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    print(f"UYARI: {self.path}:{line_number} bozuk JSON, atlandi.")
        return records

    def record_lb(self, name: str, lb_score: float) -> bool:
        """Bir deneye leaderboard skorunu ekler. Bulunursa ``True``.

        En son ayni isimli kaydi gunceller (ayni deneyi tekrar calistirmis
        olabilirsin).
        """
        records = self.load()
        for record in reversed(records):
            if record.get("name") == name:
                record["lb_score"] = lb_score
                break
        else:
            # Notebook'ta donus degeri kolayca gozden kacar. Sessizce False
            # donmek, CV-LB korelasyonunun eksik veriyle hesaplanmasina yol acar.
            available = sorted({str(record.get("name")) for record in records})
            print(
                f"UYARI: '{name}' adli deney bulunamadi, LB skoru KAYDEDILMEDI. "
                f"Mevcut deney adlari: {available[:10]}"
            )
            return False

        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

    def leaderboard(self, *, greater_is_better: bool = False, top: int = 25) -> pd.DataFrame:
        """Deneyleri CV skoruna gore siralar."""
        records = self.load()
        if not records:
            return pd.DataFrame(columns=["name", "cv_score", "lb_score", "model_kind"])

        frame = pd.DataFrame(records)
        columns = [
            column
            for column in ("name", "cv_score", "lb_score", "metric", "model_kind",
                           "n_features", "notes", "timestamp", "git_sha")
            if column in frame.columns
        ]
        return (
            frame[columns]
            .sort_values("cv_score", ascending=not greater_is_better)
            .head(top)
            .reset_index(drop=True)
        )

    def cv_lb_correlation(self) -> dict[str, Any]:
        """CV ve leaderboard skorlari arasindaki korelasyon.

        YORUM -- bu, yarismanin en onemli tek sayisidir:
          r > 0.8   -> CV'ne GUVEN. Karari CV ile ver, LB'yi yalnizca dogrulama
                       icin kullan.
          0.5 - 0.8 -> Zayif iliski. Fold sayisini artir veya CV semasini gozden
                       gecir (muhtemelen zaman/grup sizintisi var).
          r < 0.5   -> CV seman YANLIS. Duzeltmeden devam etmek, private
                       leaderboard'da coke gitmenin garantisidir.

        En az 5 eslesmis nokta gerekir; daha azi anlamsizdir.
        """
        records = [
            record
            for record in self.load()
            if record.get("lb_score") is not None and record.get("cv_score") is not None
        ]

        if len(records) < 5:
            return {
                "n": len(records),
                "correlation": None,
                "note": (
                    f"Yalnizca {len(records)} eslesmis deney var. En az 5 gerekli. "
                    "Her submission'dan sonra log.record_lb(...) cagirmayi unutma."
                ),
            }

        frame = pd.DataFrame(records)
        correlation = float(frame["cv_score"].corr(frame["lb_score"]))
        rank_correlation = float(frame["cv_score"].corr(frame["lb_score"], method="spearman"))

        if abs(rank_correlation) > 0.8:
            note = "GUCLU iliski -> CV'ne guven, kararlari CV ile ver."
        elif abs(rank_correlation) > 0.5:
            note = "ZAYIF iliski -> fold sayisini artir veya CV semasini gozden gecir."
        else:
            note = (
                "ILISKI YOK -> CV seman muhtemelen YANLIS (zaman/grup sizintisi). "
                "Once bunu duzelt; aksi halde private LB'de coke gidersin."
            )

        return {
            "n": len(records),
            "correlation": correlation,
            "rank_correlation": rank_correlation,
            "note": note,
        }
