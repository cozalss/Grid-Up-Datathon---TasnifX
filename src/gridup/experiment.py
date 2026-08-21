"""Hafif deney takibi -- 12 gunde 100+ deney yapacaksin, hangisinin ne oldugunu unutma.

NEDEN MLFLOW/W&B DEGIL: 12 gunluk bir yarismada kurulum ve ogrenme maliyeti
geri donmez. Gereken sey tek bir JSONL dosyasi ve bir tablo.

NEDEN HIC YOKTAN IYI: 8. gunde "en iyi skorumu hangi feature setiyle almistim?"
sorusuna cevap veremezsen, o skoru bir daha uretemezsin. Bu, yarismalarda
kaybedilen puanlarin sessiz kaynagidir. Ayrica juri notebook'u okurken deney
gecmisi "sistematik calistik" kanitidir.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "COMPETITION_DAYS",
    "DAILY_SUBMISSION_LIMIT",
    "DataArtifact",
    "ExperimentLog",
    "ExperimentProvenance",
    "ExperimentRecord",
    "current_git_sha",
    "sha256_file",
]


#: Gunluk submission siniri. 2023 GDZ Elektrik Datathon'unda 3'tu ve GDZ'nin
#: UC yarismasinda da (2022 case-1, case-2, 2023) istisnasiz 3 olarak olculdu
#: -- Kaggle API max_daily_submissions alanindan dogrulandi. Yarisma gunu
#: acilis yayininda TEYIT ETTIR; degisirse tek satir burada degisir.
DAILY_SUBMISSION_LIMIT = 3

#: Yarisma suresi: 21 Agustos - 1 Eylul 2026.
COMPETITION_DAYS = 12


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DataArtifact:
    """Content-addressed input or output used by an experiment."""

    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> DataArtifact:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Artifact dosyasi bulunamadi: {source}")
        return cls(
            path=str(source),
            size_bytes=source.stat().st_size,
            sha256=sha256_file(source),
        )


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


#: Parmak izi ALINAMADIGINDA kullanilan sentinel oneki.
#:
#: Ucuncu bir durumu isaretler ve bilerek hex'e BENZEMEZ: kapi bunu
#: "kaydedildi" saymamali, okuyan insan da gercek bir ozetle karistirmamali.
HESAPLANAMADI_ONEKI = "HESAPLANAMADI:"

#: ``git`` alt sureclerinin zaman asimi (saniye).
#:
#: 10'DAN 60'A CIKARILDI (2026-08-21, olculdu). Bu depoda agir olcum kosulari
#: (ablasyon: 96 ilce x 1690 gun x 16 aile x 5 tohum) butun cekirdekleri
#: doyurur ve es zamanli calismak ISTISNA DEGIL KURALDIR. O yuk altinda git
#: surecine 10 saniyede sira gelmedi; sonuc, tamamlanmis bir day_one kosusunun
#: submission YAZILDIKTAN SONRA atilmasiydi.
_GIT_ZAMAN_ASIMI_SN = 60


def _git_diff_fingerprint() -> str | None:
    """Kirli agacin diff ozeti. UC durum doner, iki degil.

    Returns:
        * ``None``            -- agac temiz, kaydedilecek bir sey yok
        * sha256 hex          -- diff yakalandi
        * ``HESAPLANAMADI:*`` -- agac kirli olabilir ama diff ALINAMADI

    Ucuncu durum eskiden ``None`` ile ayni kefeye konuyordu ve kapi ikisini
    ayirt edemiyordu. Ayirmak, yeniden uretilebilirlik guvencesini
    ZAYIFLATMAZ: kayit "denendi, olmadi, sebebi su" der. Sessizce yok
    saymaktan da, saatlerce suren bir kosuyu son adimda atmaktan da iyidir.
    """
    try:
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            capture_output=True,
            timeout=_GIT_ZAMAN_ASIMI_SN,
            check=False,
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            capture_output=True,
            timeout=_GIT_ZAMAN_ASIMI_SN,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"{HESAPLANAMADI_ONEKI}git_zaman_asimi_{_GIT_ZAMAN_ASIMI_SN}sn"
    except (OSError, subprocess.SubprocessError) as hata:
        return f"{HESAPLANAMADI_ONEKI}git_calistirilamadi_{type(hata).__name__}"
    if diff.returncode != 0 or untracked.returncode != 0:
        return f"{HESAPLANAMADI_ONEKI}git_hata_kodu_{diff.returncode}_{untracked.returncode}"
    digest = hashlib.sha256()
    changed = bool(diff.stdout or untracked.stdout)
    if not changed:
        return None
    digest.update(diff.stdout)
    for raw_path in untracked.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode(errors="surrogateescape"))
        if not relative.is_file():
            continue
        digest.update(b"\0untracked\0")
        digest.update(raw_path)
        try:
            with relative.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as hata:
            return f"{HESAPLANAMADI_ONEKI}dosya_okunamadi_{type(hata).__name__}"
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "torch",
        "optuna",
        "shap",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _redacted_command(argv: list[str]) -> tuple[str, ...]:
    sensitive = ("password", "passwd", "secret", "token", "api-key", "apikey")
    output: list[str] = []
    redact_next = False
    for value in argv:
        lowered = value.lower()
        if redact_next:
            output.append("<REDACTED>")
            redact_next = False
            continue
        if any(marker in lowered for marker in sensitive):
            if "=" in value:
                output.append(value.split("=", 1)[0] + "=<REDACTED>")
            else:
                output.append(value)
                redact_next = True
            continue
        output.append(value)
    return tuple(output)


@dataclass(frozen=True)
class ExperimentProvenance:
    """Reproduction-critical metadata with a deliberate secret-free schema."""

    recipe_fingerprint: str
    data_artifacts: tuple[DataArtifact, ...]
    feature_names: tuple[str, ...]
    fold_fingerprint: str
    git_sha: str | None
    git_dirty: bool | None
    git_diff_fingerprint: str | None
    python: str
    platform: str
    command: tuple[str, ...]
    package_versions: dict[str, str]

    @classmethod
    def capture(
        cls,
        *,
        recipe_fingerprint: str,
        data_artifacts: list[DataArtifact] | tuple[DataArtifact, ...],
        feature_names: list[str] | tuple[str, ...],
        fold_fingerprint: str,
    ) -> ExperimentProvenance:
        if not recipe_fingerprint:
            raise ValueError("recipe_fingerprint zorunludur")
        if not fold_fingerprint:
            raise ValueError("fold_fingerprint zorunludur")
        return cls(
            recipe_fingerprint=recipe_fingerprint,
            data_artifacts=tuple(data_artifacts),
            feature_names=tuple(feature_names),
            fold_fingerprint=fold_fingerprint,
            git_sha=current_git_sha(),
            git_dirty=_git_dirty(),
            git_diff_fingerprint=_git_diff_fingerprint(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            command=_redacted_command(sys.argv),
            package_versions=_package_versions(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_git_sha() -> str | None:
    """Mevcut git commit SHA'si. Repo yoksa ``None``.

    Deneyi koda baglar: bir skoru yeniden uretmek icin hangi commit'e donecegini
    bilirsin.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
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
    #: Gonderimin YAPILDIGI an. Deneyin olusturulma aninden (``timestamp``)
    #: farklidir: bir deneyi bugun kurup yarin gonderebilirsin ve gunluk
    #: submission butcesi gonderim gunune gore sayilir.
    submitted_at: str | None = None
    timestamp: str = ""
    git_sha: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    run_id: str = ""
    provenance: ExperimentProvenance | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = str(uuid.uuid4())
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

    def record_lb(self, name: str, lb_score: float, *, submitted_at: str | None = None) -> bool:
        """Bir deneye leaderboard skorunu ekler. Bulunursa ``True``.

        En son ayni isimli kaydi gunceller (ayni deneyi tekrar calistirmis
        olabilirsin).

        Args:
            submitted_at: ISO tarih/saat. Verilmezse SIMDI kaydedilir.
                ``submission_budget`` gunluk sayimi bu alandan yapar --
                deneyin OLUSTURULMA zamanindan degil, cunku bir deneyi bugun
                kurup yarin gondermis olabilirsin.
        """
        records = self.load()
        for record in reversed(records):
            if record.get("name") == name:
                record["lb_score"] = lb_score
                record["submitted_at"] = submitted_at or datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                )
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

    def submission_budget(
        self,
        *,
        daily_limit: int = DAILY_SUBMISSION_LIMIT,
        total_days: int = COMPETITION_DAYS,
        today: str | None = None,
    ) -> dict[str, Any]:
        """Gunluk ve toplam submission butcesini raporlar.

        NEDEN GEREKLI: Submission **en kit kaynaktir**. 2023 GDZ Elektrik
        Datathon'unda gunluk limit **3'tu** (GDZ'nin uc yarismasinda da ayni)
        ve 12 gunde toplam ~36 deneme demektir. Uc denemeyi ogleden once
        harcayip aksam daha iyi bir fikir bulmak, o gunu kaybetmektir.

        Ayrica **rastgele %50/%50 public/private bolmesinde** LB'ye bakarak
        secim yapmak asiri uyumdur. Butce, LB'yi ne kadar "yokladigini"
        gorunur kilar.

        Args:
            daily_limit: Gunluk submission siniri.
            total_days: Yarisma gun sayisi.
            today: ``YYYY-AA-GG``. Verilmezse bugunun UTC tarihi.

        Returns:
            ``bugun_kullanilan``, ``bugun_kalan``, ``toplam_kullanilan``,
            ``toplam_kalan``, ``gunluk_dagilim`` ve ``uyari`` anahtarlari.
        """
        today = today or datetime.now(timezone.utc).date().isoformat()
        gunluk: dict[str, int] = {}
        for record in self.load():
            if record.get("lb_score") is None:
                continue
            # Eski kayitlarda submitted_at yok -- deneyin timestamp'ine dus.
            damga = record.get("submitted_at") or record.get("timestamp") or ""
            gun = damga[:10]
            if gun:
                gunluk[gun] = gunluk.get(gun, 0) + 1

        bugun = gunluk.get(today, 0)
        toplam = sum(gunluk.values())
        toplam_hak = daily_limit * total_days

        uyari = ""
        if bugun >= daily_limit:
            uyari = (
                f"BUGUNKU {daily_limit} SUBMISSION BITTI. Kalan fikirleri CV'de "
                "dogrula, yarina sakla."
            )
        elif bugun == daily_limit - 1:
            uyari = "Bugun TEK submission kaldi -- en iyi adayina sakla."

        return {
            "tarih": today,
            "bugun_kullanilan": bugun,
            "bugun_kalan": max(0, daily_limit - bugun),
            "toplam_kullanilan": toplam,
            "toplam_kalan": max(0, toplam_hak - toplam),
            "gunluk_dagilim": dict(sorted(gunluk.items())),
            "uyari": uyari,
        }

    def budget_report(self, **kwargs: Any) -> str:
        """``submission_budget`` ciktisinin okunabilir hali."""
        butce = self.submission_budget(**kwargs)
        satirlar = [
            f"SUBMISSION BUTCESI ({butce['tarih']})",
            "-" * 42,
            f"  bugun  : {butce['bugun_kullanilan']} kullanildi, {butce['bugun_kalan']} kaldi",
            f"  toplam : {butce['toplam_kullanilan']} kullanildi, {butce['toplam_kalan']} kaldi",
        ]
        if butce["gunluk_dagilim"]:
            satirlar.append("  gunluk dagilim:")
            for gun, adet in butce["gunluk_dagilim"].items():
                satirlar.append(f"    {gun}  {'#' * adet} ({adet})")
        if butce["uyari"]:
            satirlar += ["", f"  {butce['uyari']}"]
        return "\n".join(satirlar)

    def leaderboard(self, *, greater_is_better: bool = False, top: int = 25) -> pd.DataFrame:
        """Deneyleri CV skoruna gore siralar."""
        records = self.load()
        if not records:
            return pd.DataFrame(columns=["name", "cv_score", "lb_score", "model_kind"])

        frame = pd.DataFrame(records)
        columns = [
            column
            for column in (
                "name",
                "cv_score",
                "lb_score",
                "metric",
                "model_kind",
                "n_features",
                "notes",
                "timestamp",
                "git_sha",
            )
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
