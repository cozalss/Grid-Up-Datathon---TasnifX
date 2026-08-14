"""Merkezi konfigurasyon: yollar, tohum (seed), yarisma sabitleri.

Notebook icinde dagilmis sabit degerler tekrarlanabilirligin bir numarali
dusmanidir. Her sabit BURADA yasar; notebook yalnizca import eder.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["Paths", "CompetitionConfig", "CONFIG", "set_global_seed", "PROJECT_ROOT"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Proje yollari. Frozen: kazara degistirilemez.

    Kaggle notebook'unda calisirken ``Paths.for_kaggle()`` kullan -- veri
    ``/kaggle/input/<yarisma-adi>/`` altindadir ve yazma yalnizca
    ``/kaggle/working/`` icinde mumkundur.
    """

    root: Path = PROJECT_ROOT
    data: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    raw: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    interim: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "interim")
    external: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "external")
    features: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "features")
    models: Path = field(default_factory=lambda: PROJECT_ROOT / "models")
    submissions: Path = field(default_factory=lambda: PROJECT_ROOT / "submissions")
    experiments: Path = field(default_factory=lambda: PROJECT_ROOT / "experiments")
    figures: Path = field(default_factory=lambda: PROJECT_ROOT / "reports" / "figures")

    @classmethod
    def for_kaggle(cls, competition_slug: str) -> Paths:
        """Kaggle notebook ortami icin yollar."""
        working = Path("/kaggle/working")
        return cls(
            root=working,
            data=Path("/kaggle/input") / competition_slug,
            raw=Path("/kaggle/input") / competition_slug,
            interim=working / "interim",
            external=Path("/kaggle/input"),
            features=working / "features",
            models=working / "models",
            submissions=working,
            experiments=working / "experiments",
            figures=working / "figures",
        )

    def ensure(self) -> Paths:
        """Yazilabilir dizinleri olusturur. Kendini dondurur (zincirlenebilir)."""
        for path in (
            self.interim,
            self.features,
            self.models,
            self.submissions,
            self.experiments,
            self.figures,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def is_kaggle(self) -> bool:
        return Path("/kaggle/input").exists()


@dataclass(frozen=True)
class CompetitionConfig:
    """Yarisma sabitleri.

    Veri seti 21 Agustos'ta aciklandiginda DEGISECEK alanlar ``None`` birakildi.
    O gun yalnizca bu dosya guncellenir; pipeline'in geri kalani dokunulmaz.
    """

    name: str = "Grid Up Datathon"
    organizer: str = "Coderspace x GDZ Elektrik x ADM Elektrik"

    # --- Veri aciklandiginda doldurulacak ---
    target_column: str | None = None
    id_column: str | None = None
    time_column: str | None = None
    group_column: str | None = None  # sizinti yaratabilecek grup (or. trafo_id, abone_id)
    task_type: str | None = None  # "regression" | "binary" | "multiclass"
    metric_name: str | None = None  # "rmse" | "rmsle" | "mae" | "auc" | "f1" | "mape"

    # --- Sabitler ---
    seed: int = 42
    n_folds: int = 5
    test_size: float = 0.2

    # GDZ ve ADM hizmet bolgeleri. Veri geldiginde DOGRULA -- bu liste bir
    # varsayimdir ve harici veri (hava, nufus) cekerken filtre olarak kullanilir.
    gdz_provinces: tuple[str, ...] = ("İzmir", "Manisa")
    adm_provinces: tuple[str, ...] = ("Aydın", "Denizli", "Muğla")

    @property
    def all_provinces(self) -> tuple[str, ...]:
        return self.gdz_provinces + self.adm_provinces

    @property
    def is_configured(self) -> bool:
        """Veri seti tanimlandi mi? Pipeline calistirmadan once kontrol et."""
        return all(
            value is not None
            for value in (self.target_column, self.id_column, self.task_type, self.metric_name)
        )

    def require_configured(self) -> None:
        """Yapilandirilmamissa acik bir hata firlatir -- sessiz yanlis sonuc yerine."""
        if not self.is_configured:
            missing = [
                name
                for name in ("target_column", "id_column", "task_type", "metric_name")
                if getattr(self, name) is None
            ]
            raise ValueError(
                f"CompetitionConfig eksik: {missing}. "
                "Veri seti aciklandiktan sonra src/gridup/config.py dosyasini guncelle."
            )


CONFIG = CompetitionConfig()
PATHS = Paths()


def set_global_seed(seed: int = CONFIG.seed) -> int:
    """Tum rastgelelik kaynaklarini sabitler ve kullanilan tohumu dondurur.

    Notebook'un ilk hucresinde cagir. Juri tekrarlanabilirlige bakiyor: ayni
    notebook'u iki kez calistirinca ayni skoru vermesi somut bir puan kalemidir.

    NOT: LightGBM/XGBoost ``num_threads > 1`` ile tam determinizmi GARANTI ETMEZ.
    Bitwise tekrarlanabilirlik gerekiyorsa ``deterministic=True`` (LightGBM) ve
    tek is parcacigi kullan -- yavaslama pahasina.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed
