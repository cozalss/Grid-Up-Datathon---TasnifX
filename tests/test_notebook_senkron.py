"""Commit'li notebook'lar ureticiden SAPMAMALI.

2026-08-18 denetimi olctu: ``02_baseline.ipynb`` ureticinin bir onceki
halinden uretilmisti -- ``CVRecipe`` ambargo alani yoktu, provenance kaydi
"0 gun ambargo" diyordu; jurinin okudugu notebook, gercekte kosandan BASKA
bir semayi belgeliyordu. ``test_recipe_and_provenance_contract`` uretici
kaynagini denetliyor, .ipynb'yi degil. Bu test ikisini baglar: uretici
gecici dizine yazar, hucre kaynaklari commit'li dosyayla birebir esit olmali.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
NOTEBOOKLAR = KOK / "notebooks"


def _uretici():
    yol = KOK / "scripts" / "build_notebooks.py"
    spec = importlib.util.spec_from_file_location("build_notebooks", yol)
    modul = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(modul)
    return modul


def _hucreler(yol: Path) -> list[tuple[str, str]]:
    veri = json.loads(yol.read_text(encoding="utf-8"))
    return [(h["cell_type"], "".join(h["source"])) for h in veri["cells"]]


@pytest.mark.parametrize(
    "dosya, hucre_ozniteligi",
    [("01_kesif.ipynb", "EDA_CELLS"), ("02_baseline.ipynb", "BASELINE_CELLS")],
)
def test_notebook_ureticiyle_birebir(tmp_path: Path, dosya: str, hucre_ozniteligi: str) -> None:
    m = _uretici()
    hedef = tmp_path / dosya
    m.write_notebook(getattr(m, hucre_ozniteligi), hedef)
    commitli = NOTEBOOKLAR / dosya
    assert commitli.exists(), f"{dosya} yok; python scripts/build_notebooks.py"
    assert _hucreler(hedef) == _hucreler(commitli), (
        f"{dosya} ureticiden sapmis. Calistir: python scripts/build_notebooks.py ve commit et."
    )


def test_baseline_notebook_gun1_kapilarini_iceriyor() -> None:
    """Denetimde kapatilan uc kapinin notebook'ta OLDUGUNU dogrula."""
    kaynak = "\n".join(k for _, k in _hucreler(NOTEBOOKLAR / "02_baseline.ipynb"))
    assert "sample=sample_submission" in kaynak
    assert "align_to_sample=sample_submission is not None" in kaynak
    assert "strict_provenance=not IS_KAGGLE" in kaynak
    assert "duplicated([GROUP_COLUMN, TIME_COLUMN])" in kaynak
    assert 'splitter="purged_time_series"' in kaynak
