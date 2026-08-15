"""Ortak pytest yapilandirmasi.

HYPOTHESIS PROFILLERI
---------------------
``varsayilan`` -- her kosuda calisir, hizli (150 ornek/test)
``derin``      -- kasitli derin arama (2000 ornek/test), yavas

Derin profili calistirmak::

    set HYPOTHESIS_PROFILE=derin && python -m pytest tests/test_ozellik.py

Veri gelmeden ONCE bir kez derin kos; sonra her degisiklikte varsayilan yeter.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "varsayilan",
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "derin",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "varsayilan"))
