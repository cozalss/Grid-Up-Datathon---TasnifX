# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""2025 yazi TEMSILI MI? 2020-2026 iklimolojisi + destek denetimi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(r"C:\Users\cemmo\Documents\Datahon")
sys.path.insert(0, str(KOK / "src"))
from gridup.turkish import join_key  # noqa: E402

tr = pd.read_csv(
    KOK / "data/raw/train.csv",
    usecols=["tanim", "tarih", "lokasyon"],
    encoding="utf-8",
    dtype={"tanim": str},
)
tr["ilce_key"] = tr["lokasyon"].str.split(">").str[-1].str.strip().map(join_key)
agir = tr.drop_duplicates("tanim")["ilce_key"].value_counts(normalize=True)

hava = pd.read_parquet(
    KOK / "data/external/hava_gunluk.parquet",
    columns=["ilce_key", "tarih", "sicaklik_ort", "hava_tahmin"],
).drop_duplicates(["ilce_key", "tarih"])
hava["tarih"] = pd.to_datetime(hava["tarih"])
hava["cdd22"] = (hava["sicaklik_ort"] - 22.0).clip(lower=0.0)
h = hava.merge(agir.rename("w"), left_on="ilce_key", right_index=True)
g = h.groupby("tarih").apply(
    lambda q: pd.Series(
        {
            "T": float(np.average(q["sicaklik_ort"], weights=q["w"])),
            "cdd22": float(np.average(q["cdd22"], weights=q["w"])),
            "tahmin": float(np.average(q["hava_tahmin"], weights=q["w"])),
        }
    ),
    include_groups=False,
)

print("=" * 96)
print("NIS-TEM (Nis 1 - Tem 31) IKLIMOLOJISI -- trafo agirlikli ilce ortalamasi")
print("=" * 96)
print(
    f"  {'yil':>6}{'T ort':>9}{'T std':>8}{'cdd22 ort':>11}{'cdd22 std':>11}{'Tem T':>8}{'tahmin%':>9}"
)
sat = {}
for y in range(2020, 2027):
    q = g[(g.index >= f"{y}-04-01") & (g.index <= f"{y}-07-31")]
    if len(q) < 100:
        continue
    tem = q[q.index.month == 7]["T"].mean()
    sat[y] = (q["T"].mean(), q["T"].std(), q["cdd22"].mean(), q["cdd22"].std(), tem)
    print(
        f"  {y:>6}{q['T'].mean():9.2f}{q['T'].std():8.2f}{q['cdd22'].mean():11.3f}"
        f"{q['cdd22'].std():11.3f}{tem:8.2f}{100 * q['tahmin'].mean():9.1f}"
    )
ort = np.mean([v[2] for y, v in sat.items() if y not in (2025, 2026)])
print(
    f"\n  2020-2024 ortalama cdd22 {ort:.3f}  |  2025 {sat[2025][2]:.3f} (%{100 * (sat[2025][2] / ort - 1):+.0f})"
    f"  |  2026 {sat[2026][2]:.3f} (%{100 * (sat[2026][2] / ort - 1):+.0f})"
)
sir = sorted(sat, key=lambda y: -sat[y][2])
print(f"  cdd22 siralamasi (sicaktan soguga): {' > '.join(str(y) for y in sir)}")

print("\n  DESTEK DENETIMI (2026 gunleri 2025'in araliginda mi?)")
q25 = g[(g.index >= "2025-04-02") & (g.index <= "2025-08-01")]
q26 = g[(g.index >= "2026-04-01") & (g.index <= "2026-07-31")]
for k in ("T", "cdd22"):
    print(
        f"    {k:>6}  2025 [{q25[k].min():.2f}, {q25[k].max():.2f}]   2026 [{q26[k].min():.2f}, {q26[k].max():.2f}]"
        f"   2026 destek disi gun: {int(((q26[k] > q25[k].max()) | (q26[k] < q25[k].min())).sum())}"
    )
print(f"    2026 penceresinde forecast satiri: %{100 * q26['tahmin'].mean():.2f}")
