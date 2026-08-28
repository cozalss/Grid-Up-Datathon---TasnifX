"""Sicak uzmanin GORDUGU / GORMEDIGI kolonlari sayar. Salt okuma."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa: E402

ONB = KOK / "data" / "interim" / "deney"
egitim = pd.read_parquet(ONB / "egitim.parquet")
test = pd.read_parquet(ONB / "test.parquet")

ham = [k for k in tm.oznitelikler(egitim) if k in test.columns]
kalan = [k for k in ham if not k.startswith(tm.YALIN_CIKARILAN)]
atilan = [k for k in ham if k.startswith(tm.YALIN_CIKARILAN)]

print(f"HAM       : {len(ham)}")
print(f"KALAN     : {len(kalan)}   (sicak uzmanin gordugu)")
print(f"ATILAN    : {len(atilan)}")
print()
for onek in tm.YALIN_CIKARILAN:
    grup = sorted(k for k in atilan if k.startswith(onek))
    print(f"--- {onek!r}  ({len(grup)}) ---")
    for k in grup:
        print(f"    {k}")
print()
print("=== SICAK UZMANIN GORDUGU KOLONLAR ===")
for k in sorted(kalan):
    print(f"    {k}")
print()
print("=== TAKVIM ILE ILGILI OLUP KALANLAR (arama) ===")
anahtar = ("gun", "hafta", "ay", "mevsim", "yil", "tarih", "okul", "donem")
for k in sorted(kalan):
    if any(a in k for a in anahtar):
        print(f"    {k}")
