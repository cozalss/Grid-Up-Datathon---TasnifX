#!/usr/bin/env bash
# Tam olcum zinciri. Kaggle'a HICBIR SEY gondermez.
#
#   a_gbm.py     G1/G2/R1/H1  -- farkli hedef, farkli kayip, farkli rejim siniri
#   b_klasik.py  K1..K4       -- agac olmayan model siniflari
#   c_olc.py     tek basina CV RMSLE + span geometrisi
#   d_varyant.py yon sekillendirme taramasi + BLOK-DISI tasima sinavi
#   f_ofset.py   ofset yonleri (hepsi blok-disi sinavda cokuyor -- kayit icin)
#   g_karar.py   iki rejim, iki prob; submissions/tuketim_y1_*, y2_*
#   h_ozet.py    toplam YENI dik enerji muhasebesi
set -e
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONWARNINGS=ignore
uv run python -u experiments/yeni_yon/a_gbm.py
uv run python -u experiments/yeni_yon/b_klasik.py
uv run python -u experiments/yeni_yon/c_olc.py
uv run python -u experiments/yeni_yon/d_varyant.py
uv run python -u experiments/yeni_yon/f_ofset.py
uv run python -u experiments/yeni_yon/g_karar.py
uv run python -u experiments/yeni_yon/h_ozet.py
uv run python -u scripts/kapi_denetim.py --ref submissions/tuketim_v83_sicak_optimum.csv \
    submissions/tuketim_y1_sicak_klasik.csv submissions/tuketim_y2_soguk_hedef.csv
