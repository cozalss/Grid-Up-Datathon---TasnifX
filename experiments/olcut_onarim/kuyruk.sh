set -e
cd "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
until grep -q "TAMAM" experiments/olcut_onarim/tezgah_temel.log; do sleep 30; done
uv run python experiments/olcut_onarim/tezgah.py kimlik > experiments/olcut_onarim/tezgah_kimlik.log 2>&1
uv run python experiments/olcut_onarim/tezgah.py ayar   > experiments/olcut_onarim/tezgah_ayar.log 2>&1
echo "KUYRUK BITTI"
