cd "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
until [ -f experiments/olcut_onarim/kuyruk.log ] && grep -q "KUYRUK BITTI" experiments/olcut_onarim/kuyruk.log; do sleep 30; done
uv run python experiments/olcut_onarim/tezgah.py kontrol > experiments/olcut_onarim/tezgah_kontrol.log 2>&1
echo "KUYRUK2 BITTI"
