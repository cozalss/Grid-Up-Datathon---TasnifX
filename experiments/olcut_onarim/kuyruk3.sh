cd "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
uv run python experiments/olcut_onarim/tezgah.py ekkoken > experiments/olcut_onarim/tezgah_ekkoken.log 2>&1
echo "EKKOKEN BITTI"
uv run python experiments/olcut_onarim/uret_v107.py --yeni cat_d7_lr03_rs4 --tohum 3 > experiments/olcut_onarim/uret.log 2>&1
echo "URET BITTI"
