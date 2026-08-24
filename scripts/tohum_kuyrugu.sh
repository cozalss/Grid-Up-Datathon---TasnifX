#!/usr/bin/env bash
# TOHUM KUYRUGU -- partileri sirayla, gozetimsiz kosar.
#
# NEDEN: 15 -> 30 tohum olculmus ve RISKSIZ tek kanal (docs/39 §6, ~0,0017).
# Yanliligi degistirmez; ayni tahmincinin daha kararli kestirimini verir.
# Bir parti (3 tohum) ~85 dakika, yani bes parti ~7 saat. Elle beslemek
# demek her partide bir insan beklemek demek -- kuyruk bunu kaldirir.
#
# Once calisan bir uretim kosusu varsa onun BITMESINI bekler; CPU'yu ikiye
# bolmek iki isi de yavaslatir.
#
#   bash scripts/tohum_kuyrugu.sh 118 121 124 127
set -u
cd "$(dirname "$0")/.." || exit 1
KAYIT="${TMPDIR:-/tmp}/tohum_kuyrugu.log"

for BAS in "$@"; do
  # onceki parti bitene kadar bekle
  while true; do
    N=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*tuketim_model.py*' }).Count" 2>/dev/null | tr -d '\r\n ')
    [ "${N:-0}" = "0" ] && break
    sleep 60
  done

  CIKTI="tuketim_v48_p$(( (BAS - 115) / 3 + 1 )).csv"
  echo "[$(date +%H:%M:%S)] parti basliyor: tohum $BAS..$((BAS+2))  -> $CIKTI" | tee -a "$KAYIT"
  uv run python scripts/tuketim_model.py \
      --tohum 3 --tohum-baslangic "$BAS" --dogrulama-atla \
      --cikti "$CIKTI" >> "$KAYIT" 2>&1
  KOD=$?
  echo "[$(date +%H:%M:%S)] parti bitti: tohum $BAS  cikis kodu $KOD" | tee -a "$KAYIT"
  if [ "$KOD" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] HATA -- kuyruk duruyor, kalan partiler kosulmadi" | tee -a "$KAYIT"
    exit "$KOD"
  fi
done
echo "[$(date +%H:%M:%S)] KUYRUK TAMAM -- butun partiler bitti" | tee -a "$KAYIT"
