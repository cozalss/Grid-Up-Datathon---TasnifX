#!/usr/bin/env bash
# DENEY KUYRUGU -- bir python isi bitince sirakini baslatir, gozetimsiz.
#
# NEDEN: gece boyunca CPU bos kalmamali ama iki agir egitim isini ayni anda
# kosturmak ikisini de yavaslatir. Bu betik "onceki bitsin, sonraki bassin"
# zincirini kurar ve her adimin cikis kodunu kaydeder.
#
#   bash scripts/deney_kuyrugu.sh "deney_kapasite" \
#        "uv run python scripts/deney_pg_maske.py"
#
# Ilk arguman BEKLENECEK surecin komut satirindaki desen; kalan argumanlar
# sirayla kosulacak komutlar.
set -u
cd "$(dirname "$0")/.." || exit 1
KAYIT="${TMPDIR:-/tmp}/deney_kuyrugu.log"
BEKLE="$1"; shift

bekle_bitsin() {
  local desen="$1"
  while true; do
    local n
    n=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*${desen}*' }).Count" \
      2>/dev/null | tr -d '\r\n ')
    [ "${n:-0}" = "0" ] && return 0
    sleep 60
  done
}

echo "[$(date +%H:%M:%S)] '${BEKLE}' bitmesi bekleniyor" | tee -a "$KAYIT"
bekle_bitsin "$BEKLE"
echo "[$(date +%H:%M:%S)] bitti, kuyruk basliyor" | tee -a "$KAYIT"

for KOMUT in "$@"; do
  echo "[$(date +%H:%M:%S)] KOSUYOR: $KOMUT" | tee -a "$KAYIT"
  eval "$KOMUT" >> "$KAYIT" 2>&1
  KOD=$?
  echo "[$(date +%H:%M:%S)] BITTI (cikis $KOD): $KOMUT" | tee -a "$KAYIT"
  if [ "$KOD" -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] HATA -- kuyruk duruyor" | tee -a "$KAYIT"
    exit "$KOD"
  fi
done
echo "[$(date +%H:%M:%S)] KUYRUK TAMAM" | tee -a "$KAYIT"
