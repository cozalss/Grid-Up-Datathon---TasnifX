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

sayim() {
  powershell.exe -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*$1*' }).Count" \
    2>/dev/null | tr -d '\r\n '
}

# ONCE VAR OLDUGUNU DOGRULA, SONRA BITMESINI BEKLE.
#
# Ilk surum yalnizca "sifir surec" ariyordu ve bu 2026-08-24 aksami IKI KEZ
# sessizce yanlis calisti: beklenen betik HENUZ BASLAMAMISSA sayim zaten
# sifirdir, kuyruk "bitmis" sanip hemen kosar ve iki agir is CPU'yu paylasir.
# Artik once oncul surecin GERCEKTEN calistigi dogrulanir; calismiyorsa
# kuyruk sessizce devam etmez, HATA ile durur.
bekle_bitsin() {
  local desen="$1"
  if [ "$(sayim "$desen")" = "0" ]; then
    echo "[$(date +%H:%M:%S)] HATA: '${desen}' calismiyor -- kuyruk yanlis zincirlenmis." | tee -a "$KAYIT"
    echo "[$(date +%H:%M:%S)] Sessizce kosmak yerine duruyorum (bkz. bu betigin yorumu)." | tee -a "$KAYIT"
    return 1
  fi
  while [ "$(sayim "$desen")" != "0" ]; do
    sleep 60
  done
  return 0
}

echo "[$(date +%H:%M:%S)] '${BEKLE}' bitmesi bekleniyor" | tee -a "$KAYIT"
if ! bekle_bitsin "$BEKLE"; then
  exit 1
fi
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
