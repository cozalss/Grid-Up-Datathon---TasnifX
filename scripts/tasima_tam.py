"""TAM TASIMA: baska bir makineye HER SEYI gecirir ve karsi tarafta dogrular.

``tasima_paketi.py`` yalnizca ``data/`` ve ``submissions/`` tasir. Bu betik
makineyi degistirmek icin gereken DORT parcanin hepsini toplar:

    1. VERI            data/raw + data/interim   ~600 MB
                       Yeniden indirilebilir ama pahali: data/external
                       altindaki Open-Meteo kontrol noktalari kotayla
                       alindi, yeniden yakmak gunluk limite takilir.

    2. GONDERIMLER     secili CSV'ler            ~220 MB
                       Hepsi degil -- submissions/ 1,3 GB ve cogu tarihsel.
                       Yalniz LB referansi, kazanan dosya, 15 tohumluk ham
                       tahmin ve bes tohum partisi tasinir. Partiler
                       yeniden uretilemez sayilir: her biri ~85 dakika.

    3. CLAUDE GECMISI  ~/.claude/projects/<slug>/  ~510 MB
                       Konusma gecmisi, alt-ajan dokumleri ve memory/.
                       YENIDEN URETILEMEZ. Klasor adi PROJE YOLUNUN
                       slug'idir; hedef makinede yol farkliysa yeniden
                       adlandirilmasi gerekir -- KURULUM.md bunu yazar.

    4. GIZLI DOSYALAR  .env + kaggle.json        ~1 KB
                       Depoda YOK ve olmamali. Pakete ayri bir klasore
                       konur ki karsi tarafta bilerek yerlestirilsin.

KOD TASINMAZ: GitHub'da. Hedefte ``git clone`` yeterli.
.venv TASINMAZ: Windows sanal ortami mutlak yol gomer. ``uv sync`` ile
yeniden kurulur.

DOGRULAMA
---------
Yarim kopyalanmis bir parquet okunurken degil MODEL EGITILIRKEN patlar --
ya da hic patlamaz, sessizce eksik satirla devam eder. Bu yuzden her
dosyanin SHA-256'si manifest'e yazilir ve hedefte tek tek dogrulanir.

    python scripts/tasima_tam.py --hedef E:/DATAHON_TASIMA
    python scripts/tasima_tam.py --dogrula E:/DATAHON_TASIMA   (hedef makinede)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]

#: Butunuyle tasinan dizinler. ``data`` TAMAMI tasinir, alt dizin secilmez:
#: data/external (285 MB) Open-Meteo kota kontrol noktalaridir ve yeniden
#: cekmek gunluk limite takilir; bir alt dizini elle saymak onu atlama
#: riski dogurur. data/prior'un kucuk bir kismi git'te de var, iki kez
#: tasinmasi zararsiz.
VERI_DIZINLERI = ("data",)

#: submissions/ 1,3 GB; yalniz bunlar tasinir.
GONDERIMLER = (
    "tuketim_v47_eskison.csv",    # KAZANAN, LB 1,01750
    "tuketim_v46_ham15.csv",      # 15 tohumluk HAM tahmin -> 30 tohuma buradan gidilir
    "tuketim_v30_buzme.csv",      # LB referansi 1,02639
    "tuketim_v32_ham.csv",        # tohum partisi 1 (100-102)
    "tuketim_v34_ek3tohum.csv",   # parti 2 (103-105)
    "tuketim_v38_ek3tohum.csv",   # parti 3 (106-108)
    "tuketim_v41_ek3tohum.csv",   # parti 4 (109-111)
    "tuketim_v42_ek3tohum.csv",   # parti 5 (112-114)
)

#: Depoda olmayan ama gereken gizli dosyalar: (kaynak, paket icindeki ad).
GIZLI = ((KOK / ".env", "env.txt"), (Path.home() / ".kaggle/kaggle.json", "kaggle.json"))

MANIFEST = "manifest.json"


def slug(yol: Path) -> str:
    """Claude Code'un proje klasoru adi.

    Kural: yolun harf-disi her karakteri ``-`` olur. Windows'ta SURUCU
    HARFI KUCUKTUR -- olculdu: C:/Users/cemmo/Documents/Datahon icin gercek
    klasor ``c--Users-cemmo-Documents-Datahon``, buyuk C degil. Bu tek harf
    kacsa gecmis hedef makinede acilmaz, o yuzden acikca kucultuluyor.
    """
    metin = str(yol.resolve())
    if len(metin) > 1 and metin[1] == ":":
        metin = metin[0].lower() + metin[1:]
    return "".join(c if c.isalnum() else "-" for c in metin)


def gecmis_klasoru(yol: Path) -> Path | None:
    """Once hesaplanan slug'a bakar; tutmazsa var olan klasoru arar."""
    kok = Path.home() / ".claude/projects"
    aday = kok / slug(yol)
    if aday.exists():
        return aday
    ad = yol.name
    bulunan = [d for d in kok.glob(f"*-{ad}") if d.is_dir()]
    return bulunan[0] if len(bulunan) == 1 else None


def ozet(yol: Path) -> str:
    h = hashlib.sha256()
    with yol.open("rb") as fh:
        for parca in iter(lambda: fh.read(1 << 20), b""):
            h.update(parca)
    return h.hexdigest()


def paketle(hedef_kok: Path) -> int:
    t0 = time.time()
    hedef_kok.mkdir(parents=True, exist_ok=True)
    kayitlar: list[dict] = []
    toplam = 0

    def ekle(kaynak: Path, goreli: str, grup: str) -> None:
        nonlocal toplam
        hedef = hedef_kok / goreli
        hedef.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kaynak, hedef)
        b = hedef.stat().st_size
        toplam += b
        kayitlar.append({"grup": grup, "yol": goreli, "boyut": b, "sha256": ozet(hedef)})

    print("=" * 78)
    print("TAM TASIMA PAKETI")
    print("=" * 78)

    # --- 1. VERI ---
    for d in VERI_DIZINLERI:
        kok = KOK / d
        if not kok.exists():
            print(f"  UYARI: {d} yok, atlaniyor")
            continue
        n = 0
        for p in sorted(kok.rglob("*")):
            if p.is_file():
                ekle(p, f"veri/{p.relative_to(KOK).as_posix()}", "veri")
                n += 1
        print(f"  veri  {d:20} {n:5d} dosya")

    # --- 2. GONDERIMLER ---
    eksik = []
    for ad in GONDERIMLER:
        p = KOK / "submissions" / ad
        if p.exists():
            ekle(p, f"gonderimler/{ad}", "gonderim")
        else:
            eksik.append(ad)
    print(f"  gonderim {len(GONDERIMLER) - len(eksik)}/{len(GONDERIMLER)} dosya"
          + (f"  EKSIK: {eksik}" if eksik else ""))

    # --- 3. CLAUDE GECMISI ---
    gecmis = gecmis_klasoru(KOK)
    if gecmis is not None and gecmis.exists():
        n = 0
        for p in sorted(gecmis.rglob("*")):
            if p.is_file():
                ekle(p, f"claude-gecmis/{p.relative_to(gecmis).as_posix()}", "claude")
                n += 1
        print(f"  claude gecmisi {n:5d} dosya  (kaynak klasor: {gecmis.name})")
    else:
        print(f"  UYARI: Claude gecmisi bulunamadi (slug {slug(KOK)})")

    # --- 4. GIZLI ---
    for kaynak, ad in GIZLI:
        if kaynak.exists():
            ekle(kaynak, f"gizli/{ad}", "gizli")
            print(f"  gizli  {ad}")
        else:
            print(f"  UYARI: {kaynak} yok")

    (hedef_kok / MANIFEST).write_text(
        json.dumps({"kaynak_slug": slug(KOK), "kaynak_yol": str(KOK),
                    "dosyalar": kayitlar}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (hedef_kok / "KURULUM.md").write_text(kurulum_metni(slug(KOK)), encoding="utf-8")

    print("-" * 78)
    print(f"  {len(kayitlar)} dosya, {toplam / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} dakika")
    print(f"  yazildi: {hedef_kok}")
    print("  SONRAKI ADIM: hedef makinede KURULUM.md")
    return 0


def dogrula(paket: Path) -> int:
    m = json.loads((paket / MANIFEST).read_text(encoding="utf-8"))
    print(f"  manifest: {len(m['dosyalar'])} dosya  (kaynak {m['kaynak_yol']})")
    bozuk, kayip = [], []
    for i, k in enumerate(m["dosyalar"], 1):
        p = paket / k["yol"]
        if not p.exists():
            kayip.append(k["yol"])
        elif p.stat().st_size != k["boyut"] or ozet(p) != k["sha256"]:
            bozuk.append(k["yol"])
        if i % 200 == 0:
            print(f"    {i}/{len(m['dosyalar'])} ...")
    if kayip or bozuk:
        print(f"\n  KAYIP {len(kayip)}  BOZUK {len(bozuk)}")
        for y in (kayip + bozuk)[:10]:
            print(f"    {y}")
        return 1
    print(f"\n  TAMAM: {len(m['dosyalar'])} dosyanin hepsi SHA-256 ile dogrulandi.")
    print(f"  Kaynak makinedeki proje yolu: {m['kaynak_yol']}")
    print(f"  Claude gecmis klasoru bu ada konmali: {m['kaynak_slug']}")
    print("  Hedef yol FARKLIYSA klasoru yeni yola gore adlandirin (KURULUM.md).")
    return 0


def kurulum_metni(kaynak_slug: str) -> str:
    return f"""# Laptop kurulumu

Kaynak makinedeki Claude gecmis klasoru: `{kaynak_slug}`

## 0. Once dogrula (5 dk)

```powershell
cd <paketin oldugu yer>
python dogrula_paket.py .     # ya da depodan: python scripts/tasima_tam.py --dogrula .
```

Bozuk dosya varsa DEVAM ETME -- yarim parquet egitim sirasinda patlar ya da
sessizce eksik satirla devam eder.

## 1. Depoyu klonla

```powershell
cd $env:USERPROFILE\\Documents
git clone https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git Datahon
cd Datahon
```

**Yolu ayni tutmaya calisin**: `C:\\Users\\<kullanici>\\Documents\\Datahon`.
Kullanici adi ayniysa Claude gecmisi hicbir degisiklik olmadan oturur.

## 2. Python ortami (internet gerekir, ~10 dk)

```powershell
winget install --id=astral-sh.uv -e     # uv yoksa
uv sync
```

`.venv` PAKETTE YOK -- Windows sanal ortami mutlak yol gomer, kopyalanamaz.

## 3. Veriyi yerlestir

```powershell
Copy-Item -Recurse -Force .\\veri\\data $env:USERPROFILE\\Documents\\Datahon\\
Copy-Item -Force .\\gonderimler\\*.csv $env:USERPROFILE\\Documents\\Datahon\\submissions\\
```

## 4. Gizli dosyalar

```powershell
Copy-Item .\\gizli\\env.txt $env:USERPROFILE\\Documents\\Datahon\\.env
New-Item -ItemType Directory -Force $env:USERPROFILE\\.kaggle
Copy-Item .\\gizli\\kaggle.json $env:USERPROFILE\\.kaggle\\kaggle.json
```

## 5. Claude konusma gecmisi

Hedef klasor: `%USERPROFILE%\\.claude\\projects\\<slug>`

`<slug>` proje yolunun harf-disi her karakterinin `-` ile degistirilmis
halidir. Yol `C:\\Users\\cemmo\\Documents\\Datahon` ise slug
`{kaynak_slug}` olur.

```powershell
$hedefYol = "$env:USERPROFILE\\Documents\\Datahon"
$yol = $hedefYol.Substring(0,1).ToLower() + $hedefYol.Substring(1)   # surucu harfi KUCUK
$harf = {{ if ($_ -match '[a-zA-Z0-9]') {{ $_ }} else {{ '-' }} }}
$slug = ($yol.ToCharArray() | ForEach-Object $harf) -join ''
$hedef = "$env:USERPROFILE\\.claude\\projects\\$slug"
New-Item -ItemType Directory -Force $hedef
Copy-Item -Recurse -Force .\\claude-gecmis\\* $hedef
Write-Host "Gecmis su klasore kondu: $hedef"
```

Kullanici adiniz `cemmo` ise slug birebir ayni cikar ve gecmis oldugu gibi
acilir. Farkliysa yukaridaki betik dogru adi kendisi hesaplar.

Sonra:

```powershell
cd $env:USERPROFILE\\Documents\\Datahon
claude --resume        # eski oturumlar listelenir
```

`memory/` klasoru de bu kopyayla geliyor -- kalici notlar (Kaggle gonderim
yetkisi, heredoc tuzagi, cp1254 hatasi) korunur.

## 6. Calistigini dogrula

```powershell
uv run python scripts/butunluk_son_islem.py
```

`TAMAM: uretim son islemi ile olcum tezgahi ayni sayiyi veriyor` yazmali.
Bu tek komut hem veriyi, hem onbellegi, hem de son islem kodunu sinar.

## 7. Kaldigi yerden devam

Sirada 15 -> 30 tohum var (olculmus ~0,0016, risksiz):

```powershell
uv run python scripts/tuketim_model.py --tohum 3 `
    --tohum-baslangic 115 --dogrulama-atla --cikti tuketim_v48.csv
uv run python scripts/tuketim_model.py --tohum 3 `
    --tohum-baslangic 118 --dogrulama-atla --cikti tuketim_v49.csv
# ... 5 parti = 15 yeni tohum, sonra:
uv run python scripts/birlestir_tohum.py --cikis ham30.csv `
    submissions/tuketim_v46_ham15.csv:15 yeni15.csv:15
uv run python scripts/son_islem.py --giris ham30.csv --cikis nihai30.csv
```

**son_islem_gun.py KULLANILMAZ** -- LB'de curutuldu (docs/39).
"""


def main() -> int:
    a = argparse.ArgumentParser(description="tam tasima paketi")
    a.add_argument("--hedef", help="paketin yazilacagi klasor (USB/harici disk)")
    a.add_argument("--dogrula", help="hedef makinede: paketi dogrula")
    ar = a.parse_args()
    if ar.dogrula:
        return dogrula(Path(ar.dogrula))
    if not ar.hedef:
        a.error("--hedef ya da --dogrula verin")
    h = Path(ar.hedef)
    if h.exists() and any(h.iterdir()):
        print(f"UYARI: {h} bos degil, icine yazilacak.")
    return paketle(h)


if __name__ == "__main__":
    raise SystemExit(main())
