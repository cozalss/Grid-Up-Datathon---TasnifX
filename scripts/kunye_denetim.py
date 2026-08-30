"""DIS VERI BEYANI EKSIKSIZLIK DENETIMI.

Yarisma yukumlulugu: kullanilan TUM dis veri beyan edilmeli. 30 Agustos'ta bir
eksik bulundu (EPIAS ulusal saatlik tuketim, uretim modeli girdisiydi ama
kunyede yoktu). Bu betik ayni eksigin tekrarini yakalar.

Uc yonlu karsilastirma:
  1. diskte duran dis veri dosyalari
  2. data/sources.yml'de kayitli olanlar
  3. URETIM kodunun gercekten okudugu yollar (kaynak taramasi)

Yalnizca (3) ile (2) arasindaki fark ONEMLIDIR: uretimde okunan her kaynak
beyan edilmis olmali. (1)'deki fazlalik ara-urun/arastirma dosyasi olabilir.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
ARANAN = (".parquet", ".csv", ".geojson", ".json")
VERI_KOKLERI = ("data/external", "data/reference", "data/prior")
#: Uretim boru hattini olusturan yollar. Bunlarin okudugu her sey beyan edilmeli.
URETIM = ("scripts/tuketim_model.py", "scripts/sota_tuketim_pipeline.py", "src/gridup/")
DESEN = re.compile(
    r"[\"']((?:data/)?(?:external|reference|prior)/[^\"']+\.(?:parquet|csv|geojson|json))[\"']"
)


def kayitli_kume():
    with (KOK / "data/sources.yml").open(encoding="utf-8") as fh:
        man = json.load(fh)
    return {a["path"].replace("\\", "/") for a in man["artifacts"]}


def diskteki_dosyalar(*, gizli_dahil=True):
    bulunan = []
    for kok in VERI_KOKLERI:
        d = KOK / kok
        if not d.is_dir():
            continue
        for dizin, alt, dosyalar in os.walk(d):
            if not gizli_dahil:
                alt[:] = [x for x in alt if not x.startswith(".")]
            for f in dosyalar:
                if f.endswith(ARANAN):
                    bulunan.append((Path(dizin) / f).relative_to(KOK).as_posix())
    return bulunan


def koddaki_yollar():
    bulunan = {}
    for dizin, alt, dosyalar in os.walk(KOK):
        alt[:] = [
            x for x in alt if x not in {".git", ".venv", "node_modules"} and not x.startswith(".")
        ]
        for f in dosyalar:
            if not f.endswith(".py"):
                continue
            yol = Path(dizin) / f
            try:
                icerik = yol.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for m in DESEN.findall(icerik):
                n = m if m.startswith("data/") else "data/" + m
                bulunan.setdefault(n.replace("\\", "/"), []).append(yol.relative_to(KOK).as_posix())
    return bulunan


def main():
    kayitli = kayitli_kume()
    diskte = set(diskteki_dosyalar())
    kod = koddaki_yollar()

    print("=" * 74)
    print("URETIM KODUNUN OKUDUGU DIS KAYNAKLAR")
    print("=" * 74)
    uretim_eksik = []
    for yol, kaynaklar in sorted(kod.items()):
        uretimde = [k for k in kaynaklar if any(u in k for u in URETIM)]
        if not uretimde:
            continue
        var = yol in kayitli
        if not var:
            uretim_eksik.append((yol, uretimde))
        print(f"  {'KAYITLI' if var else '*** KUNYEDE YOK ***':20s} {yol}")

    print("\n" + "=" * 74)
    print("OZET")
    print("=" * 74)
    beyansiz = [y for y, _ in uretim_eksik if (KOK / y).exists()]
    hayalet = [y for y, _ in uretim_eksik if not (KOK / y).exists()]
    print(f"  uretimde okunan, kunyede OLMAYAN ve diskte VAR : {len(beyansiz)}")
    for y in beyansiz:
        print(f"     !!! BEYAN EKSIK: {y}")
    if hayalet:
        print(f"  kodda gecen ama diskte olmayan (yorum/ornek yol): {len(hayalet)}")
        for y in hayalet:
            print(f"     (gormezden gelinir) {y}")
    if not beyansiz:
        print("  -> uretim boru hattinin okudugu her dis kaynak BEYAN EDILMIS")

    fazla = sorted(diskte - kayitli)
    grup = Counter()
    for f in fazla:
        if f.endswith(".metadata.json"):
            grup["yan kunye (*.metadata.json)"] += 1
        elif "/." in f:
            grup["gizli ara-urun dizini"] += 1
        else:
            grup[str(Path(f).parent)] += 1
    print(f"\n  diskte olup kunyede olmayan {len(fazla)} dosya (uretimde okunmuyorsa sorun degil):")
    for k, v in grup.most_common(8):
        print(f"     {v:4d}  {k}")
    return 1 if beyansiz else 0


if __name__ == "__main__":
    raise SystemExit(main())
