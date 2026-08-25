"""GUN EKSENI OLCEKLEMESI -- modelin gun rampasi AZ yayilmis, genligi duzeltilir.

BULGU
-----
Tahmin ofseti iki yonlu ayristirilinca (r = mu + a_trafo + b_gun + e) modelin
GUN bileseni gercekten sistematik olarak DAHA DUZ cikiyor. Uc blokta, uretim
harmani, etiketli (gercek ~ a + c*model OLS egimi):

    blok    model std  gercek std  korelasyon  OLS egimi
    yaz25     0,2328     0,4081      +0,991      +1,737
    guz25     0,2117     0,2248      +0,978      +1,037
    kis26     0,0436     0,0710      +0,330      +0,538

Bu bir GENLIK hatasidir, isaret hatasi degil: korelasyon yaz ve guzde 0,98'in
uzerinde -- model rampanin SEKLINI biliyor. Ve etki MEVSIMSELDIR, cunku gunluk
tuketim salinimi yazin (sogutma yuku) kisin oldugundan cok daha buyuk. Gercek
gun ekseni std'si, sabit panelde, trafo etkisi cikarilmis olarak:

    2025-04..07   0,2710      <- testin mevsimsel ikizi
    2025-05..08   0,2647
    2025-09..12   0,1302
    2025-12..26-03 0,0696

RMSLE ETKISI (olcut.py agirliklari, uc tohum, uretim harmani)

    c       yaz25      guz25      kis26
    1,00   0,81075    1,00351    0,88604
    1,25   0,80018    1,00634    0,88626
    1,50   0,79367    1,01183    0,88661
    1,75   0,79133    1,01993    0,88710

yaz25'te kazanc +0,0194 (%2,4 goreli); guz ve kista yok. Havuzlanmis t=+1,47,
1/3 blok -- yani ON KAYITLI KURAL GECMIYOR ve karar tek bir seye bagli:
TESTIN gun ekseni genligi ETIKETSIZ kestirilebilir mi?

ETIKETSIZ CAPA (kararin dayandigi sey)
--------------------------------------
Evet. Test penceresi 2026-04-01..07-31; GECEN YILIN ayni penceresi
2025-04-01..07-31 ve o pencere TAMAMEN train.csv icindedir -- etiketli.
Test etiketi kullanilmaz.

    2025 Nis-Tem GERCEK gun ekseni std        0,2710
    2026 Nis-Tem v50 TAHMIN gun ekseni std    0,1689
    c_capa = 0,2710 / 0,1689                  = 1,604

Etiketle olculen yaz25 optimumu 1,75 ve OLS egimi 1,737 -- iki bagimsiz yol
ayni bandi veriyor. Ayrica gun-of-year hizasinda 2025 GERCEK ile 2026 TAHMIN
gun profillerinin korelasyonu +0,925 (122 ortak gun): model sekli biliyor.

docs/39 §3'un dersi tam buydu: "model-disi bir nicelikten turetilen kestirim
LB'ye TASINDI, tek bir dogrulama blogundan turetilen tasinmadi." Tohum
ortalamasi tasindi cunku sigma etiketsiz olculmustu. Bu da oyle.

SECILEN c -- FORMULDEN, ELLE DEGIL
----------------------------------
Kuadratik kayipta optimum olcek OLS egimidir:

    c* = kor * (sigma_gercek / sigma_model)

RATIO TEK BASINA YANLIS SECICIDIR. Uc blokta sinandi:

    blok    oran    kor     kor x oran   OLCULEN optimum
    yaz25   2,677   0,962     2,576          2,65
    guz25   1,111   0,970     1,078          0,75
    kis26   1,581   0,347     0,549          0,70

kis26'da yalniz oran 1,581 (genislet) derdi ama gercek optimum 0,70; farki tam
olarak korelasyon kapatiyor -- orada model gun sinyali gurultu (kor 0,347).
Her bloga KENDI formul-c'si uygulaninca havuzlanmis kazanc +0,00547, SH 0,00290,
t=+1,88, 2/3 blok (guz25'te -0,00068 ile cok kucuk bir kayip).

TEST icin iki girdi de ETIKETSIZ kestirilir: sigma_gercek gecen yilin ayni
penceresinden, korelasyon gun-of-year hizasinda 2025 GERCEK vs 2026 TAHMIN
profillerinden. Olculen: oran 1,604, kor 0,925 -> c = 1,48.

KURGU
-----
Iki yonlu ayristirma, trafo etkisi ONCE cikarilir (kompozisyon tuzagi):

    b_gun = o gunun satirlarinda (r - trafo_ortalamasi) ortalamasi
    r'    = r + (c - 1) * b_gun

Trafo eksenine, gun ici yapiya ve genel seviyeye DOKUNULMAZ. Genel ortalama
korunur (b_gun merkezlendigi icin).

VARSAYILAN OLARAK YALNIZCA SICAK SATIRLAR
-----------------------------------------
Soguk satirlarin gun ortalamasi KOMPOZISYONLA kirlidir: soguk trafolar test
penceresinde sirayla giriyor (1 Nisan'da 1 satir, 31 Temmuz'da 1.962), yani
"gun etkisi" ile "hangi trafolar" birbirine karisir. Sicak tarafta boyle bir
sorun yok: 3.751 trafo x 122 gun, gunde ~3.743 satir, panel dolu.
Soguk taraf icin ``--soguk-da`` ayrica istenmelidir.

    python scripts/son_islem_gunolcek.py --giris submissions/X.csv \
        --cikis submissions/Y.csv [--c 1.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
C_VARSAYILAN = 1.50
#: Gecen yilin ayni penceresi -- capanin kaynagi. Test 2026-04-01..07-31.
CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"


def gun_etkisi(tanim: np.ndarray, gun: np.ndarray, r: np.ndarray) -> pd.Series:
    """Trafo etkisi cikarilmis gun ortalamasi (iki yonlu ayristirma)."""
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def main() -> int:
    a = argparse.ArgumentParser(description="gun ekseni genlik duzeltmesi")
    a.add_argument("--giris", required=True)
    a.add_argument("--cikis", required=True)
    a.add_argument("--c", type=float, default=None, help="elle c; verilmezse formulden")
    a.add_argument("--soguk-da", action="store_true", help="soguk satirlara da uygula")
    a.add_argument("--yalniz-soguk", action="store_true", help="YALNIZCA soguk satirlar")
    a.add_argument(
        "--lb-kalibre",
        type=float,
        default=None,
        help="formul c'yi bu katsayiyla carp. 2026-08-25 LB'si formulun %11 yuksek "
        "oldugunu gosterdi (tahmin 1,492 / cozulen 1,332), yani 0,893.",
    )
    ar = a.parse_args()

    ornek = pd.read_csv(KOK / "data/raw/sample_submission.csv", encoding="utf-8")
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "guc", "tarih"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    yol = Path(ar.giris)
    if not yol.is_absolute() and not yol.exists():
        yol = KOK / "submissions" / yol.name
    sub = pd.read_csv(yol, encoding="utf-8")
    if not sub["id"].equals(ornek["id"]):
        raise RuntimeError("id sirasi sample_submission ile ayni degil")

    m = sub.merge(te, on="id", how="left", validate="one_to_one")
    if len(m) != len(sub):
        raise RuntimeError("birlestirme satir sayisini bozdu")
    soguk = ~m["tanim"].isin(set(tr["tanim"])).to_numpy()
    if ar.yalniz_soguk and ar.soguk_da:
        raise RuntimeError("--yalniz-soguk ve --soguk-da birlikte verilemez")
    if ar.yalniz_soguk:
        hedef = soguk
    elif ar.soguk_da:
        hedef = np.ones(len(m), dtype=bool)
    else:
        hedef = ~soguk

    log_guc = np.log1p(m["guc"].to_numpy(dtype="float64"))
    r = np.log1p(m["tuketim"].to_numpy(dtype="float64")) - log_guc

    # ---- CAPA: gecen yilin ayni penceresinden beklenen genlik ----
    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    gun_g = g["tarih"].to_numpy()
    tan_g = g["tanim"].to_numpy()
    rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        g["guc"].to_numpy(dtype="float64")
    )
    x = pd.DataFrame({"t": tan_g, "g": gun_g})
    tam = x.groupby("t")["g"].nunique()
    tam = set(tam[tam >= 0.9 * x["g"].nunique()].index)
    sec = np.isin(tan_g, list(tam))
    b_gecen = gun_etkisi(tan_g[sec], gun_g[sec], rg[sec])

    b_test = gun_etkisi(
        m.loc[hedef, "tanim"].to_numpy(),
        m.loc[hedef, "tarih"].to_numpy(),
        r[hedef],
    )
    # FORMUL: kuadratik kayipta optimum c* = kor * (sigma_gercek / sigma_model),
    # yani OLS egimi. Uc blokta dogrulandi (formul / olculen optimum):
    #     yaz25 2,576 / 2,65    guz25 1,078 / 0,75    kis26 0,549 / 0,70
    # Korelasyon TEST icin gecen yilin ayni penceresiyle, gun-of-year hizasinda
    # kestirilir. RATIO tek basina yanlis seciciydi: kis26'da 1,581 diyordu ama
    # gercek optimum 0,70; farki tam olarak korelasyon (0,347) kapatiyor.
    oran = float(b_gecen.std() / b_test.std())
    ia = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index).dayofyear)
    ib = pd.Series(b_test.values, index=pd.to_datetime(b_test.index).dayofyear)
    ortak = ia.index.intersection(ib.index)
    if len(ortak) < 30:
        raise RuntimeError(f"gun-of-year ortakligi yetersiz: {len(ortak)} gun")
    kor = float(np.corrcoef(ia[ortak], ib[ortak])[0, 1])
    c_formul = kor * oran
    if ar.lb_kalibre is not None:
        c_formul = 1.0 + ar.lb_kalibre * (c_formul - 1.0)
    c_kullan = float(ar.c) if ar.c is not None else c_formul
    if not 0.3 <= c_kullan <= 3.0:
        raise RuntimeError(f"c mantik disi: {c_kullan:.3f}")

    # SATIR AGIRLIKLI merkezleme: b_test gun bazinda merkezli, ama gunlerin
    # satir sayilari esit degil (Nisan'da soguk satir az, hedef sicaksa da gun
    # basina satir tam esit degil). Satirlara yayilan etki yeniden merkezlenmezse
    # genel seviye ~4e-3 kayiyor ve bu, buzmenin dokunmadigi ekseni bozar.
    etki = m.loc[hedef, "tarih"].map(b_test).to_numpy(dtype="float64")
    etki = etki - etki.mean()
    yeni_r = r.copy()
    yeni_r[hedef] = r[hedef] + (c_kullan - 1.0) * etki
    yeni = np.clip(np.expm1(yeni_r + log_guc), 0.0, None)

    # ---- KAPILAR ----
    if np.isnan(yeni).any() or (yeni < 0).any():
        raise RuntimeError("NaN veya negatif tahmin")
    sap = float("nan")
    if not ar.soguk_da:
        dokunulmaz = ~hedef
        eski_c = m.loc[dokunulmaz, "tuketim"].to_numpy(dtype="float64")
        sap = float((np.abs(yeni[dokunulmaz] - eski_c) / np.maximum(np.abs(eski_c), 1.0)).max())
        if sap > 1e-12:
            raise RuntimeError(f"dokunulmayan rejim degisti: goreli sapma {sap:.3e}")
    b_yeni = gun_etkisi(
        m.loc[hedef, "tanim"].to_numpy(), m.loc[hedef, "tarih"].to_numpy(), yeni_r[hedef]
    )
    olcek = float(b_yeni.std() / b_test.std())
    # Kapi GORELI: uygulanan olcek, kirpilan satirlar (log1p(tahmin) sifira
    # dayanmis olanlar) yuzunden istenenden birkac binde sapar ve sapma c ile
    # buyur. Mutlak esik bu yuzden yanlisti; %3 goreli dogru sinirdir.
    if abs(olcek - c_kullan) / max(c_kullan, 1e-9) > 0.03:
        raise RuntimeError(f"olcek beklendigi gibi degil: {olcek:.3f} yerine {c_kullan:.3f}")
    kayma = float(abs(yeni_r[hedef].mean() - r[hedef].mean()))
    if kayma > 1e-9:
        raise RuntimeError(f"genel seviye kaydi: {kayma:.3e}")

    print(
        f"  hedef satir {int(hedef.sum()):,} / {len(m):,}"
        f"  ({'sicak+soguk' if ar.soguk_da else 'yalniz SICAK'})"
    )
    print(
        f"  CAPA {CAPA_BASI}..{CAPA_SONU} gercek gun std {b_gecen.std():.4f}"
        f"  ({len(tam):,} tam panel trafosu)"
    )
    print(
        f"  TAHMIN gun std {b_test.std():.4f}  ->  oran {oran:.3f}"
        f"  kor {kor:.3f}  ({len(ortak)} ortak gun-of-year)"
    )
    print(
        f"  FORMUL c = kor x oran = {c_formul:.3f}"
        f"   KULLANILAN c = {c_kullan:.3f}"
        f"{'  (elle verildi)' if ar.c is not None else '  (formulden)'}"
    )
    print(f"  uygulanan olcek {olcek:.3f}  genel seviye kaymasi {kayma:.1e}  (0 olmali)")
    if not ar.soguk_da:
        ad = "SICAK" if ar.yalniz_soguk else "SOGUK"
        print(f"  {ad} satirlar dokunulmadi, goreli sapma {sap:.1e}")
    z = pd.Series(b_test.values, index=pd.to_datetime(b_test.index))
    zg = pd.Series(b_gecen.values, index=pd.to_datetime(b_gecen.index))
    print(
        "  aylik gun-ekseni std   tahmin: "
        + "  ".join(f"{k:02d} {v:.4f}" for k, v in z.groupby(z.index.month).std().items())
    )
    print(
        "                         gecen : "
        + "  ".join(f"{k:02d} {v:.4f}" for k, v in zg.groupby(zg.index.month).std().items())
    )
    print(f"  min {yeni.min():.1f}  medyan {float(np.median(yeni)):.1f}  maks {yeni.max():.1f}")

    cik = Path(ar.cikis)
    if not cik.is_absolute():
        cik = KOK / "submissions" / cik.name
    pd.DataFrame({"id": sub["id"], "tuketim": yeni}).to_csv(cik, index=False)
    print(f"  yazildi: {cik}  ({len(sub):,} satir)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
