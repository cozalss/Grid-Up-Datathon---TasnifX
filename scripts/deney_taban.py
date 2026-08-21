"""SICAK UZMANININ HEDEF TABANI -- yonlendirmenin actigi kapi.

NEDEN BU DENEY
--------------
Hedef su ana kadar ``log1p(y) - log1p(guc)`` idi. Kapasite ofseti tek
basina -0,035 kazandirdi ve mekanizmasi sudur: agaclar bir kimlik
eslemesini merdivenlerle yaklastirir; tabani hedeften cikarmak o yuku
tamamen kaldirir.

Ayni mekanizma cok daha guclu uygulanabilir -- AMA ancak yonlendirmeden
SONRA. Once tek model hem sicak hem soguk rejime hizmet ediyordu, o yuzden
taban ikisinde de tanimli olmak zorundaydi; gecmise dayali bir taban soguk
satirlarda YOK. Yonlendirme bu kisiti kaldirdi: sicak uzmani artik yalnizca
gecmisi OLAN trafolara hizmet ediyor.

Olculdu (2026-08-21 gece, 916.781 sicak satir) -- modelin modellemesi
gereken artigin standart sapmasi:

    taban YOK (ham log1p)     2,0979
    log1p(guc)   [MEVCUT]     1,8220     <- bu -0,035 kazandirdi
    t_log_ort                 0,9139
    t_log_son90               0,8389
    t_log_son30               0,8062
    (t_log_ort + son30)/2     0,7915     <- %57 dusus

Ve kritik nokta: ``(ort+son30)/2`` HIC PARAMETRE ICERMIYOR. Sabit yari
yariya ortalama; ozet penceresinden geliyor, yani etiket penceresinden
once biter. Sicak satirlardaki RMSE'si 0,7915, yonlendirmeli harmanimizin
sicak skoru 0,8128.

Bu, aksam yapip duzelttigim hatanin TERSI: o zaman ufka gore agirliklari
degerlendirdigim satirlarin kendisine uydurmustum (ornekleme ici, haksiz).
Burada uydurulacak bir sey yok.

ADAYLAR (hepsi YALNIZCA SICAK SATIRLARDA olculur)
    W0  maske 0,15, tum satirlar, taban log1p(guc)      MEVCUT
    W1  maske 0,00, YALNIZ sicak satirlar, log1p(guc)   egitim kumesi degisimi
    W2  maske 0,00, yalniz sicak, taban t_log_ort
    W3  maske 0,00, yalniz sicak, taban (ort+son30)/2
    W4  maske 0,00, yalniz sicak, taban t_log_son30

W1 bilerek var: W2-W4'un kazancinin ne kadari TABANDAN, ne kadari
"soguk satirlari egitimden atmak"tan geliyor -- ayirmadan bilinemez.

Fit: 3 blok x 3 tohum x 5 aday = 45 CatBoost ~ 26 dakika.

Calistirma::

    python scripts/deney_taban.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402


def taban_uret(cerceve: pd.DataFrame, bicim: str) -> np.ndarray:
    """Hedeften cikarilacak log-uzayi tabanini uretir.

    Gecmise dayali tabanlarda NaN kalirsa (gecmisi var ama son 30 gunde
    kaydi yok -- sicak satirlarin %0,98'i) ``t_log_ort``a, o da yoksa
    ``log1p(guc)``e duser. Zincir bilerek boyle: her adim bir oncekinden
    daha az bilgi tasiyor ama daha genis kapsiyor.
    """
    lg = np.log1p(cerceve["guc"].to_numpy())
    if bicim == "guc":
        return lg
    ort = cerceve["t_log_ort"].to_numpy()
    son30 = cerceve["t_log_son30"].to_numpy()
    if bicim == "ort":
        ham = ort
    elif bicim == "son30":
        ham = son30
    elif bicim == "ort_son30":
        ham = np.where(np.isnan(son30), ort, 0.5 * (ort + son30))
    else:
        raise ValueError(f"bilinmeyen taban bicimi: {bicim}")
    ham = np.where(np.isnan(ham), ort, ham)
    return np.where(np.isnan(ham), lg, ham)


def egit_tabanli(
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    bicim: str,
) -> np.ndarray:
    """CatBoost'u verilen tabanla egitir, LOG UZAYINDA tahmin dondurur."""
    e_taban = taban_uret(egitim, bicim)
    h_taban = taban_uret(hedef, bicim)
    y = np.log1p(egitim[tm.HEDEF].clip(lower=0.0).to_numpy()) - e_taban
    model = di.aile_modeli("cat", tohum)
    x_e, x_h = egitim[kolonlar].copy(), hedef[kolonlar].copy()
    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
    for k in kat:
        x_e[k] = x_e[k].astype(str)
        x_h[k] = x_h[k].astype(str)
    model.fit(x_e, y, cat_features=kat)
    return model.predict(x_h) + h_taban


#: (ad, maske orani, yalnizca sicak satirlarla mi egit, taban bicimi)
ADAYLAR: tuple[tuple[str, float, bool, str], ...] = (
    ("W0 maske0.15 tum  taban=guc     [MEVCUT]", 0.15, False, "guc"),
    ("W1 maske0.00 sicak taban=guc", 0.0, True, "guc"),
    ("W2 maske0.00 sicak taban=t_log_ort", 0.0, True, "ort"),
    ("W3 maske0.00 sicak taban=(ort+son30)/2", 0.0, True, "ort_son30"),
    ("W4 maske0.00 sicak taban=t_log_son30", 0.0, True, "son30"),
)


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("SICAK UZMANININ HEDEF TABANI -- yalnizca SICAK satirlarda olculur")
    print("=" * 100)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    print(f"  egitim {len(egitim):,} satir | {len(kolonlar)} oznitelik")

    parcalar = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        parcalar[b.ad] = (kalan, dogrulama, gercek, soguk)

    print("\n  --- PARAMETRESIZ KIYAS TABANLARI (sicak satirlar) ---")
    print("  hicbir sey uydurulmuyor; ozet penceresinden dogrudan okunuyor")
    for bicim in ("guc", "ort", "son30", "ort_son30"):
        skorlar = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            taban = taban_uret(dogrulama, bicim)
            if bicim == "guc":  # kapasite tek basina seviye tasimaz; kaydir
                kalan = parcalar[b.ad][0]
                taban = taban + float(
                    (np.log1p(kalan[tm.HEDEF].clip(lower=0.0)) - np.log1p(kalan["guc"])).mean()
                )
            t = np.clip(np.expm1(taban), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[~soguk], t[~soguk])
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        print(f"  taban={bicim:12} SICAK {np.mean(list(skorlar.values())):.5f}   {detay}")

    print("\n  --- MODELLER ---")
    saklanan: dict[str, dict[str, np.ndarray]] = {}
    taban_skor = None
    for ad, maske_orani, yalniz_sicak, bicim in ADAYLAR:
        t0 = time.time()
        skorlar = {}
        saklanan[ad] = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = parcalar[b.ad]
            tahminler = []
            for tohum in di.TOHUMLAR:
                e = kalan[kalan["soguk_mu"] == 0] if yalniz_sicak else kalan
                if maske_orani > 0:
                    e = d.soguk_maskele(e, kolonlar, maske_orani, tohum)
                tahminler.append(egit_tabanli(e, dogrulama, kolonlar, tohum, bicim))
            torbali = np.mean(tahminler, axis=0)
            saklanan[ad][b.ad] = torbali
            t = np.clip(np.expm1(torbali), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[~soguk], t[~soguk])
        genel = float(np.mean(list(skorlar.values())))
        if taban_skor is None:
            taban_skor = genel
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        fark = f"{taban_skor - genel:+.5f}" if taban_skor is not None else ""
        print(f"  {ad:42} SICAK {genel:.5f}  fark {fark}   {detay}   ({time.time() - t0:.0f} sn)")

    print("\n  --- ADAYLARIN HARMANI (sicak satirlar, log uzayinda) ---")
    adlar = [a[0] for a in ADAYLAR]
    for i, ilk in enumerate(adlar):
        for ikinci in adlar[i + 1 :]:
            skorlar = {}
            for b in tm.BLOKLAR:
                _, _, gercek, soguk = parcalar[b.ad]
                karisim = 0.5 * (saklanan[ilk][b.ad] + saklanan[ikinci][b.ad])
                t = np.clip(np.expm1(karisim), 0.0, None)
                skorlar[b.ad] = tm.rmsle(gercek[~soguk], t[~soguk])
            genel = float(np.mean(list(skorlar.values())))
            print(f"  {ilk[:2]}+{ikinci[:2]} yari yariya{'':<26} SICAK {genel:.5f}")

    print("\n  Genel skora cevirme: sicak pay %77,84, soguk skor 1,7404")
    print("  genel = sqrt(0,7784*sicak^2 + 0,2216*1,7404^2)")
    for ad in adlar:
        skorlar = []
        for b in tm.BLOKLAR:
            _, _, gercek, soguk = parcalar[b.ad]
            t = np.clip(np.expm1(saklanan[ad][b.ad]), 0.0, None)
            skorlar.append(tm.rmsle(gercek[~soguk], t[~soguk]))
        s = float(np.mean(skorlar))
        print(f"  {ad:42} -> genel {np.sqrt(0.7784 * s**2 + 0.2216 * 1.7404**2):.5f}")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
