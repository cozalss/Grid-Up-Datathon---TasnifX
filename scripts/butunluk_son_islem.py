"""BUTUNLUK SINAMASI: uretim son islemi, olcum tezgahiyla AYNI seyi mi yapiyor?

NEDEN
-----
`kis26` sayilari deney betiklerinden geliyor ve o betikler donusumu YENIDEN
YAZIYOR. `son_islem_gun.py` ise gonderim dosyasini uretiyor. Ikisi arasinda
kucuk bir fark olsa -- kova kenarlari, merkezleme sirasi, ampirik-Bayes
ebeveyni -- olctugumuz sey gonderdigimiz sey OLMAZ ve fark sessizce gecer.

Bu takimin kendi kayitlarinda tam bu tur iki sessiz hizasizlik var
(docs/36: v23 tabani, uygulanmayan yama). Bu betik kapiyi kapatir.

NASIL
-----
Uretim modulunun KENDI fonksiyonlari (`son_islem_gun.hucre_etkisi` ve ayni
formul) `kis26` dogrulama kumesine uygulanir ve cikan soguk RMSLE, deney
betiginin bildirdigi sayiyla karsilastirilir.

Tablo kaynagi icin uretim kurali (``TABLO_BASLANGIC``) AYNEN uygulanir,
tek fark pencerenin hedef blogun basinda kesilmesidir -- sizinti olmasin.

    python scripts/butunluk_son_islem.py
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
import son_islem_gun as si  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
#: Beklenen sayi: deney_ikili_agirlik.py a=0,55 b=0,25 icin 1,82131 (M_gun=0)
#: bildirdi; M_gun=50'nin kis26'daki bilinen maliyeti +0,00008; tablo
#: penceresi (TABLO_BASLANGIC) kis26 RMSLE'sinde -0,00012 getiriyor
#: (1,82139 -> 1,82127, ayni cebirle olculdu).
BEKLENEN = 1.82135
ESIK = 5e-4


def uygula(r: np.ndarray, hucre: np.ndarray, gun: np.ndarray, ay: np.ndarray,
           a: float, b: float, m_gun: float) -> np.ndarray:
    """``son_islem_gun.main`` icindeki donusumun BIREBIR ayni cebiri."""
    def gruplu(v: np.ndarray, anahtar: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(anahtar).transform("mean").to_numpy()

    n_gun = pd.Series(gun).groupby(gun).transform("size").to_numpy().astype("float64")
    w = n_gun / (n_gun + m_gun) if m_gun > 0 else np.ones_like(n_gun)
    seviye = w * gruplu(r, gun) + (1.0 - w) * gruplu(r, ay)
    seviye = seviye - gruplu(seviye, ay) + gruplu(r, ay)
    h_ref = w * gruplu(hucre, gun) + (1.0 - w) * gruplu(hucre, ay)
    etki = hucre - h_ref
    etki = etki - gruplu(etki, ay)
    return seviye + a * etki + b * (r - seviye)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 92)
    print("BUTUNLUK: uretim son islemi vs olcum tezgahi")
    print("=" * 92)
    print(f"  uretim sabitleri: A_HUCRE={si.A_HUCRE}  B_MODEL={si.B_MODEL}  "
          f"M_GUN={si.M_GUN}  KOVA={si.KOVA_SAYISI}  M_ANA={si.M_ANA}  M_HUCRE={si.M_HUCRE}")

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    gun = pd.to_datetime(dg["tarih"]).to_numpy()
    ay = pd.to_datetime(dg["tarih"]).dt.to_period("M").astype(str).to_numpy()

    z = np.load(ONBELLEK)

    # --- URETIM FONKSIYONU, sizintisiz kaynak (blok oncesi ham train) ---
    ham = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    ham["t"] = pd.to_datetime(ham["tarih"])
    blok_bas = pd.Timestamp(gun.min())
    # Uretim kurali AYNEN: TABLO_BASLANGIC'ten itibaren, ama hedef bloktan ONCE
    kaynak_temiz = ham[(ham["t"] >= si.TABLO_BASLANGIC) & (ham["t"] < blok_bas)]
    hedef = pd.DataFrame({"guc": dg["guc"].to_numpy(), "lokasyon": dg["lokasyon"].to_numpy()})
    hucre_temiz = si.hucre_etkisi(kaynak_temiz, hedef)
    print(f"  tablo kaynagi (sizintisiz): {len(kaynak_temiz):,} satir, "
          f"{kaynak_temiz['t'].min().date()} .. {kaynak_temiz['t'].max().date()}")

    skorlar = []
    for t in TOHUMLAR:
        r = z[f"{t}_cat"] - log_guc
        r_yeni = uygula(r, hucre_temiz, gun, ay, si.A_HUCRE, si.B_MODEL, si.M_GUN)
        skorlar.append(tm.rmsle(y, np.clip(np.expm1(r_yeni + log_guc), 0.0, None)))
    uretim_skor = float(np.mean(skorlar))

    print(f"\n  URETIM FONKSIYONU ile   {uretim_skor:.5f}")
    print(f"  DENEY BETIGI bildirdi   {BEKLENEN:.5f}")
    fark = abs(uretim_skor - BEKLENEN)
    print(f"  fark {fark:.5f}   esik {ESIK:.5f}")

    # aylik seviye kapisi da uretimdeki gibi kontrol edilsin
    r0 = z[f"{TOHUMLAR[0]}_cat"] - log_guc
    r1 = uygula(r0, hucre_temiz, gun, ay, si.A_HUCRE, si.B_MODEL, si.M_GUN)
    ay_sapma = float(np.abs(pd.Series(r0).groupby(ay).mean()
                            - pd.Series(r1).groupby(ay).mean()).max())
    ici0 = pd.Series(r0 - pd.Series(r0).groupby(gun).transform("mean")).groupby(ay).std()
    ici1 = pd.Series(r1 - pd.Series(r1).groupby(gun).transform("mean")).groupby(ay).std()
    print(f"  aylik seviye sapmasi {ay_sapma:.2e}   gun ici yayilma "
          f"{float(ici0.mean()):.5f} -> {float(ici1.mean()):.5f}")

    if fark > ESIK:
        print("\n  HIZASIZLIK: uretim son islemi olcum tezgahiyla AYNI seyi yapmiyor.")
        return 1
    print("\n  TAMAM: uretim son islemi ile olcum tezgahi ayni sayiyi veriyor.")
    print(f"  ({(time.time() - t0) / 60:.1f} dakika)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
