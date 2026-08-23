"""SOGUK SON ISLEM: hucre agirligi ile model agirligini AYIRMAK.

BULGU
-----
``son_islem_gun.py`` tek bir ``beta`` ile iki ayri seyi birden ayarliyor.
Cebiri acinca gorunuyor -- gun ortalamasi tam korundugu icin:

    r' = taban + beta * (r - taban),   taban = gun_ort + etki
       = gun_ort + (1 - beta) * etki + beta * (r - gun_ort)

Yani EGITIMDEN gelen hucre tablosu ``1 - beta`` = 0,75 agirlik aliyor,
MODELIN kendi gun-ici sinyali ``beta`` = 0,25 aliyor. Bu iki agirligin
toplaminin 1 olmasi icin hicbir turetilmis gerekce YOK -- ikisi farkli
bilgi kaynaklari ve farkli guvenilirlikte.

Bagimsiz bir denetim (trafo-bazli holdout, 2025-04..07 mevsimsel ikizi)
hucre tablosuna verilebilecek optimal agirligi 0,545 olcmus; kod 0,75
veriyor. Bu deney ayrimi kis26 uzerinde, uretimdeki tam kurguda test eder.

    r' = gun_ort + a * etki + b * (r - gun_ort)

    a = hucre tablosunun agirligi   (egitimden gelen ilce x kova etkisi)
    b = modelin gun-ici sinyalinin agirligi
    bugunku uretim:  a = 0,75,  b = 0,25   (a + b = 1 kisitli)

Ayrica iki kusur daha sinanir:
  SEYREK GUN  n<=100 soguk satiri olan 29 gunde islem TERSINE donuyor
              (n=2-5 bandinda yayilma 3,16 KAT artiyor; n=1'de tam no-op),
              cunku gun ortalamasi o gunun kendi birkac satirindan
              hesaplaniyor. Gun ortalamasini ampirik-Bayes ile aylik
              ortalamaya dogru buzmek bunu kapatir.
  DUZLESTIRME gun ortalamasini kayan pencereyle duzlestirmek.

Karar mercii kis26. cat-only soguk harman, 3 tohum, onbellekten.

    python scripts/deney_ikili_agirlik.py
"""

from __future__ import annotations

import json
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
import deney_soguk_taban as st  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
ONBELLEK = KOK / "data" / "interim" / "deney" / f"soguk_tahmin_{BLOK}.npz"
KAYIT = KOK / "experiments" / "ikili_agirlik.jsonl"

A_DEGERLERI = (0.0, 0.25, 0.40, 0.55, 0.65, 0.75, 0.90, 1.00)
B_DEGERLERI = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
#: Gun ortalamasinin aylik ortalamaya dogru ampirik-Bayes buzmesi icin
#: onsel agirlik (satir cinsinden). n satirlik bir gun n/(n+M) agirlik alir.
M_GUN_DEGERLERI = (0.0, 50.0, 200.0, 1000.0)


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SOGUK SON ISLEM: hucre agirligi (a) ile model agirligi (b) ayriliyor -- {BLOK}")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    y = gercek[soguk]
    log_guc = np.log1p(dogrulama["guc"].to_numpy(dtype="float64"))[soguk]
    tarih = pd.to_datetime(dogrulama["tarih"]).to_numpy()[soguk]
    ay = pd.Series(tarih).dt.to_period("M").astype(str).to_numpy()

    z = np.load(ONBELLEK)
    cat = {t: z[f"{t}_cat"] for t in TOHUMLAR}
    tabanlar = st.tabanlari_kur(parca, dogrulama, soguk)
    hucre = tabanlar["ilcexkova_M2000"]

    gun_s = pd.Series(tarih)
    n_gun = gun_s.groupby(tarih).transform("size").to_numpy().astype("float64")
    print(f"  {len(y):,} soguk satir | {len(np.unique(tarih))} gun | "
          f"gunluk soguk satir: min {n_gun.min():.0f} medyan {np.median(n_gun):.0f} "
          f"maks {n_gun.max():.0f}")
    print(f"  n<=100 olan gun: {len(np.unique(tarih[n_gun <= 100]))}, "
          f"{int((n_gun <= 100).sum())} satir")

    def grup_ort(v: np.ndarray, anahtar: np.ndarray) -> np.ndarray:
        return pd.Series(v).groupby(anahtar).transform("mean").to_numpy()

    def skorla(ofs: np.ndarray) -> float:
        return tm.rmsle(y, np.clip(np.expm1(ofs + log_guc), 0.0, None))

    def uygula(m: np.ndarray, a: float, b: float, m_gun: float) -> np.ndarray:
        """r' = gun_ort + a*etki + b*(r - gun_ort); gun_ort aya dogru buzulur."""
        ham_gun = grup_ort(m, tarih)
        if m_gun > 0:
            ay_ort = grup_ort(m, ay)
            w = n_gun / (n_gun + m_gun)
            gun = w * ham_gun + (1.0 - w) * ay_ort
        else:
            gun = ham_gun
        etki = hucre - grup_ort(hucre, tarih)
        return gun + a * etki + b * (m - ham_gun)

    kayitlar: list[dict] = []
    print("\n--- A) a (hucre) x b (model),  gun ortalamasi buzmesiz ---")
    print(f"  {'a\\b':>6}" + "".join(f"{b:>10.2f}" for b in B_DEGERLERI))
    for a in A_DEGERLERI:
        satir = []
        for b in B_DEGERLERI:
            sk = [skorla(uygula(cat[t] - log_guc, a, b, 0.0)) for t in TOHUMLAR]
            satir.append(float(np.mean(sk)))
            kayitlar.append({"a": a, "b": b, "m_gun": 0.0, "rmsle": satir[-1]})
        isaret = "  <- URETIM" if abs(a - 0.75) < 1e-9 else ""
        print(f"  {a:6.2f}" + "".join(f"{v:10.5f}" for v in satir) + isaret)

    en_iyi_ab = min(kayitlar, key=lambda k: k["rmsle"])
    print(f"\n--- B) gun ortalamasi aya dogru buzulurse (en iyi a,b ile) ---")
    a, b = en_iyi_ab["a"], en_iyi_ab["b"]
    print(f"  {'M_gun':>8}{'RMSLE':>10}")
    for m_gun in M_GUN_DEGERLERI:
        sk = [skorla(uygula(cat[t] - log_guc, a, b, m_gun)) for t in TOHUMLAR]
        o = float(np.mean(sk))
        kayitlar.append({"a": a, "b": b, "m_gun": m_gun, "rmsle": o})
        print(f"  {m_gun:8.0f}{o:10.5f}")

    uretim = next(k["rmsle"] for k in kayitlar
                  if k["a"] == 0.75 and k["b"] == 0.25 and k["m_gun"] == 0.0)
    en_iyi = min(kayitlar, key=lambda k: k["rmsle"])
    kazanc = uretim - en_iyi["rmsle"]
    print(f"\n  URETIM (a=0,75 b=0,25):  {uretim:.5f}")
    print(f"  EN IYI a={en_iyi['a']:.2f} b={en_iyi['b']:.2f} M_gun={en_iyi['m_gun']:.0f}"
          f"  ->  {en_iyi['rmsle']:.5f}   kazanc {kazanc:+.5f}")
    print(f"  genel skora tahmini etki {-kazanc * 0.377:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
