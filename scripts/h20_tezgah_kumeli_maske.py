"""H20 TEZGAHI -- KUMELI soguk maskeleme, URETIM (rastgele) maskesine karsi.

HIPOTEZ
-------
``soguk_maskele`` egitimde trafolari I.I.D. RASTGELE sogutur
(``tuketim_model.py:1055``, ``rng.choice(trafolar, ...)``). Rastgele
sogutulmus bir trafonun KOMSULARI HALA SICAK, model onlara yaslanmayi
ogrenir. Ama TEST'te 1.326 trafo AYNI ANDA geliyor ve mekansal olarak
KUMELENIYOR (h20_kapilar: z(Herfindahl)=+13,94; test ilce soguk payi 0,3439
vs egitim rastgele maske 0,2206, oran 1,56).

Yani DropoutNet mekanizmasi yarim uygulanmis: test'in soguk ORANI taklit
ediliyor, KORELASYON YAPISI taklit edilmiyor.

BU TEZGAH bir gonderim dosyasi URETMEZ. Tek soruyu cevaplar:
    "Ayni maske oraninda, KUMELI maskeleme soguk tarafta daha iyi mi?"
Cevap evet ise 26 Agustos gunduzu uretim yeniden egitimine ayrilir.
Cevap hayir ise eksen TEMIZ kapanir -- bu da degerli.

PROTOKOL (pazarliksiz)
----------------------
- Bloklar: yaz25 (mevsimsel ikiz, kural 7) + kis26 (ikinci ortusmeyen
  kesme, kural 10) + guz25.
- Kollar: URETIM (rastgele, oran 0,2216) vs KUMELI (ayni oran, lokasyon
  kumesi bazinda TOPLUCA).
- 3 tohum (kural 3: soguk tarafta sinirda; sure elverirse artirilir).
- Hukum (blok, tohum) ciftlerinde ESLENIK SH ile (kural 4).
- Soguk kazanc icin trafo bazinda KIRPMA TABLOSU (kural 1).
- Uretim kodu DEGISMEZ; maske varyanti burada.

KIRILIM: ``lokasyon`` 47 essiz deger (ilce). Bir UST seviye (bolge) de
denenir -- hangisi test'in kumelenmesini daha iyi taklit ediyorsa.
"""

from __future__ import annotations

import argparse
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

ORAN = 0.2216
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
USTYAZIM: dict[str, object] = {"depth": 7}
_GECMIS = "g_"


def _gecmis_kolonlari(kolonlar: list[str]) -> list[str]:
    onek = getattr(d, "_GECMIS_ONEKI", _GECMIS)
    return [k for k in kolonlar if k.startswith(onek)]


def kumeli_maskele(
    cerceve: pd.DataFrame, kolonlar: list[str], oran: float, tohum: int, kirilim: str = "ilce"
) -> pd.DataFrame:
    """Trafolari LOKASYON KUMESI bazinda topluca sogutur.

    Kumeler rastgele sirayla secilir ve hedef ORAN'a ulasilana kadar
    eklenir. Boylece maskelenen trafolarin komsulari da SOGUK olur --
    test'teki toplu katilimin korelasyon yapisi taklit edilir.
    """
    if oran <= 0.0:
        return cerceve
    rng = np.random.default_rng(tohum)
    trafo = cerceve[[tm.GRUP, "lokasyon"]].drop_duplicates(tm.GRUP)
    if kirilim == "bolge":
        anahtar = trafo["lokasyon"].astype(str).str.split(">").str[:2].str.join(">")
    else:
        anahtar = trafo["lokasyon"].astype(str)
    trafo = trafo.assign(_k=anahtar.to_numpy())

    kumeler = trafo["_k"].unique()
    rng.shuffle(kumeler)
    hedef = int(len(trafo) * oran)
    secilen: set = set()
    for k in kumeler:
        if len(secilen) >= hedef:
            break
        secilen |= set(trafo.loc[trafo["_k"] == k, tm.GRUP])
    # son kumeyi kirp: orani ASMA (adil karsilastirma icin)
    if len(secilen) > hedef:
        fazla = list(secilen)
        rng.shuffle(fazla)
        secilen = set(fazla[:hedef])

    maske = cerceve[tm.GRUP].isin(secilen).to_numpy()
    sonuc = cerceve.copy()
    sonuc.loc[maske, _gecmis_kolonlari(kolonlar)] = np.nan
    if "soguk_mu" in sonuc.columns:
        sonuc.loc[maske, "soguk_mu"] = 1
    return sonuc


def rmsle_log(lgy: np.ndarray, tah: np.ndarray) -> float:
    return float(np.sqrt(((lgy - tah) ** 2).mean()))


def main() -> int:
    a = argparse.ArgumentParser(description="H20 kumeli maske tezgahi")
    a.add_argument("--blok", nargs="+", default=["yaz25", "kis26", "guz25"])
    a.add_argument("--tohum", type=int, nargs="+", default=[1000, 1001, 1002])
    a.add_argument("--kirilim", nargs="+", default=["ilce"])
    ar = a.parse_args()

    t0 = time.time()
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    print(
        f"cerceveler hazir ({time.time() - t0:.0f} sn)  gecmis kolonu "
        f"{len(_gecmis_kolonlari(kol))}",
        flush=True,
    )

    kayit = []
    for blok in ar.blok:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
        lgy = np.log1p(np.clip(gercek, 0, None))
        tan = dogrulama[tm.GRUP].to_numpy()
        print(
            f"\n{'=' * 92}\n{blok}  dogrulama {len(dogrulama):,}  "
            f"SOGUK {int(soguk.sum()):,} satir / "
            f"{len(set(tan[soguk])):,} trafo\n{'=' * 92}",
            flush=True,
        )

        kollar = [("URETIM", None)] + [("KUMELI-" + k, k) for k in ar.kirilim]
        for ad, kir in kollar:
            for tohum in ar.tohum:
                if kir is None:
                    mk = d.soguk_maskele(parca, kol, ORAN, tohum)
                else:
                    mk = kumeli_maskele(parca, kol, ORAN, tohum, kir)
                pay = sum(HARMAN.values())
                tah = None
                for aile, w in HARMAN.items():
                    ust = USTYAZIM if aile == "cat" else {}
                    v = di.egit_tahmin(aile, mk, dogrulama, kol, tohum, **ust)
                    tah = w * v if tah is None else tah + w * v
                tah = tah / pay
                r_s = rmsle_log(lgy[soguk], tah[soguk])
                r_t = rmsle_log(lgy, tah)
                kayit.append(
                    {
                        "blok": blok,
                        "kol": ad,
                        "tohum": tohum,
                        "soguk_rmsle": r_s,
                        "tum_rmsle": r_t,
                        "b": float((lgy[soguk] - tah[soguk]).mean()),
                    }
                )
                print(
                    f"  {ad:<14} tohum {tohum}  SOGUK {r_s:.5f}  "
                    f"tum {r_t:.5f}  b {float((lgy[soguk] - tah[soguk]).mean()):+.4f}"
                    f"   ({time.time() - t0:.0f} sn)",
                    flush=True,
                )
                pd.DataFrame(kayit).to_csv(
                    KOK / f"reports/h20_tezgah_{ar.tohum[0]}.csv", index=False
                )

        # eslenik karsilastirma
        df = pd.DataFrame([k for k in kayit if k["blok"] == blok])
        taban = df[df["kol"] == "URETIM"].set_index("tohum")["soguk_rmsle"]
        for ad in df["kol"].unique():
            if ad == "URETIM":
                continue
            alt = df[df["kol"] == ad].set_index("tohum")["soguk_rmsle"]
            ortak = taban.index.intersection(alt.index)
            fark = (alt.loc[ortak] - taban.loc[ortak]).to_numpy()
            sh = fark.std(ddof=1) / np.sqrt(len(fark)) if len(fark) > 1 else float("nan")
            print(f"\n  >>> {ad} - URETIM  (SOGUK RMSLE, eslenik)")
            print(
                f"      d = {fark.mean():+.5f}   SH {sh:.5f}   "
                f"t {fark.mean() / sh if sh else float('nan'):+.2f}   "
                f"iyilesen {int((fark < 0).sum())}/{len(fark)}",
                flush=True,
            )

    print(f"\n{'=' * 92}\nBITTI ({time.time() - t0:.0f} sn) -> reports/h20_tezgah.csv")
    print("  NEGATIF d = KUMELI daha iyi. Kural 4: (blok,tohum) eslenik SH.")
    print("  Hukum icin yaz25 (ikiz) ZORUNLU, kis26 ikinci kesme.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
