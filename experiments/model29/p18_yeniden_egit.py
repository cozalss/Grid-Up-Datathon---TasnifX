"""p18: TAM URETIM HATTI PARAMETRIK YENIDEN EGITIM.

NEDEN
-----
Simdiye kadarki yaklasim "delta yamasi": bir aileyi degistir, farki uretim
tahminine bindir (p06/p07). Kazanan yapilandirma belli oldugunda TUM hatti
o yapilandirmayla bastan egitmek daha dogru -- aileler birbirine uyum
saglar, harman yeniden dengelenir.

HATTIN IKI TARAFI (uretimden birebir kopyalandi)
------------------------------------------------
SOGUK uzman  (``scripts/uret_soguk_tahmin.py`` ile birebir):
    kaynak    = DAR egitim seti (ana bloklar; ek koken YOK)
    maske     = 1.00  (saf soguk uzman, butun t_* NaN)
    cat ust   = {"depth": 7}
    cikti     = data/interim/deney/soguk_tahmin_{blok}.npz
SICAK uzman  (``scripts/aile_onbellegi.py`` ile birebir):
    kaynak    = GENIS egitim seti (ek kokenli), blok modunda
                ``tm.kokenleri_ayikla`` ile sizintisiz
    maske     = 0.15
    cat ust   = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
    cikti     = data/interim/aile_onbellek/{blok}_{tohum}_{aile}_uretim.npy

Her iki taraf da ``deney_ileri.egit_tahmin`` uzerinden gecirilir; o fonksiyon
``tuketim_model.aile_tahmini`` ile ARITMETIK OLARAK AYNIDIR (ayni model
parametreleri, ayni kapasite ofseti). Maskeleme cagiran tarafta yapilir.

YAPILANDIRMA (ortam degiskenleri)
---------------------------------
P18_MOD              blok | test                       (varsayilan blok)
P18_TARAF            soguk,sicak                       (varsayilan ikisi)
P18_BLOKLAR          yaz25,guz25,kis26
P18_TOHUMLAR         1000,1001,1002
P18_AILELER          cat,xgb,lgbm
P18_SOGUK_CAT_KAYIP  ""(taban) | huber:0.5 | l1 | fair:1.0
P18_SOGUK_XGB_KAYIP  aynisi
P18_SOGUK_LGBM_KAYIP aynisi
P18_SICAK_CAT_KAYIP / P18_SICAK_XGB_KAYIP / P18_SICAK_LGBM_KAYIP
P18_TAU_SOGUK        yok | sayi   (ornek agirligi exp(-gecmis_gun/TAU))
P18_TAU_SICAK        yok | sayi
P18_AGAC             ""(400) | sayi        -- YALNIZ duman testi icin
P18_ALT_ORNEK        ""(1.0) | 0<f<=1      -- YALNIZ duman testi icin
P18_CIKTI            cikti dizini
P18_ETIKET           cikti alt dizini      (varsayilan "taban")
P18_DOGRULA          1 -> uretim onbellegiyle maxabs farki raporla

VARSAYILANLAR MEVCUT URETIMI BIREBIR YENIDEN URETIR. P18_DOGRULA=1 bunu
kanitlar; tutmuyorsa betik HATA verir.

KOSU
----
    set P18_TARAF=soguk & set P18_BLOKLAR=yaz25 & set P18_TOHUMLAR=1000 ^
        & set P18_AILELER=lgbm & set P18_DOGRULA=1 ^
        & python experiments/model29/p18_yeniden_egit.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOK, "scripts"))
sys.path.insert(0, os.path.join(KOK, "src"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

AO = os.path.join(KOK, "data", "interim", "aile_onbellek")
DN = os.path.join(KOK, "data", "interim", "deney")
SP_VARSAYILAN = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad/p18"
)

#: URETIM uzman ayarlari -- ``tm.REJIM_AYARLARI`` ve iki uretim betiginden
#: birebir kopyalandi. Bu sozluk uretimden SAPARSA yeniden egitim uretimi
#: yeniden uretmez.
UZMAN = {
    "soguk": {"maske": 1.00, "cat_ust": {"depth": 7}, "ek_koken": False},
    "sicak": {
        "maske": 0.15,
        "cat_ust": {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6},
        "ek_koken": True,
    },
}

#: Blok etiket pencerelerinin BASI -- yakinlik agirliginin "kesim" ani.
KESIM = {
    "yaz25": "2025-04-01",
    "guz25": "2025-08-01",
    "kis26": "2025-12-01",
    "test": "2026-04-01",
}

T0 = time.time()


def log(*a: object) -> None:
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def cev(ad: str, vars: str = "") -> str:
    return os.environ.get(ad, vars).strip()


def liste(ad: str, vars: str) -> list[str]:
    return [p.strip() for p in cev(ad, vars).split(",") if p.strip()]


# --------------------------------------------------------------- kayip


def kayip_ustyazim(aile: str, spec: str) -> dict[str, object]:
    """``"huber:0.5"`` gibi bir belirtimi aileye ozel parametreye cevirir.

    Bos / "yok" / "taban" -> {} (URETIM AYARI, hicbir seyi degistirmez).
    """
    s = spec.strip().lower()
    if s in ("", "yok", "taban", "l2", "rmse"):
        return {}
    ad, _, par = s.partition(":")
    a = float(par) if par else None
    if aile == "lgbm":
        if ad == "huber":
            return {"objective": "huber", "alpha": 1.0 if a is None else a}
        if ad == "l1":
            return {"objective": "regression_l1"}
        if ad == "fair":
            return {"objective": "fair", "fair_c": 1.0 if a is None else a}
    elif aile == "cat":
        if ad == "huber":
            return {"loss_function": f"Huber:delta={1.0 if a is None else a}"}
        if ad == "l1":
            return {"loss_function": "MAE"}
    elif aile == "xgb":
        if ad == "huber":
            return {
                "objective": "reg:pseudohubererror",
                "huber_slope": 1.0 if a is None else a,
            }
        if ad == "l1":
            return {"objective": "reg:absoluteerror"}
    raise SystemExit(f"p18: {aile} icin bilinmeyen kayip belirtimi {spec!r}")


def yakinlik_agirligi(cerceve: pd.DataFrame, kesim: str, tau: float) -> np.ndarray:
    """``w = exp(-(kesim - tarih).days / TAU)``, kesim SONRASI satirlarda 1.

    Kesimden sonraki satirlar (CV bloklarinda gelecek bloklar) icin gun
    farki negatif olurdu ve agirlik 1'in USTUNE cikardi; kirpilir.
    """
    gun = (pd.Timestamp(kesim) - pd.to_datetime(cerceve["tarih"])).dt.days.to_numpy("float64")
    return np.exp(-np.clip(gun, 0.0, None) / float(tau))


# ----------------------------------------------------------- yapilandirma


class Ayar:
    def __init__(self) -> None:
        self.mod = cev("P18_MOD", "blok")
        if self.mod not in ("blok", "test"):
            raise SystemExit(f"P18_MOD 'blok' ya da 'test' olmali, {self.mod!r} geldi")
        self.taraflar = liste("P18_TARAF", "soguk,sicak")
        self.bloklar = liste("P18_BLOKLAR", "yaz25,guz25,kis26")
        self.tohumlar = [int(t) for t in liste("P18_TOHUMLAR", "1000,1001,1002")]
        self.aileler = liste("P18_AILELER", "cat,xgb,lgbm")
        self.kayip = {
            taraf: {
                aile: cev(f"P18_{taraf.upper()}_{aile.upper()}_KAYIP")
                for aile in ("cat", "xgb", "lgbm")
            }
            for taraf in ("soguk", "sicak")
        }
        self.tau = {
            taraf: (None if cev(f"P18_TAU_{taraf.upper()}", "yok") in ("", "yok") else float(
                cev(f"P18_TAU_{taraf.upper()}")))
            for taraf in ("soguk", "sicak")
        }
        agac = cev("P18_AGAC")
        self.agac = int(agac) if agac else None
        alt = cev("P18_ALT_ORNEK")
        self.alt_ornek = float(alt) if alt else None
        self.cikti = cev("P18_CIKTI", SP_VARSAYILAN)
        self.etiket = cev("P18_ETIKET", "taban")
        self.dogrula = cev("P18_DOGRULA") == "1"

    @property
    def taban_mi(self) -> bool:
        """Hicbir sey uretimden sapmiyor mu? (birebir dogrulama ancak boyle)"""
        return (
            self.agac is None
            and self.alt_ornek is None
            and all(not v for t in self.kayip.values() for v in t.values())
            and all(v is None for v in self.tau.values())
        )

    def sozluk(self) -> dict:
        return {
            "mod": self.mod, "taraflar": self.taraflar, "bloklar": self.bloklar,
            "tohumlar": self.tohumlar, "aileler": self.aileler, "kayip": self.kayip,
            "tau": self.tau, "agac": self.agac, "alt_ornek": self.alt_ornek,
            "etiket": self.etiket, "taban_mi": self.taban_mi,
        }


# ------------------------------------------------------------- cerceveler


def cerceveleri_hazirla(ayar: Ayar):
    """``uret_soguk_tahmin.py`` ve ``aile_onbellegi.py`` ile AYNI SIRA.

    Sira onemli: ``kategorik_kodla`` yerinde calisiyor ve ``genis``
    kategorileri kodlanmis ``egitim``den aliyor.
    """
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    log(f"dar egitim {egitim.shape}  test {test.shape}  kolon {len(kol)}")

    genis = None
    if "sicak" in ayar.taraflar:
        ek = d._ek_kokenler_kur(False)
        ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
        ortak = [k for k in egitim.columns if k in ek.columns]
        genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
        for k in tm.KATEGORIK:
            genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
        log(f"genis (ek kokenli) egitim {genis.shape}")
    return egitim, test, genis, kol


def parca_ve_hedef(ayar: Ayar, taraf: str, blok: str, dar, test, genis):
    """(egitim parcasi, hedef cerceve, hedef icindeki satir maskesi)."""
    kaynak = genis if UZMAN[taraf]["ek_koken"] else dar
    if ayar.mod == "test":
        hedef = test
        secim = (test["soguk_mu"] == 1).to_numpy()
        parca = kaynak
    else:
        hedef = dar[dar["_blok"] == blok]
        secim = (hedef["soguk_mu"] == 1).to_numpy()
        if UZMAN[taraf]["ek_koken"]:
            parca = tm.kokenleri_ayikla(kaynak, blok)
        else:
            parca = kaynak[kaynak["_blok"] != blok]
    if taraf == "sicak":
        secim = ~secim
    return parca, hedef.loc[secim], secim


# ---------------------------------------------------------------- kosu


def kos(ayar: Ayar) -> dict:
    dar, test, genis, kol = cerceveleri_hazirla(ayar)
    cikti_dizin = os.path.join(ayar.cikti, ayar.etiket)
    os.makedirs(cikti_dizin, exist_ok=True)
    hedefler = ["test"] if ayar.mod == "test" else ayar.bloklar
    kayit: dict = {}

    for taraf in ayar.taraflar:
        for blok in hedefler:
            parca, hedef, secim = parca_ve_hedef(ayar, taraf, blok, dar, test, genis)
            if ayar.alt_ornek:
                parca = parca.sample(frac=ayar.alt_ornek, random_state=7)
            log(
                f"{ayar.mod}/{taraf}/{blok}: egitim {len(parca):,}  "
                f"hedef {len(hedef):,} ({int(secim.sum()):,} satir)"
            )
            yol = os.path.join(cikti_dizin, f"{taraf}_{blok}.npz")
            ham: dict[str, np.ndarray] = {}
            if os.path.exists(yol):
                z = np.load(yol)
                ham = {k: z[k] for k in z.files}
                log(f"  mevcut cikti okundu: {len(ham)} anahtar")

            tau = ayar.tau[taraf]
            for tohum in ayar.tohumlar:
                gerekli = [a for a in ayar.aileler if f"{tohum}_{a}" not in ham]
                if not gerekli:
                    log(f"  tohum {tohum}: hepsi var, atlandi")
                    continue
                # URETIM SIRASI: maskeleme tohum basina BIR KEZ.
                maskeli = d.soguk_maskele(parca, kol, UZMAN[taraf]["maske"], tohum)
                w = None
                if tau is not None:
                    w = yakinlik_agirligi(maskeli, KESIM[blok], tau)
                for aile in gerekli:
                    t0 = time.time()
                    ust: dict[str, object] = {}
                    if aile == "cat":
                        ust.update(UZMAN[taraf]["cat_ust"])
                    ust.update(kayip_ustyazim(aile, ayar.kayip[taraf][aile]))
                    if ayar.agac is not None:
                        ust["iterations" if aile == "cat" else "n_estimators"] = ayar.agac
                    ham[f"{tohum}_{aile}"] = di.egit_tahmin(
                        aile, maskeli, hedef, kol, tohum, agirlik=w, **ust
                    )
                    np.savez_compressed(yol, **ham)
                    log(f"  {tohum}_{aile:5} bitti ({time.time() - t0:.0f} sn)  ust={ust}")
                del maskeli
            kayit[f"{taraf}_{blok}"] = {"yol": yol, "anahtar": sorted(ham)}
    return kayit


# ------------------------------------------------------------ dogrulama


def dogrula(ayar: Ayar, kayit: dict) -> dict:
    """Uretim onbellekleriyle maxabs farki. Hedef < 1e-6."""
    sonuc: dict = {}
    for anahtar, bilgi in kayit.items():
        taraf, blok = anahtar.split("_", 1)
        z = np.load(bilgi["yol"])
        for k in z.files:
            tohum, aile = k.split("_")
            ref = None
            ref_ad = None
            if ayar.mod == "blok" and taraf == "soguk":
                ref_ad = os.path.join(DN, f"soguk_tahmin_{blok}.npz")
                if os.path.exists(ref_ad):
                    zr = np.load(ref_ad)
                    if k in zr.files:
                        ref = zr[k].astype("float64")
            elif ayar.mod == "blok" and taraf == "sicak":
                ref_ad = os.path.join(AO, f"{blok}_{tohum}_{aile}_uretim.npy")
                if os.path.exists(ref_ad):
                    ref = np.load(ref_ad).astype("float64")
            elif ayar.mod == "test" and taraf == "soguk":
                # p06 test-tarafi soguk aile onbellegi: (n_soguk, 3) TOHUM
                # ORTALAMASI. Tek tohum karsilastirilamaz; asagida ozel yol.
                ref_ad = "p06_test_soguk_aile.npy (tohum ortalamasi -- ayri)"
            v = z[k].astype("float64")
            if ref is None:
                sonuc[f"{ayar.mod}/{taraf}/{blok}/{k}"] = {
                    "durum": "REFERANS YOK", "referans": ref_ad
                }
                continue
            if ref.shape != v.shape:
                sonuc[f"{ayar.mod}/{taraf}/{blok}/{k}"] = {
                    "durum": "SEKIL UYUSMAZLIGI", "bizim": list(v.shape),
                    "referans_sekil": list(ref.shape), "referans": ref_ad,
                }
                continue
            mx = float(np.max(np.abs(v - ref)))
            # sicak onbellek float32 yazilmis -- kiyas o hassasiyette.
            esik = 1e-6 if taraf == "soguk" else 5e-6 * max(1.0, float(np.abs(ref).max()))
            sonuc[f"{ayar.mod}/{taraf}/{blok}/{k}"] = {
                "durum": "BIREBIR" if mx <= esik else "TUTMADI",
                "maxabs": mx, "esik": esik, "referans": os.path.basename(str(ref_ad)),
            }
    return sonuc


def main() -> int:
    ayar = Ayar()
    print("=" * 78)
    print("p18  TAM HAT YENIDEN EGITIM")
    print("=" * 78)
    print(json.dumps(ayar.sozluk(), ensure_ascii=False, indent=1))
    if ayar.dogrula and not ayar.taban_mi:
        raise SystemExit(
            "P18_DOGRULA=1 yalnizca TABAN yapilandirmasinda anlamli "
            "(kayip/tau/agac/alt_ornek verilmis)."
        )
    kayit = kos(ayar)
    R = {"ayar": ayar.sozluk(), "cikti": kayit, "sure_dk": round((time.time() - T0) / 60, 2)}
    if ayar.dogrula:
        R["dogrulama"] = dogrula(ayar, kayit)
        print("\nDOGRULAMA")
        for k, v in R["dogrulama"].items():
            print(f"  {k:36} {v.get('durum'):18} maxabs={v.get('maxabs')}")
        kotu = [k for k, v in R["dogrulama"].items() if v.get("durum") == "TUTMADI"]
        R["birebir_mi"] = not kotu and bool(R["dogrulama"])
    yol = os.path.join(ayar.cikti, f"p18_kosu_{ayar.etiket}_{ayar.mod}.json")
    os.makedirs(ayar.cikti, exist_ok=True)
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(f"\nkayit: {yol}")
    log("TAMAM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
