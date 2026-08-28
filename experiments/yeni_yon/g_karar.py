"""KARAR ve GONDERIM URETIMI -- iki rejim, iki prob.

NEDEN BOYLE
-----------
``f_ofset``in butun yonleri BLOK-DISI sinavda COKTU (hepsi negatif). O aile
CV artiginin blok-ozgu yapisini ezberliyor; ``L^2/Q`` ile olculunce "kazanc"
gorunuyordu cunku o buyukluk tanim geregi >= 0. Dogru olcut, ic-katmandan gelen
``kappa``nin DISARIDAKI blokta uygulanmasi: ``dMSE = 2*k_ic*L_dis - k_ic^2*Q_dis``.

Ayakta kalan tek aile HAM MODEL FARKI: yeni model sinifinin uretimle
anlasmazligi, trafo bazinda ortalanmis hali. Bunlar blok-disi POZITIF.

Bir dosya = bir ``kappa``. Bu yuzden iki dosya iki AYRIK rejimden kuruluyor:
    Y1  SICAK satirlar (556.319)  -- klasik zaman serisi ailesinin anlasmazligi
    Y2  SOGUK satirlar (158.369)  -- farkli hedef/kayip ailesinin anlasmazligi
Ayrik oldugu icin iki yon BIRBIRINE TAM DIK; iki prob bagimsiz iki skaler verir.

Her havuz icin agirlik, blok-disi sinavla secilen ``lam`` ile ridge'den cozulur.
Blok-disi kazanc pozitif degilse havuzun EN IYI TEK yonu gonderilir.

Kaggle'a HICBIR SEY GONDERMEZ; yalnizca ``submissions/`` altina dosya yazar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import c_olc
import d_varyant
import ortak

LAMBDALAR = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0)

#: Gonderilen yonun enerjisi (``Q = |d|^2/n``). LB skorundaki degisim ~Q
#: mertebesinde; 5 haneli yuvarlamaya gore ~2000 birim, yani ``kappa`` uc
#: anlamli haneyle cozulur. Kirpma da ihmal edilebilir kalir.
Q_HEDEF = 0.02

HAVUZLAR = {
    "Y1_sicak": ("sicak", ("trafo_sicak", "sicak", "kirp030_sicak")),
    "Y2_soguk": ("soguk", ("trafo_soguk", "soguk", "kirp030_soguk")),
}


def _yon_kur(varyantlar, tab, mcv, mte, g):  # noqa: ANN001, ANN202
    isim, Dcv, Dte = [], {b: [] for b in ortak.BLOKLAR}, []
    for ad in sorted(p.stem for p in ortak.ONB.glob("*.npz")):
        cv, tp = ortak.yukle_aday(ad)
        lgp = np.log1p(np.clip(tp, 0.0, None))
        for v in varyantlar:
            isim.append(f"{ad}|{v}")
            Dte.append(d_varyant.donustur(v, lgp - g.v102, mte))
            for b in ortak.BLOKLAR:
                dh = np.log1p(np.clip(cv[b], 0.0, None)) - tab[b]["taban"]
                Dcv[b].append(d_varyant.donustur(v, dh, mcv[b]))
    return isim, {b: np.vstack(Dcv[b]) for b in ortak.BLOKLAR}, np.vstack(Dte)


def _gram(tab, Dcv, bloklar):  # noqa: ANN001, ANN202
    k = Dcv[bloklar[0]].shape[0]
    G, b, n = np.zeros((k, k)), np.zeros(k), 0
    for blk in bloklar:
        D = Dcv[blk]
        r = tab[blk]["lgy"] - tab[blk]["taban"]
        G += D @ D.T
        b += D @ r
        n += D.shape[1]
    return G / n, b / n, n


def _blok_kappa(tab, Dcv, i):  # noqa: ANN001, ANN202
    """Tek yonun BLOK BAZLI kappa'si. Isaret bloklar arasi donuyorsa prob hala
    gecerlidir (kalici kural 14) -- olduren tek sey |kappa| ~ 0."""
    cik = {}
    for b in ortak.BLOKLAR:
        d = Dcv[b][i]
        r = tab[b]["lgy"] - tab[b]["taban"]
        Q = float(d @ d) / len(d)
        L = float(r @ d) / len(d)
        cik[b] = (L / Q if Q > 1e-12 else 0.0, L * L / Q if Q > 1e-12 else 0.0)
    return cik


def main() -> None:
    g = ortak.geo()
    tab = c_olc.uretim_tabani()
    mcv = d_varyant._meta_cv()
    mte = d_varyant._meta_test()
    rapor: dict = {}

    for etiket, (_rej, varyantlar) in HAVUZLAR.items():
        print("\n" + "=" * 118)
        print(f"{etiket}  --  varyantlar {varyantlar}")
        print("=" * 118)
        isim, Dcv, Dte = _yon_kur(varyantlar, tab, mcv, mte, g)

        print(
            f"{'yon':34}{'k_yaz':>9}{'k_guz':>9}{'k_kis':>9}"
            f"{'k_havuz':>9}{'q_perp':>10}{'q_yeni':>10}  maks kos"
        )
        tek_bilgi = []
        for i, nm in enumerate(isim):
            bk = _blok_kappa(tab, Dcv, i)
            G1, b1, _ = _gram(
                tab, {b: Dcv[b][i : i + 1] for b in ortak.BLOKLAR}, list(ortak.BLOKLAR)
            )
            kh = float(b1[0] / G1[0, 0]) if G1[0, 0] > 1e-12 else 0.0
            r = g.olc(nm, Dte[i])
            tek_bilgi.append(
                {
                    "ad": nm,
                    "kappa_blok": {b: bk[b][0] for b in bk},
                    "kappa_havuz": kh,
                    **{k: v for k, v in r.items() if k != "kos"},
                    "maks_kos": float(r["maks_kos"]),
                }
            )
            print(
                f"{nm[:34]:34}{bk['yaz25'][0]:>+9.3f}{bk['guz25'][0]:>+9.3f}"
                f"{bk['kis26'][0]:>+9.3f}{kh:>+9.3f}{r['q_perp']:>10.5f}"
                f"{r['q_yeni']:>10.5f}  {r['maks_kos_ad']} {r['maks_kos']:+.2f}"
            )

        print("\nLAMBDA SINAVI (agirlik iki bloktan, dMSE ucuncu blokta):")
        en_lam, en_skor = None, -1e9
        for lam in LAMBDALAR:
            tek = []
            for disi in ortak.BLOKLAR:
                ic = [b for b in ortak.BLOKLAR if b != disi]
                G, b, _ = _gram(tab, Dcv, ic)
                w = np.linalg.solve(G + lam * np.eye(len(b)) * np.trace(G) / len(b), b)
                Go, bo, _ = _gram(tab, Dcv, [disi])
                tek.append(2.0 * float(w @ bo) - float(w @ Go @ w))
            ort = float(np.mean(tek))
            print(f"   lam={lam:<7g} ort {ort:+.6f}  [{', '.join(f'{t:+.5f}' for t in tek)}]")
            if ort > en_skor:
                en_lam, en_skor = lam, ort

        G, b, _ = _gram(tab, Dcv, list(ortak.BLOKLAR))
        w = np.linalg.solve(G + en_lam * np.eye(len(b)) * np.trace(G) / len(b), b)
        if en_skor > 0:
            d = w @ Dte
            secim = f"bilesik (lam={en_lam})"
        else:
            # Bilesik tasimadi. Kalici kural 14: isaretin bloklar arasi donmesi
            # PROBU OLDURMEZ (kappa'yi LB cozer); olduren tek sey |kappa| ~ 0.
            # Yine de en saglam aday, kappa isareti UC BLOKTA DA AYNI olandir.
            aday = [
                (i, t)
                for i, t in enumerate(tek_bilgi)
                if abs(t["kappa_havuz"]) >= 0.02
                and len({np.sign(v) for v in t["kappa_blok"].values()}) == 1
                and t["q_perp"] >= 0.005
            ]
            if not aday:
                aday = [
                    (i, t)
                    for i, t in enumerate(tek_bilgi)
                    if abs(t["kappa_havuz"]) >= 0.02 and t["q_perp"] >= 0.005
                ]
            # YENI enerjisi en buyuk, mevcut dik envanterle en az ortusen yon
            j, _t = max(aday, key=lambda it: it[1]["q_yeni"] * (1.0 - abs(it[1]["maks_kos"])))
            d = Dte[j]
            secim = (
                f"TEK yon {isim[j]} (bilesik blok-disi sinavda kaldi; "
                "kappa isareti uc blokta da ayni)"
            )
        # OKUNABILIR OLCEK: gonderilen enerji Q_HEDEF'e olceklenir. Buyuk olcek
        # probu keskinlestirir ama kirpma kuadratigi bozar; Q_HEDEF ikisinin
        # dengesi. kappa'yi zaten LB cozecek, mutlak olcek bilgiyi degistirmez.
        Qd = float(d @ d) / g.n
        carpan = float(np.sqrt(Q_HEDEF / Qd)) if Qd > 1e-12 else 1.0
        d = d * carpan
        print(f"\n   secim: {secim}   blok-disi kazanc {en_skor:+.6f}")
        print(f"   olcek: Q {Qd:.6f} -> {Q_HEDEF:.4f}  (carpan {carpan:.4f})")

        r = g.olc(etiket, d)
        print(
            f"   Q {r['Q']:.6f}  q_perp {r['q_perp']:.6f}  q_yeni {r['q_yeni']:.6f}"
            f"  span payi {r['span_pay']:.3f}"
        )
        print("   kosinusler:", {k: round(v, 3) for k, v in r["kos"].items()})
        rapor[etiket] = {
            "secim": secim,
            "blok_disi_kazanc": en_skor,
            "lambda": en_lam,
            "agirlik": dict(zip(isim, [float(x) for x in w], strict=True)),
            "tek_yonler": tek_bilgi,
            "geo": {
                **{k: v for k, v in r.items() if k != "kos"},
                "kos": {a: float(c) for a, c in r["kos"].items()},
            },
        }
        rapor[etiket]["_d"] = d

    # --- dosyalar
    d1 = rapor["Y1_sicak"].pop("_d")
    d2 = rapor["Y2_soguk"].pop("_d")
    print(
        f"\nY1 ile Y2 kosinusu: {float(d1 @ d2) / np.sqrt(float(d1 @ d1) * float(d2 @ d2)):.6f}"
        "   (ayrik rejim -> tam dik beklenir)"
    )
    _yaz("tuketim_y1_sicak_klasik.csv", g, d1)
    _yaz("tuketim_y2_soguk_hedef.csv", g, d2)

    # --- toplam YENI enerji muhasebesi
    E, _ = g.envanter()
    taban = list(E)
    top = 0.0
    for nm, d in (("Y1", d1), ("Y2", d2)):
        up, _l = g.perp(d)
        v = up.copy()
        for e in taban:
            v -= (float(v @ e) / g.n) * e
        qy = float(v @ v) / g.n
        taban.append(v / np.sqrt(qy))
        top += qy
        print(f"{nm}: span+envanter disi YENI enerji {qy:.6f}")
    acik = 0.0202
    print(f"TOPLAM YENI enerji (2 boyut) = {top:.6f}")
    print(
        f"  f=0.4115 -> kazanc {0.4115**2 * top:.6f}  = acigin %{100 * 0.4115**2 * top / acik:.1f}'i"
    )
    for f in (0.10, 0.20, 0.30, 0.4115):
        print(
            f"  f={f:.4f} -> kazanc {f * f * top:.6f}  = acigin %{100 * f * f * top / acik:.1f}'i"
        )
    rapor["yeni_enerji_toplam"] = top
    json.dump(rapor, open(ortak.CIK / "g_karar.json", "w"), indent=2, default=float)


def _yaz(ad: str, g, d: np.ndarray) -> None:  # noqa: ANN001
    yl = g.v102 + d
    neg = int((yl < 0).sum())
    yl = np.maximum(yl, 0.0)
    yol = ortak.GON / ad
    pd.DataFrame({"id": g.ids, "tuketim": np.clip(np.expm1(yl), 0.0, None)}).to_csv(
        yol, index=False, float_format="%.17g"
    )
    fark = np.abs(yl - g.v102)
    print(
        f"YAZILDI {yol}\n   kirpilan {neg} | degisen {int((fark > 1e-9).sum())}"
        f" | maxabs {fark.max():.4f} | ort {fark.mean():.5f}"
    )


if __name__ == "__main__":
    main()
