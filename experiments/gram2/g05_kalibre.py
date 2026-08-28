"""G5 -- (a) leave-one-out ile GERCEK hata modelini kalibre et,
(b) gonderilmemis dosyalarin span-ici / span-disi ayrisimini cikar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from g03_sinav import coz_kesik, kur  # noqa: E402

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent


def main() -> None:
    H = kur("v83")
    yon, G, b, m0, D, n = H["yon"], H["G"], H["b"], H["m0"], H["D"], H["n"]
    K = len(yon)

    print("=" * 100)
    print(
        "5c) LEAVE-ONE-OUT: her yon disarida, kalanlardan tahmin. Hata ~ |c| ile olcekleniyor mu?"
    )
    print("=" * 100)
    print(" hedef  gercek    tahmin    hata       |c|_1   artik_orani  span-ici mi?")
    sat = []
    for j in range(K):
        S = [i for i in range(K) if i != j]
        Gs, bs = G[np.ix_(S, S)], b[S]
        gj = G[j, S]
        c, _kz, rr, lam, U, _beta = coz_kesik(Gs, bs, r=None)  # min-norm b cozumu
        # projeksiyon katsayilari (d_j'yi span(S)'e projekte et)
        lam2, U2 = np.linalg.eigh(Gs)
        srt = np.argsort(lam2)[::-1]
        lam2, U2 = lam2[srt], U2[:, srt]
        tut = lam2 > lam2[0] * 1e-10
        inv = np.zeros_like(lam2)
        inv[tut] = 1.0 / lam2[tut]
        cj = U2 @ (inv * (U2.T @ gj))
        artik = float(G[j, j] - gj @ cj)
        oran = artik / G[j, j] if G[j, j] > 0 else 0.0
        bh = float(gj @ c)
        mh = m0 + G[j, j] - 2 * bh
        sh = np.sqrt(max(mh, 0.0))
        ger = float(H["skorlar"][H["adlar"].index(yon[j])])
        ici = "EVET" if oran < 1e-4 else "hayir"
        sat.append((yon[j], ger, sh, sh - ger, float(np.abs(cj).sum()), oran, ici))
        print(
            f"{yon[j]:>6}  {ger:.5f}  {sh:.5f}  {sh - ger:+.5f}  {np.abs(cj).sum():8.3f}  "
            f"{oran:11.3e}  {ici}"
        )

    ici = [s for s in sat if s[6] == "EVET"]
    disi = [s for s in sat if s[6] != "EVET"]
    if ici:
        h = np.array([s[3] for s in ici])
        c1 = np.array([s[4] for s in ici])
        print(
            f"\nSPAN-ICI ({len(ici)} adet): ort hata={h.mean():+.6f}  rms={np.sqrt((h**2).mean()):.6f}  "
            f"max|hata|={np.abs(h).max():.6f}  |c|_1 araligi=[{c1.min():.2f},{c1.max():.2f}]"
        )
        # hata / |c|_1 orani -> v103'un |w|_1=3.37 icin olcekle
        oran_h = np.abs(h) / np.maximum(c1, 1e-9)
        print(f"  |hata|/|c|_1: medyan={np.median(oran_h):.3e}  max={oran_h.max():.3e}")
        print(
            f"  -> |w|_1=3.368 icin beklenen hata buyuklugu: "
            f"medyan {np.median(oran_h) * 3.368:.6f}, en kotu {oran_h.max() * 3.368:.6f}"
        )
    if disi:
        h = np.array([s[3] for s in disi])
        print(
            f"\nSPAN-DISI ({len(disi)} adet): ort hata={h.mean():+.6f}  "
            f"(hepsi >0 ise yontem span disinda KOTUMSER)"
        )
        for s in disi:
            print(f"   {s[0]:>6} artik_orani={s[5]:.4f} hata={s[3]:+.5f}")

    # ================= GONDERILMEMIS ADAYLAR =================
    print("\n" + "=" * 100)
    print("GONDERILMEMIS DOSYALAR: span-ici bilesen + YENI (span-disi) icerik")
    print("=" * 100)
    gonderilen = {
        h
        for h in [
            "tuketim_v2.csv",
            "tuketim_v7.csv",
            "tuketim_v15.csv",
            "tuketim_v16.csv",
            "tuketim_v18.csv",
            "tuketim_v25_hedge.csv",
            "tuketim_v27_v18hedge.csv",
            "tuketim_v30_buzme.csv",
            "tuketim_v46_gun.csv",
            "tuketim_v44_v27yeni.csv",
            "tuketim_v47_eskison.csv",
            "tuketim_v50_nihai30.csv",
            "tuketim_v55_gunolcek.csv",
            "tuketim_v67_c1335_olay.csv",
            "tuketim_v73_soguk_gun160.csv",
            "tuketim_v79_S3.csv",
            "tuketim_v80_optimum.csv",
            "tuketim_v81_sicak08.csv",
            "tuketim_v83_sicak_optimum.csv",
            "tuketim_v101_hepsi.csv",
            "tuketim_v102_kappa_optimum.csv",
            "gun1_baseline.csv",
        ]
    }
    # tam cozum (w) -- span icinde en iyi tahminci
    w, kz, rr, *_ = coz_kesik(G, b, r=17)
    m_opt = m0 - kz
    x0 = H["X"][H["i0"]]
    ids = H["ids"]

    lam2, U2 = np.linalg.eigh(G)
    srt = np.argsort(lam2)[::-1]
    lam2, U2 = lam2[srt], U2[:, srt]
    tut = lam2 > lam2[0] * 1e-10
    inv = np.zeros_like(lam2)
    inv[tut] = 1.0 / lam2[tut]

    kayit = []
    dosyalar = sorted(p.name for p in GON.glob("*.csv") if p.name not in gonderilen)
    print(
        f"{'dosya':38s} {'Q(v83)':>9} {'yeni%':>7} {'b_ici':>10} {'MSE_alt':>9} {'RMSLE_alt':>10}"
    )
    for dosya in dosyalar:
        try:
            d = pd.read_csv(GON / dosya)
            if list(d.columns) != ["id", "tuketim"] or len(d) != n:
                print(f"{dosya:38s}  [format uyumsuz, atlandi]")
                continue
            if not (d["id"].values == ids.values).all():
                print(f"{dosya:38s}  [id sirasi farkli, atlandi]")
                continue
            dj = np.log1p(d["tuketim"].to_numpy("f8")) - x0
            Qj = float(dj @ dj / n)
            gj = (D @ dj) / n
            cj = U2 @ (inv * (U2.T @ gj))
            artik = max(Qj - float(gj @ cj), 0.0)
            yeni = artik / Qj if Qj > 0 else 0.0
            bh = float(gj @ w)
            mh = m0 + Qj - 2 * bh
            kayit.append(
                dict(
                    dosya=dosya,
                    Q=Qj,
                    yeni_oran=yeni,
                    b_ici=bh,
                    mse_alt=mh,
                    rmsle_alt=float(np.sqrt(max(mh, 0.0))),
                )
            )
            print(
                f"{dosya:38s} {Qj:9.5f} {yeni * 100:6.2f}% {bh:+10.5f} {mh:9.5f} "
                f"{np.sqrt(max(mh, 0.0)):10.5f}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"{dosya:38s}  [hata: {e}]")

    print("\nNOT: 'MSE_alt/RMSLE_alt' = span-DISI bilesenin hedefe KATKISI SIFIR varsayimiyla")
    print("     KOTUMSER tahmin. v101 icin bu varsayim 0.0144 kotumser cikmisti.")
    print("     'yeni%' = Q'nun span disinda kalan yuzdesi = OLCUM DEGERI olan kisim.")
    print(f"     Span ici ulasilabilir en iyi: MSE={m_opt:.6f} RMSLE={np.sqrt(m_opt):.6f}")

    (CIK / "g05_adaylar.json").write_text(
        json.dumps(
            {
                "loo": [
                    dict(
                        zip(
                            ["ad", "gercek", "tahmin", "hata", "c_l1", "artik_orani", "span_ici"], s
                        )
                    )
                    for s in sat
                ],
                "adaylar": kayit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nyazildi: g05_adaylar.json")


if __name__ == "__main__":
    main()
