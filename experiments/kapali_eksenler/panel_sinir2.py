"""PANEL SINIR GUNLERI -- PENCERE KENARI ARTEFAKTI TEMIZLENMIS.

panel_sinir.py'nin bulgusu: giris/cikis gunlerinde model SISTEMATIK OLARAK
YUKSEK tahmin ediyor (yanlilik -0,24 .. -1,12), UC BLOKTA DA. Mekanizma
fiziksel: o gun sayac KISMI gun olcuyor.

AMA iki artefakt vardi:
  * kis26'nin "olum" kumesi train sonuyla (2026-03-31) dolu -> gercek olum
    degil, PENCERE KENARI. Nitekim yanliligi -0,147, guz25'in -1,389'una
    karsi.
  * test tarafinda 2026-04-01 (3.928 satir) ve 2026-07-31 (6.795 satir)
    pencere kenari; ilki train'den DEVAM eden trafolar, ikincisi bilinemez.

Bu betik ikisini de disarida birakir ve:
  1) TEST'te gercekten uygulanabilir sinir nufusunu sayar (train'e koprulu)
  2) uc blokta (d_giris, d_cikis) ORTAK optimumunu arar
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))

from ortak import BLOKLAR, SICAK_PAY, bloklari_kur, kuresel_delta, mse, taban_r  # noqa: E402

CIK = Path(__file__).resolve().parent
GUN = pd.Timedelta(days=1)
TRAIN_BAS = pd.Timestamp("2025-01-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
TEST_BAS = pd.Timestamp("2026-04-01")
TEST_SON = pd.Timestamp("2026-07-31")
pd.set_option("display.width", 240)


def bayraklar(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values(["tanim", "tarih"], kind="mergesort").copy()
    onc = d.groupby("tanim", observed=True)["tarih"].shift(1)
    son = d.groupby("tanim", observed=True)["tarih"].shift(-1)
    d["giris"] = onc.isna() | ((d["tarih"] - onc) > GUN)
    d["cikis"] = son.isna() | ((son - d["tarih"]) > GUN)
    return d


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )

    # --- TRAIN tarafi: pencere kenari HARIC
    ts = bayraklar(tr[["tanim", "tarih"]])
    ts["giris"] &= ts["tarih"] != TRAIN_BAS
    ts["cikis"] &= ts["tarih"] != TRAIN_SON
    print("TRAIN (pencere kenari cikarilmis):")
    print(
        f"  giris {int(ts['giris'].sum()):,}  cikis {int(ts['cikis'].sum()):,}  / {len(ts):,} satir"
    )

    # --- TEST tarafi: train'e KOPRULU. 2026-04-01 ancak trafo 2026-03-31'de
    #     train'de YOKSA giristir. 2026-07-31 asla cikis SAYILMAZ (bilinemez).
    ort = pd.concat([tr[["tanim", "tarih"]], te[["tanim", "tarih"]]], ignore_index=True)
    os_ = bayraklar(ort)
    os_ = os_[os_["tarih"] >= TEST_BAS]
    os_["cikis"] &= os_["tarih"] != TEST_SON
    print("\nTEST (train'e koprulu, 2026-07-31 cikis SAYILMIYOR):")
    print(
        f"  giris {int(os_['giris'].sum()):,} ({os_['giris'].mean():.5f})"
        f"  cikis {int(os_['cikis'].sum()):,} ({os_['cikis'].mean():.5f})"
        f"  / {len(os_):,} satir"
    )
    print("\n  test GIRIS gunleri (en kalabalik 8):")
    print(os_.loc[os_["giris"], "tarih"].value_counts().head(8).to_string())
    print("\n  test CIKIS gunleri (en kalabalik 5):")
    print(os_.loc[os_["cikis"], "tarih"].value_counts().head(5).to_string())

    n_g = int(os_["giris"].sum())
    n_c = int(os_["cikis"].sum())
    N = len(te)
    print(
        f"\n  Q(giris) = {n_g / N:.7f}   Q(cikis) = {n_c / N:.7f}   "
        f"kesisim {int((os_['giris'] & os_['cikis']).sum()):,}"
    )

    # --- BLOKLARDA olcum
    bloklar = bloklari_kur()
    taban = {k: taban_r(bloklar[k]) for k in BLOKLAR}
    mask: dict[str, dict[str, np.ndarray]] = {}
    print("\n" + "=" * 100)
    print("SICAK BLOKLARDA SINIR YANLILIGI (pencere kenari cikarilmis)")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        sol = pd.DataFrame(
            {
                "tanim": b.cerceve["tanim"].to_numpy(),
                "tarih": pd.to_datetime(b.cerceve["tarih"].to_numpy()),
            }
        )
        sol["_i"] = np.arange(len(sol))
        j = sol.merge(
            ts[["tanim", "tarih", "giris", "cikis"]], on=["tanim", "tarih"], how="left"
        ).sort_values("_i")
        g = j["giris"].fillna(False).to_numpy().astype("float64")
        c = j["cikis"].fillna(False).to_numpy().astype("float64")
        mask[ad] = {"giris": g, "cikis": c}
        e = b.lgy - np.maximum(taban[ad] + b.lgc, 0.0)
        for nm, m in (("GIRIS", g), ("CIKIS", c)):
            mm = m.astype(bool)
            if mm.sum() == 0:
                print(f"  {ad:6} {nm}: YOK")
                continue
            print(
                f"  {ad:6} {nm}: n={int(mm.sum()):5,}  yanlilik {e[mm].mean():+.4f}  "
                f"t={e[mm].mean() / (e[mm].std() / np.sqrt(mm.sum())):+7.2f}  "
                f"mse {float((e[mm] ** 2).mean()):.4f}"
            )

    print("\n" + "=" * 100)
    print("ORTAK OPTIMUM (d_giris, d_cikis) -- her blokta, seviye-notr, kirpmali")
    print("=" * 100)
    izgara = np.arange(-1.20, 0.201, 0.10)
    en_blok = {}
    for ad in BLOKLAR:
        b = bloklar[ad]
        r0 = taban[ad]
        m0 = mse(b, r0 + kuresel_delta(b, r0))
        en = None
        for dg in izgara:
            for dc in izgara:
                rr = r0 + dg * mask[ad]["giris"] + dc * mask[ad]["cikis"]
                v = mse(b, rr + kuresel_delta(b, rr))
                if en is None or v < en[2]:
                    en = (float(dg), float(dc), v)
        en_blok[ad] = (en[0], en[1], en[2] - m0)
        print(
            f"  {ad:6} d_giris {en[0]:+.2f}  d_cikis {en[1]:+.2f}  "
            f"kazanc {en[2] - m0:+.6f} sicak MSE  "
            f"({(en[2] - m0) * SICAK_PAY:+.6f} test MSE)"
        )

    print("\n" + "=" * 100)
    print("SABIT ADAYLAR -- ayni (d_giris, d_cikis) uc blokta birden")
    print("=" * 100)
    adaylar = [
        (-0.30, -0.30),
        (-0.40, -0.50),
        (-0.50, -0.60),
        (-0.60, -0.70),
        (-0.70, -0.60),
        (-0.50, -1.00),
        (-0.80, -0.80),
        (-0.20, -0.40),
    ]
    satirlar = []
    print(
        f"{'d_giris/d_cikis':20}{'yaz25':>11}{'guz25':>11}{'kis26':>11}"
        f"{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 88)
    for dg, dc in adaylar:
        s: dict = {"d_giris": dg, "d_cikis": dc}
        tn = td = 0.0
        for ad in BLOKLAR:
            b = bloklar[ad]
            r0 = taban[ad]
            m0 = mse(b, r0 + kuresel_delta(b, r0))
            rr = r0 + dg * mask[ad]["giris"] + dc * mask[ad]["cikis"]
            d = mse(b, rr + kuresel_delta(b, rr)) - m0
            s[ad] = d
            tn += b.n
            td += d * b.n
        s["GENEL"] = td / tn
        s["testMSE"] = s["GENEL"] * SICAK_PAY
        s["uc_ayni"] = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
        satirlar.append(s)
        karar = (
            "KABUL"
            if s["uc_ayni"] and s["testMSE"] <= -0.002
            else "red(kucuk)"
            if s["uc_ayni"] and s["testMSE"] < 0
            else "ters isaret"
            if s["testMSE"] < 0
            else "RED(zararli)"
        )
        print(
            f"{f'{dg:+.2f} / {dc:+.2f}':20}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{s['GENEL']:>+11.5f}{s['testMSE']:>+11.5f}  {karar}"
        )

    print("\n" + "=" * 100)
    print("TESTE OLCEKLEME -- blok payi ile test payi FARKLI")
    print("=" * 100)
    for ad in BLOKLAR:
        b = bloklar[ad]
        print(
            f"  {ad:6} giris payi {mask[ad]['giris'].mean():.6f}  "
            f"cikis payi {mask[ad]['cikis'].mean():.6f}"
        )
    print(f"  TEST   giris payi {n_g / N:.6f}  cikis payi {n_c / N:.6f}")
    print("\n  -> blok kazanci test payina ORANLA olceklenir; asagidaki tahmin")
    print("     her blogun optimumunu test giris/cikis paylarina tasir.")
    for ad in BLOKLAR:
        b = bloklar[ad]
        dg, dc, kz = en_blok[ad]
        # ayri ayri yeniden olc: giris ve cikis katkilarini ayirmak icin
        r0 = taban[ad]
        m0 = mse(b, r0 + kuresel_delta(b, r0))
        rg = r0 + dg * mask[ad]["giris"]
        rc = r0 + dc * mask[ad]["cikis"]
        kg = mse(b, rg + kuresel_delta(b, rg)) - m0
        kc = mse(b, rc + kuresel_delta(b, rc)) - m0
        olc_g = kg * (n_g / N) / max(mask[ad]["giris"].mean() * SICAK_PAY, 1e-12) * SICAK_PAY
        olc_c = kc * (n_c / N) / max(mask[ad]["cikis"].mean() * SICAK_PAY, 1e-12) * SICAK_PAY
        print(
            f"  {ad:6} giris {kg:+.6f} -> teste {olc_g:+.6f}   "
            f"cikis {kc:+.6f} -> teste {olc_c:+.6f}   TOPLAM {olc_g + olc_c:+.6f}"
        )

    (CIK / "panel_sinir2.json").write_text(
        json.dumps(
            {"test_giris": n_g, "test_cikis": n_c, "blok_optimum": en_blok, "adaylar": satirlar},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
