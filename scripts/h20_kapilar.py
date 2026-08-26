"""H20 UCUZ KAPILARI -- kumeli soguk maskeleme fikri ayakta mi?

HIPOTEZ
-------
``soguk_maskele`` egitimde trafolari RASTGELE sogutur. Rastgele sogutulmus
bir trafonun KOMSULARI HALA SICAK -- model onlara yaslanmayi ogrenir
(ilce profilleri, grup profilleri, ilce yapisi kolonlari). Ama TEST'te
1.326 trafo AYNI ANDA geliyor; eger mekansal olarak kumeleniyorlarsa
onlarin komsulari da SOGUK ve model servis aninda MEVCUT OLMAYAN bilgiye
yaslanmis oluyor.

DropoutNet'in yarim uygulanmasi: test'in soguk ORANI taklit ediliyor ama
KORELASYON YAPISI taklit edilmiyor.

KAPILAR (hepsi gecmeli, yoksa fikir olur)
------------------------------------------
K1. 2026-05-11 kohortu mekansal olarak KUMELENIYOR mu? (ayni buyuklukte
    rastgele orneklemle karsilastir)
K2. TEST'te o kohortun ilcelerinde SICAK komsu KALIYOR mu, yoksa ilce
    topluca mi soguk? Egitimdeki rastgele maskelenmis sogukla karsilastir.
K3. Model soguk satirlarda komsu/ilce kolonlarina gercekten yasliyor mu?
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
MASKE_ORANI = 0.2216


def herfindahl(pay: np.ndarray) -> float:
    p = pay / pay.sum()
    return float((p**2).sum())


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "lokasyon"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih", "lokasyon"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr_set = set(tr["tanim"].unique())
    te_trafo = te.drop_duplicates("tanim")[["tanim", "lokasyon"]].set_index("tanim")
    soguk_set = set(te_trafo.index) - tr_set

    ilk_te = te.groupby("tanim")["tarih"].min()
    kohort = set(ilk_te[(ilk_te == pd.Timestamp("2026-05-11"))].index) & soguk_set
    print("=" * 88)
    print("K1 -- 2026-05-11 kohortu MEKANSAL olarak kumeleniyor mu?")
    print("=" * 88)
    print(f"\n  kohort {len(kohort):,} trafo   test toplam {len(te_trafo):,} trafo")

    ilce_k = te_trafo.loc[sorted(kohort), "lokasyon"].value_counts()
    print(f"  kohortun yayildigi ilce sayisi: {len(ilce_k)} / {te_trafo['lokasyon'].nunique()}")
    print(f"  en yogun 10 ilcenin payi: {ilce_k.head(10).sum() / len(kohort):.4f}")
    print(f"  Herfindahl: {herfindahl(ilce_k.to_numpy()):.4f}")

    rng = np.random.default_rng(0)
    hs, tops = [], []
    tum = te_trafo.index.to_numpy()
    for _ in range(200):
        orn = rng.choice(tum, size=len(kohort), replace=False)
        vc = te_trafo.loc[orn, "lokasyon"].value_counts()
        hs.append(herfindahl(vc.to_numpy()))
        tops.append(vc.head(10).sum() / len(kohort))
    print("\n  RASTGELE ayni buyuklukte ornek (200 tekrar):")
    print(f"    Herfindahl  ort {np.mean(hs):.4f}  std {np.std(hs):.4f}")
    print(f"    top10 payi  ort {np.mean(tops):.4f}  std {np.std(tops):.4f}")
    z_h = (herfindahl(ilce_k.to_numpy()) - np.mean(hs)) / max(np.std(hs), 1e-12)
    z_t = (ilce_k.head(10).sum() / len(kohort) - np.mean(tops)) / max(np.std(tops), 1e-12)
    print(f"\n  >>> z(Herfindahl) = {z_h:+.2f}   z(top10) = {z_t:+.2f}")
    k1 = abs(z_h) > 3 or abs(z_t) > 3
    print(f"  K1 {'GECTI -- kumelenme VAR' if k1 else 'KALDI -- kumelenme YOK'}")

    print("\n" + "=" * 88)
    print("K2 -- TEST'te soguk trafonun ilcesinde SICAK komsu kaliyor mu?")
    print("=" * 88)
    ilce_toplam = te_trafo["lokasyon"].value_counts()
    ilce_soguk = te_trafo.loc[sorted(soguk_set), "lokasyon"].value_counts()
    ilce_sicak = (ilce_toplam - ilce_soguk.reindex(ilce_toplam.index).fillna(0)).clip(lower=0)
    pay_soguk = ilce_soguk.reindex(ilce_toplam.index).fillna(0) / ilce_toplam

    sg = te_trafo.loc[sorted(soguk_set), "lokasyon"]
    print(f"\n  TEST soguk trafolari ({len(sg):,}):")
    print(
        f"    kendi ilcesindeki SICAK trafo sayisi: "
        f"medyan {int(ilce_sicak.reindex(sg).median())}  "
        f"q10 {int(ilce_sicak.reindex(sg).quantile(0.10))}  "
        f"q90 {int(ilce_sicak.reindex(sg).quantile(0.90))}"
    )
    print(
        f"    kendi ilcesinin SOGUK PAYI: medyan "
        f"{pay_soguk.reindex(sg).median():.4f}  "
        f"q90 {pay_soguk.reindex(sg).quantile(0.90):.4f}"
    )

    # egitimdeki RASTGELE maske ile karsilastir
    tr_trafo = tr.drop_duplicates("tanim")[["tanim", "lokasyon"]].set_index("tanim")
    rng2 = np.random.default_rng(1000)
    sec = set(
        rng2.choice(tr_trafo.index.to_numpy(), size=int(len(tr_trafo) * MASKE_ORANI), replace=False)
    )
    it = tr_trafo["lokasyon"].value_counts()
    isg = tr_trafo.loc[sorted(sec), "lokasyon"].value_counts()
    isk = (it - isg.reindex(it.index).fillna(0)).clip(lower=0)
    payt = isg.reindex(it.index).fillna(0) / it
    sg2 = tr_trafo.loc[sorted(sec), "lokasyon"]
    print(f"\n  EGITIMDEKI rastgele maske ({len(sec):,} trafo, oran {MASKE_ORANI}):")
    print(
        f"    kendi ilcesindeki SICAK trafo sayisi: "
        f"medyan {int(isk.reindex(sg2).median())}  "
        f"q10 {int(isk.reindex(sg2).quantile(0.10))}  "
        f"q90 {int(isk.reindex(sg2).quantile(0.90))}"
    )
    print(
        f"    kendi ilcesinin SOGUK PAYI: medyan {payt.reindex(sg2).median():.4f}  "
        f"q90 {payt.reindex(sg2).quantile(0.90):.4f}"
    )

    m_te = float(pay_soguk.reindex(sg).median())
    m_tr = float(payt.reindex(sg2).median())
    print(
        f"\n  >>> ilce soguk payi  TEST {m_te:.4f}  vs  EGITIM MASKE {m_tr:.4f}"
        f"   oran {m_te / max(m_tr, 1e-9):.2f}"
    )
    k2 = m_te > 1.5 * m_tr
    print(f"  K2 {'GECTI -- test ilceleri COK daha soguk' if k2 else 'KALDI -- dagilimlar benzer'}")

    print("\n" + "=" * 88)
    print("K3 -- ilce granulerligi ne kadar kaba? (yaslanma potansiyeli)")
    print("=" * 88)
    print(f"\n  essiz lokasyon (ilce) sayisi: {te_trafo['lokasyon'].nunique()}")
    print(
        f"  ilce basina test trafosu: medyan "
        f"{int(ilce_toplam.median())}  ort {ilce_toplam.mean():.1f}"
    )
    print("\n  NOT: ilce yalnizca 47 essiz deger (H2'de olculdu). Bu kadar kaba")
    print("  bir granulerlikte 'komsuya yaslanma' zaten cok sinirli olabilir --")
    print("  kimlik komsulugu R2 0,019, ilce R2 0,016 (h16: dogru nufusta ~0).")

    print("\n" + "=" * 88)
    print("HUKUM")
    print("=" * 88)
    print(f"  K1 {'GECTI' if k1 else 'KALDI'}   K2 {'GECTI' if k2 else 'KALDI'}")
    if k1 and k2:
        print("  -> Mekanizma AYAKTA. Tezgah kurulabilir.")
    else:
        print("  -> Mekanizma COKTU. Deftere yaz, tezgah KURMA.")
        print("     (Kumelenme yoksa ya da test ilceleri egitimdekinden daha")
        print("      soguk degilse, kumeli maskeleme test kosullarini daha iyi")
        print("      taklit ETMIYOR demektir.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
