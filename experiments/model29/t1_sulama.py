"""T1 -- SULAMA MEVSIMSELLIGI. Model uydurulmaz, INSA edilir.

NEDEN GBM DEGIL
---------------
"Yaz genligi" kolonu dogrulama duzeneginde YAPISAL OLARAK olculemez:
  * kesim 2025-03-31 blogunda ozet penceresi Oca-Mar 2025 -- yaz HIC yok,
    kolon BOS. Testin sordugu soru ("yaz oraniyla yazi tahmin et") tam da
    bu blok, ve blok kolonu ureteMEZ.
  * kesim 2025-09-30 / 2025-11-30 bloklari kolonu doldurabiliyor ama
    SONBAHARI/KISI soruyor -- ters soru, katsayi ters isaretli ogreniliyor.
Bu yuzden GBM'e kolon vermek yerine etkiyi DOGRUDAN kurariz.

MEKANIZMA (olculdu, ilce ortalamasi yaz(6-7) - kis(1-2) log genligi):
  Gordes +1,55 · Odemis +1,01 · Bayindir +0,91 · Tire +0,64 ...
  Urla -0,14 · Bornova -0,09 · Konak -0,08
Tepe: Kucuk Menderes ve Gediz havzasinin SULANAN TARIM ilceleri (tarim_orani
korelasyonu +0,40). Test penceresi Nis-Tem tam rampanin ustunde; ozet penceresi
2026-03-31'de, sulama BASLAMADAN bitiyor.

INSA
----
  a_i   : trafonun KENDI 2025 yaz - 2025/26 kis log farki (>=20 gun/pencere)
  a_ilce: ilcedeki a_i medyani (>=8 olculmus trafo), yoksa arazi ortusunden
          kestirilen a_hat (ridge: tarim/yerlesim/su/agac/otlak oranlari)
  r(gun): SULAMAYA OZGU rampa sekli -- yuksek genlikli ceyrek ile dusuk
          genlikli ceyregin 2025'teki gunluk sapma egrilerinin FARKI
          (ortak hava/klima mevsimselligi boylece dusulur), 7 gun duzlestirilmis,
          Haz-Tem tepesine gore normalize.
  f(i,d) = (a_eff(i) - medyan) * r(gun(d))

SOGUK ODAK: soguk trafonun a_i'si YOK; a_ilce ile doldurulur -- enjeksiyonun
tamami oradadir. Sicak trafo icin m30 zaten `h_gecenyil` (2025-04..07 ortalamasi)
tasiyor, yani orada bilgi kismen mevcut; rejim paylari raporlanir.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
sys.path.insert(0, BURA)

import z1_ortak as Z  # noqa: E402

from gridup.turkish import join_key  # noqa: E402

YAZ = ("2025-06-01", "2025-08-31")
KIS = ("2025-12-01", "2026-02-28")
REF = ("2025-01-01", "2025-03-31")  # "ozet penceresi kis" durumunun 2025 esdegeri
MIN_GUN = 20
MIN_TRAFO = 8
Q_HEDEF = 0.06

tr, te = Z.yukle()
msk = Z.maskeler(tr, te)
A6 = Z.taban()
for d in (tr, te):
    d["ilce_key"] = d.lokasyon.str.split(">").str[-1].str.strip().map(join_key)


# ------------------------------------------------------------------ a_i
def _pencere(d0, d1):
    return tr[(tr.tarih >= pd.Timestamp(d0)) & (tr.tarih <= pd.Timestamp(d1))]


gy = _pencere(*YAZ).groupby("tanim").L
gk = _pencere(*KIS).groupby("tanim").L
A = pd.DataFrame({"yaz": gy.mean(), "n_yaz": gy.size(), "kis": gk.mean(), "n_kis": gk.size()})
A = A[(A.n_yaz >= MIN_GUN) & (A.n_kis >= MIN_GUN)]
A["a"] = A.yaz - A.kis
ILCE_OF = tr.drop_duplicates("tanim").set_index("tanim").ilce_key
A["ilce_key"] = A.index.map(ILCE_OF)
print(f"olculmus trafo {len(A):,} / {tr.tanim.nunique():,}")

ilce_a = A.groupby("ilce_key").a.median()
ilce_n = A.groupby("ilce_key").a.size()
gecerli = ilce_n >= MIN_TRAFO
print("ILCE GENLIGI (yaz-kis, log):")
print(ilce_a[gecerli].sort_values(ascending=False).round(3).to_string())

# ------------------------------------------------------------------ arazi ortusu kestirimi
ar = pd.read_parquet(os.path.join(KOK, "data/external/arazi_ortusu_ilce.parquet"))
ar["ilce_key"] = ar["ilce_key"].astype(object)
KOL = ["tarim_orani", "yerlesim_orani", "su_orani", "agac_orani", "otlak_orani"]
ar = ar.set_index("ilce_key")[KOL].astype(float)
ort = ilce_a[gecerli]
X = ar.reindex(ort.index)[KOL]
ok = X.notna().all(axis=1)
Xm = np.column_stack([np.ones(ok.sum()), X[ok].to_numpy()])
ym = ort[ok].to_numpy()
lam = 1e-3 * np.eye(Xm.shape[1])
lam[0, 0] = 0.0
beta = np.linalg.solve(Xm.T @ Xm + lam, Xm.T @ ym)
tahmin_ilce = pd.Series(
    np.column_stack([np.ones(len(ar)), ar[KOL].to_numpy()]) @ beta, index=ar.index
)
kor = {k: float(np.corrcoef(X[ok][k], ym)[0, 1]) for k in KOL}
r2 = float(1 - ((Xm @ beta - ym) ** 2).sum() / ((ym - ym.mean()) ** 2).sum())
print(f"arazi ridge R^2={r2:.3f}  korelasyon={ {k: round(v, 3) for k, v in kor.items()} }")

# ------------------------------------------------------------------ r(gun): sulamaya ozgu rampa
ref = _pencere(*REF).groupby("tanim").L.mean()
gec25 = tr[(tr.tarih >= pd.Timestamp("2025-04-01")) & (tr.tarih <= pd.Timestamp("2025-08-31"))]
gec25 = gec25[gec25.tanim.isin(A.index) & gec25.tanim.isin(ref.index)].copy()
gec25["dev"] = gec25.L - gec25.tanim.map(ref)
q1, q3 = A.a.quantile([0.25, 0.75])
hi = set(A.index[A.a >= q3])
lo = set(A.index[A.a <= q1])
c_hi = gec25[gec25.tanim.isin(hi)].groupby("tarih").dev.mean()
c_lo = gec25[gec25.tanim.isin(lo)].groupby("tarih").dev.mean()
c = (c_hi - c_lo).rolling(7, center=True, min_periods=3).mean()
olcek = float(c[(c.index >= "2025-06-01") & (c.index <= "2025-07-31")].mean())
r25 = c / olcek
r25.index = r25.index.dayofyear
r25 = r25.groupby(level=0).mean()
print(
    f"rampa olcegi (Haz-Tem hi-lo farki) = {olcek:.3f}; r araligi {r25.min():.2f}..{r25.max():.2f}"
)
AY_OF = (pd.Timestamp("2025-01-01") + pd.to_timedelta(np.asarray(r25.index) - 1, "D")).month
RAMPA_AY = {str(int(m)): round(float(v), 3) for m, v in r25.groupby(AY_OF).mean().items()}
print("r(ay ort):", RAMPA_AY)

# ------------------------------------------------------------------ test satirlarina uygula
doy = te.tarih.dt.dayofyear.to_numpy()
rv = pd.Series(doy).map(r25)
rv = rv.interpolate(limit_direction="both").to_numpy()

# TRAFO duzeyi genlik DENENDI ve ELENDI: kurtoz 38,2 / en kotu %1 pay %51,7
# (kislik olu sulama pompasi tek basina +5 log genlik veriyor, yon kuyruklaniyor).
# Mekanizma zaten ILCE duzeyinde (havza); ilce genligi 47 seviyeli, duzgun.
a_ilce = te.ilce_key.map(ilce_a.where(gecerli))
a_hat = te.ilce_key.map(tahmin_ilce)
kaynak = pd.Series(np.where(a_ilce.notna(), "ilce", np.where(a_hat.notna(), "arazi", "genel")))
a_eff = a_ilce.fillna(a_hat).fillna(float(ilce_a[gecerli].median()))
merkez = float(np.median(a_eff))
# SOGUK ODAK: sicak trafo icin m30 zaten `h_gecenyil` tasiyor (2025-04..07
# ortalamasi) -- duzeltme ihtiyaci orada YARIYA yakin. Soguk ve kuyruk
# trafolarin elinde HIC yaz bilgisi yok; duzeltme tam agirlikla oraya gider.
REJIM_AGIRLIK = float(os.environ.get("CEKIRDEK_AGIRLIK", "0.6"))
wr = np.where(msk["cekirdek"], REJIM_AGIRLIK, 1.0)
f = (a_eff.to_numpy() - merkez) * rv * wr

Qham = float((f**2).mean())
s = float(np.sqrt(Q_HEDEF / Qham))
rap = Z.bitir(A6 + s * f, te, msk, A6, "tuketim_t1_sulama.csv", kirp=2.0)
rap.update(
    ham_Q=Qham,
    ham_rms=float(np.sqrt(Qham)),
    olcek=s,
    merkez=merkez,
    olculmus_trafo=int(len(A)),
    kaynak_payi={k: float(v) for k, v in kaynak.value_counts(normalize=True).items()},
    ilce_genligi={
        k: round(float(v), 3) for k, v in ilce_a[gecerli].sort_values(ascending=False).items()
    },
    arazi_R2=r2,
    arazi_korelasyon=kor,
    rampa_olcegi=olcek,
    rampa_ay=RAMPA_AY,
    cekirdek_agirlik=REJIM_AGIRLIK,
    ham_rejim_payi={
        k: float((f[msk[k]] ** 2).sum() / (f**2).sum()) for k in ("soguk", "kuyruk", "cekirdek")
    },
)
json.dump(rap, open(os.path.join(BURA, "t1_sulama.json"), "w"), indent=1)
print("yazildi t1_sulama.json")
