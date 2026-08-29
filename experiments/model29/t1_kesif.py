"""T1 -- turizm verisi kesfi: kapsam, ilce kirilimi, YoY hesaplanabilirligi.

Sadece OLCER, hicbir sey yazmaz (json disinda). Amac: modelde hangi turizm
turevinin gercekten TAKVIMDEN BAGIMSIZ bilgi tasidigini gormek.
"""

import json
import os
import sys

import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
sys.path.insert(0, os.path.join(KOK, "src"))
DIS = os.path.join(KOK, "data", "external")

from gridup.turkish import join_key  # noqa: E402

rap = {}

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
for d in (tr, te):
    p = d.lokasyon.str.split(">")
    d["il_key"] = p.str[0].str.strip().map(join_key)
    d["ilce_key"] = p.str[-1].str.strip().map(join_key)

rap["panel"] = dict(
    train_satir=len(tr),
    test_satir=len(te),
    train_tarih=[str(tr.tarih.min().date()), str(tr.tarih.max().date())],
    test_tarih=[str(te.tarih.min().date()), str(te.tarih.max().date())],
    iller=sorted(te.il_key.unique().tolist()),
    ilce_sayisi=int(te.ilce_key.nunique()),
    test_ay_dagilimi={
        str(k): int(v) for k, v in te.tarih.dt.month.value_counts().sort_index().items()
    },
)

t = pd.read_parquet(os.path.join(DIS, "turizm_aylik_il.parquet"))
t["il_key"] = t["il_key"].astype(object)
BIZ = sorted(te.il_key.unique().tolist())
tb = t[t.il_key.isin(BIZ)].copy()
rap["aylik"] = dict(
    satir=len(t),
    kapsam_degerleri=sorted(t.kapsam.unique().tolist()),
    rejimler=sorted(int(x) for x in t.kapsam_rejimi.unique()),
    donem=[f"{int(t.yil.min())}-{int(t.ay.min())}", f"{int(t.yil.max())}"],
    son_donem=str(tb.assign(d=tb.yil * 100 + tb.ay).d.max()),
    bizim_il_satir=len(tb),
    bizim_iller_var=sorted(tb.il_key.unique().tolist()),
)

ib = tb[tb.kapsam == "isletme_basit"].copy()
ib["d"] = ib.yil * 100 + ib.ay
piv = ib.pivot_table(index="d", columns="il_key", values="geceleme", aggfunc="sum")
son = piv[piv.index >= 202501]
rap["geceleme_2025_2026"] = {
    str(i): {c: (None if pd.isna(v) else round(float(v))) for c, v in r.items()}
    for i, r in son.iterrows()
}

# ---- YoY: 2026 ayi / 2025 ayni ayi (isletme_basit, geceleme)
yoy = {}
for ay in range(1, 7):
    a = piv.loc[202600 + ay] if (202600 + ay) in piv.index else None
    b = piv.loc[202500 + ay] if (202500 + ay) in piv.index else None
    if a is None or b is None:
        continue
    yoy[str(ay)] = {c: round(float(a[c] / b[c]), 4) for c in piv.columns if b[c] > 0}
rap["yoy_2026_2025"] = yoy

# YoY'nin ay-ortalamasi cikarilmis hali -> takvimle dik bilesen
ym = pd.DataFrame(yoy).T  # index=ay, cols=il
rap["yoy_ay_ortalamasi"] = {str(i): round(float(v), 4) for i, v in ym.mean(axis=1).items()}
dm = ym.sub(ym.mean(axis=1), axis=0)
rap["yoy_ay_demeaned"] = {
    str(i): {c: round(float(v), 4) for c, v in r.items()} for i, r in dm.iterrows()
}
rap["yoy_il_ortalamasi_demeaned"] = {c: round(float(v), 4) for c, v in dm.mean().items()}

# ---- ilce kirilimi
gc = pd.read_parquet(os.path.join(DIS, "turizm_geceleme.parquet"))
gc["ilce_key"] = gc["ilce_key"].astype(object)
gc["il_key"] = gc["il_key"].astype(object)
gcb = gc[gc.il_key.isin(BIZ)]
top = gcb.groupby(["il_key", "yil"]).geceleme.transform("sum")
gcb = gcb.assign(pay=gcb.geceleme / top)
rap["ilce"] = dict(
    yillar=sorted(int(x) for x in gc.yil.unique()),
    bizim_ilce_satir=len(gcb),
    bizim_ilce_sayisi=int(gcb.ilce_key.nunique()),
    panelde_ilce=int(te.ilce_key.nunique()),
    eslesen=int(len(set(gcb.ilce_key) & set(te.ilce_key))),
    eslesmeyen_turizm=sorted(set(gcb.ilce_key) - set(te.ilce_key)),
)
son_yil = int(gcb.yil.max())
g25 = gcb[gcb.yil == son_yil].sort_values("pay", ascending=False)
rap["ilce"]["son_yil"] = son_yil
rap["ilce"]["en_buyuk_10_pay"] = [
    dict(ilce=r.ilce_key, il=r.il_key, pay=round(float(r.pay), 4), geceleme=int(r.geceleme))
    for r in g25.head(10).itertuples()
]
kapsanan = te.ilce_key.isin(set(g25.ilce_key)).mean()
rap["ilce"]["test_satir_kapsami"] = round(float(kapsanan), 4)

# ---- ilce payi x il aylik profil: takvimle dik mi? il-ay profili ile ilce payi carpimi
# il aylik pay (yil ici pay), 2025 tam yil
p25 = ib[ib.yil == 2025].pivot_table(index="ay", columns="il_key", values="geceleme")
p25 = p25 / p25.sum()
rap["il_ay_payi_2025"] = {
    str(i): {c: round(float(v), 4) for c, v in r.items()} for i, r in p25.iterrows()
}

# test penceresindeki aylar icin ilce duzeyi yogunluk yayilimi
aylar = sorted(te.tarih.dt.month.unique().tolist())
pay_map = g25.set_index("ilce_key").pay
il_map = g25.set_index("ilce_key").il_key
yayilim = {}
for ay in aylar:
    v = {}
    for ilce in te.ilce_key.unique():
        if ilce in pay_map.index:
            il = il_map[ilce]
            if il in p25.columns:
                v[ilce] = float(pay_map[ilce] * p25.loc[ay, il])
    s = pd.Series(v)
    yayilim[str(ay)] = dict(
        n=int(len(s)),
        maks=round(float(s.max()), 5),
        medyan=round(float(s.median()), 6),
        maks_ilce=str(s.idxmax()),
        std=round(float(s.std()), 5),
    )
rap["ilce_ay_yogunluk_yayilimi"] = yayilim

# ---- TATIL x turizm: test penceresindeki tatil gunleri
TATIL = pd.to_datetime(
    [
        "2026-04-23",
        "2026-05-01",
        "2026-05-19",
        "2026-05-27",
        "2026-05-28",
        "2026-05-29",
        "2026-06-06",
        "2026-07-15",
    ]
)
tt = te.tarih.isin(set(TATIL))
rap["tatil"] = dict(
    test_tatil_satir=int(tt.sum()),
    test_tatil_pay=round(float(tt.mean()), 4),
    gunler=[str(d.date()) for d in TATIL if ((te.tarih == d).any())],
    egitimde_kurban=[
        str(d.date())
        for d in pd.to_datetime(["2025-06-06", "2025-06-07", "2025-06-08", "2025-06-09"])
        if (tr.tarih == d).any()
    ],
)

print(json.dumps(rap, indent=1, ensure_ascii=False))
json.dump(
    rap,
    open(os.path.join(BURA, "t1_kesif.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
