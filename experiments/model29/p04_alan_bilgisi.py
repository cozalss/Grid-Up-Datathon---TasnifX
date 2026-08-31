"""p04: ALAN BILGISI (bayram / sulama / turizm / CDD) -- yaz25'te SIZINTISIZ olcum.

SIZINTI KONTROLU (p03 ile ayni cerceve)
  * yaz25 hedefi (tuketim) yalnizca ARTIK HESABINDA kullanilir; hicbir
    duzeltme katsayisi yaz25'ten kestirilmez.
  * Her duzeltmenin katsayisi SADECE guz25 + kis26 artiklarindan (blok disi)
    en kucuk kareyle bulunur, sonra yaz25'e UYGULANIR ve orada olculur.
  * Bayram katsayisi icin kullanilan gunler: Ramazan Bayrami 2026-03-19..22
    (kis26 icinde), 30 Agustos / 29 Ekim 2025 (guz25), 1 Ocak 2026 (kis26).
    Kurban 2025-06-05..09 SADECE degerlendirmede kullanilir.
  * Tahminler onbellekten okunur; hicbir model yeniden egitilmez.

OLCUT: agirlikli MSE (soguk satirlarin agirligi HEDEF_SOGUK=0.222'ye
cekilir -- gercek testin soguk payi). Kazanc = sqrt(m0) - sqrt(m1).
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
HEDEF_SOGUK = 0.222

KOL = [
    "tanim", "tarih", "tuketim", "ilce_key", "il_key", "soguk_mu", "_blok",
    "ufuk_gun", "guc", "tarim_orani", "yerlesim_orani", "nufus",
    "cdd18", "cdd22", "cdd24", "sicaklik_ort", "sicaklik_max",
    "et0_toplam", "toprak_nem_ort", "vpd_ort", "gunes_radyasyon",
    "tatil_mi", "tatil_agirligi", "tatil_kod", "tatil_mesafe",
    "ramazan_ayi", "ulusal_gunluk", "trafo_basina_nufus",
]

e = pd.read_parquet(os.path.join(DN, "egitim.parquet"), columns=KOL)
print("yuklendi", e.shape, flush=True)


def blok_artik(ad):
    """Bir blogun (yaz25/guz25/kis26) tahminini onbellekten kurar, artigi dondurur."""
    blk = e[e._blok == ad]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t in (1000, 1001, 1002)
        for aa in ("cat", "xgb", "lgbm")
        if os.path.exists(os.path.join(AO, f"{ad}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{ad}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    bf["p"] = pb
    bf["L"] = np.log1p(bf.tuketim.values.astype(np.float64))
    bf["r"] = bf.L - bf.p
    s = bf.soguk_mu.values.astype(np.float64)
    w = np.where(s == 1, HEDEF_SOGUK / s.mean(), (1 - HEDEF_SOGUK) / (1 - s.mean()))
    bf["w"] = w / w.mean()
    print(f"  {ad}: n={len(bf)} aile={len(P)} agirlikli_MSE={float((bf.w*bf.r**2).mean()):.6f}",
          flush=True)
    return bf.reset_index(drop=True)


B = {a: blok_artik(a) for a in ("yaz25", "guz25", "kis26")}
Y = B["yaz25"]
DIS = pd.concat([B["guz25"], B["kis26"]], ignore_index=True)
m0 = float((Y.w * Y.r**2).mean())
print(f"\nYAZ25 TABAN agirlikli MSE={m0:.6f}  RMSLE={np.sqrt(m0):.6f}\n", flush=True)

R = {"taban_yaz25_mse": m0, "taban_yaz25_rmsle": float(np.sqrt(m0))}


def kazanc(duz, ad, kirp=None):
    """duz: yaz25 satirlarina eklenecek log-uzayi duzeltmesi. Kazanc = RMSLE dususu."""
    d = np.asarray(duz, dtype=np.float64)
    d = np.where(np.isfinite(d), d, 0.0)
    if kirp is not None:
        d = np.clip(d, -kirp, kirp)
    m1 = float((Y.w * (Y.r - d) ** 2).mean())
    g = float(np.sqrt(m0) - np.sqrt(m1))
    print(f"  {ad:44s} kazanc={g:+.6f}  (|d|ort={np.abs(d).mean():.5f}, "
          f"dokunan={float((d != 0).mean()):.3f})", flush=True)
    return g


# ===================== 1. TANI: bayram gunleri yaz25 artiginda =====================
KURBAN25 = pd.to_datetime(["2025-06-05", "2025-06-06", "2025-06-07", "2025-06-08",
                           "2025-06-09"])
TEKGUN25 = pd.to_datetime(["2025-04-23", "2025-05-01", "2025-05-19", "2025-07-15"])

Y["tarih"] = pd.to_datetime(Y.tarih)
DIS["tarih"] = pd.to_datetime(DIS.tarih)
tot = float((Y.w * Y.r**2).sum())
tani = {}
for ad, gunler in (("kurban25", KURBAN25), ("tekgun25", TEKGUN25)):
    m = Y.tarih.isin(gunler)
    tani[ad] = {
        "satir_payi": float(m.mean()),
        "sse_payi": float((Y.w[m] * Y.r[m] ** 2).sum() / tot),
        "ort_artik": float(np.average(Y.r[m], weights=Y.w[m])),
        "ort_artik_disi": float(np.average(Y.r[~m], weights=Y.w[~m])),
    }
    print(f"{ad}: satir%={tani[ad]['satir_payi']:.4f} SSE%={tani[ad]['sse_payi']:.4f} "
          f"ort_artik={tani[ad]['ort_artik']:+.4f} (digerleri {tani[ad]['ort_artik_disi']:+.4f})",
          flush=True)

gunluk = Y.groupby("tarih").apply(
    lambda g: pd.Series({"ort": np.average(g.r, weights=g.w), "n": len(g)}),
    include_groups=False)
print("\nKurban civari gunluk agirlikli ortalama artik:", flush=True)
print(gunluk.loc["2025-05-30":"2025-06-15"].round(4).to_string(), flush=True)
R["gunluk_kurban_civari"] = {str(k.date()): float(v) for k, v in
                             gunluk.loc["2025-05-30":"2025-06-15"]["ort"].items()}
R["tani_bayram"] = tani

print("\nAylik agirlikli ortalama artik (yaz25):", flush=True)
ay = Y.groupby(Y.tarih.dt.month).apply(
    lambda g: float(np.average(g.r, weights=g.w)), include_groups=False)
print(ay.round(4).to_string(), flush=True)
R["aylik_yanlilik_yaz25"] = {int(k): float(v) for k, v in ay.items()}

# ===================== 2. BAYRAM: blok-disi katsayi -> yaz25 =====================
# Blok disi (guz25+kis26) coklu-gun dini/resmi bayramlar
DIS_BAYRAM = {
    "ramazan26": pd.to_datetime(["2026-03-19", "2026-03-20", "2026-03-21", "2026-03-22"]),
    "yilbasi26": pd.to_datetime(["2025-12-31", "2026-01-01"]),
    "z30agu25": pd.to_datetime(["2025-08-30"]),
    "z29eki25": pd.to_datetime(["2025-10-28", "2025-10-29"]),
}
PENCERE, MIN_TABAN = 10, 2


def sapma_tablosu(df, gunler, tum_tatil):
    """Trafo basina: tatil gunu artigi eksi +-10 gunde AYNI HAFTA GUNU tabani."""
    par = []
    for g in gunler:
        pen = df[(df.tarih >= g - pd.Timedelta(days=PENCERE))
                 & (df.tarih <= g + pd.Timedelta(days=PENCERE))
                 & (df.tarih.dt.dayofweek == g.dayofweek)
                 & (~df.tarih.isin(tum_tatil))]
        tb = pen.groupby("tanim").r.agg(["mean", "size"])
        tb = tb[tb["size"] >= MIN_TABAN]["mean"]
        gun = df[df.tarih == g].groupby("tanim").r.mean()
        ort = gun.index.intersection(tb.index)
        if len(ort):
            par.append((gun.loc[ort] - tb.loc[ort]).rename(str(g.date())))
    if not par:
        return pd.Series(dtype=float)
    return pd.concat(par, axis=1).mean(axis=1)


TUM_TATIL_DIS = pd.to_datetime(sorted(set().union(*[set(v) for v in DIS_BAYRAM.values()])
                                      | set(pd.to_datetime(["2025-09-01"]))))
ILCE_OF = e.drop_duplicates("tanim").set_index("tanim").ilce_key

bayram_kats = {}
for ad, gunler in DIS_BAYRAM.items():
    sp = sapma_tablosu(DIS, gunler, TUM_TATIL_DIS)
    if not len(sp):
        continue
    k = sp.groupby(sp.index.map(ILCE_OF)).median()
    k = k[sp.groupby(sp.index.map(ILCE_OF)).size() >= 8]
    k = k - k.median()
    bayram_kats[ad] = k
    print(f"\n{ad}: {len(sp)} trafo, {len(k)} ilce, global_sapma={sp.median():+.4f}, "
          f"ilce yayilimi std={k.std():.4f}", flush=True)

# 2a. GLOBAL bayram kaymasi (blok disi olculur, Kurban'a uygulanir)
kurban_mask = Y.tarih.isin(KURBAN25).values
tekgun_mask = Y.tarih.isin(TEKGUN25).values
glob_dini = float(np.median(sapma_tablosu(DIS, DIS_BAYRAM["ramazan26"], TUM_TATIL_DIS)))
glob_tek = float(np.median(sapma_tablosu(
    DIS, pd.to_datetime(list(DIS_BAYRAM["z30agu25"]) + list(DIS_BAYRAM["z29eki25"])),
    TUM_TATIL_DIS)))
print(f"\nBLOK DISI global kayma: dini_bayram={glob_dini:+.4f} tekgun={glob_tek:+.4f}",
      flush=True)
R["blok_disi_global_kayma"] = {"dini": glob_dini, "tekgun": glob_tek}

d = np.zeros(len(Y))
d[kurban_mask] = glob_dini
R["fikir_bayram_global_kurban"] = kazanc(d, "F1 bayram GLOBAL kayma (Kurban)")
d = np.zeros(len(Y))
d[tekgun_mask] = glob_tek
R["fikir_bayram_global_tekgun"] = kazanc(d, "F2 bayram GLOBAL kayma (tek gun)")

# 2b. ILCE x bayram katsayisi (blok disi Ramazan26'dan)
if "ramazan26" in bayram_kats:
    kats = bayram_kats["ramazan26"]
    ky = Y.ilce_key.map(kats).fillna(0.0).values
    d = np.zeros(len(Y))
    d[kurban_mask] = (glob_dini + ky)[kurban_mask]
    R["fikir_bayram_ilce_kurban"] = kazanc(d, "F3 bayram ILCE katsayisi (Ramazan26->Kurban)")
    # katsayinin YONU dogru mu? yaz25 Kurban'da olculen ilce sapmasiyla korelasyon
    sp_y = sapma_tablosu(Y, KURBAN25, pd.to_datetime(list(KURBAN25) + list(TEKGUN25)))
    ky_gercek = sp_y.groupby(sp_y.index.map(ILCE_OF)).median()
    ort = kats.index.intersection(ky_gercek.index)
    kor = float(np.corrcoef(kats.loc[ort], ky_gercek.loc[ort])[0, 1])
    print(f"  ILCE katsayisi kor(Ramazan26, Kurban25) = {kor:+.4f}  n={len(ort)}", flush=True)
    R["kor_ilce_ramazan26_kurban25"] = kor
    R["ilce_kurban25_gercek"] = {k: round(float(v), 4) for k, v in ky_gercek.items()}

# ===================== 3. SULAMA / TURIZM / CDD: blok-disi dogrusal ayar ==========
def dis_katsayi(x_dis, x_yaz):
    """Blok disinda artigi x uzerine regresyon; katsayiyi yaz25'e tasi."""
    xd = np.asarray(x_dis, dtype=np.float64)
    ok = np.isfinite(xd)
    xd = xd[ok] - xd[ok].mean()
    wd, rd = DIS.w.values[ok], DIS.r.values[ok]
    den = float((wd * xd * xd).sum())
    if den <= 0:
        return 0.0, np.zeros(len(x_yaz))
    b = float((wd * xd * rd).sum() / den)
    xy = np.asarray(x_yaz, dtype=np.float64)
    xy = np.where(np.isfinite(xy), xy, np.nanmean(xy))
    return b, b * (xy - xy.mean())


KIYI = {"cesme", "karaburun", "urla", "seferihisar", "foca", "dikili", "selcuk",
        "guzelbahce", "menderes", "aliaga"}
SULAMA = {"saruhanli", "salihli", "alasehir", "turgutlu", "akhisar", "kinik",
          "bergama", "menemen", "torbali", "odemis", "tire", "bayindir",
          "sarigol", "kirkagac", "golmarmara", "kula", "gordes", "selendi",
          "demirci", "soma", "beydag", "kiraz", "kemalpasa"}

adaylar = {}
for nm in ("cdd18", "cdd22", "cdd24", "sicaklik_ort", "et0_toplam",
           "toprak_nem_ort", "vpd_ort", "gunes_radyasyon"):
    adaylar[f"H_{nm}"] = (DIS[nm].values, Y[nm].values)

ta_d, ta_y = DIS.tarim_orani.fillna(0).values, Y.tarim_orani.fillna(0).values
for nm in ("et0_toplam", "toprak_nem_ort", "cdd24", "vpd_ort"):
    adaylar[f"S_tarim_x_{nm}"] = (ta_d * DIS[nm].values, ta_y * Y[nm].values)

ki_d = DIS.ilce_key.isin(KIYI).astype(float).values
ki_y = Y.ilce_key.isin(KIYI).astype(float).values
su_d = DIS.ilce_key.isin(SULAMA).astype(float).values
su_y = Y.ilce_key.isin(SULAMA).astype(float).values
adaylar["T_kiyi_x_cdd24"] = (ki_d * DIS.cdd24.values, ki_y * Y.cdd24.values)
adaylar["T_kiyi_x_gunuzunlugu"] = (ki_d * DIS.sicaklik_ort.values, ki_y * Y.sicaklik_ort.values)
adaylar["S_sulama_x_et0"] = (su_d * DIS.et0_toplam.values, su_y * Y.et0_toplam.values)
adaylar["S_sulama_x_topraknem"] = (su_d * DIS.toprak_nem_ort.values,
                                   su_y * Y.toprak_nem_ort.values)

print("\n=== DOGRUSAL ADAYLAR (katsayi blok DISINDA, olcum yaz25'te) ===", flush=True)
R["dogrusal"] = {}
for ad, (xd, xy) in adaylar.items():
    b, d = dis_katsayi(xd, xy)
    R["dogrusal"][ad] = {"beta_dis": b, "kazanc_yaz25": kazanc(d, ad, kirp=1.0)}

# ===================== 4. Aylik yanlilik: blok disi tasinabilir mi? ==============
print("\n=== AYLIK YANLILIK TASINABILIRLIGI ===", flush=True)
ay_dis = DIS.groupby(DIS.tarih.dt.month).apply(
    lambda g: float(np.average(g.r, weights=g.w)), include_groups=False)
print("blok disi aylik yanlilik:", ay_dis.round(4).to_dict(), flush=True)
R["aylik_yanlilik_dis"] = {int(k): float(v) for k, v in ay_dis.items()}

# Ufuk (kesimden uzaklik) uzerinden tasima: blok disi ufuk->artik egimi
b_uf, d_uf = dis_katsayi(DIS.ufuk_gun.values.astype(float), Y.ufuk_gun.values.astype(float))
print(f"beta(ufuk_gun) blok disi = {b_uf:+.6f}", flush=True)
R["fikir_ufuk_egimi"] = {"beta": b_uf, "kazanc": kazanc(d_uf, "F4 ufuk_gun dogrusal ayar")}

with open(os.path.join(BURA, "p04_alan_bilgisi.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1)
print("\nyazildi: p04_alan_bilgisi.json", flush=True)
