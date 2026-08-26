"""H4 -- BORU HATTI / BUTUNLUK DENETIMI.

Sekiz bagimsiz denetim, her biri icin GERCEK SAYI ve GECTI/KALDI.
Hicbir sey uretmez, hicbir dosyayi degistirmez -- sadece OLCER.

    uv run python scripts/h4_butunluk_denetimi.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]

GONDERIMLER = [
    "tuketim_v67_c1335_olay.csv",
    "tuketim_v55_gunolcek.csv",
    "tuketim_v50_nihai30.csv",
    "tuketim_v66_c1335.csv",
    "tuketim_v69_prob05.csv",
    "tuketim_v70_prob_soguk12.csv",
]

sonuclar: list[tuple[str, str, str]] = []


def kayit(ad: str, gecti: bool, detay: str) -> None:
    sonuclar.append((ad, "GECTI" if gecti else "KALDI", detay))
    print(f"[{'GECTI' if gecti else 'KALDI'}] {ad}: {detay}", flush=True)


def basli(s: str) -> None:
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------- veri yukle
basli("VERI YUKLENIYOR")
tr = pd.read_csv(KOK / "data" / "raw" / "train.csv")
te = pd.read_csv(KOK / "data" / "raw" / "test.csv")
ss = pd.read_csv(KOK / "data" / "raw" / "sample_submission.csv")
print(f"train {tr.shape}  test {te.shape}  sample {ss.shape}")
print(f"train kolon: {list(tr.columns)}")
print(f"test  kolon: {list(te.columns)}")
print(f"sample kolon: {list(ss.columns)}")

tr["tarih"] = pd.to_datetime(tr["tarih"])
te["tarih"] = pd.to_datetime(te["tarih"])


# ============================================================== 1. ID SIRASI
basli("1. ID SIRASI / SATIR SAYISI / NaN / NEGATIF")

ss_id = ss["id"].to_numpy()
n_ref = len(ss_id)
print(f"sample_submission satir = {n_ref}")
print(f"sample id dtype = {ss['id'].dtype}, ilk 3 = {ss_id[:3]}, son 3 = {ss_id[-3:]}")
print(f"sample id mukerrer = {int(pd.Series(ss_id).duplicated().sum())}")

# test.csv id'si sample ile ayni sirada mi?
te_id = te["id"].to_numpy()
te_ayni = len(te_id) == n_ref and bool(np.array_equal(te_id, ss_id))
kayit(
    "1a test.csv id sirasi == sample_submission",
    te_ayni,
    f"satir {len(te_id)} vs {n_ref}, birebir sira {te_ayni}",
)

hedef_kol = [c for c in ss.columns if c != "id"]
print(f"hedef kolon adi = {hedef_kol}")

for ad in GONDERIMLER:
    yol = KOK / "submissions" / ad
    if not yol.exists():
        kayit(f"1.{ad}", False, "DOSYA YOK")
        continue
    sub = pd.read_csv(yol)
    n = len(sub)
    kol_ok = list(sub.columns) == list(ss.columns)
    sid = sub["id"].to_numpy()
    sira_ok = n == n_ref and bool(np.array_equal(sid, ss_id))
    set_ok = n == n_ref and set(sid.tolist()) == set(ss_id.tolist())
    muk = int(pd.Series(sid).duplicated().sum())
    v = sub[sub.columns[-1]].to_numpy(dtype="float64")
    nan = int(np.isnan(v).sum())
    inf = int(np.isinf(v).sum())
    neg = int((v < 0).sum())
    sifir = int((v == 0).sum())
    vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
    gecti = n == n_ref and sira_ok and muk == 0 and nan == 0 and inf == 0 and neg == 0 and kol_ok
    kayit(
        f"1.{ad}",
        gecti,
        f"n={n} sira={sira_ok} set={set_ok} muk={muk} kolon={kol_ok} "
        f"NaN={nan} inf={inf} neg={neg} sifir={sifir} min={vmin:.6g} max={vmax:.6g}",
    )


# ================================================== 2. GUC TUTARLILIGI (BUTUNLUK)
basli("2. GUC TUTARLILIGI -- ayni tanim, farkli guc?")

tr_g = tr.groupby("tanim")["guc"].agg(["nunique", "min", "max", "first"])
te_g = te.groupby("tanim")["guc"].agg(["nunique", "min", "max", "first"])
print(
    f"train: tanim sayisi {len(tr_g)}, ic-tutarsiz (nunique>1) = {int((tr_g['nunique'] > 1).sum())}"
)
print(
    f"test : tanim sayisi {len(te_g)}, ic-tutarsiz (nunique>1) = {int((te_g['nunique'] > 1).sum())}"
)

ortak = tr_g.index.intersection(te_g.index)
print(f"ortak tanim = {len(ortak)}")
a = tr_g.loc[ortak, "first"].to_numpy(dtype="float64")
b = te_g.loc[ortak, "first"].to_numpy(dtype="float64")
farkli = ~np.isclose(a, b, rtol=0, atol=1e-9)
n_farkli = int(farkli.sum())
print(f"train/test guc FARKLI olan ortak tanim = {n_farkli} ({100 * n_farkli / len(ortak):.3f}%)")
if n_farkli:
    ornek = pd.DataFrame({"tanim": ortak[farkli], "guc_train": a[farkli], "guc_test": b[farkli]})
    ornek["oran"] = ornek["guc_test"] / ornek["guc_train"]
    print(ornek.head(15).to_string(index=False))
    print(f"oran ozeti: {ornek['oran'].describe().to_dict()}")
    # kac TEST SATIRI etkileniyor
    etkilenen = set(ornek["tanim"])
    n_satir = int(te["tanim"].isin(etkilenen).sum())
    print(f"etkilenen TEST satiri = {n_satir} ({100 * n_satir / len(te):.4f}%)")
kayit(
    "2 guc tutarliligi",
    n_farkli == 0
    and int((tr_g["nunique"] > 1).sum()) == 0
    and int((te_g["nunique"] > 1).sum()) == 0,
    f"ic-tutarsiz train={int((tr_g['nunique'] > 1).sum())} test={int((te_g['nunique'] > 1).sum())}, "
    f"ortakta farkli={n_farkli}",
)


# ============================================================ 3. MUKERRER ID
basli("3. MUKERRER ID")
muk_te = int(te["id"].duplicated().sum())
# (tanim,tarih) mukerrer?
muk_tt = int(te.duplicated(subset=["tanim", "tarih"]).sum())
muk_tr = int(tr.duplicated(subset=["tanim", "tarih"]).sum())
kayit(
    "3 mukerrer",
    muk_te == 0 and muk_tt == 0 and muk_tr == 0,
    f"test.id muk={muk_te}, test(tanim,tarih) muk={muk_tt}, train(tanim,tarih) muk={muk_tr}",
)
# e4_NIHAI_RECETE kontrolu
for aday in list((KOK / "reports").rglob("*NIHAI_RECETE*")) + list(
    (KOK / "submissions").rglob("*RECETE*")
):
    try:
        dd = pd.read_csv(aday)
        if "id" in dd.columns:
            print(f"  {aday.name}: n={len(dd)} mukerrer_id={int(dd['id'].duplicated().sum())}")
    except Exception as e:  # noqa: BLE001
        print(f"  {aday.name}: okunamadi {e}")


# ============================================================ 4. KIRPMA KAYBI
basli("4. KIRPMA KAYBI -- negatif tahmin var mi, kirpma MSLE'yi ne kadar bozuyor")

kirp_bilgi = []
for ad in GONDERIMLER:
    yol = KOK / "submissions" / ad
    if not yol.exists():
        continue
    v = pd.read_csv(yol)[hedef_kol[0]].to_numpy(dtype="float64")
    neg = int((v < 0).sum())
    kucuk = int((v < 1e-6).sum())
    kirp_bilgi.append((ad, neg, kucuk, float(v.min())))
    print(f"  {ad:34s} neg={neg:6d}  <1e-6={kucuk:6d}  min={v.min():.8g}")
kayit(
    "4 kirpma",
    all(x[1] == 0 for x in kirp_bilgi),
    "gonderimlerde negatif tahmin yok" if all(x[1] == 0 for x in kirp_bilgi) else "NEGATIF VAR",
)

# zincirin ic adimlarinda kirpma oluyor mu -- son_islem betiklerinde clip ara
import subprocess  # noqa: E402

r = subprocess.run(
    ["grep", "-rn", "-E", r"clip|maximum\(0|np\.where\(.*<\s*0", str(KOK / "scripts")],
    capture_output=True,
    encoding="utf-8",
)
satirlar = [s for s in (r.stdout or "").splitlines() if "son_islem" in s]
print(f"son_islem*.py icinde kirpma satiri = {len(satirlar)}")
for s in satirlar[:25]:
    print("   ", s.strip())


# ==================================================== 5. expm1/log1p GIDIS-DONUS
basli("5. expm1/log1p GIDIS-DONUSU -- zincir adimlari arasi kayma")


def oku(ad: str) -> np.ndarray | None:
    y = KOK / "submissions" / ad
    if not y.exists():
        return None
    return pd.read_csv(y)[hedef_kol[0]].to_numpy(dtype="float64")


v50 = oku("tuketim_v50_nihai30.csv")
v66 = oku("tuketim_v66_c1335.csv")
v67 = oku("tuketim_v67_c1335_olay.csv")
v55 = oku("tuketim_v55_gunolcek.csv")
v69 = oku("tuketim_v69_prob05.csv")
v70 = oku("tuketim_v70_prob_soguk12.csv")

# float64 gidis-donus kesinligi
for ad, v in [("v50", v50), ("v66", v66), ("v67", v67)]:
    if v is None:
        continue
    rt = np.expm1(np.log1p(v))
    hata = float(np.max(np.abs(rt - v)))
    goreli = float(np.max(np.abs(rt - v) / np.maximum(v, 1e-12)))
    print(f"  {ad}: max|expm1(log1p(x))-x| = {hata:.3e}  goreli {goreli:.3e}")

# CSV yazma hassasiyeti: dosyada kac anlamli hane var
ilk = (
    (KOK / "submissions" / "tuketim_v67_c1335_olay.csv")
    .read_text(encoding="utf-8")
    .splitlines()[1:6]
)
print("  v67 ham satirlar:", ilk)

# v66 -> v67 (olay gunu) beklenen delta: sadece ~4069 satir degismeli
if v66 is not None and v67 is not None:
    d = np.log1p(v67) - np.log1p(v66)
    degisen = int((np.abs(d) > 1e-9).sum())
    print(f"  v66->v67 degisen satir = {degisen} (beklenen ~4069)")
    print(
        f"    delta ozeti: min={d.min():+.5f} max={d.max():+.5f} "
        f"ort(degisen)={d[np.abs(d) > 1e-9].mean():+.5f}"
    )
    ok67 = 3500 <= degisen <= 4600
    kayit("5a v66->v67 olay gunu kapsami", ok67, f"degisen={degisen}, beklenen 4069")

# v50 -> v66 (gun ekseni c=1.335): TUM satirlar degismeli, gun bazli sabit
if v50 is not None and v66 is not None:
    d = np.log1p(v66) - np.log1p(v50)
    degisen = int((np.abs(d) > 1e-9).sum())
    print(f"  v50->v66 degisen satir = {degisen} / {len(d)}")
    print(f"    delta: min={d.min():+.5f} max={d.max():+.5f} ort={d.mean():+.6f} std={d.std():.5f}")
    # gun bazli mi? tarih basina std
    gd = pd.DataFrame({"tarih": te["tarih"].to_numpy(), "d": d})
    icgun = gd.groupby("tarih")["d"].std()
    print(
        f"    gun-ici std: ort={icgun.mean():.6f} max={icgun.max():.6f}  (gun ekseni ise ~0 olmali)"
    )
    print(f"    gunler arasi std = {gd.groupby('tarih')['d'].mean().std():.5f}")

# v67 -> v69 (delta_sicak=0.05) ve v67 -> v70 (delta_soguk=0.12)
tr_tanim = set(tr["tanim"].unique())
sicak_maske = te["tanim"].isin(tr_tanim).to_numpy()
p_sicak = float(sicak_maske.mean())
p_soguk = 1.0 - p_sicak
print(f"  p_sicak={p_sicak:.5f} p_soguk={p_soguk:.5f} (dokumante 0.77841 / 0.22159)")

if v67 is not None and v69 is not None:
    d = np.log1p(v69) - np.log1p(v67)
    print(
        f"  v67->v69: sicak ort delta={d[sicak_maske].mean():+.6f} "
        f"(std {d[sicak_maske].std():.2e}), soguk ort={d[~sicak_maske].mean():+.6f} "
        f"(std {d[~sicak_maske].std():.2e})"
    )
    ok69 = abs(d[sicak_maske].mean() - 0.05) < 1e-4 and abs(d[~sicak_maske].mean()) < 1e-6
    kayit(
        "5b v69 = v67 + 0.05 (yalniz sicak)",
        ok69,
        f"sicak {d[sicak_maske].mean():+.6f}, soguk {d[~sicak_maske].mean():+.2e}",
    )

if v67 is not None and v70 is not None:
    d = np.log1p(v70) - np.log1p(v67)
    print(
        f"  v67->v70: soguk ort delta={d[~sicak_maske].mean():+.6f} "
        f"(std {d[~sicak_maske].std():.2e}), sicak ort={d[sicak_maske].mean():+.6f} "
        f"(std {d[sicak_maske].std():.2e})"
    )
    ok70 = abs(d[~sicak_maske].mean() - 0.12) < 1e-4 and abs(d[sicak_maske].mean()) < 1e-6
    kayit(
        "5c v70 = v67 + 0.12 (yalniz soguk)",
        ok70,
        f"soguk {d[~sicak_maske].mean():+.6f}, sicak {d[sicak_maske].mean():+.2e}",
    )

# CSV YUVARLAMA KAYBI -- yazilan hane sayisi dMSE'yi ne kadar bozar?
if v67 is not None:
    ham = (KOK / "submissions" / "tuketim_v67_c1335_olay.csv").read_text(encoding="utf-8")
    ikinci = ham.splitlines()[1].split(",")[-1]
    print(f"  v67 ilk deger metni = '{ikinci}' (uzunluk {len(ikinci)})")
    # yuvarlama simulasyonu: 6 haneye yuvarlarsak log-uzayda ne kayar
    for hane in (4, 6, 8):
        yv = np.round(v67, hane)
        dd = np.log1p(yv) - np.log1p(v67)
        print(f"    {hane} ondalik -> max|dlog|={np.abs(dd).max():.3e}  dMSE~{(dd**2).mean():.3e}")


# ================================================= 6. SON ISLEM ZINCIR SIRASI
basli("6. SON ISLEM ZINCIRININ SIRASI")
for ad in ["son_islem_gunolcek.py", "son_islem_olay.py", "son_islem_seviye.py"]:
    p = KOK / "scripts" / ad
    if not p.exists():
        print(f"  {ad}: YOK")
        continue
    t = p.read_text(encoding="utf-8")
    ipuclari = [
        s.strip()
        for s in t.splitlines()
        if ("SONRA" in s or "ONCE" in s or "sonra kosul" in s.lower())
    ]
    print(f"  --- {ad} ---")
    for s in ipuclari[:8]:
        print(f"      {s}")


# ============================================================ 7. TARIH KAPSAMASI
basli("7. TARIH KAPSAMASI")
te_gun = te.groupby("tanim")["tarih"].agg(["nunique", "min", "max"])
n_gun_bekl = int(te["tarih"].nunique())
print(f"test essiz gun = {n_gun_bekl} ({te['tarih'].min().date()} .. {te['tarih'].max().date()})")
dag = te_gun["nunique"].value_counts().sort_index()
print("test trafo basina gun sayisi dagilimi (ilk/son 10):")
print(dag.head(10).to_string())
print("  ...")
print(dag.tail(10).to_string())
tam = int((te_gun["nunique"] == n_gun_bekl).sum())
print(f"TAM kapsamali trafo = {tam} / {len(te_gun)} ({100 * tam / len(te_gun):.2f}%)")
kismi = te_gun[te_gun["nunique"] < n_gun_bekl]
print(
    f"KISMI trafo = {len(kismi)}, toplam satir = {int(kismi['nunique'].sum())} "
    f"({100 * kismi['nunique'].sum() / len(te):.3f}% test satiri)"
)
if len(kismi):
    kismi2 = kismi.copy()
    kismi2["sicak"] = kismi2.index.isin(tr_tanim)
    print(
        f"  kismilerin {int(kismi2['sicak'].sum())} tanesi SICAK, "
        f"{int((~kismi2['sicak']).sum())} tanesi SOGUK"
    )
    print("  baslangic tarihi dagilimi (ilk 10):")
    print(kismi2["min"].value_counts().sort_index().head(10).to_string())
    print("  bitis tarihi dagilimi (son 10):")
    print(kismi2["max"].value_counts().sort_index().tail(10).to_string())
    # bosluk var mi (baslangic-bitis arasi eksik gun)?
    kismi2["beklenen"] = (kismi2["max"] - kismi2["min"]).dt.days + 1
    bosluklu = int((kismi2["nunique"] < kismi2["beklenen"]).sum())
    print(f"  IC BOSLUKLU (kesintili) trafo = {bosluklu}")
kayit(
    "7 tarih kapsamasi",
    True,
    f"{tam}/{len(te_gun)} tam, {len(kismi)} kismi ({100 * kismi['nunique'].sum() / len(te):.3f}% satir)",
)


# ================================================================ 8. LOKASYON
basli("8. LOKASYON NORMALIZASYONU")


def norm(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip()
    # Turkce-duyarli buyuk/kucuk: once I/i esle
    x = x.str.replace("İ", "i", regex=False).str.replace("I", "i", regex=False)
    x = x.str.replace("ı", "i", regex=False)
    x = x.str.lower()
    x = x.str.replace(r"\s+", " ", regex=True)
    return x


tr_lok = tr["lokasyon"].astype(str)
te_lok = te["lokasyon"].astype(str)
u_tr, u_te = set(tr_lok.unique()), set(te_lok.unique())
print(f"train essiz lokasyon = {len(u_tr)}")
print(f"test  essiz lokasyon = {len(u_te)}")
print(f"birlesim = {len(u_tr | u_te)}, kesisim = {len(u_tr & u_te)}")
print(f"YALNIZ test'te olan = {len(u_te - u_tr)}  ornek: {sorted(u_te - u_tr)[:8]}")
print(f"YALNIZ train'de olan = {len(u_tr - u_te)}  ornek: {sorted(u_tr - u_te)[:8]}")

n_tr, n_te = norm(tr_lok), norm(te_lok)
nu_tr, nu_te = set(n_tr.unique()), set(n_te.unique())
print(f"NORMALIZE: train {len(u_tr)} -> {len(nu_tr)}, test {len(u_te)} -> {len(nu_te)}")
print(f"NORMALIZE birlesim {len(u_tr | u_te)} -> {len(nu_tr | nu_te)}")
print(f"NORMALIZE yalniz-test {len(u_te - u_tr)} -> {len(nu_te - nu_tr)}")

# hangi cift cakisti?
from collections import defaultdict  # noqa: E402

grup = defaultdict(set)
for h in sorted(u_tr | u_te):
    grup[norm(pd.Series([h])).iloc[0]].add(h)
cakisan = {k: v for k, v in grup.items() if len(v) > 1}
print(f"normalize edilince BIRLESEN grup sayisi = {len(cakisan)}")
for k, v in list(cakisan.items())[:20]:
    print(f"   '{k}' <- {sorted(v)}")

# whitespace / bos / NaN
print(
    f"train lokasyon NaN = {int(tr['lokasyon'].isna().sum())}, "
    f"test = {int(te['lokasyon'].isna().sum())}"
)
bos_tr = int((tr_lok.str.strip() != tr_lok).sum())
bos_te = int((te_lok.str.strip() != te_lok).sum())
print(f"bastaki/sondaki bosluk iceren satir: train {bos_tr}, test {bos_te}")

n_cakisan_satir = int(te_lok.isin({h for v in cakisan.values() for h in v}).sum())
print(
    f"cakisan lokasyona ait TEST satiri = {n_cakisan_satir} "
    f"({100 * n_cakisan_satir / len(te):.4f}%)"
)
kayit(
    "8 lokasyon normalizasyonu",
    len(cakisan) == 0 and bos_tr == 0 and bos_te == 0,
    f"{len(u_tr | u_te)} essiz -> {len(nu_tr | nu_te)} normalize; birlesen grup={len(cakisan)}; "
    f"etkilenen test satiri={n_cakisan_satir}",
)


# =================================================================== OZET
basli("OZET")
for ad, h, d in sonuclar:
    print(f"{h:6s}  {ad}")
kaldi = [s for s in sonuclar if s[1] == "KALDI"]
print(f"\nTOPLAM: {len(sonuclar)} denetim, {len(kaldi)} KALDI")
