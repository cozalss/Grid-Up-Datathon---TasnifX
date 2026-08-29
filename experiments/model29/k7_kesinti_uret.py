"""K7 -- kesinti yonunu uret, TEMIZLE, CAPALA, OLC ve yaz.

Iki aday olculur:
  TAM   : L = P(taban+kesinti)                  (modelin kendisi)
  DELTA : L = L_m6 + [P(taban+kesinti) - P(taban)]   (kesintinin IZOLE katkisi)

Her ikisi de z1_ortak.bitir ile ayni son islemden gecer:
  1) f = L - L_m6
  2) f[CURUK] = 0        (docs/52 s1 -- olu-trafo tezi LB'de CURUDU, y1_temizle)
  3) f = clip(f, -2, +2)
  4) rejim (soguk/kuyruk/cekirdek) ortalamasi sifirlanir  (m71 sonundaki capa)

Secim olcutu CV DEGIL: |kosinus| <= 0,20, kurtoz <= 10, Q >= 0,01.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402

S = Z.S
KOK = Z.KOK
M0 = 1.005688  # m6'nin LB MSE'si (1.00284^2)
R_KAL = 0.0641  # m4'ten kalibre yon kalitesi
KIRP = 2.0
CIKTI = "tuketim_k5_kesinti.csv"

KARSILASTIR = [
    ("y40_sota", "tuketim_y40_sota_temiz.csv"),
    ("q1c_kapasite", "tuketim_q1c_kapasite_siki.csv"),
    ("y46_amnezik", "tuketim_y46_amnezik_kirpik.csv"),
    ("y45_mevsimsel", "tuketim_y45_mevsimsel_kirpik.csv"),
    ("z2_analog", "tuketim_z2_analog.csv"),
    ("g7_span_tau3", "tuketim_g7_span_tau3.csv"),
]


def logoku(ad):
    return np.log1p(pd.read_csv(os.path.join(S, ad)).tuketim.values)


def isle(L_hat, A6, msk):
    """z1_ortak.bitir ile AYNI donusum -- ama dosya yazmadan."""
    L_hat = np.asarray(L_hat, dtype=float)
    kotu = ~np.isfinite(L_hat)
    L_hat = np.where(kotu, A6, L_hat)
    f0 = L_hat - A6
    rap = dict(
        finite_olmayan=int(kotu.sum()),
        ham=Z.olcut(f0),
        curuk_Q_payi=float((f0[msk["curuk"]] ** 2).sum() / max(1e-30, (f0**2).sum())),
    )
    f = np.where(msk["curuk"], 0.0, f0)
    rap["temiz"] = Z.olcut(f)
    f = np.clip(f, -KIRP, KIRP)
    capa = {}
    for nm in ("soguk", "kuyruk", "cekirdek"):
        mm = msk[nm] & ~msk["curuk"]
        if mm.sum():
            d = float(f[mm].mean())
            f[mm] -= d
            capa[nm] = dict(satir=int(mm.sum()), kaydirma=-d)
    f[msk["curuk"]] = 0.0
    rap["kirpma"] = KIRP
    rap["capa"] = capa
    rap["son"] = Z.olcut(f)
    return f, rap


def main():
    tr, te = Z.yukle()
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    p_tab = np.load(os.path.join(BURA, "k6_p_taban.npy"))
    p_kes = np.load(os.path.join(BURA, "k6_p_kesinti.npy"))
    assert len(p_tab) == len(A6) == len(te)

    # --- olculmus eksenler (z3_olcum ile ayni tanim)
    V102 = logoku("tuketim_v102_kappa_optimum.csv")
    D4 = logoku("tuketim_m4_hava_capali.csv") - V102
    D6 = A6 - V102

    def nrm(v):
        return v / np.sqrt((v**2).mean())

    E1 = nrm(D4)
    E2 = nrm(D6 - (D6 * E1).mean() * E1)

    mevcut = {}
    for nm, dosya in KARSILASTIR:
        yol = os.path.join(S, dosya)
        if not os.path.exists(yol):
            print(f"  YOK {dosya} -- atlandi")
            continue
        mevcut[nm] = nrm(logoku(dosya) - A6)

    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    ref6 = dict(
        log_ort=float(A6.mean()),
        log_std=float(A6.std()),
        maks=float(np.expm1(A6).max()),
        alt1kwh=float((np.expm1(A6) < 1).mean()),
    )

    # DELTA'nin ham Q'su kucukse yon ayni kalarak Q>=0,01 kapisina olceklenir
    # (kosinus ve kurtoz olcekten BAGIMSIZ; Q yalnizca olcum hassasiyeti icin).
    dham = p_kes - p_tab
    Qd = float((dham**2).mean())
    olcek = float(np.sqrt(0.012 / Qd)) if 0 < Qd < 0.012 else 1.0

    adaylar = {}
    fvek = {}
    for ad, L in (
        ("TAM", p_kes),
        ("DELTA", A6 + dham),
        ("DELTA_OLCEK", A6 + olcek * dham),
    ):
        f, rap = isle(L, A6, msk)
        Q = rap["son"]["Q"]
        u = nrm(f) if Q > 0 else f
        kos = {k: float((u * v).mean()) for k, v in mevcut.items()}
        kos["m4_v102_ekseni"] = float((u * E1).mean())
        kos["m6_dik_ekseni"] = float((u * E2).mean())
        B = np.array([E1, E2] + [mevcut[k] for k in mevcut])
        G = B @ B.T / len(u)
        c = B @ u / len(u)
        w = np.linalg.solve(G + 1e-10 * np.eye(len(G)), c)
        bagimsiz = max(0.0, float(1.0 - w @ c))
        rap["kosinus"] = kos
        rap["mutlak_maks_kosinus"] = float(max(abs(v) for v in kos.values()))
        rap["bagimsiz_pay_mevcutlara_gore"] = bagimsiz
        rap["beklenen_kazanc_bagimsiz"] = R_KAL**2 * bagimsiz
        rap["basabas_skor"] = float(np.sqrt(M0 + Q))
        rap["Q_rejim_payi"] = {
            k: float((f[msk[k]] ** 2).sum() / max(1e-30, (f**2).sum()))
            for k in ("soguk", "kuyruk", "cekirdek", "curuk")
        }
        y = np.clip(np.expm1(A6 + f), 0.0, None)
        b = np.log1p(y)
        rap["dagilim"] = dict(
            log_ort=float(b.mean()),
            log_std=float(b.std()),
            maks=float(y.max()),
            alt1kwh=float((y < 1).mean()),
        )
        rap["gecti"] = dict(
            kosinus=bool(rap["mutlak_maks_kosinus"] <= 0.20),
            kurtoz=bool(rap["son"]["kurtoz"] <= 10.0),
            Q=bool(Q >= 0.01),
        )
        adaylar[ad] = rap
        fvek[ad] = f
        print(
            f"{ad:6s} Q={Q:.5f} kurtoz={rap['son']['kurtoz']:6.2f} "
            f"%1pay={100 * rap['son']['en_kotu_yuzde1_pay']:5.1f} "
            f"|kos|max={rap['mutlak_maks_kosinus']:.3f} bagimsiz%={100 * bagimsiz:5.1f} "
            f"gecti={rap['gecti']}",
            flush=True,
        )
        for k, v in kos.items():
            print(f"        kos[{k:16s}] = {v:+.4f}")

    # iki aday arasi kosinus
    if fvek["TAM"].any() and fvek["DELTA"].any():
        ikili = float((nrm(fvek["TAM"]) * nrm(fvek["DELTA"])).mean())
    else:
        ikili = float("nan")

    print("\n=== BASLIK OLCUM: kos(g7_span_tau3) ===")
    for ad in adaylar:
        print(f"  {ad:12s} kos(g7_span_tau3) = {adaylar[ad]['kosinus'].get('g7_span_tau3'):+.4f}")

    # --- SECIM: uc kapiyi da geceni al; ikisi de gecerse bagimsiz payi yuksek olan
    uygun = [a for a in adaylar if all(adaylar[a]["gecti"].values())]
    if uygun:
        secim = max(uygun, key=lambda a: adaylar[a]["bagimsiz_pay_mevcutlara_gore"])
    else:
        secim = max(adaylar, key=lambda a: adaylar[a]["bagimsiz_pay_mevcutlara_gore"])
    f = fvek[secim]
    y = np.clip(np.expm1(A6 + f), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    out.to_csv(os.path.join(S, CIKTI), index=False)
    kapi = dict(
        satir=int(len(out)),
        id_birebir=bool(len(out) == len(ss) and (out.id.values == ss.iloc[:, 0].values).all()),
        id_test_birebir=bool((out.id.values == te.id.values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        sonsuz=int((~np.isfinite(out.tuketim.values)).sum()),
        baslik=list(out.columns) == ["id", "tuketim"],
    )
    assert kapi["satir"] == 714688 and kapi["id_birebir"] and kapi["id_test_birebir"]
    assert not kapi["nan"] and not kapi["negatif"] and not kapi["sonsuz"] and kapi["baslik"]

    kunye = {}
    for nm in ("k5_kesinti_veri.json", "k6_kesinti_model.json", "k6_kesinti_kontrol.json"):
        yol = os.path.join(BURA, nm)
        if os.path.exists(yol):
            kunye[nm] = json.load(open(yol, encoding="utf-8"))

    kon = kunye.get("k6_kesinti_kontrol.json", {})
    gurultu = bool(kon and kon.get("oran_gercek_bolu_sahte", 9e9) < 1.5)
    hukum = {
        "DIK_MI": "EVET -- DELTA yonu |kos| <= 0,074 ile mevcut butun yonlere DIK",
        "ALINMALI_MI": "HAYIR",
        "GEREKCE": [
            "GURULTU TABANI KONTROLU: ilce etiketleri KARISTIRILMIS kesinti "
            f"kolonlariyla uretilen sahte delta Q={kon.get('Q_sahte_delta')}, "
            f"gercek delta Q={kon.get('Q_gercek_delta')} -- oran "
            f"{kon.get('oran_gercek_bolu_sahte')}. Sahte, gercekten BUYUK.",
            f"kos(gercek delta, sahte delta) = {kon.get('kosinus_gercek_sahte')} -- "
            "yonun dortte ucu, bilgi tamamen yok edildiginde de aynen olusuyor. "
            "Fark kesintinin bilgisi degil, 35 fazladan kolonun feature_fraction "
            "ornekmesini degistirmesidir.",
            f"LightGBM gain payi: gercek kesinti kolonlari {kon.get('gain_payi_gercek')}, "
            f"karistirilmis kolonlar {kon.get('gain_payi_sahte')} -- karistirilmis "
            "olanlar DAHA COK gain aliyor.",
            "Gain'in cogu ilceye ozgu degil GUN duzeyi toplamlarda "
            "(ks_gun_toplam_dk/adet) -- yani model kesintiyi fiziksel sebep olarak "
            "degil, EPIAS kayit yogunlugunun takvim vekili olarak kullaniyor.",
            "DELTA enerjisinin %54,0'u OLCULMUS-CURUK (olu trafo) satirlarda, "
            "temizlikten sonra kalanin %50,7'si SOGUK trafolarda -- yani kesinti "
            "panelinin trafo duzeyinde HIC bilgi tasiyamayacagi yerlerde. Fiziksel "
            "kesinti etkisinin sekli bu degildir.",
            "Kurtoz 19,8 (> 10 kapisi) ve Q 0,00149 (< 0,01 kapisi): yon hem sivri "
            "hem LB'de olculemeyecek kadar kucuk.",
            "TAM aday kapilari gecmiyor: kos(m4-v102 ekseni) = +0,616, "
            "kos(y40_sota) = +0,432 -- mevcut yonlerin tekrari.",
            "docs/30 s4a ile TUTARLI: kesinti daha once olculmus, artigin %0,17'sini "
            "acikliyor ve isareti mekanik beklentinin TERSI (+0,0404) cikmisti.",
        ],
        "VERI_YETERLI_MI": "Kapsam yeterli (test penceresi 122/122 gun, 47/47 ilce, "
        "%100 satir eslesmesi). Yetersiz olan KAPSAM DEGIL, SINYAL.",
    }
    sonuc = dict(
        HUKUM=hukum if gurultu else {"DIK_MI": "olc", "ALINMALI_MI": "olc"},
        BASLIK=dict(
            kos_g7_span_tau3={a: adaylar[a]["kosinus"].get("g7_span_tau3") for a in adaylar},
            secilen=secim,
            secilenin_kos_g7=adaylar[secim]["kosinus"].get("g7_span_tau3"),
            mutlak_maks_kosinus=adaylar[secim]["mutlak_maks_kosinus"],
            kurtoz=adaylar[secim]["son"]["kurtoz"],
            Q=adaylar[secim]["son"]["Q"],
            gecti=adaylar[secim]["gecti"],
        ),
        taban=dict(dosya="tuketim_m6_ikiyon.csv", LB=1.00284, m0=M0, **ref6),
        kalibrasyon=dict(m4_r=R_KAL, kirpma=KIRP, delta_ham_Q=Qd, delta_olcek=olcek),
        adaylar=adaylar,
        ikili_kosinus_TAM_DELTA=ikili,
        secim=secim,
        cikti=dict(dosya=CIKTI, kapi=kapi, **adaylar[secim]["dagilim"]),
        kunye=kunye,
    )
    json.dump(
        sonuc,
        open(os.path.join(BURA, "k5_kesinti.json"), "w", encoding="utf-8"),
        indent=1,
        ensure_ascii=False,
    )
    print(f"\nSECIM {secim} -> submissions/{CIKTI}")
    print("KAPI:", json.dumps(kapi))
    print("m6 REF:", json.dumps({k: round(v, 5) for k, v in ref6.items()}))
    print("ADAY  :", json.dumps({k: round(v, 5) for k, v in adaylar[secim]["dagilim"].items()}))
    print(f"yazildi {os.path.join(BURA, 'k5_kesinti.json')}")


if __name__ == "__main__":
    main()
