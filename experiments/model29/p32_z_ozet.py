"""p32-Z: NIHAI HUKUM -- katman basina kapi degerlendirmesi ve P(LB>=0.00628).

Birlesik dagilim: K3e'nin trafo-kumeli onyukleme dagilimi + K2'nin p15'te
olculmus sabit harman kazanci (+0.00111 dMSE, kendi belirsizligi p15'te
raporlanmamis oldugu icin nokta olarak eklenir -- bu, birlesimin
olasiligini HAFIF ABARTIR).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_d_hcapa import wins_h
from p32_e_gecmis_uzunlugu import capa_ve_kapsam
from p32_f_test_kohort import hucre_gen
from p32_ortak import BLOKLAR, PK

N = 20000
HEDEF = 0.00628
SICAK_PAY = PB.SICAK_PAY
BOLEN = 2 * 1.00115
K2_dMSE = 0.00111  # p15_ozet: lgbm huber a=2.0 sicak harman kazanci


def main() -> None:
    B = PB.veri_kur()
    E = pd.read_parquet(os.path.join(PB.DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(PB.DN, "test.parquet"))
    ts = T[T["soguk_mu"] == 0]
    _, _, tn, tay = capa_ve_kapsam("TEST", ts["tanim"].astype(str).to_numpy())
    tp = pd.Series(hucre_gen(ts["p_gun_sayisi"].to_numpy("float64"),
                             ts["guc"].to_numpy("float64"),
                             ts["ufuk_gun"].to_numpy("float64"), tn, tay)
                   ).value_counts(normalize=True)
    HH, TN, CAPA, P0 = [], [], {}, {}
    for b in BLOKLAR:
        bb = B[b]
        mu, sd, n, ay = capa_ve_kapsam(b, bb["tanim"])
        CAPA[b] = (mu, sd)
        P0[b] = PB.harman(bb, PB.ADAYLAR["URETIM"])
        ds = E[(E["_blok"] == b) & (E["soguk_mu"] == 0)]
        HH.append(hucre_gen(ds["p_gun_sayisi"].to_numpy("float64"),
                            ds["guc"].to_numpy("float64"),
                            ds["ufuk_gun"].to_numpy("float64"), n, ay))
        TN.append(bb["tanim"])
    HH, TN = np.concatenate(HH), np.concatenate(TN)
    hp = pd.Series(HH).value_counts(normalize=True)
    w = np.array([tp.get(k, 0.0) / hp[k] for k in HH])
    w = w / w.mean()

    rng = np.random.default_rng(3600)
    tas = rng.uniform(0.28, 0.76, size=N)
    out = {}
    for k in (8.0, 12.0, 20.0):
        d2 = np.concatenate([
            ((B[b]["y"] - P0[b]) ** 2 - (B[b]["y"] - wins_h(P0[b], k, "cift", CAPA[b])) ** 2)
            for b in BLOKLAR]) * w
        s = pd.Series(d2).groupby(TN)
        top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
        r2 = np.random.default_rng(3601)
        m = len(top)
        idx = r2.integers(0, m, size=(N, m))
        o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
        for ad, dm in (("K3e_tek", o), ("K3e+K2", o + K2_dMSE)):
            lb = tas * (dm * SICAK_PAY) / BOLEN
            out[f"{ad}_k{int(k)}"] = {
                "dMSE_ort": round(float(dm.mean()), 6),
                "dMSE_GA95": [round(float(np.quantile(dm, 0.025)), 6),
                              round(float(np.quantile(dm, 0.975)), 6)],
                "LB_kazanci_ort": round(float(lb.mean()), 5),
                "LB_GA95": [round(float(np.quantile(lb, 0.025)), 5),
                            round(float(np.quantile(lb, 0.975)), 5)],
                "beklenen_LB_skoru@tasima0.5": round(
                    1.00115 - float((0.5 * (dm * SICAK_PAY) / BOLEN).mean()), 5),
                "P_LB_>0": round(float((lb > 0).mean()), 3),
                "P_LB_>=_0.00628": round(float((lb >= HEDEF).mean()), 3),
            }
    # K2 tek basina
    lb2 = tas * (K2_dMSE * SICAK_PAY) / BOLEN
    out["K2_tek"] = {
        "dMSE_ort": K2_dMSE,
        "LB_kazanci_ort": round(float(lb2.mean()), 5),
        "P_LB_>=_0.00628": 0.0,
        "not": "p15'te olculdu; nokta tahmin, dagilim yok.",
    }

    R = {
        "00_HEDEF": {"ilk_3_icin_gereken_LB_kazanci": HEDEF, "mevcut_LB": 1.00115,
                     "3_sira": 0.99487},
        "01_KATMAN_HUKUMLERI": {
            "K1_olu_trafo": {
                "HUKUM": "CURUDU -- GONDERILMEZ",
                "kanit": (
                    "URETIM harmanli tezgahta W=30/c=0.5 kurali UC BLOKTA DA KAYBEDIYOR "
                    "(agr dMSE yaz25 -0.0052, guz25 -0.0306, kis26 -0.0151; isaret 0/3, "
                    "tohum 0/9). Agresif uc (c=0.0) 10 kat daha kotu (ORT -0.120). "
                    "Maske docs/79'unkiyle BIREBIR ortusuyor (yaz25 11.770 satir, "
                    "kesinlik %96.76, ort gercek 65.4) -- yani ayni kural, TERS sonuc."
                ),
                "neden_docs79_farkli": (
                    "Maskeli satirlarda modelin tahmin ortalamasi ZATEN gercege esit "
                    "(yaz25 log-uzayinda p=0.189 vs y=0.215). Daha da kucultmek "
                    "ASAGI YANLILIK ekliyor. docs/52 (28 Agustos) bu tezi ILERI "
                    "PENCERELI kor tekrarla ZATEN CURUTMUSTU (5 kesmede 5/5 "
                    "sifirlama ZARARLI: +0.016..+0.106). docs/79 §4 / docs/80 §6'daki "
                    "'KABUL' o curutmeyle CELISIYOR ve yeniden uretilemedi."
                ),
                "365_gun_ekseni": (
                    "OLCULEMEZ. train.csv 2025-01-01'de basliyor: yaz25 gecmisi 90 gun, "
                    "guz25 212, kis26 334. W=365 UC BLOKTA DA BOS maske veriyor. "
                    "W=180 tek olculebilir uzun pencere: yaz25 OLCULEMEZ (gecmis 90 gun), "
                    "guz25 +0.00015, kis26 +0.00072 -- yani en fazla 2/3 ve buyuklugu "
                    "gerekenin ~%3'u. 'Uzun pencere daha iyi' hipotezi bu veriyle "
                    "SINANAMAZ; disaridaki -0.01527 rakami bize TASINMAZ."
                ),
            },
            "K2_lgbm_huber_a2.0": {
                "HUKUM": "GECERLI ama KUCUK -- uretildi",
                "yol": "DELTA (test tarafinda hicbir aile onbellegi YOK; taban ve huber "
                       "kollari AYNI p18 kosusunda egitildi, boylece docs/80 §8'deki "
                       "onbellek sapmasi delta'da birinci dereceden iptal olur)",
                "kazanc": "p15: +0.00111 harman dMSE -> LB +0.00056 (tasima 0.5)",
                "gerekenin_yuzdesi": "%9",
            },
            "K3_winsorization": {
                "GLOBAL_sigma": "ELENDI -- 2.0/2.4'te 1/3, ORT NEGATIF (-0.16 / -0.03)",
                "TRAFO_ICI_global_sigma": "KONUSUZ -- hicbir satiri budamiyor (no-op)",
                "GECMIS_CAPALI_kucuk_k": (
                    "ELENDI. k in [1.5, 3.0] blok-esit ortalamada +0.008..+0.012 "
                    "gorunuyor ama TEST KOHORTUNA agirliklandirilinca guz25 ISARET "
                    "DONDURUYOR (+0.008 -> -0.029) ve havuz isareti 1/3'e dusuyor. "
                    "(MEMORY: 'ham blok kazanci sisik cikiyor'.)"
                ),
                "KUYRUK_KAPAGI_buyuk_k": {
                    "HUKUM": "DORT KAPIYI DA GECEN TEK KATMAN",
                    "kural": "SICAK satirda log1p(tahmin), trafonun egitim gecmisi "
                             "log1p ort mu_i +- k*sd_i bandina kirpilir (k=8..20)",
                    "kapi_a_gercek_blok_CV": "EVET -- URETIM sicak harmani (3,1,1,1.4)",
                    "kapi_b_blok_disi": "EVET -- k=8 plato icinde 2/3 katta secildi; "
                                        "durust blok-disi ort +0.0147 (kohortlu)",
                    "kapi_c_isaret": "3/3 HAM ve 3/3 TEST-KOHORTLU (k=8..20)",
                    "kapi_d_tohum": "9/9 (k>=10), 8/9 (k=8)",
                    "onyukleme_k8": "P(dMSE>0)=1.000, GA95 [+0.0012, +0.0545]",
                    "MEKANIZMA": (
                        "Kazanc INCE AYARDAN DEGIL, BIR AVUC UC DEGERDEN geliyor: "
                        "k=12'de satirlarin yalnizca %0.35'i kirpilirken kis26 kazanci "
                        "hala +0.018. Model kucuk bir alt kumede trafonun tum gecmisiyle "
                        "bagdasmayan tahminler uretiyor; kapak bunlari yakaliyor. "
                        "Test'te 4.263 (k=20) - 9.592 (k=8) satir etkileniyor, HEPSI SICAK."
                    ),
                },
                "SOGUK_winsorization": (
                    "YOK. Tek gecerli blok kis26 (ezber kanali): G_1.5/2.0/2.4 hepsi "
                    "NEGATIF (-0.025/-0.013/-0.003); yalniz G_3.0 +0.00098 ve o da "
                    "satirlarin %0.2'sine dokunuyor -- ihmal edilebilir."
                ),
            },
        },
        "02_BIRLESIM_ve_P_hedef": out,
        "03_CAKISMA": (
            "K1 uretilmedi. K3e (9.592 sicak satir) ile K2 (556.319 sicak satir) "
            "KESISIYOR. Birlesik dosyalarda SIRA: once K2 kaydirmasi, SONRA K3e "
            "kirpmasi -- tersi olsa K2 deltasi kirpilan satirlari bandin disina "
            "tasirdi. Soguk satirlara HICBIR katman dokunmuyor (kesisim 0), yani "
            "p20/p21 gibi soguk adaylarla da cakismaz."
        ),
        "04_ILK_3_ICIN_YETERLI_MI": {
            "CEVAP": "HAYIR -- olasilik dusuk ama SIFIR DEGIL.",
            "en_iyi_aday": "p32_kuyruk_cift_k8_l2",
            "beklenen_LB_tasima_0.5": "1.00115 - 0.00412 = 0.99703 (tasima 0.5); GA95 [0.98807, 1.00081]",
            "P_hedefi_tutturma": "0.215 (k=8, K3e+K2; tasima orani belirsizligi dahil). k=12 icin 0.192, k=20 icin 0.158.",
            "aciklama": (
                "Beklenen kazanc gerekenin ~%70'i. Hedefe ulasmak, hem tasima oraninin "
                "bandinin ust ucunda (>=0.7) hem de kuyruk kapagi kazancinin onyukleme "
                "dagiliminin ust yarisinda olmasini gerektiriyor. Bu KOSULLU bir "
                "bahis, beklenti degil."
            ),
        },
        "05_DURUSTLUK_NOTU": (
            "Bu turda ALTI ADAY'DAN DORDU ELENDI: p08 olu trafo (curudu, docs/52 "
            "ile ayni yonde), global winsorization, trafo-ici winsorization (no-op), "
            "kucuk-k gecmis capali winsorization (kohort agirliginda isaret dondu). "
            "365-gun ekseni OLCULEMEZ (egitim gecmisi yetersiz). Ayakta kalan: "
            "kuyruk kapagi (gercek, dort kapi gecti, ama tek basina yetmiyor) ve "
            "lgbm huber (gercek, cok kucuk)."
        ),
    }

    yol = os.path.join(PK, "p32_katmanlar.json")
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
    mevcut["ZZ_HUKUM"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"02": out, "04": R["04_ILK_3_ICIN_YETERLI_MI"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
