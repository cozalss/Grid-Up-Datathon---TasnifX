"""Denetim ozeti: d1/d2/d3 ciktilarindan iddia-basina GECTI/KALDI tablosu uretir.
Cikti: d1_denetim.json (UTF-8)
"""

import json
import os

BURA = os.path.dirname(os.path.abspath(__file__))


def yuk(n):
    for enc in ("utf-8", "cp1254"):
        try:
            return json.load(open(os.path.join(BURA, n), encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(n)


d1 = yuk("d1_denetim.json")
d2 = yuk("d2_derin.json")
d3 = yuk("d3_ucgen.json")
d4 = yuk("d3_sapma.json")

O = {
    "_ne": "TasnifX 29 Agustos iddialarinin bagimsiz denetimi",
    "_tarih": "2026-08-29",
    "_kaynak": ["d1_hesap.py", "d2_derin.py", "d3_ucgen.py", "d3_sapma.py"],
    "IDDIA_1_Q_toplam": {
        "hukum": "GECTI",
        "iddia": 0.121396,
        "olculen": d1["3_diklik"]["Q_dogrudan"],
        "fark": d1["3_diklik"]["Q_dogrudan"] - 0.121396,
        "not": "d = log1p(m4) - log1p(v102), N=714688 tam bolen. 6 hanede birebir.",
    },
    "IDDIA_2_L_toplam": {
        "hukum": "GECTI",
        "iddia": 0.022319,
        "olculen_tam": d2["A_dosya_kimligi"]["LTOT_tam"],
        "yuvarlama_hatasi": d2["A_dosya_kimligi"]["LTOT_yuvarlama_hatasi"],
        "turetim": (
            "MSE(k)=m0-2kL+k^2 Q ; L:=-<a-y,d>/N ; k=1 -> L=(m0+Q-M4^2)/2. "
            "Isaretler DOGRU, sayisal ozdeslik hatasi 2.2e-16."
        ),
        "kusur": (
            "L_toplam kodda 0.022319 diye ELLE YAZILDI (m93/m94/m95). Tam deger "
            "0.0223185465. Skora etkisi 9.3e-12 RMSLE -- ihmal edilebilir ama "
            "kappa'lari 5. hanede kaydiriyor (kc 0.336880 yerine 0.336857)."
        ),
    },
    "IDDIA_3_maske_ve_Q_ayrisimi": {
        "hukum": "GECTI",
        "soguk_satir": d1["2_maske"]["soguk_satir"],
        "Q_sicak": d1["3_diklik"]["Q_sicak"],
        "Q_soguk": d1["3_diklik"]["Q_soguk"],
        "toplam_esitligi": d1["3_diklik"]["fark"],
        "nan_tanim": 0,
        "bosluk_farki": 0,
        "metinli_tanim_test": len(d2["E_maske"]["test_metinli"]),
        "metinli_tanim_train": len(d2["E_maske"]["train_metinli"]),
        "not": (
            "12 metinli tanim var, 9'u train'de de var -> tam-string karsilastirmasi "
            "DOGRU sonucu veriyor. Sayisal on-ek cikarimi yapilsaydi 490 satir "
            "YANLIS sekilde SOGUK'a giderdi (Q'nun %0.022'si, MSE etkisi 1.2e-6)."
        ),
    },
    "IDDIA_4_p51_dosyasi": {
        "hukum": "GECTI",
        "formul": "p51 = v102 + 0.50*d_sicak + 0.1838536*d_soguk",
        "maks_log_fark": d2["A_dosya_kimligi"]["p51_maks_log_fark"],
        "tc": 0.022319 / (d1["3_diklik"]["Q_sicak"] + d1["3_diklik"]["Q_soguk"]),
        "not": "Diskteki dosya bit duzeyinde bu formul (4.4e-16).",
    },
    "IDDIA_5_L_ayrisimi": {
        "hukum": "GECTI",
        "L_sicak": 0.010605162105012665,
        "L_soguk": 0.011713837894987334,
        "not": "m95'in cozucusu cebirsel olarak dogru, bagimsiz turetimle birebir.",
    },
    "IDDIA_6_m6_dosyasi": {
        "hukum": "GECTI",
        "kw": d2["A_dosya_kimligi"]["kw"],
        "kc": d2["A_dosya_kimligi"]["kc"],
        "maks_log_fark": d2["A_dosya_kimligi"]["m6_maks_log_fark"],
        "diklik": "d_sicak ve d_soguk ayrik destekli -> ic carpim TAM 0 (yapisal).",
    },
    "IDDIA_7_ongoru_ve_sapma": {
        "hukum": "GECTI (ongoru) / KALDI (sapmanin ACIKLAMASI)",
        "ongoru": d2["A_dosya_kimligi"]["ongoru_kod"],
        "gerceklesen": 1.00284,
        "sapma": d4["2_bugun_m6"]["sapma"],
        "yuvarlama_bandi": d1["9_yuvarlama_bandi"]["bant"],
        "sapma_bant_yarim_genisliginin_kati": d1["9_yuvarlama_bandi"][
            "sapma_bant_genisliginin_kati"
        ],
        "KUSUR": (
            "docs/55 'sapma girdi skorlarinin 5 haneye yuvarlanmasindan geliyor' "
            "diyor. YANLIS: yuvarlama bandi [1.002907, 1.002931], yarim genislik "
            "1.18e-5; gozlenen sapma 7.9e-5 = bandin 6.7 KATI, bant DISINDA."
        ),
        "GERCEK_KAYNAK": (
            "Q TUM 714688 satirda olculuyor, LB skorlari ise PUBLIC %50'de. "
            "Q_public != Q_tum. Turetilen ozdeslik: "
            "gerceklesen_MSE - ongorulen_MSE = 0.2770*(Qw_tum-Qw_pub) + 0.3252*(Qc_tum-Qc_pub). "
            "Bu kanalin sd'si 9.28e-5 RMSLE; gozlenen sapma 0.85 sd. TAMAMEN NORMAL."
        ),
        "gereken_bagil_Q_kaymasi": d4["2_bugun_m6"]["gereken_bagil_Q_kaymasi"],
        "sistematik_mi": (
            "HAYIR. Simulasyon ortalamasi -5.5e-6 (ihmal edilebilir onyargi), sd 1.02e-4. "
            "Gerceklesen ongoruden IYI cikmasi SANS. Model L'yi sistematik kucuk "
            "tahmin ETMIYOR."
        ),
        "neden_v102_4e-7_idi": (
            "v102 ongorusu TEK parametreliydi ve o yonun sd'si 3.27e-5 idi; gozlenen "
            "2.0e-7 = 0.006 sd -- SANS eseri tam isabet, arac 'kesin' oldugu icin degil. "
            "v83 ucgeninde sapma 5.37e-6 = public-yarim sd'sinin 14.7 KATI ama yuvarlama "
            "bandi icinde (o yonun Q'su 0.0047, cok kucuk). "
            "m6 ise 2 parametreli, 1.58x buyutmeli ve d'nin kurtozu 20.6 -> sd 9.3e-5."
        ),
        "SONUC": (
            "'0.00008 sapma yuvarlamadan' iddiasi CURUTULDU. Ongoru araci +-1e-5 degil, "
            "+-9e-5 (1 sd) hassasiyetindedir. Yarinki planlamada bu bant kullanilmalidir."
        ),
    },
    "PUBLIC_PRIVATE_RISKI": {
        "hukum": "DUSUK RISK",
        "asiri_uyum_turetimi": (
            "L_j'nin public orneklem hatasi -> beklenen ek private MSE = 4*m0/N "
            "PARCA BASINA, parcanin buyuklugunden BAGIMSIZ. = 5.66e-6 MSE = 2.82e-6 RMSLE."
        ),
        "parca_basina_RMSLE_bagimsizlik": d2["D_asiri_uyum"]["parca_basina_RMSLE_bagimsiz"],
        "parca_basina_RMSLE_kotu_durum": d2["D_asiri_uyum"]["parca_basina_RMSLE_kotu_durum"],
        "mevcut_2_parca_private_kaybi_RMSLE": d2["D_asiri_uyum"][
            "mevcut_2_parca_beklenen_private_kaybi_RMSLE"
        ],
        "private_m6_tahmini": d2["D_asiri_uyum"]["private_m6_tahmini"],
        "J_parca_bedeli_RMSLE": d2["D_asiri_uyum"]["J_bedeli_RMSLE"],
        "J_esigi_4_siraya_dusme": d2["D_asiri_uyum"]["esik_4_siraya_dusme"],
        "kosul_sayilari": {k: v["kosul_sayisi"] for k, v in d4["3_J_parca_riski"].items()},
        "J_toplam_ongoru_sd": {k: v["toplam_sd_RMSLE"] for k, v in d4["3_J_parca_riski"].items()},
        "SAYI": (
            "Asiri uyum acisindan TEHLIKE ESIGI ~43 parca (kotu durum) / ~695 parca "
            "(bagimsizlik). 3-4 parca RISKSIZ. GERCEK kisit: (a) her ek parca 1 PROB "
            "GONDERIMI yiyor (9 hak var -> en fazla J=4-5 makul), (b) ongoru sd'si "
            "J'den bagimsiz ~9.5e-5'te sabit kaliyor; marjinal kazanc 2e-4'un altina "
            "duserse olcum gurultusunun icinde kaybolur."
        ),
    },
    "DAGILIM_SAGLAMASI": {
        "hukum": "TEMIZ",
        "m6_min": 0.0118720947776849,
        "m6_maks": 180734.95004543543,
        "m6_negatif": 0,
        "m6_nan": 0,
        "m6_kirpilan_satir": 0,
        "log_uzayi_seviye": (
            "m6 log-ort 6.6421 vs train Nis-Tem 2025 ayni trafolar 6.6318 (+0.010). "
            "Aritmetik ort 2260 vs 3328 farki kuyruk etkisi; RMSLE log uzayinda oldugu "
            "icin sorun YOK."
        ),
        "dalga_kohortu_2026_05_11": d2["G_kohort"]["dalga_2026-05-11"],
        "soguk_d_ortalamasi": d2["G_kohort"]["soguk"]["d_ort"],
        "not_soguk_d_ort": (
            "Soguk tarafta d'nin ortalamasi TAM 0 (-4.5e-16): m4'un soguk kanadi "
            "v102'yi log uzayinda ORTALAMA KORUYARAK yeniden dagitiyor. Hata degil, "
            "ama 'soguk yon salt yeniden dagitim' demek -- seviye bilgisi tasimiyor."
        ),
    },
    "KIRPMA_expm1": {
        "hukum": "GECTI (m6) / KUSUR (belge)",
        "m6_kirpilan": 0,
        "p51_kirpilan": 0,
        "neden": "kw,kc in [0,1] ve a,b >= 0 oldugundan harman hep >= 0.",
        "m4_kirpilan_satir": d3["m4_kirpma"]["sifir_satir"],
        "m4_kirpma_Q_kaybi": d3["m4_kirpma"]["fark"],
        "KUSUR": (
            "m4'un KENDISI 6120 satirda (0.86%) sifira kirpildi. m71 Q'yu kirpma "
            "ONCESI olctu (0.121581) ve bu deger docs/53, docs/54 ve m90_on_kayit.json'a "
            "girdi. Dogru (dosya) degeri 0.121396. Fark 1.85e-4. "
            "EGER bu deger kullanilsaydi L=0.022411 (+0.4%), kappa=0.18433 olurdu. "
            "NEYSE KI m92/m93/m94/m95 Q'yu DOSYADAN yeniden hesapladi -> gonderilen "
            "zincir TEMIZ. Ama docs/54 kilavuzu YANLIS SAYI tasiyor; yarin elle "
            "kullanilirsa hata uretir."
        ),
    },
    "HIZALAMA": {
        "hukum": "GECTI",
        "satir": 714688,
        "id_sample_submission_ile_birebir": True,
        "id_sutunundaki_tanim_uyusmazligi": 0,
        "csv_gidis_donus_kaybi": "0 (repr ile yazilmis, tam)",
    },
    "UC_DEGERLER": {
        "hukum": "UYARI",
        "d_maks": d2["F_uc_degerler"]["d_maks"],
        "d_kurtozu": d2["F_uc_degerler"]["d_kurtozu"],
        "Q_ust_yuzde1_payi": d2["F_uc_degerler"]["Q_ust_yuzde1"],
        "Q_ust_1000_satir_payi": d2["F_uc_degerler"]["Q_ust_1000_satir"],
        "not": (
            "Q'nun %35'i satirlarin %1'inden geliyor, kurtoz 20.6. Bu yuzden "
            "Q_public ile Q_tum arasindaki fark buyuk (bagil sd %0.66-0.87) ve "
            "ongoru bandi genis. Daha DAR yonler (kucuk kurtoz) daha guvenilir ongoru verir."
        ),
    },
    "GECMIS_UCGENLER": d3["ucgenler"],
}

json.dump(
    O,
    open(os.path.join(BURA, "d1_denetim.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print(json.dumps(O, indent=1, ensure_ascii=False))
