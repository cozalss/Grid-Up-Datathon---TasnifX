"""p23-A: 2026-05-11 PARTISINI BAGIMSIZ DOGRULA (yalniz csv + p06 maskesi).

Baska oturumun iddiasi (kaba tabanda):
  * test'te 2026-05-11 gunu 2.222 trafo ayni anda doguyor
  * 1.326'si train'de YOK (soguk-parti, 108.253 satir)
  * 896'si train'de VAR (KOPRU, 72.785 satir)

Cikti:
  * aday_csv/p23_parti_soguk_maske.npy  (714688 bool, test.csv id sirasi)
  * aday_csv/p23_parti_kopru_maske.npy  (714688 bool, test.csv id sirasi)
  * p23_parti.json  "adim1_parti_dogrulama" bolumu
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")
JSON_YOL = os.path.join(PK, "p23_parti.json")

test = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
train = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    dtype={"tanim": str},
    usecols=["tanim", "tarih"],
)
assert len(test) == 714688, f"test satir {len(test)}"

# --- gunluk satir sayilari (10-11 Mayis gecisi)
gunluk = test.groupby("tarih").size()
gecis = {g: int(gunluk.get(g, 0)) for g in ["2026-05-09", "2026-05-10", "2026-05-11", "2026-05-12"]}

# --- her tanim'in test'teki ILK tarihi
ilk = test.groupby("tanim")["tarih"].min()
dogum_sayim = ilk.value_counts().sort_index()
en_buyuk_5 = {str(k): int(v) for k, v in dogum_sayim.nlargest(5).items()}

parti_tanimlar = set(ilk[ilk == "2026-05-11"].index)

train_tanimlar = set(train["tanim"].unique())
parti_kopru = parti_tanimlar & train_tanimlar
parti_soguk = parti_tanimlar - train_tanimlar

# --- train'de tek gunde dogan en buyuk parti (organiklik kontrolu)
train_ilk = train.groupby("tanim")["tarih"].min()
train_dogum = train_ilk.value_counts()
train_en_buyuk = {str(k): int(v) for k, v in train_dogum.nlargest(3).items()}

# --- satir sayilari (test.csv id sirasinda maskeler)
m_soguk = test["tanim"].isin(parti_soguk).to_numpy()
m_kopru = test["tanim"].isin(parti_kopru).to_numpy()
assert not (m_soguk & m_kopru).any()

# --- p06 soguk maskesiyle tutarlilik: parti-soguk MUTLAKA soguk olmali
p06 = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
assert p06.shape == (714688,)
ihlal_soguk = int((m_soguk & ~p06).sum())   # parti-soguk ama p06'ya gore sicak -> 0 olmali
kopru_p06_soguk = int((m_kopru & p06).sum())  # kopru p06'ya gore soguk olmamali -> 0 beklenir

# --- p21 aday csv id sirasi test.csv ile birebir mi
p21 = pd.read_csv(os.path.join(AC, "p21_harman311_olu50.csv"))
id_ayni = bool((p21["id"].to_numpy() == test["id"].to_numpy()).all())

sonuc = {
    "gunluk_satir_gecis": gecis,
    "test_dogum_en_buyuk_5_gun": en_buyuk_5,
    "train_dogum_en_buyuk_3_gun": train_en_buyuk,
    "parti_trafo": len(parti_tanimlar),
    "parti_kopru_trafo": len(parti_kopru),
    "parti_soguk_trafo": len(parti_soguk),
    "parti_soguk_satir": int(m_soguk.sum()),
    "parti_kopru_satir": int(m_kopru.sum()),
    "parti_soguk_satir_pay": round(float(m_soguk.sum()) / len(test), 4),
    "parti_kopru_satir_pay": round(float(m_kopru.sum()) / len(test), 4),
    "iddia": {"trafo": 2222, "soguk": 1326, "kopru": 896,
              "soguk_satir": 108253, "kopru_satir": 72785},
    "p06_tutarlilik": {
        "parti_soguk_ama_p06_sicak": ihlal_soguk,
        "kopru_ama_p06_soguk": kopru_p06_soguk,
        "p06_soguk_toplam": int(p06.sum()),
    },
    "p21_id_sirasi_test_ile_birebir": id_ayni,
}

if ihlal_soguk != 0:
    sonuc["DUR"] = "parti-soguk p06 soguk maskesinin ALT KUMESI DEGIL -- devam etme"
else:
    np.save(os.path.join(AC, "p23_parti_soguk_maske.npy"), m_soguk)
    np.save(os.path.join(AC, "p23_parti_kopru_maske.npy"), m_kopru)
    sonuc["maskeler"] = ["aday_csv/p23_parti_soguk_maske.npy",
                        "aday_csv/p23_parti_kopru_maske.npy"]

R = {}
if os.path.exists(JSON_YOL):
    with open(JSON_YOL, encoding="utf-8") as fh:
        R = json.load(fh)
R["adim1_parti_dogrulama"] = sonuc
with open(JSON_YOL, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False)

print(json.dumps(sonuc, indent=1, ensure_ascii=False))
