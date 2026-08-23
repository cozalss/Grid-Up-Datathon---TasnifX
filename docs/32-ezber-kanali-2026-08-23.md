# CV'yi bozan şey bulundu: `tanim_num` ezber kanalı (2026-08-23)

Bugün LB, dün gece CV'nin onayladığı iki değişikliğin **zararlı** olduğunu
söyledi (+0,0145). Sebebi arandı ve bulundu. Hipotezim yanlıştı; gerçek
mekanizma daha ağır.

## 1. Hipotezim çürütüldü

"Doğrulamada soğuk satırlar maskeleme ile üretiliyor" demiştim. **Yanlış.**
`soguk_maskele` yalnızca **eğitim** çerçevesine uygulanıyor
(`tuketim_model.py:1119`, `deney.py:226`); doğrulama ve test çerçeveleri hiç
maskelenmiyor. Değerlendirilen soğuk satırlar **doğal** soğuk.

Ayrıca "maskeli satır ayırt edici iz bırakıyor mu" sorusu ölçüldü: 33 `t_*`
kolonunun **tamamı** maskeli, doğrulama ve test kümelerinin üçünde de %100
boş. Maskeleme eksiklik desenini birebir yeniden üretiyor. İz yok.

## 2. Gerçek mekanizma: kimlik maskelemeden sağ çıkıyor

Maskeleme trafonun **geçmiş özetini** siliyor ama **kimliğini** silmiyor.

```
tanim_num  ->  %100 bire-bir trafo kimligi
               (5.300 benzersiz / 5.309 trafo, tanim_num basina max 1 trafo)
           ->  "t_" onekiyle BASLAMADIGI icin maskelemeden SAG CIKIYOR
           ->  soguk uzmaninin 107 kolonu ICINDE
```

Ve CV'de "soğuk" sayılan trafoların neredeyse tamamı, **başka eğitim
katlarında mevcut**:

```
blok     soguk trafolarin egitim katlarinda bulunma orani   medyan satir
yaz25    %94,1 trafo  /  %97,2 satir                         51
guz25    %94,9 trafo  /  %97,7 satir                        121
kis26    %0,0                                                 0
------------------------------------------------------------------
CV soguk degerlendirme satirlarinin %48,0'i EZBERLENEBILIR
TEST                                                        %0,0
```

Yani model, "soğuk" dediğimiz trafoların gerçek tüketim seviyesini
`tanim_num` üzerinden **ezberleyebiliyor**. Ölçülen değeri:

```
yaz25  trafo-ezberi   RMSE 1,2111  r=0,741
       mesru yedek    RMSE 1,7224  r=0,312    (ilce x guc)
                      -> 0,51 log birimlik haksiz avantaj
```

Testte 2.024 soğuk trafonun **2.024'ü** eğitimde hiç yok. Kanal kapalı.

## 3. Kırılmanın tam deseni bununla açıklanıyor

```
blok     ezber orani    dunku degisiklik ne dedi
yaz25       %97,2       -0,0130   IYILESTI     <- kirli
guz25       %97,7        (ara)                 <- kirli
kis26        %0,0       +0,0165   KOTULESTI    <- TEK DURUST KAT
LB            —         +0,0145   KOTULESTI    <- kis26 ile AYNI ISARET
```

**`kis26` LB'yi doğru tahmin etti.** Son blok olduğu için ondan sonra kat
yok, dolayısıyla ezber kanalı kapalı — ve tam da bu yüzden dürüst.

Kalibrasyon (`LB ≈ yaz25_CV + 0,0423`) iki noktada tutmuştu çünkü o
değişiklikler ezber kanalını **kımıldatmıyordu**. Dünkü iki değişiklik
(hafta günü ve harman ağırlığının cat'ten xgb/lgbm'e kayması) doğrudan
modelin kıt ayırt-edici kolonları kullanma biçimini değiştirdi — yani ezber
kanalını kullanan tarafı güçlendirdi. Sapmanın 0,0006'dan 0,029'a fırlaması
bu yüzden.

## 4. Koddaki gerekçe ölçümle yanlış çıktı

`REJIM_AYARLARI` içindeki "soğuk uzmanı maske 1,00'da çalışıyor ve elinde
trafoyu ayırt eden **hiçbir şey yok**" gerekçesi **yanlış**. Elinde
`tanim_num` var ve o bire-bir kimlik.

Hafta günü kararı da bu yüzden çöktü: hafta-sonu etkisi kVA bandına göre
**işaret değiştiriyor** (<160 kVA +0,0624, 1000-1600 kVA −0,0879), ve
`yaz25` ile `kis26` hafta günü profilleri **negatif** korele (−0,210). Yani
+0,00274 / t=+2,39 sonucu kararlı bir sinyali değil, 678 trafoluk küçük ve
kat içinde bile tutarsız bir alt kümenin takvim gürültüsünü ölçüyordu.

## 5. İki doğrudan sonuç

**A. Soğuk rejim için karar mercii `yaz25` değil, `kis26` olmalı.**
`yaz25` soğuk için geçersiz bir ölçüm aracı; mevsimsel ikiz olması bu kusuru
kapatmıyor.

**B. `tanim_*` kolonları soğuk uzmanından çıkarılmalı.** Ezber kanalını
kapatmak hem CV'yi dürüstleştirir hem de testte zaten kullanılamayan bir
kanala model bağımlılığını keser.

İkisi de ölçülecek. Sıra: sinir ağı ölçümü bitince `kis26` üzerinde
`tanim_*` çıkarma deneyi.
