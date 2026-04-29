# WhenOA | Kestirimci Tedarik Zinciri Analitiği

[Click to read in English (İngilizce okumak için tıklayın)](readme.md)

### Bir sipariş oluşturulduğu anda geç teslim edilip edilmeyeceğini tahmin eden makine öğrenmesi modeli.

## 🧭 Problem

Lojistik şirketleri teslimat gecikmelerini yaşandıktan sonra öğrenir. Ancak bir sipariş sisteme girildiği anda kargo modu, ürün türü, sipariş saati, bölge gibi bilgiler zaten elimizdedir. Bu bilgilerle **önceden** tahmin yapılabilir mi?

**Bu projenin sorusu:** Sipariş anında elimizdeki bilgilerle, o sipariş geç teslim edilecek mi?

Bu soru finans sektöründe "bu kredi geri ödenir mi?", üretimde "bu makine arızalanacak mı?", e-ticarette "bu kullanıcı churn olacak mı?" sorusunun lojistikteki karşılığıdır. Hepsi aynı yapı: geçmişte bilinen değişkenlerle gelecekte gerçekleşecek bir olayı tahmin etmek.

---

## ⚠️ Neden Çok Yüksek Skorlar Riskli?

Gerçekçi değildir. Veri setinde `days_for_shipping_real`, `delivery_status` ve `order_status` gibi sütunlar bulunuyor. Bu sütunlar **teslimat gerçekleştikten sonra** oluşan bilgileri içerir. Bunları modele dahil etmek, sınav sorularını önceden vermekle eşdeğerdir. Model gerçek bir şey öğrenemez, sadece cevabı ezberler.

Buna **data leakage** denir. Bir modeli canlıya aldığınızda bu bilgiler elinizde olmaz. Model çöker.

Bu projede 4 leakage kararı verildi:

| Sütun | Neden çıkarıldı |
|---|---|
| `days_for_shipping_real` | Teslimat bittikten sonra belli olur |
| `delivery_status` | Teslimat sonrası bilgileri içerir |
| `late_delivery_risk` | Hedef değişken |
| `order_status` | `COMPLETE` veya `CLOSED` görmek için siparişin kapanması gerekir - tahmin anında bu bilgi yoktur |

Leakage temizlendiğinde AUC **0.796**'ya düştü. Bu daha düşük bir skor, ama **güvenilir** bir skor.

---

## 📦 Veri Seti

**DataCo Smart Supply Chain:** 180.519 sipariş kaydı, 53 özellik.  
Kaynak: Fabian Constante et al., Mendeley Data, 2019.

Veri 2015–2018 yıllarını kapsar: 5 pazar (LATAM, Europe, Pacific Asia, USCA, Africa), 23 bölge, 51 ürün kategorisi, 4 kargo modu.

---

## 🧹 Veri Temizliği | Alınan Kararlar

Temizlik adımları "çalıştı geçti" mantığıyla değil, **her karar gerekçelendirilerek** yapıldı.

- **Mükerrer sütunlar** - 6 sütun çifti birebir aynı içeriğe sahip olduğu `==` operatörüyle kanıtlandı ve silindi.

```python
pairs = [
    ('order_item_cardprod_id', 'product_card_id'),
    ('order_profit_per_order', 'benefit_per_order'),
    # ...
]
for col1, col2 in pairs:
    print(f"{col1} == {col2} → {(df[col1] == df[col2]).all()}")
# Tümü → True
```

- **PII & anlamsız sütunlar:** `customer_password`, `customer_email`, `customer_street` gizlilik gerekçesiyle; `product_status` tüm değerleri 0 olduğu için (sıfır bilgi içeriyor) silindi.

- **Kategori tutarsızlığı:** `Electronics` isimli kategori iki farklı `category_id` ile eşleşiyordu. Biri ayakkabı, diğeri golf topu ürünlerini içeriyordu. `groupby('category_name')['category_id'].nunique() > 1` sorgusuyla tespit edildi ve her ID gerçek içeriğine göre yeniden isimlendirildi.

- **Lokasyon anomalisi:** `customer_state` sütununda eyalet kısaltması yerine zip code değerleri (95758, 91732) içeren 3 satır tespit edilip silindi.

- **İptal siparişler:** `delivery_status == 'Shipping canceled'` olan 7.754 sipariş modelden çıkarıldı. Gerekçe: iptal edilmiş bir siparişin gecikmesini tahmin etmek anlamsız. Model sadece teslim edilebilir siparişler üzerinde çalışıyor (180.516 → 172.762 satır).

---

## 🔍 Keşifsel Analiz | Ne Öğrendik?

- **Gecikme yapısal bir sorundur.** Aylık gecikme oranı tüm yıl %54–57 bandında sabit seyrediyor; mevsimsellik veya iyileşme trendi yok. Bu demek ki gecikme rastgele değil, sistemin kendisinden kaynaklanıyor.

- **Kargo modu en güçlü sinyal.** First Class kargoda gecikme oranı %95'i geçiyor. Standard Class'ta ise %38'e düşüyor. Neden? Muhtemelen First Class kısa süreli taahhütler veriyor ama lojistik altyapı bunu karşılamıyor.

- **Planlanan süre azaldıkça risk dramatik artıyor.** 1 günlük planlanan teslimat süresinde gecikme oranı %95, 4 günde %38. Bu ilişki güçlü ama dikkatli olunması gerekiyor. Modelin bu sütuna bağımlı olması overfitting riskini beraberinde getiriyordu, bu sorun feature engineering ile çözüldü.

**Korelasyon analizi leakage sinyallerini netleştirdi.**

```
real_days       →  +0.40  (güçlü pozitif - LEAKAGE)
scheduled_days  →  -0.37  (güçlü negatif - dikkatli kullanılmalı)
order_hour      →  +0.047 (zayıf ama anlamlı)
```

---

## ⚙️ Feature Engineering | Diktatörlüğü Kırmak

İlk model eğitimi denemesinde `scheduled_days` sütunu %80 feature importance aldı. Model tek sütuna bakıp karar veriyordu. Bu hem overfitting riski hem de zayıf genelleme anlamına geliyordu.

Çözüm: `scheduled_days`'i direkt modele vermek yerine, bilgisini diğer değişkenlerle harmanlayan iki hibrit feature türetmek.

```python
time_factor = 4 / (df['scheduled_days'] + 1)

# Kısa sürede, pahalı ve hacimli ürün → yüksek risk
df['load_risk_score'] = (df['product_price'] * df['item_quantity']) * time_factor

# Akşam siparişi + kısa süre → yüksek stres
df['temporal_stress'] = (df['order_hour'] > 17).astype(int) * time_factor

# Dominant sütun çıkarıldı
X = df.drop(columns=['late_delivery', 'scheduled_days'])
```

`time_factor`'daki 4 sabiti: `scheduled_days` maksimum 4 olduğundan payda 1–5 arasında kalıyor, ölçeği 0–4 bandında tutuyor, diğer sayısal özelliklerle karşılaştırılabilir büyüklükte.

Sonuç: Model artık hiçbir sütuna ezici biçimde bağımlı değil, `load_risk_score`, `sales`, `order_hour`, `order_country` dengeli katkı yapıyor.

---

## 🤖 Model Seçimi

4 model karşılaştırıldı:

| Model | Test Acc | Train-Test Gap | AUC |
|---|---|---|---|
| Logistic Regression | %59.9 | %0.33 | 0.666 |
| Random Forest | %72.4 | %3.22 | 0.800 |
| XGBoost | %72.5 | %2.23 | 0.801 |
| **LightGBM** | **%72.5** | **%0.26** | **0.796** |

**LightGBM seçildi.** XGBoost'tan AUC farkı yalnızca 0.005, istatistiksel olarak anlamsız. Ama LightGBM'in train-test gap'i sadece %0.26, bu model farklı veri dilimlerinde tutarlı davranıyor.

**Neden AUC önemli?** Accuracy tek başına yanıltıcıdır. Eğer veri setinin %57'si geç teslimatsa, "her şeyi geç de" diyen bir model %57 accuracy alır. AUC ise modelin iki sınıfı birbirinden ne kadar iyi ayırt ettiğini ölçer, sınıf dengesizliğinden etkilenmez.

**5-Fold Cross-Validation ile doğrulama:**

```
CV AUC Skorları : [0.789, 0.793, 0.793, 0.795, 0.795]
Ortalama        : 0.7931 ± 0.0021
```

Standart sapma 0.002 - model hangi veri dilimini görürse görsün benzer performans veriyor. Şansa bağlı değil.

**Bu veri setinde gerçekçi tavan nedir?** Leakage temizlenmiş tabular ML modelleri için AUC ~0.82–0.85. Elde edilen 0.796, bu tavanın yaklaşık %90'ında. Kalanı için hava durumu, taşıyıcı kapasitesi, depo doluluk oranı gibi ölçülmemiş değişkenler gerekiyor.

![Feature Importance](assets/feature_importance.png)

---

## 🎯 Threshold | İş Mantığı ile Karar

Varsayılan threshold 0.50 ile Recall 0.60 çıkıyordu: geç teslimatların %40'ı kaçırılıyordu.

**İş sorusu:** Geç teslimatı kaçırmak mı daha maliyetli, yoksa yanlış alarm üretmek mi?

Lojistikte cevap nettir: kaçırılan geç teslimat müşteri kaybına, ceza ücretlerine, itibar hasarına yol açar. Yanlış alarm ise operasyonel bir rahatsızlıktır. Bu nedenle **Recall önceliklendirildi** ve threshold 0.40'a düşürüldü.

| Threshold | Recall(1) | Precision(1) | Accuracy |
|---|---|---|---|
| 0.50 | 0.60 | 0.89 | %72.5 |
| **0.40** | **0.82** | **0.69** | **%68.0** |

%4.5 accuracy feda edilerek geç teslimat yakalama oranı %60'tan %82'ye çıkarıldı.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 📊 Final Sonuçlar

```
Model           : LightGBM | Threshold: 0.40
Test Accuracy   : %72.5 (threshold 0.50)
AUC-ROC         : 0.7963
CV AUC (5-Fold) : 0.7931 ± 0.0021
Recall (Geç)    : 0.82
Train-Test Gap  : %0.26
```

---

## 🔬 Model Doğrulama | Tutarlılık Testleri

Deploy sonrası model sistematik testlerle doğrulandı. Kargo modu, miktar ve saat bazlı testler EDA bulgularıyla örtüştü:

- **Kargo modu:** Same Day ve First Class %99+ gecikme: EDA'daki %95+ bulgusuyla örtüşüyor ✅
- **Miktar (1→5):** Monoton ve düzgün gecikme artışı, sıçrama yok ✅
- **Saat & indirim:** Dengeli dağılım, tüm aralıkta tutarlı ✅

**Tespit edilen sınırlılık | Fiyat:** 50–450$ aralığında model tutarlı ve sezgisel sonuçlar üretiyor. Ancak bu aralığın dışında güvenilirlik azalıyor.

```
Fiyat aralığı  | Eğitimdeki sipariş sayısı | Model davranışı
---------------|---------------------------|----------------
0 – 450$       | 179,544  (%99.5)          | Tutarlı ✅
500 – 1000$    | 515      (%0.3)           | Dikkatli ⚠️
1000 – 2000$   | 457      (%0.25)          | Güvenilmez ❌
```

Bu durum tüm ML modellerinin temel bir özelliğidir: model eğitildiği dağılımın dışında güvenilir tahmin üretemez. Önemli olan bunu test etmek, ölçmek ve belgelemektir.

---

## 🚀 Canlı Demo

**[whenoa.com](https://whenoa.com)** | Sipariş bilgilerini girerek anlık gecikme riski tahmini yapabilirsiniz.

| Gecikmeli Tahmin | Zamanında Tahmin |
|---|---|
| ![Gecikmeli](assets/demo_late.png) | ![Zamanında](assets/demo_ontime.png) |

---

## 🗂️ Proje Yapısı

```
supply-chain-predictive-analytics/
│
├── README.md
├── readme.tr.md
├── requirements.txt
├── .gitignore
│
├── 01_data/
│   ├── 01_raw_data/          # Ham veri push edilmedi (bkz. .gitignore)
│   └── 02_processed_data/    # İşlenmiş veri push edilmedi (bkz. .gitignore)
│
├── 02_notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_modeling_and_evaluation.ipynb
│
├── 03_app/
│   ├── app.py
│   ├── logic.py
│   ├── constants.py
│   └── style.css
│
├── 04_models/
│   └── lightgbm_late_delivery.pkl
│
└── assets/
    ├── demo_late.png
    ├── demo_ontime.png
    ├── feature_importance.png
    └── confusion_matrix.png
```