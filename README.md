# Trendyol Hackathon 2025 — Search Ranking Pipeline

Bu repo, Trendyol Hackathon kapsamında kullanıcı-aranan terim-oturum özelinde ürün sıralama modeli üretmek için hazırlanan uçtan uca veri işleme ve modelleme pipeline'ını içerir.

## Problem
Katılımcılar, moda ile ilişkili popüler arama verileri ve kullanıcı davranışlarını kullanarak arama sonuçlarının tıklama ve satın alıma dönüşüm oranını artırmayı hedefler. Bir oturum, (kullanıcı, arama terimi, tarih) üçlüsü olarak tanımlanır. Amaç, her oturum için ürünlerin:
- clicked: Tıklanma olasılığı
- ordered: Sipariş olasılığı

üzerinden sıralamalar üretmektir.

## Değerlendirme
Oturum bazında tıklama ve sipariş AUC değerleri ayrı ayrı hesaplanır. Final skor, bu AUC değerlerinin ağırlıklı toplamıdır (sipariş AUC’si daha yüksek ağırlıktadır). 1’e yakın AUC, modelin ilgili ürünleri üst sıralara daha iyi yerleştirdiğini gösterir.

Bu repo ile elde edilen private skor: 0.64677

## Proje Yapısı
- `trendyol_hackathon_ranking.ipynb`: Veri birleştirme, özellik mühendisliği, model eğitimi, doğrulama ve submission üretimi.
- `data/`: Yerel veri klasörü (bu dizine veri dosyalarınızı yerleştiriniz). Notebook içerisindeki tüm yollar `data/` ile başlar.

## Kurulum
Ortamı hazırlamak için (önerilen):

```bash
python -m venv .venv
.\\.venv\\Scripts\\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Kullanım
1. `data/` klasörüne sağlanan parquet dosyalarını dizin yapısına göre yerleştirin.
2. `trendyol_hackathon_ranking.ipynb` dosyasını açın ve sırayla çalıştırın:
   - Veri yükleme ve birleştirme (Polars, lazy → eager)
   - Metin temizleme ve label encoding
   - Downcast ve veri temizliği
   - Doğrulama (TabNet) ve skorlama
   - Tüm veri ile eğitim ve test üzerinde tahmin (submission.csv)
3. Çıktılar: `submission.csv` dosyası proje köküne yazılır.

## Notlar
- Notebook içindeki yollar Kaggle yerine yerel `data/` klasörüne güncellenmiştir.
- Tanı/diagnostic ve runtime kurulum (pip install) hücreleri kaldırılmıştır. Bağımlılıklar `requirements.txt` ile yönetilir.
- İsteğe bağlı iyileştirmeler: model hiperparametreleri, özellik seçimi, farklı modellerle karşılaştırma.

## Lisans
Bu çalışma hackathon kullanımına yöneliktir. Veri setlerinin lisans ve paylaşım koşulları orijinal sağlayıcıya aittir.
