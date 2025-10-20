import pandas as pd
import os

# Birleştirilecek dosyaların adları
files_to_merge = [
    'anadolu_ajansi_haberler.csv', 
    'trt_haberler.csv'
]

# Çıktı dosyasının adı
output_filename = 'final_dataset.csv'

# Boş bir DataFrame oluşturarak başla
final_df = pd.DataFrame()

print("Veri setleri birleştiriliyor...")

# Her bir dosyayı oku ve ana DataFrame'e ekle
for file in files_to_merge:
    if os.path.exists(file):
        print(f"-> '{file}' okunuyor...")
        df = pd.read_csv(file)
        final_df = pd.concat([final_df, df], ignore_index=True)
    else:
        print(f"Uyarı: '{file}' dosyası bulunamadı, atlanıyor.")

if not final_df.empty:
    print(f"\nBirleştirme sonrası toplam satır sayısı: {len(final_df)}")

    # Temizleme Adımları
    # 1. URL'ye göre mükerrer kayıtları temizle
    initial_rows = len(final_df)
    final_df.drop_duplicates(subset=['url'], inplace=True, keep='first')
    print(f"Tekrar eden kayıtlar temizlendi. Kalan satır sayısı: {len(final_df)}")

    # 2. 'article_text' veya 'summary' sütunları boş olan satırları temizle
    final_df.dropna(subset=['article_text', 'summary'], inplace=True)
    print(f"Boş metin/özet içeren satırlar temizlendi. Kalan satır sayısı: {len(final_df)}")

    # Nihai veri setini kaydet
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nİşlem tamamlandı! Toplam {len(final_df)} adet temiz veri '{output_filename}' dosyasına kaydedildi.")
else:
    print("Birleştirilecek veri bulunamadı.")