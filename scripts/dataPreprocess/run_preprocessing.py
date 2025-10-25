import pandas as pd
from preprocess import preprocess_turkish_text  # dosyadan fonksiyonu import et
import os

def main():
    print("Veri Ön İşleme Script'i Başlatıldı.")
    
    base_dir = os.path.dirname(__file__) # Şu anki dosyanın klasörü (scripts/dataPreprocess)
    raw_data_path = os.path.join(base_dir, '../../data/raw/final_dataset.csv')
    processed_data_path = os.path.join(base_dir, '../../data/processed/cleaned_dataset.csv')
    
    # 1. Ham verinin hepsini yükle
    print(f"Ham veri yükleniyor: {raw_data_path}")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"HATA: Ham veri dosyası bulunamadı. Beklenen konum: {raw_data_path}")
        return
    print(f"Toplam {len(df)} satır veri yüklendi.")

    # 2. Güvenlik Kontrolü: Gerekli sütunlarda boş veri varsa o satırları uçur
    df.dropna(subset=['article_text', 'summary'], inplace=True)
    print(f"Eksik veriler temizlendi, kalan satır sayısı: {len(df)}")
    
    # 3. Temizleme fonksiyonunu tüm veri setine uygula
    print("Ön işleme başlıyor (Bu işlem uzun sürebilir)")
    df['cleaned_article'] = df['article_text'].apply(preprocess_turkish_text)
    df['cleaned_summary'] = df['summary'].apply(preprocess_turkish_text)
    print("Ön işleme tamamlandı.")

    # 4. Sadece gerekli sütunları içeren yeni bir DataFrame oluştur, bu sütunlardan sadece 2 tanesini fine tune ederken kullanacağız ama ileride referans karşılaştırma işlemleri vb için diğerlerini de saklıyoruz
    final_df = df[['url', 'title', 'summary', 'article_text', 'cleaned_summary', 'cleaned_article']]
    
    # 5. Temizlenmiş veriyi yeni bir dosyaya kaydet
    # 'processed' klasörünün var olduğundan emin ol
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    
    final_df.to_csv(processed_data_path, index=False)
    print(f"Temizlenmiş veri başarıyla kaydedildi: {processed_data_path}")

if __name__ == '__main__':
    main()