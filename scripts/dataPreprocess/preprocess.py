import re
import nltk
from nltk.corpus import stopwords

# Gerekli nltk veri setlerini indir
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

turkish_stop_words = set(stopwords.words('turkish'))

def preprocess_turkish_text(text):
    """
    Türkçe bir metni ön işleme tabi tutar:
    - HTML etiketlerini kaldırır
    - URL'leri kaldırır
    - Noktalama işaretlerini ve sayıları kaldırır
    - Türkçe'ye özel küçük harfe çevirir
    - Stopword'leri (etkisiz kelimeler) kaldırır
    - Fazla boşlukları temizler
    """
    # 0. Güvenlik: Eğer veri string değilse, boş metin döndür
    if not isinstance(text, str):
        return ""
    
    # 1. HTML etiketlerini temizle
    text = re.sub(r'<.*?>', '', text)
    
    # 2. URL'leri ve e-postaları temizle
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text, flags=re.MULTILINE)
    
    # 3. Sadece harfleri ve boşlukları bırak
    text = re.sub(r'[^a-zA-ZçÇğĞıİöÖşŞüÜ\s]', '', text)
    
    # 4. Türkçe'ye özel küçük harfe çevirme (I -> ı)
    text = text.replace('I', 'ı').lower()

    # 5. Birden fazla boşluğu tek boşluğa indir
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Stop words temizliği
    words = text.split() # Cümleyi kelimelere ayır
    filtered_words = [word for word in words if word not in turkish_stop_words]
    text = ' '.join(filtered_words)
    
    return text

if __name__ == '__main__':
    # Bu dosya doğrudan çalıştırılırsa bir test yapsın
    print("Test modunda çalışıyor...")
    test_metin = "   <p>Bu, <b>123</b> adet gereksiz kelime içeren bir test metnidir! https://www.test.com   "
    temiz_metin = preprocess_turkish_text(test_metin)
    print(f"Orijinal: {test_metin}")
    print(f"Temizlenmiş: {temiz_metin}")