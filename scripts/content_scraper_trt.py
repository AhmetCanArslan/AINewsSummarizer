import requests
import time
from bs4 import BeautifulSoup
import csv
import os
import urllib3

# SSL sertifika doğrulama hatalarında çıkan uyarıyı gizler
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def read_urls_from_csv(filename="urls_trt.csv"):
    """
    Verilen CSV dosyasından URL'leri okur.
    """
    if not os.path.exists(filename):
        print(f"Hata: '{filename}' dosyası bulunamadı. Lütfen önce linkleri toplayan script'i çalıştırın.")
        return []
    
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader) # Başlık satırını atla
        urls = [row[0] for row in reader]
        print(f"'{filename}' dosyasından {len(urls)} adet URL okundu.")
        return urls

def parse_article_trt(url):
    """
    Tek bir TRT Haber URL'sini ayrıştırır ve verileri bir dictionary olarak döndürür.
    """
    try:
        # SSL doğrulamasını atlamak için verify=False kullanıyoruz
        response = requests.get(url, timeout=15, verify=False)
        response.raise_for_status() # Hatalı isteklerde (404, 500 vb.) hata fırlatır
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Daha önce analiz ettiğimiz etiket ve class'ları kullanarak verileri buluyoruz
        title_tag = soup.find('h1', class_='news-title')
        title = title_tag.get_text(strip=True) if title_tag else None
        
        summary_tag = soup.find('h2', class_='news-spot')
        summary = summary_tag.get_text(strip=True) if summary_tag else None
        
        article_body = soup.find('div', class_='news-content')
        article_text = ''
        if article_body:
            # Haber metniyle ilgisi olmayan etiketler bölümünü metinden çıkar
            if tags_div := article_body.find('div', class_='news-tags'):
                tags_div.decompose()
            
            # Tüm paragrafları bul ve metinlerini birleştir
            paragraphs = article_body.find_all('p')
            article_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)

        # Eğer başlık veya ana metin bulunamazsa, bu linki geçersiz say
        if not title or not article_text:
            print(f"--> Başlık veya içerik bulunamadı. URL atlanıyor: {url}")
            return None
        
        # Tarih bilgisini çek
        date_tag = soup.find('span', class_='created-date')
        # Tarih metnindeki "HABER GİRİŞ" kısmını temizle
        publish_date = date_tag.get_text(strip=True).replace("HABER GİRİŞ", "").strip() if date_tag else None

        return {
            'url': url,
            'title': title,
            'summary': summary,
            'publish_date': publish_date,
            'article_text': article_text
        }

    except Exception as e:
        print(f"Hata: {url} parse edilirken bir sorun oluştu. Sebep: {e}")
        return None

# --- Ana Script ---
if __name__ == "__main__":
    article_urls = read_urls_from_csv("urls_trt.csv")
    
    if not article_urls:
        print("İşleme devam etmek için URL bulunamadı.")
    else:
        output_csv_file = 'trt_haberler.csv'
        fieldnames = ['url', 'title', 'summary', 'publish_date', 'article_text']
        
        # CSV dosyasını yazma modunda aç
        with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader() # Başlıkları yaz
            
            total_urls = len(article_urls)
            for i, url in enumerate(article_urls):
                print(f"[{i+1}/{total_urls}] İşleniyor: {url}")
                data = parse_article_trt(url)
                
                # Eğer veri başarıyla çekildiyse dosyaya yaz
                if data:
                    writer.writerow(data)
                
                
        print(f"\nİşlem tamamlandı. Veriler '{output_csv_file}' dosyasına kaydedildi.")