import requests
import time
from bs4 import BeautifulSoup
import csv
import os
import urllib3

# SSL uyarısını bastırmak için
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def read_urls_from_csv(filename="../urls/urls.csv"):
    if not os.path.exists(filename):
        print(f"Hata: '{filename}' dosyası bulunamadı.")
        return []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        return [row[0] for row in reader]

def parse_article(url):
    """
    anahaberajansi.com.tr'nin HTML yapısına göre verileri ayrıştırır.
    """
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 1. Başlık: Bu sitede başlık <h1 class="name post-title entry-title"> içinde
        title_tag = soup.find('h1', class_='post-title')
        title = title_tag.get_text(strip=True) if title_tag else None
        
        # 2. Ana Metin: İçerik <div class="entry-content entry clearfix"> içinde
        article_body = soup.find('div', class_='entry-content')
        
        # 3. Özet: Bu sitede belirgin bir özet alanı yok.
        # Bu yüzden ana metnin ilk paragrafını özet olarak alacağız.
        summary = ''
        article_text = ''
        
        if article_body:
            # İstenmeyen elementleri (script, style etiketleri vb.) temizle
            for unwanted_tag in article_body.find_all(['script', 'style']):
                unwanted_tag.decompose()
            
            # Tüm paragrafları bul
            paragraphs = article_body.find_all('p')
            
            if paragraphs:
                # İlk paragrafı özet olarak al
                summary = paragraphs[0].get_text(strip=True)
                # Tüm paragrafları birleştirerek ana metni oluştur
                article_text = '\n'.join(p.get_text(strip=True) for p in paragraphs)

        # Temel veriler yoksa bu linki atla
        if not title or not article_text:
            print("--> Başlık veya içerik bulunamadı. Bu URL atlanıyor.")
            return None

        # Tarih bilgisi bu sitede kolayca çekilebilecek bir formatta değil, şimdilik boş bırakıyoruz.
        publish_date = None

        return {'url': url, 'title': title, 'summary': summary, 'publish_date': publish_date, 'article_text': article_text}

    except requests.exceptions.RequestException as e:
        print(f"Hata: {url} adresine ulaşılamadı. Sebep: {e}")
        return None
    except Exception as e:
        print(f"Hata: {url} parse edilirken bir sorun oluştu. Sebep: {e}")
        return None

# --- Ana Script---
if __name__ == "__main__":
    article_urls = read_urls_from_csv("../urls/urls.csv")
    
    if not article_urls:
        print("İşleme devam etmek için URL bulunamadı. Script sonlandırılıyor.")
    else:
        output_csv_file = '../data/anadolu_ajansi_haberler.csv'
        fieldnames = ['url', 'title', 'summary', 'publish_date', 'article_text']
        
        with open(output_csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            total_urls = len(article_urls)
            successful_scrapes = 0
            for i, url in enumerate(article_urls):
                print(f"[{i+1}/{total_urls}] İşleniyor: {url}")
                data = parse_article(url)
                
                if data:
                    writer.writerow(data)
                    successful_scrapes += 1
                
                
        print(f"\nİşlem tamamlandı. {total_urls} linkten {successful_scrapes} tanesi başarıyla çekilip '{output_csv_file}' dosyasına kaydedildi.")