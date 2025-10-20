import requests
import xml.etree.ElementTree as ET
import csv
import urllib3

# SSL uyarısını bastırmak için
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Çekmek istediğimiz maksimum URL sayısı
URL_LIMIT = 5000
# Ana site haritası endeksinin adresi
ROOT_SITEMAP_URL = "https://www.anahaberajansi.com.tr/sitemap.xml"

def get_article_urls_from_all_sitemaps():
    """
    Ana sitemap endeksinden başlayarak tüm alt sitemap'leri gezer
    ve belirlenen limite kadar haber URL'lerini toplar.
    """
    all_article_urls = []
    
    try:
        print(f"Ana sitemap endeksi indiriliyor: {ROOT_SITEMAP_URL}")
        response = requests.get(ROOT_SITEMAP_URL, timeout=15, verify=False)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Endeks içindeki tüm alt sitemap linklerini bul
        sitemap_links = [
            elem.text for elem in root.findall('sitemap:sitemap/sitemap:loc', namespace)
            if 'post-sitemap' in elem.text # Sadece haberleri içerenleri hedef al
        ]
        
        print(f"Toplam {len(sitemap_links)} adet 'post' sitemap'i bulundu. Tarama başlıyor...")
        
        # Her bir alt sitemap'i gez
        for sitemap_url in sitemap_links:

            print(f"-> Alt sitemap işleniyor: {sitemap_url}")
            sub_response = requests.get(sitemap_url, timeout=15, verify=False)
            sub_root = ET.fromstring(sub_response.content)
            
            # Bu alt sitemap içindeki haber linklerini topla
            article_links_in_sub = [
                elem.text for elem in sub_root.findall('sitemap:url/sitemap:loc', namespace)
            ]
            
            for article_url in article_links_in_sub:
                all_article_urls.append(article_url)
                if len(all_article_urls) >= URL_LIMIT:
                    break # Limite ulaştıysak bu sitemap'i işlemeyi bırak
        
        print(f"Tarama tamamlandı. Toplam {len(all_article_urls)} adet haber URL'si çekildi.")
        return all_article_urls[:URL_LIMIT] # Tam olarak limitte döndüğünden emin ol

    except requests.exceptions.RequestException as e:
        print(f"Hata: Sitemap'e ulaşılamadı. Sebep: {e}")
        return all_article_urls
    except ET.ParseError as e:
        print(f"Hata: XML parse edilirken bir sorun oluştu. Sebep: {e}")
        return all_article_urls

def save_urls_to_csv(urls, filename="urls.csv"):
    if not urls:
        print("Kaydedilecek URL bulunamadı.")
        return
        
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['url'])
        for url in urls:
            writer.writerow([url])
    
    print(f"Tüm URL'ler başarıyla '{filename}' dosyasına kaydedildi.")

# --- Ana Script ---
if __name__ == "__main__":
    # 1. Adım: Tüm sitemap'lerden URL'leri çek
    final_urls = get_article_urls_from_all_sitemaps()
    
    # 2. Adım: Çekilen URL'leri CSV dosyasına kaydet
    save_urls_to_csv(final_urls)