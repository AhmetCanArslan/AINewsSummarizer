from fileinput import filename
import requests
import xml.etree.ElementTree as ET
import csv

URL_LIMIT = 5000 # Her sitemap dosyasından çekilecek maksimum URL sayısı, bu kadar veri çekmek istiyoruz

def fetch_urls_from_sitemap(sitemap_url):
    """Verilen sitemap url'sinden tüm <loc> etiketlerindeki URL'leri çeker."""

    try:
        print(f"Sitemap'ten URL'ler çekiliyor: {sitemap_url}")
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()  # HTTP hatası olunca exception fırlatır

        root = ET.fromstring(response.content)# XML içeriğini ayrıştırır
        namespace = {'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9'} # Namespace tanımı

        urls = []
        # Tüm <loc> etiketlerini bul ve URL'leri listeye ekle
        for url_element in root.findall('sitemap:url', namespace):

            if len(urls) >= URL_LIMIT:
                print(f"Belirlenen {URL_LIMIT} URL limitine ulaşıldı. Arama durduruluyor.")
                break

            loc = url_element.find('sitemap:loc', namespace)
            if loc is not None:
                urls.append(loc.text) # boş değilse URL'yi listeye ekle

        return urls
    
    #detaylı exception handling
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return urls
    except requests.Timeout:
        print("İstek zaman aşımına uğradı.")
        return urls
    except requests.RequestException as e:
        print(f"İstek hatası: {e}")
        return urls
    except ET.ParseError:
        print("XML ayrıştırma hatası.")
        return urls 
    
def save_urls_to_csv(urls, csv_filename):
    """URL listesini CSV dosyasına kaydeder."""

    if not urls:
        print("Kaydedilecek URL bulunamadı.")
        return

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['url'])
        for url in urls:
            writer.writerow([url])
    
    print(f"Tüm URL'ler başarıyla '{filename}' dosyasına kaydedildi.")    

if __name__ == "__main__":
    
    SITEMAP_URL = "https://www.anahaberajansi.com.tr/post-sitemap.xml"
    
    # 1. Adım: Site haritasından URL'leri çek
    all_urls = fetch_urls_from_sitemap(SITEMAP_URL)
    
    # 2. Adım: Çekilen URL'leri CSV dosyasına kaydet
    save_urls_to_csv(all_urls, "sitemap_urls.csv")
