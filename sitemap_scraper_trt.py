import requests
import xml.etree.ElementTree as ET
import csv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Toplamak istediğimiz maksimum URL sayısı
URL_LIMIT = 3000

# TRT'nin robots.txt dosyasında listelenen ve haber içerme potansiyeli olan sitemap'ler
SITEMAP_URLS_TO_CHECK = [
    "https://www.trthaber.com/sitemap_haber.xml",
    "https://www.trthaber.com/sitemaps/news.xml",
    "https://www.trthaber.com/sitemaps/trt-news.xml"
]

def fetch_urls_from_sitemaps():
    """
    Belirlenen sitemap listesindeki tüm linkleri, limite ulaşana kadar çeker.
    Mükerrer linkleri önlemek için set kullanır.
    """
    unique_urls = set()
    
    for sitemap_url in SITEMAP_URLS_TO_CHECK:
        if len(unique_urls) >= URL_LIMIT:
            print("Limite ulaşıldı, daha fazla sitemap taranmayacak.")
            break
            
        try:
            print(f"Site haritası indiriliyor: {sitemap_url}")
            response = requests.get(sitemap_url, timeout=15, verify=False)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            # Bu sitemap içindeki tüm <loc> etiketlerini bul
            for url_element in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                unique_urls.add(url_element.text)
                if len(unique_urls) >= URL_LIMIT:
                    break
            
            print(f"-> Bu sitemap'ten sonra toplam {len(unique_urls)} uniek URL bulundu.")

        except Exception as e:
            print(f"Hata: {sitemap_url} işlenirken bir sorun oluştu. Sebep: {e}")
            continue # Bir sitemap hata verirse diğerine geç

    print(f"\nTarama tamamlandı. Toplam {len(unique_urls)} adet uniek URL bulundu.")
    return list(unique_urls)[:URL_LIMIT] # Limiti aşmadığından emin ol

def save_urls_to_csv(urls, filename="urls_trt.csv"):
    if not urls:
        print("Kaydedilecek URL bulunamadı.")
        return
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['url'])
        for url in urls:
            writer.writerow([url])
    print(f"Tüm URL'ler başarıyla '{filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    all_urls = fetch_urls_from_sitemaps()
    save_urls_to_csv(all_urls)