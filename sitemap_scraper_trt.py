import requests
import xml.etree.ElementTree as ET
import csv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Toplamak istediğimiz maksimum URL sayısı
URL_LIMIT = 3000

# TRT'nin arşiv sitemap'lerini listeleyen ana endeks dosyaları
TOP_LEVEL_SITEMAPS = [
    "https://www.trthaber.com/sitemaps/news.xml",
    "https://www.trthaber.com/sitemaps/trt-news.xml"
]

def fetch_all_urls():
    """
    İki kademeli tarama yaparak tüm haber URL'lerini toplar.
    """
    unique_article_urls = set()
    
    print("1. Kademe: Aylık arşiv sitemap'leri toplanıyor...")
    monthly_archive_urls = []
    
    # Ana endeksleri gezerek aylık arşiv linklerini topla
    for index_url in TOP_LEVEL_SITEMAPS:
        try:
            response = requests.get(index_url, timeout=15, verify=False)
            root = ET.fromstring(response.content)
            # Endeks içindeki diğer .xml linklerini bul
            for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                if loc.text.endswith('.xml'):
                    monthly_archive_urls.append(loc.text)
        except Exception as e:
            print(f"Hata: {index_url} işlenemedi. Sebep: {e}")

    print(f"Toplam {len(monthly_archive_urls)} adet aylık arşiv sitemap'i bulundu.")
    print("\n2. Kademe: Aylık arşivlerden haber linkleri çekiliyor...")

    # Aylık arşivleri gezerek haber linklerini topla
    for archive_url in monthly_archive_urls:
        if len(unique_article_urls) >= URL_LIMIT:
            print("Limite ulaşıldı, tarama durduruluyor.")
            break
        try:
            print(f"-> Arşiv işleniyor: {archive_url}")
            response = requests.get(archive_url, timeout=15, verify=False)
            root = ET.fromstring(response.content)
            
            for loc in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc'):
                # Sadece haber linklerini al, diğer .xml linklerini atla
                if loc.text.endswith('.html'):
                    unique_article_urls.add(loc.text)
                    if len(unique_article_urls) % 100 == 0: # Her 100 linkte bir bilgi ver
                         print(f"   {len(unique_article_urls)} link toplandı...")
                    if len(unique_article_urls) >= URL_LIMIT:
                        break
        except Exception as e:
            print(f"Hata: {archive_url} işlenemedi. Sebep: {e}")

    print(f"\nTarama tamamlandı. Toplam {len(unique_article_urls)} adet uniek URL bulundu.")
    return list(unique_article_urls)

def save_urls_to_csv(urls, filename="urls_trt.csv"):
    if not urls:
        print("Kaydedilecek URL bulunamadı.")
        return
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['url'])
        writer.writerows([[url] for url in urls]) # Daha hızlı yazma
    print(f"Tüm URL'ler başarıyla '{filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    final_urls = fetch_all_urls()
    save_urls_to_csv(final_urls)