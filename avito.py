from bs4 import BeautifulSoup as BeSo
import requests
import time
import random
import re

def parse_all_sofa_pages():
    base_url = "https://www.avito.ru/simferopol/divany_i_kresla"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }
    
    all_ads = []
    page = 1
    
    while True:
        # Формируем URL для каждой страницы
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}?p={page}"
        
        print(f"Парсим страницу {page}...")
        
        try:
            # Задержка между запросами
            time.sleep(random.uniform(2, 4))
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Ошибка: статус код {response.status_code}")
                break
                
            soup = BeSo(response.text, 'html.parser')
            
            # Проверяем блокировку
            if "Доступ ограничен" in response.text:
                print("Доступ ограничен. Попробуйте позже.")
                break
            
            # Ищем все объявления на странице
            ads = soup.find_all('div', {'data-marker': 'item'})
            
            if not ads:
                print("Объявления не найдены. Конец выдачи.")
                break
                
            print(f"Найдено {len(ads)} объявлений")
            
            # Парсим каждое объявление
            for ad in ads:
                ad_data = parse_ad_from_list(ad)
                if ad_data:
                    all_ads.append(ad_data)
                    print(f"  - ID: {ad_data['id']}, Цена: {ad_data['price']}")
            
            # Проверяем наличие следующей страницы
            next_button = soup.find('a', {'data-marker': 'pagination/next'})
            if not next_button or 'disabled' in next_button.get('class', []):
                print("Достигнута последняя страница")
                break
                
            page += 1
            
        except Exception as e:
            print(f"Ошибка при парсинге страницы {page}: {e}")
            break
    
    return all_ads

def parse_ad_from_list(ad):
    """Парсит данные объявления из списка"""
    try:
        # ID объявления
        ad_id = ad.get('data-item-id', '')
        if not ad_id:
            return None
        
        # Название
        title_elem = ad.find('a', {'data-marker': 'item-title'})
        title = title_elem.get_text(strip=True) if title_elem else "Название не указано"
        
        # Цена
        price_elem = ad.find('span', {'data-marker': 'item-price'})
        price = price_elem.get_text(strip=True) if price_elem else "Цена не указана"
        
        # Срок публикации
        date_elem = ad.find('div', {'data-marker': 'item-date'})
        date = date_elem.get_text(strip=True) if date_elem else "Дата не указана"
        
        return {
            'id': ad_id,
            'title': title,
            'price': price,
            'date': date
        }
        
    except Exception as e:
        print(f"Ошибка парсинга объявления: {e}")
        return None

def save_results(ads, filename='sofas_simferopol.txt'):
    """Сохраняет результаты в файл"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Всего собрано объявлений: {len(ads)}\n\n")
        for ad in ads:
            f.write(f"ID: {ad['id']}\n")
            f.write(f"Название: {ad['title']}\n")
            f.write(f"Цена: {ad['price']}\n")
            f.write(f"Дата публикации: {ad['date']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"Результаты сохранены в {filename}")

# Основной код
if __name__ == "__main__":
    print("Начинаем сбор всех объявлений о диванах в Симферополе...")
    
    all_ads = parse_all_sofa_pages()
    
    print(f"\n=== РЕЗУЛЬТАТЫ ===")
    print(f"Всего собрано объявлений: {len(all_ads)}")
    
    if all_ads:
        # Сохраняем в файл
        save_results(all_ads)
        
        # Выводим первые 5 объявлений для примера
        print("\nПервые 5 объявлений:")
        for i, ad in enumerate(all_ads[:5], 1):
            print(f"{i}. ID: {ad['id']}")
            print(f"   Название: {ad['title']}")
            print(f"   Цена: {ad['price']}")
            print(f"   Дата: {ad['date']}\n")
    else:
        print("Не удалось собрать объявления")