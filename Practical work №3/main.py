from bs4 import BeautifulSoup as BeSo
import requests

url = 'https://almaz-meb.ru/divany/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
}

# Проверяем отвечает ли сайт
response = requests.get(url, headers=headers)
print(f"Статус: {response.status_code}")
soup = BeSo(response.text, "html.parser")

# Ссылки на товары
links = [a["href"] for a in soup.find_all("a", class_="product-title", href=True)]

# Будем парсить 3 товара
for link in links[:4]:
    if link.startswith("/"):
        product_url = "https://almaz-meb.ru" + link
    else:
        product_url = link

    res = requests.get(product_url, headers=headers)
    p_soup = BeSo(res.text, "html.parser")

    # Цена и ID
    price_tag = p_soup.find("span", id=lambda x: x and x.startswith('sec_discounted_price_'))
    if price_tag:
        price = price_tag.get_text(strip=True)
        element_id = price_tag.get("id")
        item_id = element_id.split("_")[-1]
        print(f"ID: {item_id}")
        print(f"Цена: {price}")
    else:
        print("Объявление не найдено")

    # Название
    title_tag = p_soup.find("h1", class_="ut2-pb__title")
    if title_tag:
        title = title_tag.get_text(strip=True)
        print(f"Название: {title}")
    else:
        print("Объявление не найдено")
    
    # Код товара
    code_tag = p_soup.find("span", id=lambda x: x and x.startswith("product_code_"))
    if code_tag:
        code = code_tag.get_text(strip=True)
        print(f"Код товара: {code}")
    else:
        print("Объявление не найдено")
    
    print(f"id: {item_id} | Цена: {price} | {title}, | Код товара: {code}")