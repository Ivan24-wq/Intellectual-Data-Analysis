from bs4 import BeautifulSoup as BeSo
import requests

url = 'https://www.avito.ru/ekaterinburg/mebel_i_interer/divan_raskladnoy_bu_7667869453?context=H4sIAAAAAAAA_wEmANn_YToxOntzOjE6IngiO3M6MTY6IlF1MUJaZWI1T05OU1FNZ0MiO33Eb5tHJgAAAA'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
}

response = requests.get(url, headers=headers)
print(f"Статус: {response.status_code}")

soup = BeSo(response.text, "html.parser")
price_elem = soup.find('span', {'data-marker': 'item-view/item-price'})

if price_elem:
    price_text = price_elem.get_text(strip=True)  # исправлена опечатка
    print(f"Цена: {price_text}")
else:
    print("Цена не найдена!")

#дата публикации
publishe_date = soup.find('span', {'data-marker': 'item-view/item-date'})
if publishe_date:
    publishe_text = publishe_date.get_text(strip=True)
    print(f"Дата публикации: {publishe_text}")
else:
    print("Дата публикации не найдена")

title = soup.find('h1', {'data-marker': 'item-view/title-info'})
if title:
    title_info = title.get_text(strip=True)
    print(f"Название: {title_info}")
    
id_add = soup.find('div', {'data-item-id': True})
if id_add:
    id_info = id_add.get('data-item-id')
    print(f"Id: {id_info}")