from bs4 import BeautifulSoup as BeSo
import requests
url = 'https://www.avito.ru/simferopol/mebel_i_interer/krovat_dvuspalnaya_160h200_velyur_4338589507?context=H4sIAAAAAAAA_wE_AMD_YToyOntzOjEzOiJsb2NhbFByaW9yaXR5IjtiOjA7czoxOiJ4IjtzOjE2OiJzeHBqNEhZN0VjSE9WendTIjt9cOZTUz8AAAA'
response = requests.get(url).text
soup = BeSo(response, "html.parser")
price_elem = soup.find('span', {'data-marker': 'item-view/item-price'})
if price_elem:
    price_value = price_elem.get('content')
    pricw_text = price_elem.get_text(strip=True)
    
    if price_value:
        prive_number = int(price_value)
        print(f"Цена: {pricw_text}")