from bs4 import BeautifulSoup as BeSo
import requests
url = 'https://www.avito.ru/?ysclid=mgb89jzvdl330113690'
response = requests.get(url).text
print(response)