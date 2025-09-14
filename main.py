from newspaper import Article
import string

url = 'https://habr.com/ru/companies/selectel/articles/946726/'


article = Article(url)

article.download()
article.parse()

text = article.text
#Очищаем текст
clean_text = "".join(ch for ch in text if ch not in string.punctuation)
print(article.title)
print(clean_text)