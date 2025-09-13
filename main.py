import nltk
from nltk.corpus import stopwords
from collections import Counter
import string
import pymorphy2
nltk.download('stopwords')

#загрущка стоп слов
russian_stopwords = stopwords.words('russian')

#Чтение файла
with open('F:\ИАД\\paper.txt', 'r', encoding='utf-8') as file:
    text = file.read()

#Разбиваем на слова
words = text.lower().split()
cleaned_words = [] #Удаление пунктуации
for word in words:
    word = word.strip(string.punctuation + '<<>>-')
    if word and word not in russian_stopwords:
        cleaned_words.append(word) #Проверка на стопслово


#приведения слов к нормальной форме
morph = pymorphy2.MorphAnalyzer()
lemmatized_words = [morph.parse(word)[0].normal_form for word in cleaned_words]

word_counter = Counter(lemmatized_words)

#Вывод самых часто использующих слов
print("Самые используемые слова: ")
for word, count in word_counter.most_common(10):
    print(f'{word}: {count}')

#Уникальные слова
print(f"\n Уникальыне слова: {len(word_counter)}")