import string
import math
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Загрузка ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


def preprocess_text(
        text: str,
        remove_stopwords: bool = True,
        case_sensitive: bool = False,
        min_word_length: int = 2,
        language: str = "russian"
) -> List[str]:
    """
    Предварительная обработка текста с расширенными возможностями фильтрации.

    Args:
        text (str): Входной текст для обработки
        remove_stopwords (bool): Удалять ли стоп-слова
        case_sensitive (bool): Учитывать ли регистр
        min_word_length (int): Минимальная длина слова
        language (str): Язык для стоп-слов

    Returns:
        List[str]: Список обработанных токенов
    """
    # Приведение к нижнему регистру, если не требуется учет регистра
    if not case_sensitive:
        text = text.lower()

    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Токенизация
    tokens = word_tokenize(text)

    # Выбор стоп-слов
    stop_words = set()
    if remove_stopwords:
        if language in ["russian", "ru"]:
            stop_words.update(stopwords.words('russian'))
        if language in ["english", "en"]:
            stop_words.update(stopwords.words('english'))
        if language == "auto":
            stop_words.update(stopwords.words('russian') + stopwords.words('english'))

    # Фильтрация токенов
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words and len(word) >= min_word_length
    ]

    return filtered_tokens


def calculate_tf(word: str, document: List[str]) -> int:
    """
    Расчет частоты термина (Term Frequency) в документе.

    Args:
        word (str): Слово для подсчета
        document (List[str]): Список токенов документа

    Returns:
        int: Количество вхождений слова в документ
    """
    return document.count(word)


def calculate_idf(word: str, documents: List[List[str]]) -> float:
    """
    Расчет обратной частоты документа (Inverse Document Frequency).

    Args:
        word (str): Слово для подсчета
        documents (List[List[str]]): Список документов

    Returns:
        float: Значение IDF
    """
    # Количество документов, содержащих слово
    document_count = sum(1 for doc in documents if word in doc)

    # Защита от деления на ноль и случая одного документа
    total_documents = max(len(documents), 2)
    if document_count == 0:
        return 0.0

    return math.log(total_documents / document_count)


def process_multiple_files(
        file_paths: List[str],
        remove_stopwords: bool = True,
        case_sensitive: bool = False,
        min_word_length: int = 2,
        language: str = "russian"
) -> List[Dict[str, float]]:
    """
    Расчет TF-IDF для коллекции документов.

    Args:
        file_paths (List[str]): Пути к файлам
        remove_stopwords (bool): Удалять ли стоп-слова
        case_sensitive (bool): Учитывать ли регистр
        min_word_length (int): Минимальная длина слова
        language (str): Язык для обработки

    Returns:
        List[Dict[str, float]]: Список словарей со статистикой слов
    """
    document_words = []
    all_words = set()

    # Чтение и обработка документов
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

                # Предобработка текста
                tokens = preprocess_text(
                    text,
                    remove_stopwords=remove_stopwords,
                    case_sensitive=case_sensitive,
                    min_word_length=min_word_length,
                    language=language
                )

                # Подсчет слов в документе
                doc_word_count = {}
                for token in tokens:
                    doc_word_count[token] = doc_word_count.get(token, 0) + 1

                document_words.append(doc_word_count)
                all_words.update(doc_word_count.keys())

        except Exception as e:
            print(f"Ошибка при обработке файла {file_path}: {e}")

    # Проверка наличия обработанных документов
    if not document_words:
        return []

    total_documents = len(document_words)

    # Создаем список всех токенов для расчета IDF
    all_document_tokens = [list(doc.keys()) for doc in document_words]

    word_stats = []

    # Расчет метрик для каждого слова
    for word in all_words:
        # Общая частота слова (TF)
        tf = sum(doc.get(word, 0) for doc in document_words)

        # Расчет IDF с учетом случая одного документа
        idf = calculate_idf(word, all_document_tokens)

        # Расчет TF-IDF
        tfidf = tf * idf

        word_stats.append({
            'word': word,
            'tf': tf,  # Общая частота
            'idf': idf,  # Обратная документная частота
            'tfidf': tfidf  # TF-IDF
        })

    # Сортировка по IDF (по убыванию), затем по TF-IDF (по убыванию)
    word_stats.sort(key=lambda x: (-x['idf'], -x['tfidf']))

    return word_stats