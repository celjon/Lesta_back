import re
import math
import string
from collections import Counter
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def preprocess_text(text, remove_stopwords=True, case_sensitive=False, min_word_length=2, language="russian"):
    """
    Предварительная обработка текста:
    - приведение к нижнему регистру (опционально)
    - удаление знаков пунктуации
    - удаление стоп-слов (опционально)
    - токенизация
    """
    # Приведение к нижнему регистру, если не case_sensitive
    if not case_sensitive:
        text = text.lower()

    # Удаление знаков пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Токенизация
    tokens = word_tokenize(text)

    # Удаление стоп-слов, если необходимо
    if remove_stopwords:
        stop_words = set()
        if language == "russian" or language == "ru":
            stop_words.update(stopwords.words('russian'))
        if language == "english" or language == "en":
            stop_words.update(stopwords.words('english'))
        if language == "auto":
            stop_words.update(stopwords.words('russian') + stopwords.words('english'))

        filtered_tokens = [word for word in tokens if word.lower() not in stop_words and len(word) >= min_word_length]
    else:
        filtered_tokens = [word for word in tokens if len(word) >= min_word_length]

    return filtered_tokens


def calculate_tf(word, document):
    """
    Расчет TF (частота термина) для слова в документе.
    TF = (Количество повторений слова в документе) / (Общее количество слов в документе)
    """
    word_count = document.count(word)
    return word_count


def calculate_idf(word, documents_list):
    """
    Расчет IDF (обратная частота документа) для слова.
    IDF = log(Общее количество документов / Количество документов, содержащих слово)
    """
    # Для одного документа можно разделить его на части, например, на абзацы
    doc_count = sum(1 for doc in documents_list if word in doc)
    if doc_count == 0:
        return 0
    return math.log(len(documents_list) / doc_count)


def process_text_file(file_path, remove_stopwords=True, case_sensitive=False, min_word_length=2, language="russian"):
    """
    Обработка текстового файла и расчет TF-IDF для всех слов.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Разделение текста на "документы" (абзацы или предложения)
    # Для простоты можно считать каждый абзац отдельным документом
    documents = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]

    # Если текст не содержит явных абзацев, разделим его на предложения
    if len(documents) <= 1:
        documents = [sent.strip() for sent in sent_tokenize(text) if sent.strip()]

    # Предобработка каждого документа с учетом параметров
    processed_docs = [preprocess_text(
        doc,
        remove_stopwords=remove_stopwords,
        case_sensitive=case_sensitive,
        min_word_length=min_word_length,
        language=language
    ) for doc in documents]

    # Получение всех уникальных слов
    all_words = set()
    for doc in processed_docs:
        all_words.update(doc)

    # Подсчет TF и IDF для каждого слова
    word_stats = []

    # Объединяем все токены для подсчета общей частоты
    all_tokens = [token for doc in processed_docs for token in doc]
    total_words = len(all_tokens)
    word_counts = Counter(all_tokens)

    for word in all_words:
        tf = word_counts[word]
        idf = calculate_idf(word, processed_docs)
        tfidf = tf * idf

        word_stats.append({
            'word': word,
            'tf': tf,
            'idf': idf,
            'tfidf': tfidf
        })

    # Сортировка по убыванию IDF
    word_stats = sorted(word_stats, key=lambda x: x['idf'], reverse=True)

    return word_stats


def calculate_tfidf_sklearn(file_path, remove_stopwords=True, min_word_length=2, language="russian"):
    """
    Альтернативный метод расчета TF-IDF с использованием sklearn.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Разделение текста на "документы"
    documents = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]
    if len(documents) <= 1:
        documents = [sent.strip() for sent in sent_tokenize(text) if sent.strip()]

    # Подготовка стоп-слов
    stop_words = None
    if remove_stopwords:
        if language == "russian" or language == "ru":
            stop_words = stopwords.words('russian')
        elif language == "english" or language == "en":
            stop_words = stopwords.words('english')
        elif language == "auto":
            stop_words = stopwords.words('russian') + stopwords.words('english')

    # Регулярное выражение для токенизации с учетом min_word_length
    token_pattern = rf'\b[а-яА-Яa-zA-Z]{{{min_word_length},}}\b'

    # Используем TfidfVectorizer из sklearn
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        token_pattern=token_pattern
    )

    # Применяем векторизатор к нашим документам
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Получаем имена функций (слова)
    feature_names = vectorizer.get_feature_names_out()

    # Создаем DataFrame для хранения результатов
    results = []

    # Получаем средние значения TF-IDF для каждого слова по всем документам
    tfidf_means = tfidf_matrix.mean(axis=0).A1

    # Получаем document frequency для расчета IDF
    df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names
    ).astype(bool).sum(axis=0)

    # Расчет IDF
    idf = pd.Series(
        {term: math.log(len(documents) / (df[term])) for term in feature_names}
    )

    # Считаем общую частоту слов в зависимости от настроек регистра
    term_counts = Counter()
    for doc in documents:
        if language == "russian" or language == "ru":
            pattern = rf'\b[а-яА-Я]{{{min_word_length},}}\b'
        elif language == "english" or language == "en":
            pattern = rf'\b[a-zA-Z]{{{min_word_length},}}\b'
        else:  # auto или другие
            pattern = rf'\b[а-яА-Яa-zA-Z]{{{min_word_length},}}\b'

        words = re.findall(pattern, doc.lower())
        term_counts.update(words)

    for term in feature_names:
        if term in term_counts:
            results.append({
                'word': term,
                'tf': term_counts[term],
                'idf': idf[term],
                'tfidf': term_counts[term] * idf[term]
            })

    # Сортировка по убыванию IDF
    results = sorted(results, key=lambda x: x['idf'], reverse=True)

    return results


def extract_keywords(text, top_n=10, remove_stopwords=True, language="auto"):
    """
    Извлекает ключевые слова из текста на основе TF-IDF
    """
    # Разделение на предложения
    sentences = sent_tokenize(text)

    # Подготовка стоп-слов
    stop_words = None
    if remove_stopwords:
        if language == "russian" or language == "ru":
            stop_words = stopwords.words('russian')
        elif language == "english" or language == "en":
            stop_words = stopwords.words('english')
        elif language == "auto":
            stop_words = stopwords.words('russian') + stopwords.words('english')

    # Используем TfidfVectorizer
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r'\b[а-яА-Яa-zA-Z]{2,}\b'
    )

    # Применяем к предложениям
    tfidf_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    # Суммируем значения TF-IDF по всем предложениям для каждого слова
    tfidf_sums = tfidf_matrix.sum(axis=0).A1

    # Создаем словарь {слово: значение}
    keywords = {feature_names[i]: tfidf_sums[i] for i in range(len(feature_names))}

    # Сортируем слова по значению TF-IDF и возвращаем топ
    sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


def similar_words(word, words_data, top_n=5):
    """
    Находит слова, похожие на указанное,
    на основе простого сравнения расстояния Левенштейна
    """
    from nltk.metrics.distance import edit_distance

    # Получаем все слова из данных
    all_words = [item['word'] for item in words_data]

    # Рассчитываем расстояние Левенштейна между указанным словом и всеми остальными
    distances = [(w, edit_distance(word, w)) for w in all_words if w != word]

    # Сортируем по расстоянию (меньше = более похоже)
    sorted_distances = sorted(distances, key=lambda x: x[1])

    # Возвращаем топ наиболее похожих слов
    return [w for w, d in sorted_distances[:top_n]]