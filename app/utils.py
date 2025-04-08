import os
import re
import math
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use('Agg')  # Используем backend без GUI

import nltk
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def detect_language(text: str) -> Tuple[str, float]:
    """
    Определяет основной язык текста с учетом процентного соотношения символов.

    Args:
        text (str): Текст для анализа

    Returns:
        Кортеж (язык, доля символов)
    """
    # Подсчет символов разных алфавитов
    russian_chars = len(re.findall(r'[а-яА-Я]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', text))

    total_chars = russian_chars + english_chars + japanese_chars

    if total_chars == 0:
        return "unknown", 0.0

    # Определение доминирующего языка
    if japanese_chars > 0 and japanese_chars / total_chars > 0.3:
        return "ja", japanese_chars / total_chars
    elif russian_chars > english_chars:
        return "ru", russian_chars / total_chars
    else:
        return "en", english_chars / total_chars


def detect_languages_distribution(text: str) -> Dict[str, float]:
    """
    Анализирует распределение языков в тексте по абзацам.

    Args:
        text (str): Текст для анализа

    Returns:
        Словарь с процентным распределением языков
    """
    # Разделение на абзацы
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Подсчет языков
    language_counts = {}
    total_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
        lang, _ = detect_language(paragraph)
        language_counts[lang] = language_counts.get(lang, 0) + 1

    # Расчет процентного соотношения
    return {
        lang: round(count / total_paragraphs * 100, 2)
        for lang, count in language_counts.items()
    }


def count_syllables_en(word: str) -> int:
    """
    Приблизительный подсчет слогов в английском слове.

    Args:
        word (str): Слово для подсчета слогов

    Returns:
        Количество слогов
    """
    word = word.lower()
    if len(word) <= 3:
        return 1

    # Удаление окончания 'e'
    if word.endswith('e'):
        word = word[:-1]

    # Подсчет гласных как приближение к слогам
    vowels = "aeiouy"
    return max(1, sum(
        1 for i in range(1, len(word))
        if word[i] in vowels and word[i - 1] not in vowels
    ))


def get_readability_stats(text: str) -> Dict[str, Optional[float]]:
    """
    Расчет статистики читабельности текста.

    Args:
        text (str): Текст для анализа

    Returns:
        Словарь со статистическими показателями
    """
    # Подсчет слов, предложений, символов
    import nltk
    sentences = nltk.sent_tokenize(text)
    words = text.split()

    sentence_count = len(sentences)
    word_count = len(words)
    char_count = len(text)

    # Расчет средних значений
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    # Расчет индекса Флеша-Кинкейда (только для английского)
    lang, _ = detect_language(text)
    flesch_reading_ease = None
    flesch_kincaid_grade = None

    if lang == 'en':
        syllable_count = sum(count_syllables_en(word) for word in words)
        if sentence_count > 0 and word_count > 0:
            flesch_reading_ease = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
            flesch_kincaid_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        else:
            flesch_reading_ease = 0
            flesch_kincaid_grade = 0
    else:
        flesch_reading_ease = None
        flesch_kincaid_grade = None

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "char_count": char_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_word_length": round(avg_word_length, 2),
        "flesch_reading_ease": round(flesch_reading_ease, 2) if flesch_reading_ease is not None else None,
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 2) if flesch_kincaid_grade is not None else None,
    }


def generate_base64_image(plt_figure: plt.Figure, format: str = 'png') -> Optional[str]:
    """
    Преобразует matplotlib figure в base64 изображение.

    Args:
        plt_figure (plt.Figure): Matplotlib фигура
        format (str): Формат изображения

    Returns:
        Base64 закодированное изображение или None
    """
    try:
        # Сохраняем изображение в буфер
        img_buffer = BytesIO()
        plt_figure.savefig(img_buffer, format=format)
        img_buffer.seek(0)

        # Кодируем изображение в base64
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(plt_figure)

        return encoded_img
    except Exception as e:
        print(f"Ошибка при создании изображения: {e}")
        return None


def generate_wordcloud_image(words_data: List[Dict], max_words: int = 100) -> Optional[str]:
    """
    Создает изображение облака слов на основе данных TF-IDF.

    Args:
        words_data (List[Dict]): Данные о словах
        max_words (int): Максимальное количество слов

    Returns:
        Base64 закодированное изображение или None
    """
    try:
        from wordcloud import WordCloud

        # Создаем словарь {слово: частота}
        word_freq = {item['word']: item['tf'] for item in words_data[:max_words]}

        # Создаем облако слов
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            contour_width=1,
            contour_color='steelblue'
        ).generate_from_frequencies(word_freq)

        # Создаем и сохраняем изображение
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        return generate_base64_image(plt.gcf())
    except ImportError:
        print("WordCloud библиотека не установлена")
        return None


def generate_tfidf_chart(words_data: List[Dict], top_n: int = 20) -> Optional[str]:
    """
    Создает график сравнения TF и IDF для топ-N слов.

    Args:
        words_data (List[Dict]): Данные о словах
        top_n (int): Количество слов для визуализации

    Returns:
        Base64 закодированное изображение или None
    """
    try:
        # Подготовка данных
        data = words_data[:top_n]
        df = pd.DataFrame([
            {
                'word': item['word'],
                'tf': item['tf'],
                'idf': item['idf'],
                'tfidf': item['tfidf']
            } for item in data
        ])

        # Создание графика
        plt.figure(figsize=(12, 6))
        ax1 = plt.subplot(111)
        bar_width = 0.35

        # TF на левой оси
        ax1.bar(df.index - bar_width / 2, df['tf'], bar_width, label='TF', color='skyblue')
        ax1.set_xlabel('Слова')
        ax1.set_ylabel('TF (частота)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # IDF на правой оси
        ax2 = ax1.twinx()
        ax2.bar(df.index + bar_width / 2, df['idf'], bar_width, label='IDF', color='lightcoral')
        ax2.set_ylabel('IDF', color='firebrick')
        ax2.tick_params(axis='y', labelcolor='firebrick')

        # Настройка осей
        ax1.set_xticks(df.index)
        ax1.set_xticklabels(df['word'], rotation=45, ha='right')

        # Легенда
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Сравнение TF и IDF для топ слов')
        plt.tight_layout()

        return generate_base64_image(plt.gcf())
    except Exception as e:
        print(f"Ошибка при создании графика: {e}")
        return None


def export_to_csv(words_data: List[Dict], filename: str = "tfidf_results.csv") -> str:
    """
    Экспортирует результаты анализа в CSV файл.

    Args:
        words_data (List[Dict]): Данные о словах
        filename (str): Имя файла для экспорта

    Returns:
        Путь к созданному CSV файлу
    """
    # Создаем DataFrame с корректными столбцами
    df = pd.DataFrame([
        {
            'Слово': item['word'],
            'TF (частота)': item['tf'],
            'IDF': item['idf'],
            'TF-IDF': item['tfidf']
        } for item in words_data
    ])

    # Создаем временный файл
    temp_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'downloads', filename)

    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)

    # Экспортируем с корректными параметрами
    df.to_csv(
        temp_file,
        index=False,
        encoding='utf-8-sig',
        sep=';',
        decimal='.',
        float_format='%.4f'
    )

    return temp_file


def export_to_excel(words_data: List[Dict], filename: str = "tfidf_results.xlsx") -> str:
    """
    Экспортирует результаты анализа в Excel файл.

    Args:
        words_data (List[Dict]): Данные о словах
        filename (str): Имя файла для экспорта

    Returns:
        Путь к созданному Excel файлу
    """
    # Создаем DataFrame с корректными столбцами
    df = pd.DataFrame([
        {
            'Слово': item['word'],
            'TF (частота)': item['tf'],
            'IDF': item['idf'],
            'TF-IDF': item['tfidf']
        } for item in words_data
    ])

    # Создаем временный файл
    temp_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'downloads', filename)

    # Создаем директорию, если она не существует
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)

    # Экспортируем в Excel
    df.to_excel(
        temp_file,
        index=False,
        sheet_name='TF-IDF результаты'
    )

    return temp_file