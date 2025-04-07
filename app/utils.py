import os
import re
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk

# Загрузка необходимых ресурсов NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def detect_language(text):
    """
    Определяет язык текста
    """
    # Более точная эвристика для определения языка
    russian_chars = len(re.findall(r'[а-яА-Я]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    japanese_chars = len(re.findall(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]', text))

    total_chars = russian_chars + english_chars + japanese_chars

    if total_chars == 0:
        return "unknown", 0

    if japanese_chars > 0 and japanese_chars / total_chars > 0.3:
        return "ja", japanese_chars / total_chars
    elif russian_chars > english_chars:
        return "ru", russian_chars / total_chars
    else:
        return "en", english_chars / total_chars


def detect_languages_distribution(text):
    """
    Определяет распределение языков в тексте по абзацам
    """
    # Разделение на абзацы
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    # Анализ языка каждого абзаца
    language_counts = Counter()
    total_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
        lang, _ = detect_language(paragraph)
        language_counts[lang] += 1

    # Расчет процентного соотношения
    language_distribution = {}
    for lang, count in language_counts.items():
        language_distribution[lang] = round(count / total_paragraphs * 100, 2)

    return language_distribution


def count_sentences(text):
    """
    Подсчитывает количество предложений в тексте
    """
    sentences = sent_tokenize(text)
    return len(sentences)


def count_syllables_en(word):
    """
    Подсчет слогов в английском слове (приблизительно)
    """
    word = word.lower()
    if len(word) <= 3:
        return 1

    # Удаление окончания 'e'
    if word.endswith('e'):
        word = word[:-1]

    # Подсчет гласных как приближение к слогам
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel

    return max(1, count)


def get_readability_stats(text):
    """
    Рассчитывает статистику читабельности текста
    """
    # Подсчет слов, предложений, символов
    sentences = sent_tokenize(text)
    words = text.split()

    sentence_count = len(sentences)
    word_count = len(words)
    char_count = len(text)

    # Средняя длина предложения в словах
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Средняя длина слова в символах
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

    # Расчет индекса Флеша-Кинкейда (для английского языка)
    lang, _ = detect_language(text)
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


def generate_wordcloud_image(words_data, max_words=100):
    """
    Создает изображение облака слов на основе данных TF-IDF
    """
    try:
        import matplotlib.pyplot as plt
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

        # Создаем изображение
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        # Сохраняем изображение в буфер
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Кодируем изображение в base64 для отображения на странице
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()

        return encoded_img
    except ImportError:
        return None


def generate_tfidf_chart(words_data, top_n=20):
    """
    Создает график сравнения TF и IDF для топ-N слов
    """
    try:
        data = words_data[:top_n]

        # Создаем DataFrame из данных
        df = pd.DataFrame([
            {
                'word': item['word'],
                'tf': item['tf'],
                'idf': item['idf'],
                'tfidf': item['tfidf']
            } for item in data
        ])

        # Создаем график
        plt.figure(figsize=(12, 6))

        # Строим бар-график с двойной осью Y
        ax1 = plt.subplot(111)
        bar_width = 0.35

        # TF на левой оси
        bars1 = ax1.bar(df.index - bar_width / 2, df['tf'], bar_width, label='TF', color='skyblue')
        ax1.set_xlabel('Слова')
        ax1.set_ylabel('TF (частота)', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        # IDF на правой оси
        ax2 = ax1.twinx()
        bars2 = ax2.bar(df.index + bar_width / 2, df['idf'], bar_width, label='IDF', color='lightcoral')
        ax2.set_ylabel('IDF', color='firebrick')
        ax2.tick_params(axis='y', labelcolor='firebrick')

        # Настройка оси X
        ax1.set_xticks(df.index)
        ax1.set_xticklabels(df['word'], rotation=45, ha='right')

        # Добавляем легенду
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Сравнение TF и IDF для топ слов')
        plt.tight_layout()

        # Сохраняем изображение в буфер
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        # Кодируем изображение в base64 для отображения на странице
        encoded_img = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close()

        return encoded_img
    except Exception as e:
        print(f"Ошибка при создании графика: {str(e)}")
        return None


def export_to_csv(words_data, filename="tfidf_results.csv"):
    """
    Экспортирует результаты анализа в CSV файл
    """
    df = pd.DataFrame([
        {
            'word': item['word'],
            'tf': item['tf'],
            'idf': item['idf'],
            'tfidf': item['tfidf']
        } for item in words_data
    ])

    # Создаем временный файл
    temp_file = os.path.join(tempfile.gettempdir(), filename)

    # Экспортируем с корректными заголовками и разделителями
    # Используем точку с запятой как разделитель (стандарт для европейских/русских Excel)
    df.to_csv(temp_file, index=False, encoding='utf-8-sig', sep=';',
              header=['Слово', 'TF (частота)', 'IDF', 'TF-IDF'])

    return temp_file


def export_to_excel(words_data, filename="tfidf_results.xlsx"):
    """
    Экспортирует результаты анализа в Excel файл
    """
    df = pd.DataFrame([
        {
            'word': item['word'],
            'tf': item['tf'],
            'idf': item['idf'],
            'tfidf': item['tfidf']
        } for item in words_data
    ])

    # Переименовываем столбцы
    df.columns = ['Слово', 'TF (частота)', 'IDF', 'TF-IDF']

    # Создаем временный файл
    temp_file = os.path.join(tempfile.gettempdir(), filename)

    # Экспортируем как Excel
    df.to_excel(temp_file, index=False, engine='openpyxl')

    return temp_file