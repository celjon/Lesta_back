from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator


class FileType(str):
    """Перечисление поддерживаемых типов файлов."""
    TXT = "txt"
    CSV = "csv"


class AnalysisOptions(BaseModel):
    """Параметры для анализа текста."""
    remove_stopwords: bool = Field(
        default=True,
        description="Удалять ли стоп-слова из текста"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Учитывать ли регистр слов"
    )
    min_word_length: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Минимальная длина слова для анализа"
    )
    max_features: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Максимальное количество анализируемых слов"
    )
    language: str = Field(
        default="russian",
        description="Язык для стоп-слов и анализа"
    )

    @field_validator('language')
    @classmethod
    def validate_language(cls, lang):
        """Валидация выбранного языка."""
        valid_languages = ['russian', 'ru', 'english', 'en', 'auto']
        if lang.lower() not in valid_languages:
            raise ValueError(f"Неподдерживаемый язык. Допустимые значения: {', '.join(valid_languages)}")
        return lang.lower()


class WordData(BaseModel):
    """Данные о слове после анализа TF-IDF."""
    word: str = Field(description="Слово")
    tf: int = Field(ge=0, description="Частота термина (Term Frequency)")
    idf: float = Field(ge=0, description="Обратная частота документа (Inverse Document Frequency)")
    tfidf: float = Field(description="Произведение TF и IDF")


class TextAnalysisResult(BaseModel):
    """Результаты анализа текста."""
    filenames: List[str] = Field(description="Имена обработанных файлов")
    total_words: int = Field(ge=0, description="Общее количество слов")
    unique_words: int = Field(ge=0, description="Количество уникальных слов")
    top_words: List[WordData] = Field(description="Топ-слова по TF-IDF")
    document_count: int = Field(ge=1, description="Количество обработанных документов")
    language_stats: Dict[str, float] = Field(description="Статистика по языкам")
    analysis_options: AnalysisOptions = Field(description="Параметры анализа")


class PaginationParams(BaseModel):
    """Параметры для пагинации."""
    page: int = Field(
        default=1,
        ge=1,
        description="Номер страницы"
    )
    items_per_page: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Количество элементов на странице"
    )


class LanguageDetectionResult(BaseModel):
    """Результат определения языка."""
    language: str = Field(description="Основной язык текста")
    confidence: float = Field(
        ge=0,
        le=1,
        description="Уверенность в определении языка (от 0 до 1)"
    )