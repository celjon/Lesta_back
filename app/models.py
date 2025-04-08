from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum


class FileType(str, Enum):
    """Поддерживаемые типы файлов"""
    TXT = "txt"
    CSV = "csv"


class AnalysisOptions(BaseModel):
    """Параметры для анализа текста"""
    remove_stopwords: bool = True
    case_sensitive: bool = False
    min_word_length: int = 2
    max_features: int = 50
    language: str = "russian"  # Язык для стоп-слов


class WordData(BaseModel):
    """Данные о слове после анализа TF-IDF"""
    word: str
    tf: int
    idf: float
    tfidf: float


class TextAnalysisResult(BaseModel):
    """Результаты анализа текста"""
    filenames: List[str]
    total_words: int
    unique_words: int
    top_words: List[WordData]
    document_count: int
    language_stats: Dict[str, float]  # Статистика по языкам
    analysis_options: AnalysisOptions


class PaginationParams(BaseModel):
    """Параметры для пагинации"""
    page: int = 1
    items_per_page: int = 50