from fastapi import FastAPI, Request, File, UploadFile, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pathlib import Path
import tempfile
import shutil
import uvicorn
from typing import Optional
import json
import uuid
from datetime import datetime, timedelta

from app.models import AnalysisOptions, PaginationParams
from app.text_processing import process_text_file, calculate_tfidf_sklearn
from app.utils import detect_language, detect_languages_distribution, get_readability_stats
from app.utils import generate_wordcloud_image, generate_tfidf_chart, export_to_csv, export_to_excel

app = FastAPI(title="Анализатор текста - TF-IDF")

# Настройка шаблонов и статических файлов
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR.parent / "static")), name="static")

# Для хранения обработанных файлов и результатов
TEMP_DIR = tempfile.gettempdir()

# Кэш для хранения результатов анализа
# {session_id: {data: [...], expires: datetime}}
RESULTS_CACHE = {}

# Время хранения кэша (в минутах)
CACHE_EXPIRY = 30


def clean_expired_cache():
    """Очистка устаревших данных в кэше"""
    now = datetime.now()
    expired_keys = [key for key, value in RESULTS_CACHE.items()
                    if value['expires'] < now]

    for key in expired_keys:
        del RESULTS_CACHE[key]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница с формой для загрузки файла"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(
        request: Request,
        file: UploadFile = File(None),
        page: int = Form(1),
        items_per_page: int = Form(50),
        remove_stopwords: bool = Form(True),
        case_sensitive: bool = Form(False),
        min_word_length: int = Form(2),
        language: str = Form("auto"),
        session_id: str = Form(None)
):
    """Обработка загруженного файла и отображение результатов"""
    # Проверяем наличие сессии
    if session_id and session_id in RESULTS_CACHE:
        # Используем кэшированные данные
        cache_data = RESULTS_CACHE[session_id]
        words_data = cache_data['data']
        filename = cache_data['filename']
        language_distribution = cache_data['language_distribution']
        readability_stats = cache_data['readability_stats']
        language = cache_data['language']
        wordcloud_image = cache_data.get('wordcloud_image')
        tfidf_chart = cache_data.get('tfidf_chart')

        # Обновляем время жизни кэша
        RESULTS_CACHE[session_id]['expires'] = datetime.now() + timedelta(minutes=CACHE_EXPIRY)
    else:
        # Создаем новую сессию при загрузке файла
        if not file:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Файл не выбран"}
            )

        if not file.filename.endswith(('.txt', '.csv')):
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Пожалуйста, загрузите текстовый файл (.txt или .csv)"}
            )

        # Сохраняем временный файл
        temp_file_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            # Читаем содержимое файла для определения языка
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Определяем язык, если указан "auto"
            if language == "auto":
                detected_lang, confidence = detect_language(file_content)
                language = detected_lang

            # Получаем распределение языков
            language_distribution = detect_languages_distribution(file_content)

            # Получаем статистику читабельности
            readability_stats = get_readability_stats(file_content)

            # Создаем параметры анализа
            analysis_options = AnalysisOptions(
                remove_stopwords=remove_stopwords,
                case_sensitive=case_sensitive,
                min_word_length=min_word_length,
                language=language
            )

            # Обрабатываем файл и получаем результаты TF-IDF
            words_data = process_text_file(
                temp_file_path,
                remove_stopwords=remove_stopwords,
                case_sensitive=case_sensitive,
                min_word_length=min_word_length,
                language=language
            )

            # Генерируем облако слов (если доступно)
            wordcloud_image = generate_wordcloud_image(words_data)

            # Генерируем график TF-IDF
            tfidf_chart = generate_tfidf_chart(words_data)

            # Сохраняем в кэш
            session_id = str(uuid.uuid4())
            RESULTS_CACHE[session_id] = {
                'data': words_data,
                'filename': file.filename,
                'language': language,
                'language_distribution': language_distribution,
                'readability_stats': readability_stats,
                'wordcloud_image': wordcloud_image,
                'tfidf_chart': tfidf_chart,
                'expires': datetime.now() + timedelta(minutes=CACHE_EXPIRY)
            }

            filename = file.filename

        except Exception as e:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": f"Ошибка при обработке файла: {str(e)}"}
            )
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    # Очистка устаревших данных
    clean_expired_cache()

    # Расчет для пагинации
    total_items = len(words_data)
    total_pages = (total_items + items_per_page - 1) // items_per_page

    # Проверка корректности номера страницы
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages

    # Получаем данные для текущей страницы
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    current_page_data = words_data[start_idx:end_idx]

    # Базовая статистика
    unique_words = len(words_data)
    total_words = sum(item['tf'] for item in words_data)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "words_data": current_page_data,
            "filename": filename,
            "current_page": page,
            "total_pages": total_pages,
            "items_per_page": items_per_page,
            "total_items": total_items,
            "unique_words": unique_words,
            "total_words": total_words,
            "language": language,
            "language_distribution": language_distribution,
            "readability_stats": readability_stats,
            "wordcloud_image": wordcloud_image,
            "tfidf_chart": tfidf_chart,
            "session_id": session_id,
            "analysis_options": {
                "remove_stopwords": remove_stopwords,
                "case_sensitive": case_sensitive,
                "min_word_length": min_word_length,
                "language": language
            }
        }
    )


@app.get("/page/{page}")
async def get_page(
        request: Request,
        page: int,
        session_id: str,
        items_per_page: int = 50
):
    """Обработка запроса на переход к определенной странице"""
    if session_id not in RESULTS_CACHE:
        return RedirectResponse(url="/")

    # Получаем данные из кэша
    cache_data = RESULTS_CACHE[session_id]
    words_data = cache_data['data']

    # Обновляем время жизни кэша
    RESULTS_CACHE[session_id]['expires'] = datetime.now() + timedelta(minutes=CACHE_EXPIRY)

    # Расчет для пагинации
    total_items = len(words_data)
    total_pages = (total_items + items_per_page - 1) // items_per_page

    # Проверка корректности номера страницы
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages

    # Получаем данные для текущей страницы
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    current_page_data = words_data[start_idx:end_idx]

    # Базовая статистика
    unique_words = len(words_data)
    total_words = sum(item['tf'] for item in words_data)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "words_data": current_page_data,
            "filename": cache_data['filename'],
            "current_page": page,
            "total_pages": total_pages,
            "items_per_page": items_per_page,
            "total_items": total_items,
            "unique_words": unique_words,
            "total_words": total_words,
            "language": cache_data['language'],
            "language_distribution": cache_data['language_distribution'],
            "readability_stats": cache_data['readability_stats'],
            "wordcloud_image": cache_data.get('wordcloud_image'),
            "tfidf_chart": cache_data.get('tfidf_chart'),
            "session_id": session_id,
            "analysis_options": {
                "remove_stopwords": True,  # Заглушка, в кэше нет этих данных
                "case_sensitive": False,
                "min_word_length": 2,
                "language": cache_data['language']
            }
        }
    )


@app.get("/download-csv/")
async def download_results_csv(
        filename: str = Query(...),
        session_id: str = Query(...)
):
    """Скачивание результатов в формате CSV"""
    try:
        # Проверяем наличие сессии
        if session_id not in RESULTS_CACHE:
            raise HTTPException(status_code=404, detail="Данные не найдены")

        # Получаем данные из кэша
        cache_data = RESULTS_CACHE[session_id]
        words_data = cache_data['data']

        # Экспортируем результаты в CSV
        output_filename = f"tfidf_results_{Path(filename).stem}.csv"
        csv_path = export_to_csv(words_data, output_filename)

        # Возвращаем файл для скачивания
        return FileResponse(csv_path, filename=output_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@app.get("/download-excel/")
async def download_results_excel(
        filename: str = Query(...),
        session_id: str = Query(...)
):
    """Скачивание результатов в формате Excel"""
    try:
        # Проверяем наличие сессии
        if session_id not in RESULTS_CACHE:
            raise HTTPException(status_code=404, detail="Данные не найдены")

        # Получаем данные из кэша
        cache_data = RESULTS_CACHE[session_id]
        words_data = cache_data['data']

        # Экспортируем результаты в Excel
        output_filename = f"tfidf_results_{Path(filename).stem}.xlsx"
        excel_path = export_to_excel(words_data, output_filename)

        # Возвращаем файл для скачивания
        return FileResponse(excel_path, filename=output_filename)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка экспорта: {str(e)}")


@app.get("/about/", response_class=HTMLResponse)
async def about(request: Request):
    """Страница с информацией о проекте"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/advanced/", response_class=HTMLResponse)
async def advanced_settings(request: Request):
    """Страница с расширенными настройками анализа"""
    return templates.TemplateResponse("advanced.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)