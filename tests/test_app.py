import pytest
from fastapi.testclient import TestClient
import tempfile
import os
from pathlib import Path

# Импортируем основное приложение
from app.main import app
from app.text_processing import preprocess_text, calculate_tf, calculate_idf

# Создаем тестовый клиент
client = TestClient(app)


def test_index_route():
    """Тест главной страницы"""
    response = client.get("/")
    assert response.status_code == 200
    assert "Анализатор текста - TF-IDF" in response.text


def test_about_route():
    """Тест страницы 'О проекте'"""
    response = client.get("/about/")
    assert response.status_code == 200
    assert "Что такое TF-IDF?" in response.text


def test_upload_no_file():
    """Тест загрузки без файла"""
    response = client.post("/upload/", data={"page": 1, "items_per_page": 50})
    assert response.status_code == 422  # Unprocessable Entity


def test_upload_invalid_file_type():
    """Тест загрузки файла неверного типа"""
    # Создаем временный файл с неверным расширением
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
        tmp_file.write(b"Test image content")
        tmp_file.flush()

        # Загружаем файл
        with open(tmp_file.name, "rb") as f:
            response = client.post(
                "/upload/",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"page": 1, "items_per_page": 50}
            )

        assert response.status_code == 200
        assert "Пожалуйста, загрузите текстовый файл" in response.text


def test_upload_valid_text_file():
    """Тест загрузки корректного текстового файла"""
    # Создаем временный текстовый файл
    test_text = """
    Это тестовый текст для проверки работы TF-IDF анализатора.
    Текст содержит несколько повторяющихся слов, чтобы проверить расчет TF.
    TF-IDF анализатор должен корректно обрабатывать этот тестовый текст.
    """

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(test_text.encode('utf-8'))
        tmp_file.flush()

        # Загружаем файл
        with open(tmp_file.name, "rb") as f:
            response = client.post(
                "/upload/",
                files={"file": ("test.txt", f, "text/plain")},
                data={"page": 1, "items_per_page": 50}
            )

        # Удаляем временный файл
        os.unlink(tmp_file.name)

        assert response.status_code == 200
        assert "Результаты анализа TF-IDF" in response.text
        # Проверяем, что таблица результатов содержит нужные заголовки
        assert "Слово" in response.text
        assert "TF (частота)" in response.text
        assert "IDF" in response.text


def test_preprocess_text():
    """Тест функции предварительной обработки текста"""
    test_text = "Это тестовый текст, содержащий знаки препинания и стоп-слова."
    tokens = preprocess_text(test_text)

    # Проверяем, что стоп-слова и пунктуация удалены
    assert "и" not in tokens
    assert "," not in tokens

    # Проверяем, что все токены преобразованы в нижний регистр
    assert all(token == token.lower() for token in tokens)

    # Проверяем наличие ожидаемых токенов
    assert "тестовый" in tokens
    assert "текст" in tokens


def test_calculate_tf():
    """Тест расчета TF (частоты термина)"""
    document = ["это", "тестовый", "текст", "тестовый", "документ"]

    # Проверка частоты для слова "тестовый" (встречается 2 раза)
    tf_testoviy = calculate_tf("тестовый", document)
    assert tf_testoviy == 2

    # Проверка частоты для слова "текст" (встречается 1 раз)
    tf_text = calculate_tf("текст", document)
    assert tf_text == 1

    # Проверка частоты для слова, которого нет в документе
    tf_none = calculate_tf("отсутствует", document)
    assert tf_none == 0


def test_calculate_idf():
    """Тест расчета IDF (обратной частоты документа)"""
    documents = [
        ["это", "первый", "документ"],
        ["это", "второй", "тестовый", "документ"],
        ["третий", "документ", "содержит", "другие", "слова"]
    ]

    # Слово "документ" встречается во всех трех документах
    idf_doc = calculate_idf("документ", documents)
    assert idf_doc == pytest.approx(0.0)  # log(3/3) = 0

    # Слово "это" встречается в двух документах
    idf_eto = calculate_idf("это", documents)
    assert idf_eto == pytest.approx(0.4054651081081644)  # log(3/2) ≈ 0.405

    # Слово "первый" встречается только в одном документе
    idf_perv = calculate_idf("первый", documents)
    assert idf_perv == pytest.approx(1.0986122886681098)  # log(3/1) ≈ 1.099

    # Слово, которого нет ни в одном документе
    idf_none = calculate_idf("отсутствует", documents)
    assert idf_none == 0  # Определено в функции, чтобы избежать деления на 0