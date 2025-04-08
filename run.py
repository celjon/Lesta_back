import uvicorn
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем функцию запуска тестов
from tests.conftest import run_tests


def main():
    # Запускаем тесты перед стартом сервера
    if not run_tests():
        # Если тесты не пройдены, завершаем приложение
        print("Не удалось запустить приложение из-за невыполненных тестов.")
        sys.exit(1)

    # Если тесты пройдены, запускаем сервер
    uvicorn.run("app.main:app", host="127.0.0.1", port=8002, reload=True)


if __name__ == "__main__":
    main()