import pytest
import subprocess
import sys
import os


def run_tests():
    """Функция для запуска тестов"""
    print("Запуск тестов...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_app.py'],
                            capture_output=True,
                            text=True)

    # Вывод результатов тестов
    print(result.stdout)

    # Проверка статуса тестов
    if result.returncode != 0:
        print("ТЕСТЫ НЕ ПРОЙДЕНЫ!")
        print(result.stderr)
        return False
    else:
        print("Все тесты пройдены успешно!")
        return True


def pytest_configure(config):
    """Дополнительная конфигурация pytest"""
    config.addinivalue_line(
        "markers", "run_on_startup: mark test to run on startup"
    )