import uvicorn

if __name__ == "__main__":
    """
    Запуск приложения через этот файл
    Пример использования:
    $ python run.py
    """
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)