document.addEventListener('DOMContentLoaded', function() {
    // Функции для работы с формой загрузки файлов
    const fileInput = document.getElementById('files');
    const uploadForm = document.querySelector('form');

    if (fileInput && uploadForm) {
        // Максимальный размер файла (5 МБ)
        const MAX_FILE_SIZE = 5 * 1024 * 1024;

        // Допустимые расширения файлов
        const ALLOWED_EXTENSIONS = ['txt', 'csv'];

        // Валидация файлов при выборе
        fileInput.addEventListener('change', function(event) {
            const files = event.target.files;
            let errorMessages = [];

            // Проверка количества файлов
            if (files.length > 10) {
                errorMessages.push('Можно загрузить не более 10 файлов');
            }

            // Проверка каждого файла
            Array.from(files).forEach(file => {
                // Проверка расширения
                const fileExtension = file.name.split('.').pop().toLowerCase();
                if (!ALLOWED_EXTENSIONS.includes(fileExtension)) {
                    errorMessages.push(`Недопустимый формат файла: ${file.name}. Используйте .txt или .csv`);
                }

                // Проверка размера
                if (file.size > MAX_FILE_SIZE) {
                    errorMessages.push(`Файл "${file.name}" слишком большой. Максимальный размер: 5 МБ`);
                }
            });

            // Показ ошибок или сброс формы
            if (errorMessages.length > 0) {
                alert(errorMessages.join('\n'));
                fileInput.value = ''; // Очистка выбранных файлов
                return false;
            }
        });

        // Блокировка кнопки при отправке формы
        uploadForm.addEventListener('submit', function(event) {
            const submitButton = this.querySelector('button[type="submit"]');
            const fileInput = this.querySelector('input[type="file"]');

            // Проверка наличия файлов перед отправкой
            if (!fileInput.files || fileInput.files.length === 0) {
                event.preventDefault();
                alert('Пожалуйста, выберите файлы для загрузки');
                return false;
            }

            // Блокировка кнопки и изменение её вида
            submitButton.disabled = true;
            submitButton.innerHTML = `
                <span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                Обработка...
            `;
        });
    }

    // Анимация загрузки страницы
    const mainContent = document.querySelector('.container');
    if (mainContent) {
        mainContent.style.opacity = '0';
        mainContent.style.transform = 'translateY(20px)';
        mainContent.style.transition = 'opacity 0.5s ease, transform 0.5s ease';

        setTimeout(function() {
            mainContent.style.opacity = '1';
            mainContent.style.transform = 'translateY(0)';
        }, 100);
    }

    // Функции для страницы результатов
    function initResultsPage() {
        const exportCsvBtn = document.getElementById('exportBtn');
        const exportExcelBtn = document.getElementById('exportExcelBtn');
        const sessionIdField = document.getElementById('sessionIdField');

        // Экспорт в CSV
        if (exportCsvBtn && sessionIdField) {
            exportCsvBtn.addEventListener('click', function() {
                const filename = document.querySelector('.lead').textContent.replace('Файл: ', '');
                const sessionId = sessionIdField.value;

                // Создаем ссылку для скачивания
                window.location.href = `/download-csv/?filename=${encodeURIComponent(filename)}&session_id=${sessionId}`;
            });
        }

        // Экспорт в Excel
        if (exportExcelBtn && sessionIdField) {
            exportExcelBtn.addEventListener('click', function() {
                const filename = document.querySelector('.lead').textContent.replace('Файл: ', '');
                const sessionId = sessionIdField.value;

                // Создаем ссылку для скачивания
                window.location.href = `/download-excel/?filename=${encodeURIComponent(filename)}&session_id=${sessionId}`;
            });
        }

        // Переключение вкладок
        const tabButtons = document.querySelectorAll('.nav-link');
        const tabPanes = document.querySelectorAll('.tab-pane');

        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const targetTab = this.getAttribute('data-bs-target');

                // Убираем активные классы
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabPanes.forEach(pane => {
                    pane.classList.remove('show', 'active');
                });

                // Добавляем активные классы
                this.classList.add('active');
                document.querySelector(targetTab).classList.add('show', 'active');
            });
        });
    }

    // Инициализация страницы результатов, если она есть
    if (document.querySelector('.results-page')) {
        initResultsPage();
    }

    // Функция для копирования текста в буфер обмена
    function initCopyToClipboard() {
        const copyButtons = document.querySelectorAll('.copy-to-clipboard');

        copyButtons.forEach(button => {
            button.addEventListener('click', function() {
                const textToCopy = this.getAttribute('data-copy');

                // Создаем временный textarea для копирования
                const tempTextArea = document.createElement('textarea');
                tempTextArea.value = textToCopy;
                document.body.appendChild(tempTextArea);

                // Выделяем и копируем текст
                tempTextArea.select();
                document.execCommand('copy');

                // Удаляем временный элемент
                document.body.removeChild(tempTextArea);

                // Временное изменение текста кнопки
                const originalText = this.textContent;
                this.textContent = 'Скопировано!';
                this.disabled = true;

                setTimeout(() => {
                    this.textContent = originalText;
                    this.disabled = false;
                }, 2000);
            });
        });
    }

    // Вызываем функцию копирования, если есть кнопки
    initCopyToClipboard();
});