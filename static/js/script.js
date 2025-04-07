// Функции для основной страницы
document.addEventListener('DOMContentLoaded', function() {
    // Проверка размера файла
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const maxFileSize = 5 * 1024 * 1024; // 5MB
            if (this.files[0] && this.files[0].size > maxFileSize) {
                alert('Файл слишком большой. Максимальный размер: 5MB');
                this.value = '';
            }
        });
    }

    // Анимация при загрузке страницы
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

    // Форма загрузки файла - добавление класса для стилизации при фокусе
    const formInputs = document.querySelectorAll('.form-control, .form-select');
    formInputs.forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('input-focused');
        });

        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('input-focused');
        });
    });
});

// Определение темного режима по предпочтениям пользователя
function checkDarkMode() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        document.body.classList.add('dark-mode');
    }
}

// Обработка загрузки файла
function handleFileUpload() {
    const form = document.querySelector('form');
    const submitButton = form.querySelector('button[type="submit"]');

    form.addEventListener('submit', function() {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Обработка...';
    });
}