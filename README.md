# 📱 Real-Time Lens Flare Removal

<div align="center">

![Android](https://img.shields.io/badge/Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)
![Kotlin](https://img.shields.io/badge/Kotlin-0095D5?&style=for-the-badge&logo=kotlin&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

**Приложение для удаления бликов с объектива в реальном времени на Android с использованием машинного обучения**

[Установка](#-установка) • [Возможности](#-возможности) • [Архитектура](#-архитектура) • [Использование](#-использование) • [Требования](#-требования)

</div>

---

## 🎯 Возможности

- **🔍 Удаление бликов в реальном времени** - Обработка видео с камеры на лету
- **🤖 Машинное обучение** - Использование PyTorch модели для качественного удаления бликов
- **📊 Два режима работы**:
  - **Обычный режим** - Просмотр камеры без обработки
  - **Режим удаления шума** - Обработка изображения с настраиваемыми параметрами
- **⚙️ Настраиваемые параметры**:
  - Выбор типа шума (Гауссовский, "Соль и перец", Импульсный)
  - Размер ядра фильтра (3x3 до 7x7)
  - Параметр sigma для фильтрации
- **📱 Современный UI** - Интуитивный интерфейс с элементами управления

## 🏗️ Архитектура

### Основные компоненты:

```
📁 app/src/main/java/com/example/lensflare/
├── 📄 MainActivity.kt          # Главная активность с UI и камерой
├── 📄 LensFlareProcessor.kt    # Обработчик бликов с ML моделью
└── 📄 NoiseRemovalProcessor.kt # Обработчик шума с OpenCV
```

### Технологический стек:

- **🎥 CameraX** - Современный API для работы с камерой
- **🤖 PyTorch Mobile** - Машинное обучение на устройстве
- **🔧 OpenCV** - Обработка изображений и фильтрация
- **📱 Jetpack Compose** - Современный UI фреймворк
- **⚡ Kotlin Coroutines** - Асинхронная обработка

### ML Модель:

- **Файл**: `model_specular_removal.ptl` (132MB)
- **Архитектура**: Специализированная модель для удаления бликов
- **Вход**: RGB изображение 128x128 пикселей
- **Выход**: Обработанное изображение без бликов


### Метрики (на среднем устройстве)

| Операция | Время |
|----------|-------|
| Загрузка модели | ~2-3 сек |
| Обработка кадра | ~50-100 мс |
| Переключение режимов | ~100 мс |

## 🤝 Вклад в проект

Мы приветствуем вклад в развитие проекта! 

1. **Fork** репозитория
2. Создайте **feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit** изменения (`git commit -m 'Add some AmazingFeature'`)
4. **Push** в branch (`git push origin feature/AmazingFeature`)
5. Откройте **Pull Request**

