# 📱 Обнаружение и удаление шумов в видеопотоке

**Предмет:** Технологии мультимедиа  
**Тема:** Разработка мобильных приложений с применением технологии компьютерного зрения  
**Задание:** Обнаружение и удаление шумов (Гаусса, Импульсный шум, Соль и перец) в видеопотоке от видеокамеры в реальном масштабе времени  

---

## 🎯 Задание

### Цель работы
Разработка мобильного приложения для Android с применением технологии компьютерного зрения, способного обнаруживать и удалять различные типы шумов в видеопотоке от камеры в реальном времени.

### Требования
- **Платформа:** Android (Android Studio)
- **Технологии:** OpenCV, машинное обучение (PyTorch Mobile)
- **Функциональность:** Обработка видеопотока в реальном времени
- **Типы шумов:** Гауссовский, Импульсный, "Соль и перец"

---

## Краткое описание математики и алгоритмов

### 1. Математические основы фильтрации шумов

#### Гауссовский шум
Гауссовский шум описывается нормальным распределением:
```
N(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
```
где:
- `μ` - математическое ожидание (обычно 0)
- `σ` - стандартное отклонение
- `x` - значение пикселя

#### Импульсный шум ("Соль и перец")
Импульсный шум характеризуется случайными пикселями с экстремальными значениями:
```
I_noisy(x,y) = {
    I_max, с вероятностью p/2
    I_min, с вероятностью p/2  
    I_original(x,y), с вероятностью (1-p)
}
```

### 2. Алгоритмы удаления шумов

#### Гауссовский фильтр
```kotlin
// Применение гауссовского фильтра
Imgproc.GaussianBlur(
    sourceMat,    // исходное изображение
    destinationMat, // результат
    Size(kernelSize, kernelSize), // размер ядра
    sigma // стандартное отклонение
)
```

#### Медианный фильтр
```kotlin
// Применение медианного фильтра
Imgproc.medianBlur(
    sourceMat,    // исходное изображение
    destinationMat, // результат
    kernelSize    // размер ядра
)
```

#### Адаптивный фильтр
```kotlin
// Адаптивная фильтрация на основе локальной статистики
fun adaptiveFilter(input: Mat, output: Mat, kernelSize: Int) {
    val localMean = Mat()
    val localVariance = Mat()
    
    // Вычисление локального среднего
    Imgproc.boxFilter(input, localMean, -1, Size(kernelSize, kernelSize))
    
    // Вычисление локальной дисперсии
    val inputSquared = Mat()
    Core.multiply(input, input, inputSquared)
    Imgproc.boxFilter(inputSquared, localVariance, -1, Size(kernelSize, kernelSize))
    Core.subtract(localVariance, Core.multiply(localMean, localMean), localVariance)
    
    // Применение адаптивного фильтра
    val noiseVariance = estimateNoiseVariance(input)
    val adaptiveWeight = localVariance / (localVariance + noiseVariance)
    Core.multiply(adaptiveWeight, input, output)
    Core.add(output, Core.multiply(Core.subtract(Mat.ones(input.size(), input.type()), adaptiveWeight), localMean), output)
}
```

---

## Архитектура приложения

### Структура проекта
```
📁 app/src/main/java/com/example/lensflare/
├── 📄 MainActivity.kt          # Главная активность
├── 📄 LensFlareProcessor.kt    # Обработчик бликов (ML)
└── 📄 NoiseRemovalProcessor.kt # Обработчик шумов (OpenCV)
```

### Основные компоненты

#### 1. MainActivity.kt
```kotlin
class MainActivity : ComponentActivity() {
    // Управление камерой и UI
    // Переключение между режимами
    // Обработка разрешений
}
```

#### 2. NoiseRemovalProcessor.kt
```kotlin
class NoiseRemovalProcessor {
    enum class NoiseType {
        GAUSSIAN,      // Гауссовский шум
        SALT_PEPPER,   // Шум "соль и перец"
        IMPULSE        // Импульсный шум
    }
    
    fun processImage(input: Mat, noiseType: NoiseType, kernelSize: Int, sigma: Double): Mat
}
```

#### 3. LensFlareProcessor.kt
```kotlin
class LensFlareProcessor(private val context: Context) {
    private var module: Module? = null
    
    fun processImage(bgrMat: Mat): Mat {
        // Загрузка ML модели
        // Предобработка изображения
        // Применение нейросети
        // Постобработка результата
    }
}
```

---

## 🛠️ Технологический стек

### Основные технологии
- **Android Studio** - IDE для разработки
- **Kotlin** - язык программирования
- **OpenCV** - библиотека компьютерного зрения
- **PyTorch Mobile** - машинное обучение на устройстве
- **CameraX** - современный API для работы с камерой

### Зависимости
```gradle
// Основные
implementation 'androidx.core:core-ktx:1.12.0'
implementation 'androidx.appcompat:appcompat:1.6.1'

// Машинное обучение
implementation 'org.pytorch:pytorch_android_lite:1.13.1'
implementation 'org.pytorch:pytorch_android_torchvision_lite:1.13.1'

// Камера
implementation 'androidx.camera:camera-core:1.3.1'
implementation 'androidx.camera:camera-camera2:1.3.1'
implementation 'androidx.camera:camera-lifecycle:1.3.1'
implementation 'androidx.camera:camera-view:1.3.1'

// Обработка изображений
implementation project(':opencv')
```

---

## 📚 Список литературы

1. OpenCV Documentation - https://docs.opencv.org/
2. PyTorch Mobile Documentation - https://pytorch.org/mobile/
3. Android CameraX Guide - https://developer.android.com/training/camerax
4. "Digital Image Processing" - Gonzalez, Woods
5. "Computer Vision: Algorithms and Applications" - Szeliski

---



