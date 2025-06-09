package com.example.lensflare

import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class NoiseRemovalProcessor {
    enum class NoiseType {
        GAUSSIAN,
        SALT_PEPPER,
        IMPULSE
    }

    private var currentNoiseType = NoiseType.GAUSSIAN
    private var kernelSize = 3
    private var sigmaX = 1.0
    private var sigmaY = 1.0

    fun setNoiseType(type: NoiseType) {
        currentNoiseType = type
        Log.d("NoiseRemovalProcessor", "Установлен тип шума: $type")
    }

    fun setKernelSize(size: Int) {
        if (size % 2 == 1 && size >= 3) {
            kernelSize = size
            Log.d("NoiseRemovalProcessor", "Установлен размер ядра: $size")
        }
    }

    fun setSigma(sigma: Double) {
        sigmaX = sigma
        sigmaY = sigma
        Log.d("NoiseRemovalProcessor", "Установлено значение sigma: $sigma")
    }

    fun processImage(inputMat: Mat): Mat {
        val resultMat = Mat()
        
        when (currentNoiseType) {
            NoiseType.GAUSSIAN -> {
                // Гауссовский фильтр для удаления гауссовского шума
                Imgproc.GaussianBlur(
                    inputMat,
                    resultMat,
                    Size(kernelSize.toDouble(), kernelSize.toDouble()),
                    sigmaX,
                    sigmaY
                )
            }
            NoiseType.SALT_PEPPER -> {
                // Медианный фильтр для удаления шума типа "соль и перец"
                Imgproc.medianBlur(inputMat, resultMat, kernelSize)
            }
            NoiseType.IMPULSE -> {
                // Адаптивный медианный фильтр для удаления импульсного шума
                val tempMat = Mat()
                Imgproc.medianBlur(inputMat, tempMat, kernelSize)
                
                // Применяем билатеральный фильтр для сохранения границ
                Imgproc.bilateralFilter(
                    tempMat,
                    resultMat,
                    kernelSize,
                    sigmaX * 2,
                    sigmaY * 2
                )
                tempMat.release()
            }
        }

        return resultMat
    }

    fun release() {
        // Освобождаем ресурсы, если необходимо
    }
} 