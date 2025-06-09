package com.example.lensflare

import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Core
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.nio.ByteBuffer
import android.graphics.Bitmap

class MainActivity : ComponentActivity() {
    private var cameraExecutor: ExecutorService? = null
    private var isNoiseRemovalEnabled = false
    private var noiseAnalyzer: NoiseAnalyzer? = null
    private lateinit var previewView: PreviewView
    private lateinit var processedImageView: ImageView
    private lateinit var noiseTypeSpinner: Spinner
    private lateinit var kernelSizeSeekBar: SeekBar
    private lateinit var sigmaSeekBar: SeekBar
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Требуется разрешение на использование камеры", Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d("MainActivity", "Starting onCreate")
        Log.d("OpenCV", "Trying to initialize OpenCV")
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed")
            Toast.makeText(this, "Ошибка инициализации OpenCV", Toast.LENGTH_LONG).show()
            return
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully")
        }

        // Инициализируем views
        previewView = findViewById(R.id.previewView)
        processedImageView = findViewById(R.id.processedImageView)
        noiseTypeSpinner = findViewById(R.id.noiseTypeSpinner)
        kernelSizeSeekBar = findViewById(R.id.kernelSizeSeekBar)
        sigmaSeekBar = findViewById(R.id.sigmaSeekBar)

        // Настраиваем Spinner для выбора типа шума
        val noiseTypes = NoiseRemovalProcessor.NoiseType.values()
        val adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            noiseTypes.map { when(it) {
                NoiseRemovalProcessor.NoiseType.GAUSSIAN -> "Гауссовский шум"
                NoiseRemovalProcessor.NoiseType.SALT_PEPPER -> "Шум 'соль и перец'"
                NoiseRemovalProcessor.NoiseType.IMPULSE -> "Импульсный шум"
            }}
        )
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        noiseTypeSpinner.adapter = adapter

        // Настраиваем SeekBar для размера ядра
        kernelSizeSeekBar.max = 7 // Максимальный размер ядра 7x7
        kernelSizeSeekBar.progress = 3 // Начальный размер ядра 3x3
        kernelSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val size = if (progress % 2 == 0) progress + 1 else progress
                noiseAnalyzer?.setKernelSize(size)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Настраиваем SeekBar для sigma
        sigmaSeekBar.max = 10 // Максимальное значение sigma
        sigmaSeekBar.progress = 1 // Начальное значение sigma
        sigmaSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                noiseAnalyzer?.setSigma(progress.toDouble())
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        // Настраиваем кнопку переключения режима
        findViewById<Button>(R.id.modeButton).setOnClickListener {
            isNoiseRemovalEnabled = !isNoiseRemovalEnabled
            it as Button
            it.text = if (isNoiseRemovalEnabled) "Режим удаления шума" else "Обычный режим"
            
            // Переключаем видимость элементов
            if (isNoiseRemovalEnabled) {
                previewView.visibility = android.view.View.GONE
                processedImageView.visibility = android.view.View.VISIBLE
                findViewById<LinearLayout>(R.id.controlsLayout).visibility = android.view.View.VISIBLE
            } else {
                previewView.visibility = android.view.View.VISIBLE
                processedImageView.visibility = android.view.View.GONE
                findViewById<LinearLayout>(R.id.controlsLayout).visibility = android.view.View.GONE
            }
            
            // Обновляем анализатор с новым режимом
            noiseAnalyzer?.updateMode(isNoiseRemovalEnabled)
            
            Toast.makeText(
                this,
                if (isNoiseRemovalEnabled) "Режим удаления шума включен" else "Обычный режим",
                Toast.LENGTH_SHORT
            ).show()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                
                val preview = Preview.Builder().build()
                noiseAnalyzer = NoiseAnalyzer(isNoiseRemovalEnabled, applicationContext) { bitmap ->
                    runOnUiThread {
                        processedImageView.setImageBitmap(bitmap)
                    }
                }
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(cameraExecutor!!, noiseAnalyzer!!)
                    }
                
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                
                try {
                    cameraProvider.unbindAll()
                    preview.setSurfaceProvider(previewView.surfaceProvider)
                    cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis
                    )
                } catch (exc: Exception) {
                    Log.e("Camera", "Failed to bind camera", exc)
                    Toast.makeText(this, "Ошибка при запуске камеры", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e("Camera", "Error in camera setup", e)
                Toast.makeText(this, "Ошибка настройки камеры: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    override fun onDestroy() {
        super.onDestroy()
        try {
            noiseAnalyzer?.release()
            cameraExecutor?.shutdown()
            cameraExecutor = null
        } catch (e: Exception) {
            Log.e("MainActivity", "Error shutting down camera executor", e)
        }
    }
}

class NoiseAnalyzer(
    private var isNoiseRemovalEnabled: Boolean,
    private val context: android.content.Context,
    private val onImageProcessed: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {
    private val processor = NoiseRemovalProcessor()
    
    fun updateMode(enabled: Boolean) {
        isNoiseRemovalEnabled = enabled
    }

    fun setNoiseType(type: NoiseRemovalProcessor.NoiseType) {
        processor.setNoiseType(type)
    }

    fun setKernelSize(size: Int) {
        processor.setKernelSize(size)
    }

    fun setSigma(sigma: Double) {
        processor.setSigma(sigma)
    }
    
    override fun analyze(imageProxy: ImageProxy) {
        if (!isNoiseRemovalEnabled) {
            imageProxy.close()
            return
        }
        
        var bgrMat: Mat? = null
        var processedMat: Mat? = null
        
        try {
            val yPlane = imageProxy.planes[0]
            val uPlane = imageProxy.planes[1]
            val vPlane = imageProxy.planes[2]

            val yBuffer = yPlane.buffer.apply { rewind() }
            val uBuffer = uPlane.buffer.apply { rewind() }
            val vBuffer = vPlane.buffer.apply { rewind() }

            val ySize = yBuffer.remaining()
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height

            val yuv_I420_data = ByteArray(imageWidth * imageHeight * 3 / 2)

            yBuffer.get(yuv_I420_data, 0, ySize)

            var destOffset = ySize
            val uRowStride = uPlane.rowStride
            val uPixelStride = uPlane.pixelStride
            val chromaWidth = imageWidth / 2
            val chromaHeight = imageHeight / 2

            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    yuv_I420_data[destOffset++] = uBuffer.get(row * uRowStride + col * uPixelStride)
                }
            }

            val vRowStride = vPlane.rowStride
            val vPixelStride = vPlane.pixelStride
            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    yuv_I420_data[destOffset++] = vBuffer.get(row * vRowStride + col * vPixelStride)
                }
            }

            val yuvMat = Mat(imageHeight * 3 / 2, imageWidth, CvType.CV_8UC1)
            yuvMat.put(0, 0, yuv_I420_data)

            bgrMat = Mat()
            org.opencv.imgproc.Imgproc.cvtColor(yuvMat, bgrMat, org.opencv.imgproc.Imgproc.COLOR_YUV2BGR_I420)
            yuvMat.release()
            
            Core.rotate(bgrMat, bgrMat, Core.ROTATE_90_CLOCKWISE)
            
            processedMat = processor.processImage(bgrMat)
            
            processedMat?.let { nonNullProcessed ->
                val rgba = Mat()
                org.opencv.imgproc.Imgproc.cvtColor(nonNullProcessed, rgba, org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA)
                
                val bitmap = Bitmap.createBitmap(
                    rgba.cols(),
                    rgba.rows(),
                    Bitmap.Config.ARGB_8888
                )
                Utils.matToBitmap(rgba, bitmap)
                
                rgba.release()
                
                onImageProcessed(bitmap)
            }
        } catch (e: Exception) {
            Log.e("NoiseAnalyzer", "Error processing frame", e)
        } finally {
            bgrMat?.release()
            processedMat?.release()
            imageProxy.close()
        }
    }
    
    fun release() {
        processor.release()
    }
} 