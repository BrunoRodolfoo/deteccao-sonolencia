package br.pucpr.sonolencia

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult

/**
 * Encapsula o MediaPipe FaceLandmarker (Tasks Vision 0.10.x) em modo LIVE_STREAM.
 * Tenta o delegate GPU e cai para CPU se a criação falhar.
 *
 * Referência: developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/android
 */
class FaceLandmarkerHelper(
    private val context: Context,
    private val onResult: (FaceLandmarkerResult, MPImage) -> Unit,
    private val onError: (RuntimeException) -> Unit,
) {
    private var landmarker: FaceLandmarker? = null

    /** "GPU" ou "CPU" após [setup]; null se não inicializado. */
    var delegateName: String? = null
        private set

    val isReady: Boolean get() = landmarker != null

    /**
     * Cria o FaceLandmarker. Retorna null em caso de sucesso ou a mensagem de erro.
     */
    fun setup(): String? {
        if (!TfliteLoader.assetExists(context, MODEL_ASSET)) {
            return "asset $MODEL_ASSET ausente"
        }
        val gpuError = try {
            landmarker = create(Delegate.GPU)
            delegateName = "GPU"
            null
        } catch (e: Exception) {
            Log.w(TAG, "Delegate GPU indisponível, usando CPU: ${e.message}")
            e.message ?: e.toString()
        }
        if (landmarker == null) {
            try {
                landmarker = create(Delegate.CPU)
                delegateName = "CPU"
            } catch (e: Exception) {
                Log.e(TAG, "Falha ao criar FaceLandmarker (CPU)", e)
                return "GPU: $gpuError; CPU: ${e.message}"
            }
        }
        return null
    }

    private fun create(delegate: Delegate): FaceLandmarker {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_ASSET)
            .setDelegate(delegate)
            .build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(1)
            .setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setOutputFaceBlendshapes(false)
            .setOutputFacialTransformationMatrixes(false)
            .setResultListener { result, input -> onResult(result, input) }
            .setErrorListener { error -> onError(error) }
            .build()
        return FaceLandmarker.createFromOptions(context, options)
    }

    /**
     * Envia um quadro para detecção assíncrona. [timestampMs] deve ser estritamente
     * crescente (use SystemClock.uptimeMillis()).
     */
    fun detectAsync(bitmap: Bitmap, timestampMs: Long) {
        val lm = landmarker ?: return
        val mpImage: MPImage = BitmapImageBuilder(bitmap).build()
        lm.detectAsync(mpImage, timestampMs)
    }

    fun close() {
        try {
            landmarker?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Erro ao fechar FaceLandmarker", e)
        }
        landmarker = null
        delegateName = null
    }

    companion object {
        const val TAG = "FaceLandmarkerHelper"
        const val MODEL_ASSET = "face_landmarker.task"
    }
}
