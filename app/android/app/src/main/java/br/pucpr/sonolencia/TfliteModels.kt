package br.pucpr.sonolencia

import android.content.Context
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

private const val TAG = "TfliteModels"

/** Utilitários de carga de modelos TFLite a partir de assets (mmap, sem cópia). */
object TfliteLoader {
    fun assetExists(context: Context, name: String): Boolean = try {
        context.assets.openFd(name).use { true }
    } catch (e: Exception) {
        false
    }

    fun assetSizeBytes(context: Context, name: String): Long = try {
        context.assets.openFd(name).use { it.declaredLength }
    } catch (e: Exception) {
        -1L
    }

    /** Mapeia o arquivo do asset em memória (requer asset sem compressão: ver noCompress). */
    fun loadModelFile(context: Context, name: String): MappedByteBuffer {
        context.assets.openFd(name).use { fd ->
            FileInputStream(fd.fileDescriptor).use { input ->
                val channel: FileChannel = input.channel
                return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    fun newInterpreter(context: Context, name: String, threads: Int = 2): Interpreter {
        val opts = Interpreter.Options().setNumThreads(threads)
        return Interpreter(loadModelFile(context, name), opts)
    }
}

/**
 * Classificador temporal `blink_temporal.tflite`: entrada [1,30,4] float32, saída [1,3]
 * (softmax: alerta, pouca vigilância, sonolento). Se a saída for [1,1] (regressão 0-10 como
 * no artigo original), discretiza em <3.3 / 3.3-6.6 / >6.6 e devolve um one-hot.
 */
class BlinkTemporalModel(context: Context, assetName: String = ASSET) : TemporalClassifier, AutoCloseable {
    val interpreter: Interpreter = TfliteLoader.newInterpreter(context, assetName)
    val metadata: ModelMetadata = ModelMetadata.load(context, assetName.replace(".tflite", ".json"))
    private val windowSize: Int
    private val featureSize: Int
    private val outputSize: Int
    private val input: ByteBuffer
    private val output: Array<FloatArray>

    init {
        val inShape = interpreter.getInputTensor(0).shape()  // [1,30,4]
        require(inShape.size == 3) { "Entrada esperada [1,30,4], obtida ${inShape.toList()}" }
        require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32) { "Entrada deve ser float32" }
        windowSize = inShape[1]
        featureSize = inShape[2]
        val outShape = interpreter.getOutputTensor(0).shape()   // [1,3] ou [1,1]
        outputSize = outShape.last()
        input = ByteBuffer.allocateDirect(4 * windowSize * featureSize).order(ByteOrder.nativeOrder())
        output = arrayOf(FloatArray(outputSize))
        if (metadata.featureOrder.isNotEmpty() &&
            metadata.featureOrder.map { it.lowercase().take(3) } != listOf("fre", "amp", "dur", "vel")
        ) {
            Log.w(TAG, "feature_order do JSON difere da ordem Freq/Amp/Dur/Vel esperada: ${metadata.featureOrder}")
        }
        Log.i(TAG, "blink_temporal: in=${inShape.toList()} out=${outShape.toList()}")
    }

    override fun predict(window: Array<FloatArray>): FloatArray {
        require(window.size == windowSize && window[0].size == featureSize)
        input.rewind()
        for (t in 0 until windowSize) for (f in 0 until featureSize) input.putFloat(window[t][f])
        input.rewind()
        interpreter.run(input, output)
        val y = output[0]
        return if (outputSize == 3) {
            y.copyOf()
        } else {
            val score = y[0]
            when {
                score < 3.3f -> floatArrayOf(1f, 0f, 0f)
                score <= 6.6f -> floatArrayOf(0f, 1f, 0f)
                else -> floatArrayOf(0f, 0f, 1f)
            }
        }
    }

    override fun close() = interpreter.close()

    companion object {
        const val ASSET = "blink_temporal.tflite"
    }
}

/**
 * CNN de estado do olho (`eye_state_int8.tflite` ou `eye_state_fp32.tflite`): entrada
 * [1,H,W,1] (48x48 cinza por padrão) float32/int8/uint8; saída [1,1] (sigmoide = P(aberto))
 * ou [1,2] (softmax; índice da classe "aberto" vem do metadata, padrão 1).
 *
 * A quantização da entrada/saída é lida do próprio interpretador
 * (Tensor.quantizationParams: scale, zeroPoint), então o mesmo código serve às duas versões.
 */
class EyeStateModel(context: Context, val assetName: String) : AutoCloseable {
    val interpreter: Interpreter = TfliteLoader.newInterpreter(context, assetName)
    val metadata: ModelMetadata = ModelMetadata.load(context, assetName.replace(".tflite", ".json"))
    val height: Int
    val width: Int
    private val channels: Int
    private val inType: DataType
    private val inScale: Float
    private val inZero: Int
    private val outType: DataType
    private val outScale: Float
    private val outZero: Int
    private val outputSize: Int
    private val input: ByteBuffer
    private val outputBuf: ByteBuffer

    init {
        val inT = interpreter.getInputTensor(0)
        val inShape = inT.shape()  // [1,H,W,C]
        require(inShape.size == 4) { "Entrada esperada [1,H,W,C], obtida ${inShape.toList()}" }
        height = inShape[1]; width = inShape[2]; channels = inShape[3]
        inType = inT.dataType()
        val q = inT.quantizationParams()
        inScale = q.scale; inZero = q.zeroPoint
        val outT = interpreter.getOutputTensor(0)
        outputSize = outT.shape().last()
        outType = outT.dataType()
        val oq = outT.quantizationParams()
        outScale = oq.scale; outZero = oq.zeroPoint
        val inBytes = if (inType == DataType.FLOAT32) 4 else 1
        input = ByteBuffer.allocateDirect(inBytes * height * width * channels).order(ByteOrder.nativeOrder())
        val outBytes = if (outType == DataType.FLOAT32) 4 else 1
        outputBuf = ByteBuffer.allocateDirect(outBytes * outputSize).order(ByteOrder.nativeOrder())
        Log.i(TAG, "$assetName: in=${inShape.toList()} $inType scale=$inScale zp=$inZero; out=$outputSize $outType")
    }

    /**
     * @param gray pixels em cinza 0..255 (row-major, tamanho H*W), já redimensionados.
     * @return probabilidade de olho ABERTO em [0,1]
     */
    fun predictOpenProb(gray: FloatArray): Float {
        require(gray.size == height * width) { "Esperados ${height * width} pixels" }
        input.rewind()
        for (v in gray) {
            var x = if (metadata.inputRange255) v else v / 255f
            x = (x - metadata.mean) / metadata.std
            for (c in 0 until channels) {
                when (inType) {
                    DataType.FLOAT32 -> input.putFloat(x)
                    DataType.INT8 -> {
                        val qv = (x / inScale).roundToInt() + inZero
                        input.put(qv.coerceIn(-128, 127).toByte())
                    }
                    DataType.UINT8 -> {
                        val qv = (x / inScale).roundToInt() + inZero
                        input.put(qv.coerceIn(0, 255).toByte())
                    }
                    else -> throw IllegalStateException("Tipo de entrada não suportado: $inType")
                }
            }
        }
        input.rewind()
        outputBuf.rewind()
        interpreter.run(input, outputBuf)
        outputBuf.rewind()
        val y = FloatArray(outputSize) {
            when (outType) {
                DataType.FLOAT32 -> outputBuf.getFloat()
                DataType.INT8 -> (outputBuf.get().toInt() - outZero) * outScale
                DataType.UINT8 -> ((outputBuf.get().toInt() and 0xFF) - outZero) * outScale
                else -> throw IllegalStateException("Tipo de saída não suportado: $outType")
            }
        }
        return if (outputSize == 1) y[0].coerceIn(0f, 1f)
        else y[metadata.openIndex.coerceIn(0, outputSize - 1)].coerceIn(0f, 1f)
    }

    override fun close() = interpreter.close()

    companion object {
        const val ASSET_INT8 = "eye_state_int8.tflite"
        const val ASSET_FP32 = "eye_state_fp32.tflite"
    }
}
