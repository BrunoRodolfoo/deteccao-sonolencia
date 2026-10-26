package br.pucpr.sonolencia

/**
 * Uma piscada detectada. Índices de quadro são contadores globais de quadros com rosto
 * (0-based); tempos em milissegundos.
 */
data class Blink(
    val startFrame: Long,
    val bottomFrame: Long,
    val endFrame: Long,
    val earStart: Float,
    val earBottom: Float,
    val earEnd: Float,
    val startTimeMs: Long,
    val endTimeMs: Long,
) {
    /** Duração em quadros: end - start + 1 (Ghoddoosian et al., 2019, eq. 2). */
    val durationFrames: Long get() = endFrame - startFrame + 1
    val durationMs: Long get() = endTimeMs - startTimeMs
}

/**
 * Detector de piscadas por limiar adaptativo sobre o EAR médio dos dois olhos.
 *
 * - limiar = 0.75 x mediana do EAR de olho aberto medido na calibração (ver
 *   [DrowsinessEngine]);
 * - piscada = sequência contígua de quadros com EAR < limiar;
 * - start = primeiro quadro abaixo, bottom = argmin do EAR, end = último quadro abaixo;
 * - duração mínima de [minFrames] quadros (descarta ruído de 1 quadro);
 * - duração máxima de [maxDurationMs] (fechamentos longos NÃO viram piscada: são tratados
 *   pelo alerta de olhos fechados).
 */
class BlinkDetector(
    private val minFrames: Int = 2,
    private val maxDurationMs: Long = 1500L,
) {
    var threshold: Float? = null

    private var inBlink = false
    private var overlong = false
    private var startFrame = 0L
    private var startEar = 0f
    private var startTime = 0L
    private var bottomFrame = 0L
    private var bottomEar = 0f
    private var lastFrame = 0L
    private var lastEar = 0f
    private var lastTime = 0L

    /** true enquanto o EAR está abaixo do limiar (olho considerado fechado). */
    val isClosed: Boolean get() = inBlink || overlong

    fun reset() {
        inBlink = false
        overlong = false
    }

    /**
     * Processa um quadro. Retorna a piscada concluída neste quadro (quando o olho reabre)
     * ou null.
     */
    fun update(frameIndex: Long, timeMs: Long, ear: Float): Blink? {
        val thr = threshold ?: return null
        if (ear < thr) {
            if (overlong) return null
            if (!inBlink) {
                inBlink = true
                startFrame = frameIndex
                startEar = ear
                startTime = timeMs
                bottomFrame = frameIndex
                bottomEar = ear
            } else if (ear < bottomEar) {
                bottomFrame = frameIndex
                bottomEar = ear
            }
            lastFrame = frameIndex
            lastEar = ear
            lastTime = timeMs
            if (timeMs - startTime > maxDurationMs) {
                // Fechamento longo: descarta como piscada, mas continua "fechado".
                inBlink = false
                overlong = true
            }
            return null
        }
        // EAR >= limiar: olho aberto.
        if (overlong) {
            overlong = false
            return null
        }
        if (!inBlink) return null
        inBlink = false
        val blink = Blink(
            startFrame = startFrame, bottomFrame = bottomFrame, endFrame = lastFrame,
            earStart = startEar, earBottom = bottomEar, earEnd = lastEar,
            startTimeMs = startTime, endTimeMs = lastTime,
        )
        if (blink.durationFrames < minFrames) return null
        if (blink.durationMs > maxDurationMs) return null
        return blink
    }
}

/**
 * As 4 características por piscada de Ghoddoosian et al. (CVPRW 2019), na ORDEM usada
 * pelas features do UTA-RLDD (Preprocessing.py, normalize_blinks):
 * [0]=Frequency, [1]=Amplitude, [2]=Duration, [3]=Eye-opening Velocity.
 */
data class BlinkFeatures(
    val frequency: Float,
    val amplitude: Float,
    val duration: Float,
    val velocity: Float,
) {
    fun toArray() = floatArrayOf(frequency, amplitude, duration, velocity)

    companion object {
        const val SIZE = 4

        /**
         * @param blink piscada i
         * @param blinkCountUpToThis número de piscadas até (e incluindo) a piscada i
         * @param framesUntilEnd número de quadros processados até end_i (end_i + 1, 0-based)
         *
         * Duration_i = end_i - start_i + 1
         * Amplitude_i = (EAR[start_i] - 2 EAR[bottom_i] + EAR[end_i]) / 2
         * Velocity_i = (EAR[end_i] - EAR[bottom_i]) / (end_i - bottom_i)
         *   (denominador com piso 1 quadro quando end_i == bottom_i, como em blink_video.py)
         * Frequency_i = 100 x (piscadas até i) / (quadros até end_i)
         */
        fun fromBlink(blink: Blink, blinkCountUpToThis: Long, framesUntilEnd: Long): BlinkFeatures {
            val dur = blink.durationFrames.toFloat()
            val amp = (blink.earStart - 2f * blink.earBottom + blink.earEnd) / 2f
            val denom = maxOf(blink.endFrame - blink.bottomFrame, 1L).toFloat()
            val vel = (blink.earEnd - blink.earBottom) / denom
            val freq = 100f * blinkCountUpToThis.toFloat() / maxOf(framesUntilEnd, 1L).toFloat()
            return BlinkFeatures(freq, amp, dur, vel)
        }
    }
}

/**
 * Normalização por sujeito (z-score) de cada característica, com média e desvio obtidos
 * das piscadas de calibração em estado alerta (equivalente ao "terço do vídeo de alerta"
 * do protocolo do UTA-RLDD).
 */
class FeatureNormalizer(
    val mean: FloatArray,
    val std: FloatArray,
) {
    init {
        require(mean.size == BlinkFeatures.SIZE && std.size == BlinkFeatures.SIZE)
    }

    fun normalize(f: FloatArray): FloatArray =
        FloatArray(BlinkFeatures.SIZE) { i -> (f[i] - mean[i]) / std[i] }

    companion object {
        const val STD_FLOOR = 1e-6f

        /** Estatísticas populacionais (ddof=0, como numpy.std) com piso no desvio. */
        fun fit(samples: List<FloatArray>): FeatureNormalizer {
            require(samples.isNotEmpty()) { "Sem amostras para calibrar" }
            val n = samples.size.toFloat()
            val mean = FloatArray(BlinkFeatures.SIZE)
            val std = FloatArray(BlinkFeatures.SIZE)
            for (i in 0 until BlinkFeatures.SIZE) {
                var s = 0f
                for (x in samples) s += x[i]
                mean[i] = s / n
                var v = 0f
                for (x in samples) {
                    val d = x[i] - mean[i]
                    v += d * d
                }
                std[i] = maxOf(kotlin.math.sqrt(v / n), STD_FLOOR)
            }
            return FeatureNormalizer(mean, std)
        }
    }
}
