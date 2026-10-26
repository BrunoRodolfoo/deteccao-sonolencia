package br.pucpr.sonolencia

/**
 * Classificador temporal sobre a sequência das últimas 30 piscadas normalizadas.
 * Implementado no Android por [BlinkTemporalModel] (TFLite); nos testes, por um fake.
 */
interface TemporalClassifier {
    /**
     * @param window matriz [30][4] (piscadas x características, ordem Freq/Amp/Dur/Vel),
     *               preenchida com zeros à esquerda quando há menos de 30 piscadas
     *               (como `unroll_in_time` em Preprocessing.py).
     * @return probabilidades das 3 classes [alerta, pouca vigilância, sonolento]
     */
    fun predict(window: Array<FloatArray>): FloatArray
}

enum class EnginePhase { IDLE, THRESHOLD_CALIB, BLINK_CALIB, RUNNING }

enum class AlertType { EYES_CLOSED, PERCLOS, DROWSY }

data class EngineConfig(
    /** Quadros com rosto usados para estimar o EAR de olho aberto (mediana). */
    val thresholdCalibFrames: Int = 90,
    /** Limiar de piscada = fator x mediana do EAR aberto. */
    val thresholdFactor: Float = 0.75f,
    /** Nº de piscadas em estado alerta usadas para o z-score por sujeito. */
    val calibBlinks: Int = 20,
    /** Tempo máximo da calibração de piscadas (ms); ao expirar, usa o que tiver (>= minCalibBlinks). */
    val calibMaxMs: Long = 120_000L,
    val minCalibBlinks: Int = 5,
    val windowSize: Int = 30,
    val minBlinkFrames: Int = 2,
    val maxBlinkDurationMs: Long = 1500L,
    val perclosWindowMs: Long = 60_000L,
    /** Cobertura mínima da janela antes de disparar o alerta de PERCLOS. */
    val perclosMinCoverageMs: Long = 10_000L,
    val perclosThreshold: Float = 0.15f,
    val eyesClosedAlertMs: Long = 1000L,
    /** Suavização exponencial das probabilidades: p = alpha*novo + (1-alpha)*anterior. */
    val smoothingAlpha: Float = 0.3f,
    /** Janelas consecutivas com classe "sonolento" para disparar o alerta. */
    val drowsySustainedWindows: Int = 3,
)

/** Saída do motor para um quadro. */
data class EngineOutput(
    val phase: EnginePhase,
    val faceDetected: Boolean,
    val ear: Float,
    val mar: Float,
    val threshold: Float?,
    val closed: Boolean,
    val closedMs: Long,
    val perclos: Float,
    val blinkCount: Long,
    val sequenceLength: Int,
    val calibFramesDone: Int,
    val calibBlinksDone: Int,
    val calibElapsedMs: Long,
    /** Probabilidades suavizadas [alerta, pouca vigilância, sonolento] ou null. */
    val probabilities: FloatArray?,
    /** 0 alerta, 1 pouca vigilância, 2 sonolento, -1 indefinido. */
    val classIndex: Int,
    val alerts: Set<AlertType>,
    /** Piscada concluída neste quadro (características brutas), se houver. */
    val newBlink: BlinkFeatures?,
)

/**
 * Motor de detecção de sonolência: lógica pura (sem Android), testável na JVM.
 *
 * Fluxo: [start] -> THRESHOLD_CALIB (mediana do EAR aberto, ~3 s) -> BLINK_CALIB
 * (primeiras N piscadas em estado alerta => média/desvio por característica) -> RUNNING
 * (cada nova piscada normalizada entra no buffer de 30 e dispara o classificador temporal).
 * Os alertas de olhos fechados e PERCLOS funcionam desde que o limiar exista.
 */
class DrowsinessEngine(
    val config: EngineConfig = EngineConfig(),
    private val classifier: TemporalClassifier? = null,
) {
    var phase: EnginePhase = EnginePhase.IDLE
        private set

    private val detector = BlinkDetector(config.minBlinkFrames, config.maxBlinkDurationMs)
    private val perclos = PerclosWindow(config.perclosWindowMs)
    private val calibEars = ArrayList<Float>()
    private val calibFeatures = ArrayList<FloatArray>()
    private var normalizer: FeatureNormalizer? = null
    private val sequence = ArrayList<FloatArray>()
    private var frameIndex = 0L
    private var blinkCount = 0L
    private var calibStartMs = 0L
    private var closedSinceMs: Long? = null
    private var smoothed: FloatArray? = null
    private var drowsyStreak = 0
    private var lastProbs: FloatArray? = null

    val threshold: Float? get() = detector.threshold
    val blinkTotal: Long get() = blinkCount
    val normalizerOrNull: FeatureNormalizer? get() = normalizer

    /** Inicia (ou reinicia) a calibração completa. */
    fun start() {
        reset()
        phase = EnginePhase.THRESHOLD_CALIB
    }

    /** Refaz apenas a calibração das piscadas (mantém o limiar de EAR). */
    fun recalibrateBlinks(nowMs: Long) {
        if (detector.threshold == null) {
            start(); return
        }
        calibFeatures.clear()
        sequence.clear()
        normalizer = null
        smoothed = null
        lastProbs = null
        drowsyStreak = 0
        blinkCount = 0
        frameIndex = 0
        calibStartMs = nowMs
        phase = EnginePhase.BLINK_CALIB
    }

    fun stop() {
        reset()
        phase = EnginePhase.IDLE
    }

    private fun reset() {
        detector.threshold = null
        detector.reset()
        perclos.reset()
        calibEars.clear()
        calibFeatures.clear()
        normalizer = null
        sequence.clear()
        frameIndex = 0
        blinkCount = 0
        calibStartMs = 0
        closedSinceMs = null
        smoothed = null
        lastProbs = null
        drowsyStreak = 0
    }

    /** Quadro sem rosto detectado: mantém estado, não conta quadro. */
    fun processNoFace(timeMs: Long): EngineOutput {
        closedSinceMs = null
        detector.reset()
        return output(false, 0f, 0f, timeMs, null, emptySet())
    }

    /**
     * Processa um quadro com rosto.
     * @param timeMs relógio monotônico em ms (SystemClock.uptimeMillis no Android)
     */
    fun processFrame(timeMs: Long, leftEar: Float, rightEar: Float, mar: Float): EngineOutput {
        if (phase == EnginePhase.IDLE) {
            return output(true, (leftEar + rightEar) / 2f, mar, timeMs, null, emptySet())
        }
        val ear = (leftEar + rightEar) / 2f
        var newBlink: BlinkFeatures? = null

        if (phase == EnginePhase.THRESHOLD_CALIB) {
            calibEars.add(ear)
            if (calibEars.size >= config.thresholdCalibFrames) {
                detector.threshold = config.thresholdFactor * median(calibEars)
                phase = EnginePhase.BLINK_CALIB
                calibStartMs = timeMs
                frameIndex = 0
            }
            return output(true, ear, mar, timeMs, null, emptySet())
        }

        // BLINK_CALIB ou RUNNING: limiar definido.
        val blink = detector.update(frameIndex, timeMs, ear)
        val closed = detector.isClosed
        frameIndex++

        // Olhos fechados contínuos
        if (closed) {
            if (closedSinceMs == null) closedSinceMs = timeMs
        } else {
            closedSinceMs = null
        }
        perclos.update(timeMs, closed)

        if (blink != null) {
            blinkCount++
            val feats = BlinkFeatures.fromBlink(blink, blinkCount, blink.endFrame + 1)
            newBlink = feats
            onBlink(feats, timeMs)
        }
        if (phase == EnginePhase.BLINK_CALIB) {
            val elapsed = timeMs - calibStartMs
            val enough = calibFeatures.size >= config.calibBlinks ||
                (elapsed >= config.calibMaxMs && calibFeatures.size >= config.minCalibBlinks)
            if (enough) finishCalibration()
        }

        val alerts = HashSet<AlertType>()
        val closedMs = closedSinceMs?.let { timeMs - it } ?: 0L
        if (closedMs >= config.eyesClosedAlertMs) alerts.add(AlertType.EYES_CLOSED)
        if (perclos.coverageMs >= config.perclosMinCoverageMs && perclos.value > config.perclosThreshold) {
            alerts.add(AlertType.PERCLOS)
        }
        if (drowsyStreak >= config.drowsySustainedWindows) alerts.add(AlertType.DROWSY)

        return output(true, ear, mar, timeMs, newBlink, alerts)
    }

    private fun onBlink(feats: BlinkFeatures, timeMs: Long) {
        val raw = feats.toArray()
        when (phase) {
            EnginePhase.BLINK_CALIB -> calibFeatures.add(raw)
            EnginePhase.RUNNING -> {
                val norm = normalizer ?: return
                pushSequence(norm.normalize(raw))
                runClassifier()
            }
            else -> Unit
        }
    }

    private fun finishCalibration() {
        val norm = FeatureNormalizer.fit(calibFeatures)
        normalizer = norm
        // As piscadas de calibração (estado alerta) também entram na sequência, como no
        // protocolo do UTA-RLDD, em que o terço de calibração do vídeo de alerta é normalizado
        // com os próprios parâmetros.
        sequence.clear()
        for (f in calibFeatures) pushSequence(norm.normalize(f))
        phase = EnginePhase.RUNNING
        runClassifier()
    }

    private fun pushSequence(x: FloatArray) {
        sequence.add(x)
        while (sequence.size > config.windowSize) sequence.removeAt(0)
    }

    /** Janela [30][4] com zeros à esquerda quando há menos de 30 piscadas. */
    fun currentWindow(): Array<FloatArray> {
        val w = Array(config.windowSize) { FloatArray(BlinkFeatures.SIZE) }
        val offset = config.windowSize - sequence.size
        for (i in sequence.indices) w[offset + i] = sequence[i].copyOf()
        return w
    }

    private fun runClassifier() {
        val clf = classifier ?: return
        if (sequence.isEmpty()) return
        val p = clf.predict(currentWindow())
        require(p.size == 3) { "Classificador deve devolver 3 probabilidades" }
        lastProbs = p
        val s = smoothed
        smoothed = if (s == null) p.copyOf() else FloatArray(3) { i ->
            config.smoothingAlpha * p[i] + (1f - config.smoothingAlpha) * s[i]
        }
        val cls = argmax(smoothed!!)
        drowsyStreak = if (cls == 2) drowsyStreak + 1 else 0
    }

    private fun output(
        face: Boolean, ear: Float, mar: Float, timeMs: Long,
        newBlink: BlinkFeatures?, alerts: Set<AlertType>,
    ): EngineOutput {
        val probs = smoothed?.copyOf()
        return EngineOutput(
            phase = phase,
            faceDetected = face,
            ear = ear,
            mar = mar,
            threshold = detector.threshold,
            closed = face && detector.isClosed,
            closedMs = closedSinceMs?.let { timeMs - it } ?: 0L,
            perclos = perclos.value,
            blinkCount = blinkCount,
            sequenceLength = sequence.size,
            calibFramesDone = calibEars.size,
            calibBlinksDone = calibFeatures.size,
            calibElapsedMs = if (phase == EnginePhase.BLINK_CALIB) timeMs - calibStartMs else 0L,
            probabilities = probs,
            classIndex = probs?.let { argmax(it) } ?: -1,
            alerts = alerts,
            newBlink = newBlink,
        )
    }

    companion object {
        val CLASS_NAMES = arrayOf("alerta", "pouca_vigilancia", "sonolento")

        fun argmax(a: FloatArray): Int {
            var best = 0
            for (i in 1 until a.size) if (a[i] > a[best]) best = i
            return best
        }

        fun median(values: List<Float>): Float {
            require(values.isNotEmpty())
            val s = values.sorted()
            val n = s.size
            return if (n % 2 == 1) s[n / 2] else (s[n / 2 - 1] + s[n / 2]) / 2f
        }
    }
}
